"""Safe, versioned VietToxic state export/import tooling.

The default CLI behavior is read-only.  Export always snapshots SQLite before
portable-reference conversion, and import validates an entire directory bundle
before preparing any target staging paths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from backend.artifact_refs import inspect_artifact_ref
from backend.mlflow_legacy_export import (
    ARTIFACTS_FILENAME,
    CHECKSUMS_FILENAME as LEGACY_CHECKSUMS_FILENAME,
    INVENTORY_FILENAME,
)
from backend.portability_migration import DB_FIELDS, LOG_PATH_KEYS, ROW_KEYS, apply_migration
from backend.runtime_paths import (
    get_data_dir,
    get_feedback_db_path,
    get_mlflow_artifact_root,
    get_mlflow_db_path,
    get_mlflow_evidence_dir,
    get_model_options_dir,
    get_runtime_dir,
)


SCHEMA_VERSION = 1
PROJECT = "viettoxic"
CHECKSUMS_FILENAME = "checksums.sha256"
MANIFEST_FILENAME = "manifest.json"
ALLOWED_COMPONENTS = {
    "application",
    "models",
    "persistent_artifacts",
    "active_mlflow",
    "legacy_mlflow_evidence",
    "metadata",
}
LEGACY_EVIDENCE_FILES = (INVENTORY_FILENAME, ARTIFACTS_FILENAME, LEGACY_CHECKSUMS_FILENAME)
REQUIRED_APPLICATION_TABLES = {
    "system_setting",
    "mlflow_comment_item",
    "mlflow_training_artifact",
    "mlflow_model_version",
    "mlflow_production_slot",
    "mlflow_do_run",
    "training_tracker_phase",
    "training_tracker_task",
}
COUNT_TABLES = (
    "feedback_page",
    "feedback_segment",
    "synthetic_dataset_row",
    "system_setting",
    "mlflow_crawl_batch",
    "mlflow_comment_item",
    "mlflow_comment_prediction",
    "mlflow_training_artifact",
    "mlflow_model_version",
    "mlflow_production_slot",
    "mlflow_promotion_event",
    "mlflow_automation_state",
    "mlflow_automation_event",
    "mlflow_do_run",
    "mlflow_kaggle_ingestion",
    "training_tracker_phase",
    "training_tracker_group",
    "training_tracker_task",
    "training_tracker_result",
)
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
MACHINE_PATH_RE = re.compile(r"(?:^[A-Za-z]:[\\/]|^/Users/|^file:(?://)?(?:/[A-Za-z]:|/Users/))", re.IGNORECASE)
PORTABLE_REFERENCE_PREFIXES = ("data://", "runtime://", "model://")


class StateBundleError(RuntimeError):
    """Raised when state cannot be bundled or installed safely."""


@dataclass(frozen=True)
class SourcePaths:
    data_dir: Path
    runtime_dir: Path
    model_options_dir: Path
    feedback_db: Path
    active_mlflow_db: Path
    active_mlflow_artifacts: Path
    legacy_evidence_dir: Path

    @property
    def artifact_roots(self) -> dict[str, Path]:
        return {
            "data": self.data_dir,
            "runtime": self.runtime_dir,
            "model": self.model_options_dir,
        }


@dataclass(frozen=True)
class TargetPaths:
    data_dir: Path
    runtime_dir: Path
    model_options_dir: Path

    @property
    def feedback_db(self) -> Path:
        return self.data_dir / "processed" / "feedback" / "feedback.db"

    @property
    def mlflow_dir(self) -> Path:
        return self.data_dir / "mlflow"

    @property
    def artifact_roots(self) -> dict[str, Path]:
        return {
            "data": self.data_dir,
            "runtime": self.runtime_dir,
            "model": self.model_options_dir,
        }


def resolve_source_paths() -> SourcePaths:
    return SourcePaths(
        data_dir=get_data_dir(),
        runtime_dir=get_runtime_dir(),
        model_options_dir=get_model_options_dir(),
        feedback_db=get_feedback_db_path(),
        active_mlflow_db=get_mlflow_db_path(),
        active_mlflow_artifacts=get_mlflow_artifact_root(),
        legacy_evidence_dir=get_mlflow_evidence_dir(),
    )


def resolve_target_paths(
    *,
    data_dir: Path | None = None,
    runtime_dir: Path | None = None,
    model_options_dir: Path | None = None,
) -> TargetPaths:
    return TargetPaths(
        data_dir=(data_dir or get_data_dir()).expanduser().resolve(),
        runtime_dir=(runtime_dir or get_runtime_dir()).expanduser().resolve(),
        model_options_dir=(model_options_dir or get_model_options_dir()).expanduser().resolve(),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _safe_display_reference(value: object) -> str:
    """Return a diagnostic-only reference value that cannot disclose URI credentials.

    Application references remain unchanged in the private SQLite payload.  This
    function is only for CLI reports and non-DB bundle metadata.
    """
    raw = str(value or "").strip()
    if not raw or raw.startswith(PORTABLE_REFERENCE_PREFIXES):
        return raw
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return "<redacted>"
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        if not parsed.hostname:
            return "<redacted>"
        try:
            host = parsed.hostname
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
        except ValueError:
            return "<redacted>"
        netloc = f"<redacted>@{host}" if parsed.username is not None or parsed.password is not None else host
        display = urlunsplit((scheme, netloc, parsed.path, "", ""))
        return f"{display}?<redacted>" if parsed.query or parsed.fragment else display
    if scheme and "://" in raw:
        return f"{scheme}://<redacted>"
    return raw


def _safe_reference_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a reference record for diagnostics without retaining a raw URI."""
    safe = dict(record)
    display = _safe_display_reference(record.get("logical_reference", record.get("reference", "")))
    safe["display_value"] = display
    for key in ("reference", "logical_reference"):
        if key in safe:
            safe[key] = display
    return safe


def _safe_reference_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_safe_reference_record(record) for record in records]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
        or value.startswith(("/", "//"))
    ):
        raise StateBundleError(f"Unsafe bundle-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StateBundleError(f"Unsafe bundle-relative path: {value!r}")
    return path


def _bundle_path(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    candidate = root.joinpath(*safe.parts)
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise StateBundleError(f"Bundle path escapes root: {relative!r}") from exc
    return candidate


def _reject_symlink(path: Path, *, label: str) -> None:
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    if path.is_symlink() or is_junction:
        raise StateBundleError(f"Symlinks/junctions are not allowed in {label}: {path}")


def _walk_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    _reject_symlink(root, label="state component")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in list(directories):
            _reject_symlink(current_path / directory, label="state component")
        for name in names:
            path = current_path / name
            _reject_symlink(path, label="state component")
            if not path.is_file():
                raise StateBundleError(f"Unsupported non-file state member: {path}")
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _copy_path(source: Path, destination: Path) -> None:
    _reject_symlink(source, label="source state")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    if not source.is_dir():
        raise StateBundleError(f"Referenced state is neither a file nor directory: {source}")
    if destination.exists():
        raise StateBundleError(f"Duplicate bundle destination: {destination}")
    destination.mkdir(parents=True)
    for file_path in _walk_files(source):
        relative = file_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)


def _sqlite_uri(path: Path, *, immutable: bool = False) -> str:
    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    return path.expanduser().resolve(strict=True).as_uri() + suffix


def _sqlite_integrity(path: Path, *, immutable: bool = False) -> str:
    try:
        with closing(sqlite3.connect(_sqlite_uri(path, immutable=immutable), uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            row = connection.execute("PRAGMA integrity_check").fetchone()
    except (OSError, sqlite3.Error) as exc:
        raise StateBundleError(f"SQLite integrity check failed for {path}: {type(exc).__name__}") from exc
    result = str(row[0]) if row else "no result"
    if result.lower() != "ok":
        raise StateBundleError(f"SQLite integrity check failed for {path}: {result}")
    return result


def _sqlite_snapshot(source: Path, destination: Path) -> None:
    _sqlite_integrity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise StateBundleError(f"Snapshot destination already exists: {destination}")
    try:
        with closing(sqlite3.connect(_sqlite_uri(source), uri=True)) as source_connection, closing(
            sqlite3.connect(destination)
        ) as target:
            source_connection.execute("PRAGMA query_only = ON")
            source_connection.backup(target)
            target.commit()
    except sqlite3.Error as exc:
        raise StateBundleError(f"SQLite backup failed for {source}: {type(exc).__name__}") from exc
    _sqlite_integrity(destination, immutable=True)


def _sqlite_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        with closing(sqlite3.connect(_sqlite_uri(path), uri=True)) as source, closing(
            sqlite3.connect(":memory:")
        ) as snapshot:
            source.execute("PRAGMA query_only = ON")
            row = source.execute("PRAGMA integrity_check").fetchone()
            if not row or str(row[0]).lower() != "ok":
                raise StateBundleError(f"SQLite integrity check failed for {path}")
            source.backup(snapshot)
            if hasattr(snapshot, "serialize"):
                digest = hashlib.sha256(snapshot.serialize()).hexdigest()
            else:  # pragma: no cover - Python 3.11+ supplies serialize
                digest = hashlib.sha256("\n".join(snapshot.iterdump()).encode("utf-8")).hexdigest()
    except sqlite3.Error as exc:
        raise StateBundleError(f"Could not fingerprint SQLite database {path}: {type(exc).__name__}") from exc
    return {
        "integrity": "ok",
        "logical_sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }


def _sqlite_schema(path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(_sqlite_uri(path, immutable=True), uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables: dict[str, Any] = {}
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            name = str(row["name"])
            columns = [
                {
                    "name": str(item[1]),
                    "type": str(item[2]),
                    "not_null": bool(item[3]),
                    "primary_key": int(item[5]),
                }
                for item in connection.execute(f'PRAGMA table_info("{name}")')
            ]
            indexes = [
                str(item[1])
                for item in connection.execute(f'PRAGMA index_list("{name}")')
                if item[1]
            ]
            tables[name] = {"columns": columns, "indexes": sorted(indexes)}
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    return {"user_version": user_version, "tables": tables}


def _table_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(_sqlite_uri(path, immutable=True), uri=True)) as connection:
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in COUNT_TABLES:
            if table in tables:
                counts[table] = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        if "mlflow_comment_item" in tables:
            counts["mlflow_comment_item.selected_for_training"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM mlflow_comment_item WHERE COALESCE(selected_for_training, 1) = 1"
                ).fetchone()[0]
            )
    return counts


def _application_analysis(path: Path) -> dict[str, Any]:
    _sqlite_integrity(path, immutable=True)
    schema = _sqlite_schema(path)
    table_names = set(schema["tables"])
    missing_required = sorted(REQUIRED_APPLICATION_TABLES - table_names)
    automation_enabled = False
    sensitive = "unknown"
    with closing(sqlite3.connect(_sqlite_uri(path, immutable=True), uri=True)) as connection:
        if "system_setting" in table_names:
            rows = connection.execute("SELECT key FROM system_setting").fetchall()
            names = [str(row[0]) for row in rows]
            # Values are intentionally never read into the report. Any persisted
            # setting makes the application snapshot private migration material.
            sensitive = bool(names)
            automation = connection.execute(
                "SELECT value FROM system_setting WHERE key='MLFLOW_AUTOMATION_ENABLED'"
            ).fetchone()
            automation_enabled = bool(
                automation and str(automation[0]).strip().lower() in {"1", "true", "yes", "on"}
            )
    return {
        "integrity": "ok",
        "schema": schema,
        "missing_required_tables": missing_required,
        "state_counts": _table_counts(path),
        "contains_sensitive_application_state": sensitive,
        "automation_enabled": automation_enabled,
    }


def _iter_path_references(db_path: Path, *, roots: Mapping[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with closing(sqlite3.connect(_sqlite_uri(db_path, immutable=True), uri=True)) as connection:
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, fields in DB_FIELDS.items():
            if table not in tables:
                continue
            key = ROW_KEYS[table]
            for row in connection.execute(f'SELECT "{key}", {", ".join(fields)} FROM "{table}"'):
                for index, field in enumerate(fields, start=1):
                    if row[index] is not None and str(row[index]).strip():
                        records.append(
                            {"table": table, "row_identity": str(row[0]), "field": field, "reference": str(row[index])}
                        )
            if table == "mlflow_do_run":
                for row in connection.execute("SELECT run_id, logs_json FROM mlflow_do_run WHERE logs_json IS NOT NULL"):
                    try:
                        payload = json.loads(str(row[1]))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, list):
                        continue
                    for item_index, item in enumerate(payload):
                        if not isinstance(item, dict):
                            continue
                        for field in sorted(LOG_PATH_KEYS.intersection(item)):
                            value = item.get(field)
                            if isinstance(value, str) and value.strip():
                                records.append(
                                    {
                                        "table": table,
                                        "row_identity": str(row[0]),
                                        "field": f"logs_json[{item_index}].{field}",
                                        "reference": value,
                                    }
                                )

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        identity = (record["table"], record["row_identity"], record["field"], record["reference"])
        if identity in seen:
            continue
        seen.add(identity)
        inspected = inspect_artifact_ref(record["reference"], roots=roots)
        classified = dict(record)
        classified["classification"] = inspected.classification
        classified["error_code"] = None
        classified["persistence"] = "external"
        classified["exists"] = inspected.path.exists() if inspected.path is not None else None
        classified["logical_reference"] = record["reference"]
        if inspected.classification == "portable":
            scheme, suffix = record["reference"].split("://", 1)
            if scheme == "model":
                classified["persistence"] = "model-component"
            elif scheme == "runtime" and suffix.startswith("model_registry/"):
                classified["persistence"] = "persistent-required"
            elif scheme == "runtime":
                classified["persistence"] = "ephemeral" if suffix.startswith("kaggle_real_jobs/") else "persistent-optional"
            elif scheme == "data" and suffix.startswith("mlflow/"):
                classified["persistence"] = "active-mlflow-component"
            else:
                classified["persistence"] = (
                    "persistent-required"
                    if record["table"] in {"mlflow_training_artifact", "mlflow_model_version"}
                    else "persistent-optional"
                )
            if classified["exists"] is False and classified["persistence"] == "persistent-required":
                classified["error_code"] = "MISSING_REFERENCED_ARTIFACT"
        elif inspected.classification in {"external_absolute", "relative_unmanaged"}:
            classified["error_code"] = "EXTERNAL_REFERENCE_REQUIRES_REVIEW"
        elif inspected.classification == "invalid":
            classified["error_code"] = "INVALID_REFERENCE"
        result.append(classified)
    return result


def _stale_managed_references(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return references that still contain managed machine-specific paths.

    Classification is authoritative here. Protected URIs and external paths
    may look like machine paths, but they are intentionally opaque values and
    must not be rewritten or rejected as stale managed state.
    """
    managed_classifications = {
        "managed_absolute",
        "legacy_managed_absolute",
        "legacy_managed_relative",
    }
    return [item for item in references if item["classification"] in managed_classifications]


def _model_inventory(root: Path, *, include_hashes: bool) -> dict[str, Any]:
    if not root.exists():
        return {"status": "missing", "file_count": 0, "total_bytes": 0, "files": [], "contracts": []}
    files: list[dict[str, Any]] = []
    total = 0
    for path in _walk_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        record: dict[str, Any] = {
            "relative_path": relative,
            "size_bytes": size,
            "model_family": relative.split("/", 1)[0] if "/" in relative else "unknown",
            "logical_role": "model_artifact",
        }
        if include_hashes:
            record["sha256"] = _sha256_file(path)
        files.append(record)
        total += size
    return {
        "status": "present",
        "preservation": "full",
        "hashes_computed": include_hashes,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
        "contracts": _verify_model_contracts(root),
    }


def _verify_model_contracts(root: Path) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    if not root.exists():
        return contracts
    for family in sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()):
        for model_dir in sorted(path for path in family.iterdir() if path.is_dir() and not path.is_symlink()):
            names = {item.name for item in model_dir.iterdir() if item.is_file() and not item.is_symlink()}
            if family.name == "tfidf_lr":
                valid = {"vectorizer.pkl", "model_lr.pkl"}.issubset(names)
                requirement = "vectorizer.pkl + model_lr.pkl"
            elif family.name == "phobert":
                valid = "config.json" in names and bool({"model.safetensors", "pytorch_model.bin"}.intersection(names))
                requirement = "config.json + model.safetensors|pytorch_model.bin"
            else:
                continue
            contracts.append(
                {
                    "model": model_dir.relative_to(root).as_posix(),
                    "model_family": family.name,
                    "requirement": requirement,
                    "valid": valid,
                }
            )
    return contracts


def _tree_signature(root: Path) -> list[tuple[str, int, int]]:
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in _walk_files(root)
    ] if root.exists() else []


def _active_mlflow_analysis(db_path: Path, artifact_root: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "status": "not_initialized",
            "database_present": False,
            "artifact_tree_present": artifact_root.exists(),
            "artifact_file_count": len(_walk_files(artifact_root)) if artifact_root.exists() else 0,
        }
    _sqlite_integrity(db_path)
    unsafe: list[dict[str, str]] = []
    running: list[str] = []
    alembic_revision = None
    with closing(sqlite3.connect(_sqlite_uri(db_path), uri=True)) as connection:
        connection.execute("PRAGMA query_only = ON")
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "runs" in tables:
            running = [str(row[0]) for row in connection.execute("SELECT run_uuid FROM runs WHERE upper(status)='RUNNING'")]
            for run_id, uri in connection.execute("SELECT run_uuid, artifact_uri FROM runs WHERE artifact_uri IS NOT NULL"):
                value = str(uri)
                if MACHINE_PATH_RE.search(value) or ("://" not in value and not value.startswith("mlflow-artifacts:/")):
                    unsafe.append(
                        {
                            "kind": "run",
                            "identity": str(run_id),
                            "reference": _safe_display_reference(value),
                            "display_value": _safe_display_reference(value),
                        }
                    )
        if "experiments" in tables:
            for experiment_id, uri in connection.execute(
                "SELECT experiment_id, artifact_location FROM experiments WHERE artifact_location IS NOT NULL"
            ):
                value = str(uri)
                if MACHINE_PATH_RE.search(value) or ("://" not in value and not value.startswith("mlflow-artifacts:/")):
                    unsafe.append(
                        {
                            "kind": "experiment",
                            "identity": str(experiment_id),
                            "reference": _safe_display_reference(value),
                            "display_value": _safe_display_reference(value),
                        }
                    )
        if "alembic_version" in tables:
            row = connection.execute("SELECT version_num FROM alembic_version LIMIT 1").fetchone()
            alembic_revision = str(row[0]) if row else None
    missing_tables = sorted({"experiments", "runs"} - tables)
    status = "unsafe" if unsafe or running or missing_tables or not artifact_root.is_dir() else "ready"
    return {
        "status": status,
        "database_present": True,
        "database_integrity": "ok",
        "alembic_revision": alembic_revision,
        "running_run_ids": running,
        "unsafe_references": unsafe,
        "missing_required_tables": missing_tables,
        "artifact_tree_present": artifact_root.exists(),
        "artifact_file_count": len(_walk_files(artifact_root)) if artifact_root.exists() else 0,
    }


def _environment_requirements() -> dict[str, Any]:
    from backend.system_settings import SETTING_DEFINITIONS

    return {
        "required_on_target": [
            {"name": "VIETTOXIC_ADMIN_USERNAME", "description": "Backend administrator username", "secret": False},
            {"name": "VIETTOXIC_ADMIN_PASSWORD", "description": "Backend administrator credential", "secret": True},
            {"name": "VIETTOXIC_ADMIN_SESSION_SECRET", "description": "Backend session signing secret", "secret": True},
        ],
        "required_for_kaggle": [
            {"name": "KAGGLE_USERNAME", "description": "Kaggle account name", "secret": False},
            {"name": "KAGGLE_KEY", "description": "Kaggle API credential", "secret": True},
            {"name": "KAGGLE_API_TOKEN", "description": "Kaggle API token for current CLI releases", "secret": True},
        ],
        "required_for_gemini": [
            {"name": "GEMINI_API_KEY", "description": "Gemini API credential", "secret": True}
        ],
        "optional_path_overrides": [
            {"name": "APP_DATA_DIR", "default": "<project>/data"},
            {"name": "APP_RUNTIME_DIR", "default": "<project>/.runtime"},
            {"name": "VIETTOXIC_MODEL_OPTIONS_DIR", "default": "<project>/models/options"},
            {"name": "MLFLOW_ARTIFACT_ROOT", "default": "APP_DATA_DIR/mlflow/artifacts"},
        ],
        "application_setting_names": [
            {
                "name": definition.key,
                "required": definition.required,
                "secret": definition.secret,
                "default_available": definition.default is not None and not definition.secret,
                "source": "feedback.db or target environment",
            }
            for definition in SETTING_DEFINITIONS
        ],
        "intentionally_excluded": [
            "backend/.env.local",
            "comprehensive_ui/.env.local",
            "environment values and secrets",
            "browser localStorage/sessionStorage/admin session token/theme",
        ],
    }


def _source_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _mlflow_version() -> str | None:
    try:
        return importlib.metadata.version("mlflow")
    except importlib.metadata.PackageNotFoundError:
        return None


def _component_for_path(relative: str) -> str:
    if relative.startswith("mlflow/legacy_evidence/"):
        return "legacy_mlflow_evidence"
    first = _safe_relative(relative).parts[0]
    return {
        "application": "application",
        "models": "models",
        "persistent_artifacts": "persistent_artifacts",
        "mlflow": "active_mlflow",
        "metadata": "metadata",
    }.get(first, "metadata")


def _file_records(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in _walk_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in {MANIFEST_FILENAME, CHECKSUMS_FILENAME}:
            continue
        result.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "component": _component_for_path(relative),
                "role": "state_file",
            }
        )
    return result


def _content_identity(file_records: Sequence[Mapping[str, Any]], state_counts: Mapping[str, int]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT,
        "files": [
            {key: item[key] for key in ("path", "size_bytes", "sha256", "component")}
            for item in file_records
            if not str(item["path"]).startswith("metadata/")
        ],
        "state_counts": dict(state_counts),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_checksums(root: Path) -> None:
    members = [path for path in _walk_files(root) if path.name != CHECKSUMS_FILENAME]
    lines = [f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in members]
    (root / CHECKSUMS_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _copy_persistent_artifacts(
    references: list[dict[str, Any]], *, roots: Mapping[str, Path], bundle_root: Path
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    destinations: set[str] = set()
    for reference in references:
        if reference["persistence"] != "persistent-required" or reference["exists"] is not True:
            continue
        logical = str(reference["logical_reference"])
        if "://" not in logical:
            continue
        scheme, suffix = logical.split("://", 1)
        if scheme not in {"data", "runtime"}:
            continue
        safe_suffix = _safe_relative(suffix)
        bundle_relative = PurePosixPath("persistent_artifacts", scheme, *safe_suffix.parts).as_posix()
        if bundle_relative in destinations:
            continue
        destinations.add(bundle_relative)
        source = roots[scheme].joinpath(*safe_suffix.parts)
        destination = _bundle_path(bundle_root, bundle_relative)
        _copy_path(source, destination)
        copied.append(
            {
                "logical_reference": logical,
                "bundle_path": bundle_relative,
                "source_kind": "directory" if source.is_dir() else "file",
                "table": reference["table"],
                "row_identity": reference["row_identity"],
                "field": reference["field"],
            }
        )
    return copied


def _copy_active_mlflow(paths: SourcePaths, destination: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis["status"] == "not_initialized":
        return analysis
    if analysis["status"] != "ready":
        raise StateBundleError("Active MLflow store is not safe to export")
    before_db = _sqlite_fingerprint(paths.active_mlflow_db)
    before_artifacts = _tree_signature(paths.active_mlflow_artifacts)
    db_destination = destination / "mlflow.db"
    _sqlite_snapshot(paths.active_mlflow_db, db_destination)
    if paths.active_mlflow_artifacts.exists():
        _copy_path(paths.active_mlflow_artifacts, destination / "artifacts")
    after_db = _sqlite_fingerprint(paths.active_mlflow_db)
    after_artifacts = _tree_signature(paths.active_mlflow_artifacts)
    if before_db != after_db or before_artifacts != after_artifacts:
        raise StateBundleError("Active MLflow changed during export; no bundle was activated")
    copied_analysis = _active_mlflow_analysis(db_destination, destination / "artifacts")
    if copied_analysis["status"] != "ready":
        raise StateBundleError("Exported active MLflow snapshot failed portability validation")
    return {
        **analysis,
        "source_unchanged": True,
        "snapshot_integrity": "ok",
        "snapshot_logical_sha256": _sqlite_fingerprint(db_destination)["logical_sha256"],
    }


def _prepare_feedback_snapshot(paths: SourcePaths, destination: Path) -> dict[str, Any]:
    before = _sqlite_fingerprint(paths.feedback_db)
    _sqlite_snapshot(paths.feedback_db, destination)
    snapshot_before_migration = _sqlite_fingerprint(destination)
    backup, migration_plan = apply_migration(destination, roots=paths.artifact_roots)
    backup.unlink(missing_ok=True)
    snapshot_after_migration = _sqlite_fingerprint(destination)
    application = _application_analysis(destination)
    after = _sqlite_fingerprint(paths.feedback_db)
    source_changed = before != after
    return {
        "source_before": before,
        "source_after": after,
        "source_changed_during_export": source_changed,
        "snapshot_semantics": "point-in-time SQLite backup",
        "snapshot_before_migration": snapshot_before_migration,
        "snapshot_after_migration": snapshot_after_migration,
        "portability_updates": sum(1 for item in migration_plan if item["new"] != item["old"]),
        "application": application,
        "warnings": (
            ["SOURCE_CHANGED_DURING_EXPORT: bundle contains a valid point-in-time snapshot"]
            if source_changed
            else []
        ),
    }


def _legacy_evidence_inventory(source: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in LEGACY_EVIDENCE_FILES:
        path = source / name
        if path.is_file() and not path.is_symlink():
            files.append({"name": name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        else:
            missing.append(name)
    return {"status": "complete" if not missing else "partial", "files": files, "missing": missing}


def _copy_legacy_evidence(source: Path, destination: Path) -> None:
    for name in LEGACY_EVIDENCE_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise StateBundleError(f"Required legacy MLflow evidence is missing: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination / name)


def _bundle_status(references: list[dict[str, Any]], legacy: dict[str, Any]) -> tuple[str, list[str]]:
    issues = sorted({str(item["error_code"]) for item in references if item.get("error_code")})
    required_missing = any(
        item.get("error_code") == "MISSING_REFERENCED_ARTIFACT"
        and item.get("persistence") == "persistent-required"
        for item in references
    )
    if legacy["status"] != "complete":
        issues.append("MISSING_LEGACY_MLFLOW_EVIDENCE")
        required_missing = True
    if required_missing:
        return "invalid", sorted(set(issues))
    if issues:
        return "partial", sorted(set(issues))
    return "complete", []


def _inventory_only(paths: SourcePaths, workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    feedback_destination = workspace / "application" / "feedback.db"
    feedback = _prepare_feedback_snapshot(paths, feedback_destination)
    references = _iter_path_references(feedback_destination, roots=paths.artifact_roots)
    models = _model_inventory(paths.model_options_dir, include_hashes=False)
    active_mlflow = _active_mlflow_analysis(paths.active_mlflow_db, paths.active_mlflow_artifacts)
    legacy = _legacy_evidence_inventory(paths.legacy_evidence_dir)
    status, issues = _bundle_status(references, legacy)
    return (
        {
            "mode": "dry-run",
            "bundle_status": status,
            "issues": issues,
            "feedback": feedback,
            "models": {key: value for key, value in models.items() if key != "files"},
            "persistent_references": _safe_reference_records(references),
            "active_mlflow": active_mlflow,
            "legacy_mlflow_evidence": legacy,
            "environment_requirements": _environment_requirements(),
            "actions": [
                "WOULD create a SQLite backup snapshot of feedback.db",
                "WOULD apply portable-reference conversion to the snapshot only",
                "WOULD preserve the full models/options tree",
                "WOULD copy required referenced managed artifacts only",
                "WOULD snapshot active MLflow if initialized and safe",
                "WOULD preserve deterministic legacy MLflow evidence",
                "WOULD write a checksummed private directory bundle",
            ],
        },
        {"models": models},
    )


def export_bundle(
    *,
    output: Path | None = None,
    dry_run: bool = True,
    source_paths: SourcePaths | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Inspect or export source state.  Dry-run is the safe default."""
    paths = source_paths or resolve_source_paths()
    if not paths.feedback_db.is_file():
        raise StateBundleError(f"Application database does not exist: {paths.feedback_db}")
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="viettoxic-state-dry-run-") as temporary:
            report, _ = _inventory_only(paths, Path(temporary))
            return report
    if output is None:
        raise StateBundleError("An explicit --output is required for export")
    destination = output.expanduser().resolve()
    if destination.exists():
        raise StateBundleError(f"Export destination already exists: {destination}")
    for source_root in (paths.data_dir, paths.runtime_dir, paths.model_options_dir):
        try:
            destination.relative_to(source_root.expanduser().resolve())
        except ValueError:
            continue
        raise StateBundleError(f"Export destination cannot be inside managed source state: {source_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    activated = False
    try:
        feedback_destination = stage / "application" / "feedback.db"
        feedback = _prepare_feedback_snapshot(paths, feedback_destination)
        references = _iter_path_references(feedback_destination, roots=paths.artifact_roots)
        models = _model_inventory(paths.model_options_dir, include_hashes=False)
        active_analysis = _active_mlflow_analysis(paths.active_mlflow_db, paths.active_mlflow_artifacts)
        legacy = _legacy_evidence_inventory(paths.legacy_evidence_dir)
        status, issues = _bundle_status(references, legacy)
        if status != "complete" and not allow_partial:
            raise StateBundleError(
                f"Source state is {status}; resolve or explicitly review issues before export: {', '.join(issues)}"
            )
        if models["status"] != "present":
            raise StateBundleError("MODEL_OPTIONS_DIR is missing; full model preservation cannot be satisfied")
        invalid_contracts = [item["model"] for item in models["contracts"] if not item["valid"]]
        if invalid_contracts:
            raise StateBundleError(f"Model contract validation failed: {', '.join(invalid_contracts)}")

        _copy_path(paths.model_options_dir, stage / "models" / "options")
        copied_persistent = _copy_persistent_artifacts(
            references, roots=paths.artifact_roots, bundle_root=stage
        )
        active_mlflow = _copy_active_mlflow(paths, stage / "mlflow" / "active", active_analysis)
        _copy_legacy_evidence(paths.legacy_evidence_dir, stage / "mlflow" / "legacy_evidence")
        environment = _environment_requirements()
        source_inventory = {
            "bundle_status": status,
            "issues": issues,
            "feedback": feedback,
            "models": models,
            "persistent_references": _safe_reference_records(references),
            "active_mlflow": active_mlflow,
            "legacy_mlflow_evidence": legacy,
            "runtime_exclusions": [
                "logs",
                "cache",
                "pytest temp directories",
                "build verification directories",
                "frontend build output",
                "Python __pycache__",
                "temporary Kaggle workspaces",
            ],
        }
        _write_json(stage / "metadata" / "source_inventory.json", source_inventory)
        _write_json(stage / "metadata" / "environment_requirements.json", environment)
        file_records = _file_records(stage)
        content_identity = _content_identity(file_records, feedback["application"]["state_counts"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "project": PROJECT,
            "source_platform": platform.system().lower() or "unknown",
            "source_git_commit": _source_git_commit(),
            "created_at": _utc_now(),
            "content_identity": {"algorithm": "sha256", "value": content_identity, "excludes": ["created_at", MANIFEST_FILENAME, CHECKSUMS_FILENAME]},
            "components": {
                "application": {"status": "complete", "path": "application/feedback.db"},
                "models": {"status": models["status"], "path": "models/options", "preservation": "full"},
                "persistent_artifacts": {"status": "complete", "items": copied_persistent},
                "active_mlflow": {"status": active_mlflow["status"], "path": "mlflow/active" if active_mlflow["status"] != "not_initialized" else None},
                "legacy_mlflow_evidence": {"status": legacy["status"], "path": "mlflow/legacy_evidence"},
                "metadata": {"status": "complete", "path": "metadata"},
            },
            "compatibility": {
                "python_version": platform.python_version(),
                "mlflow_version": _mlflow_version(),
                "application_required_tables": sorted(REQUIRED_APPLICATION_TABLES),
                "application_missing_required_tables": feedback["application"]["missing_required_tables"],
                "active_mlflow_alembic_revision": active_mlflow.get("alembic_revision"),
            },
            "sensitivity": {
                "private_migration_material": True,
                "contains_sensitive_application_state": feedback["application"]["contains_sensitive_application_state"],
                "environment_secret_values_included": False,
            },
            "statistics": {
                "application_state_counts": feedback["application"]["state_counts"],
                "model_artifact_count": models["file_count"],
                "model_artifact_total_bytes": models["total_bytes"],
                "persistent_artifact_count": len(copied_persistent),
                "bundle_payload_file_count": len(file_records),
                "bundle_payload_total_bytes": sum(item["size_bytes"] for item in file_records),
            },
            "bundle_status": status,
            "issues": issues,
            "files": file_records,
        }
        _write_json(stage / MANIFEST_FILENAME, manifest)
        _write_checksums(stage)
        verify_bundle(stage)
        os.replace(stage, destination)
        activated = True
        return {
            "mode": "export",
            "bundle": str(destination),
            "schema_version": SCHEMA_VERSION,
            "content_identity": content_identity,
            "bundle_status": status,
            "statistics": manifest["statistics"],
            "source_feedback_unchanged": not feedback["source_changed_during_export"],
            "warnings": feedback["warnings"],
        }
    finally:
        if not activated and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _parse_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / CHECKSUMS_FILENAME
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise StateBundleError(f"Missing safe checksum inventory: {CHECKSUMS_FILENAME}")
    records: dict[str, str] = {}
    for line_number, raw in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise StateBundleError(f"Malformed checksum line {line_number}")
        relative = _safe_relative(parts[1]).as_posix()
        if relative == CHECKSUMS_FILENAME or relative in records:
            raise StateBundleError(f"Duplicate or recursive checksum member: {relative}")
        records[relative] = parts[0]
    return records


def verify_bundle(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise StateBundleError("Only non-symlink directory bundles are supported")
    checksums = _parse_checksums(root)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in _walk_files(root)
        if path.name != CHECKSUMS_FILENAME
    }
    if set(checksums) != actual_files:
        missing = sorted(set(checksums) - actual_files)
        unexpected = sorted(actual_files - set(checksums))
        raise StateBundleError(f"Bundle member mismatch; missing={missing}, unexpected={unexpected}")
    for relative, expected in checksums.items():
        path = _bundle_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise StateBundleError(f"Unsafe or missing bundle member: {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise StateBundleError(f"SHA-256 mismatch for {relative}")
    try:
        manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateBundleError("Manifest is missing or malformed") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise StateBundleError(f"Unsupported bundle schema version: {manifest.get('schema_version')!r}")
    if manifest.get("project") != PROJECT:
        raise StateBundleError(f"Unexpected bundle project: {manifest.get('project')!r}")
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise StateBundleError("Manifest components must be an object")
    unexpected_components = set(components) - ALLOWED_COMPONENTS
    if unexpected_components:
        raise StateBundleError(f"Unexpected bundle component(s): {', '.join(sorted(unexpected_components))}")
    missing_components = ALLOWED_COMPONENTS - set(components)
    if missing_components:
        raise StateBundleError(f"Missing bundle component(s): {', '.join(sorted(missing_components))}")
    manifest_records = manifest.get("files")
    if not isinstance(manifest_records, list):
        raise StateBundleError("Manifest files must be an array")
    recorded_paths: set[str] = set()
    for record in manifest_records:
        if not isinstance(record, dict):
            raise StateBundleError("Manifest file record must be an object")
        relative = _safe_relative(str(record.get("path") or "")).as_posix()
        if relative in recorded_paths:
            raise StateBundleError(f"Duplicate manifest file: {relative}")
        recorded_paths.add(relative)
        path = _bundle_path(root, relative)
        if not path.is_file():
            raise StateBundleError(f"Manifest file is missing: {relative}")
        if int(record.get("size_bytes", -1)) != path.stat().st_size:
            raise StateBundleError(f"Size mismatch for {relative}")
        if str(record.get("sha256")) != checksums.get(relative):
            raise StateBundleError(f"Manifest checksum metadata mismatch for {relative}")
    expected_payload = actual_files - {MANIFEST_FILENAME}
    if recorded_paths != expected_payload:
        raise StateBundleError("Manifest file inventory does not exactly match bundle payload")

    application_path = root / "application" / "feedback.db"
    application = _application_analysis(application_path)
    if application["missing_required_tables"]:
        raise StateBundleError(
            "Application DB is incompatible; missing tables: "
            + ", ".join(application["missing_required_tables"])
        )
    expected_identity = _content_identity(manifest_records, application["state_counts"])
    if manifest.get("content_identity", {}).get("value") != expected_identity:
        raise StateBundleError("Bundle content identity mismatch")
    model_contracts = _verify_model_contracts(root / "models" / "options")
    invalid_models = [item["model"] for item in model_contracts if not item["valid"]]
    if invalid_models:
        raise StateBundleError(f"Model contract validation failed: {', '.join(invalid_models)}")
    active = manifest["components"].get("active_mlflow", {})
    if active.get("status") != "not_initialized":
        active_analysis = _active_mlflow_analysis(
            root / "mlflow" / "active" / "mlflow.db", root / "mlflow" / "active" / "artifacts"
        )
        if active_analysis["status"] != "ready":
            raise StateBundleError("Bundled active MLflow component is unsafe")
    return {
        "valid": True,
        "schema_version": manifest["schema_version"],
        "project": manifest["project"],
        "content_identity": manifest.get("content_identity", {}).get("value"),
        "bundle_status": manifest.get("bundle_status"),
        "file_count": len(actual_files),
        "application_integrity": application["integrity"],
        "model_contracts": model_contracts,
        "manifest": manifest,
    }


def inspect_bundle(bundle: Path) -> dict[str, Any]:
    verified = verify_bundle(bundle)
    manifest = verified.pop("manifest")
    return {
        **verified,
        "source_platform": manifest.get("source_platform"),
        "source_git_commit": manifest.get("source_git_commit"),
        "created_at": manifest.get("created_at"),
        "components": manifest.get("components"),
        "statistics": manifest.get("statistics"),
        "sensitivity": manifest.get("sensitivity"),
        "issues": manifest.get("issues", []),
    }


def _persistent_install_groups(bundle_root: Path, target: TargetPaths) -> list[tuple[str, Path, Path]]:
    groups: list[tuple[str, Path, Path]] = []
    persistent = bundle_root / "persistent_artifacts"
    if not persistent.exists():
        return groups
    for scheme in ("runtime", "data"):
        scheme_root = persistent / scheme
        if not scheme_root.exists():
            continue
        destination_root = target.runtime_dir if scheme == "runtime" else target.data_dir
        for first in sorted(scheme_root.iterdir(), key=lambda item: item.name):
            _reject_symlink(first, label="persistent artifact")
            if scheme == "data" and first.name == "processed":
                for second in sorted(first.iterdir(), key=lambda item: item.name):
                    if second.name == "feedback":
                        raise StateBundleError("Persistent artifact overlaps application feedback destination")
                    groups.append((f"persistent-{scheme}-{first.name}-{second.name}", second, destination_root / first.name / second.name))
            elif scheme == "data" and first.name == "mlflow":
                raise StateBundleError("Persistent artifact overlaps active MLflow destination")
            else:
                groups.append((f"persistent-{scheme}-{first.name}", first, destination_root / first.name))
    return groups


def _install_plan(bundle_root: Path, target: TargetPaths, manifest: Mapping[str, Any]) -> list[tuple[str, Path, Path]]:
    items: list[tuple[str, Path, Path]] = [
        (
            "application-feedback",
            bundle_root / "application",
            target.feedback_db.parent,
        ),
        ("models-options", bundle_root / "models" / "options", target.model_options_dir),
    ]
    mlflow_stage = bundle_root / "mlflow"
    active_status = manifest["components"]["active_mlflow"]["status"]
    combined_mlflow = None
    if active_status != "not_initialized" or (mlflow_stage / "legacy_evidence").exists():
        combined_mlflow = mlflow_stage
        items.append(("mlflow-state", combined_mlflow, target.mlflow_dir))
    items.extend(_persistent_install_groups(bundle_root, target))
    destinations: list[tuple[str, Path]] = []
    for label, source, destination in items:
        key = os.path.normcase(str(destination.resolve(strict=False)))
        if any(key == existing for existing, _ in destinations):
            raise StateBundleError(f"Duplicate import destination: {destination}")
        resolved_destination = destination.resolve(strict=False)
        resolved_source = source.resolve(strict=False)
        if resolved_source == resolved_destination or resolved_source in resolved_destination.parents or resolved_destination in resolved_source.parents:
            raise StateBundleError(f"Bundle source overlaps import destination: {label}")
        for _, other in destinations:
            if resolved_destination in other.parents or other in resolved_destination.parents:
                raise StateBundleError(f"Overlapping import destinations: {other} and {resolved_destination}")
        destinations.append((key, resolved_destination))
    return items


def _missing_environment_names(manifest: Mapping[str, Any]) -> list[str]:
    environment = manifest.get("components", {}).get("metadata", {})
    _ = environment  # Manifest component is validated; requirements are read from the bundle below.
    names = (
        "VIETTOXIC_ADMIN_USERNAME",
        "VIETTOXIC_ADMIN_PASSWORD",
        "VIETTOXIC_ADMIN_SESSION_SECRET",
    )
    return [name for name in names if not os.getenv(name)]


def import_dry_run(bundle: Path, *, target_paths: TargetPaths | None = None) -> dict[str, Any]:
    target = target_paths or resolve_target_paths()
    root = bundle.expanduser().resolve(strict=True)
    verified = verify_bundle(root)
    manifest = verified["manifest"]
    items = _install_plan(root, target, manifest)
    collisions = [str(destination) for _, _, destination in items if destination.exists()]
    application = _application_analysis(root / "application" / "feedback.db")
    warnings: list[str] = []
    if application["automation_enabled"]:
        warnings.append("Imported state contains enabled automation. Review target environment before starting services.")
    return {
        "mode": "dry-run",
        "zero_target_writes": True,
        "bundle_schema": manifest["schema_version"],
        "source_platform": manifest.get("source_platform"),
        "source_git_commit": manifest.get("source_git_commit"),
        "components": manifest["components"],
        "checksums": "verified",
        "target_paths": {
            "data_dir": str(target.data_dir),
            "runtime_dir": str(target.runtime_dir),
            "model_options_dir": str(target.model_options_dir),
        },
        "collisions": collisions,
        "existing_target_state": bool(collisions),
        "application_db_integrity": application["integrity"],
        "model_count": manifest["statistics"]["model_artifact_count"],
        "active_mlflow_status": manifest["components"]["active_mlflow"]["status"],
        "missing_environment_requirements": _missing_environment_names(manifest),
        "actions": [
            f"WOULD install {label} at {destination}" for label, _, destination in items
        ],
        "warnings": warnings,
        "manual_readiness": [
            "Create the Python environment and install backend dependencies",
            "Create backend/.env.local and comprehensive_ui/.env.local manually",
            "Configure Kaggle and Gemini credentials if those features are used",
            "Start the MLflow server explicitly if active tracking is required",
            "Browser session and frontend local state are intentionally not migrated",
        ],
    }


def _copy_install_source(source: Path, staging: Path, *, label: str) -> None:
    if label == "mlflow-state":
        active = source / "active"
        legacy = source / "legacy_evidence"
        staging.mkdir(parents=True)
        if active.exists():
            for child in sorted(active.iterdir(), key=lambda item: item.name):
                _copy_path(child, staging / child.name)
        if legacy.exists():
            _copy_path(legacy, staging / "evidence")
        return
    if source.is_file():
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staging)
    else:
        _copy_path(source, staging)


def _backup_existing(source: Path, backup: Path, *, label: str) -> None:
    if source.is_file():
        if source.name.endswith(".db"):
            _sqlite_snapshot(source, backup)
        else:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)
        return
    backup.mkdir(parents=True, exist_ok=True)
    for file_path in _walk_files(source):
        relative = file_path.relative_to(source)
        destination = backup / relative
        if file_path.suffix.lower() == ".db":
            _sqlite_snapshot(file_path, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination)


def import_bundle(
    bundle: Path,
    *,
    target_paths: TargetPaths | None = None,
    apply: bool = False,
    replace_existing: bool = False,
    backup_dir: Path | None = None,
    _fail_after_promotions: int | None = None,
) -> dict[str, Any]:
    """Dry-run or atomically staged local import.  Apply must be explicit."""
    target = target_paths or resolve_target_paths()
    if not apply:
        return import_dry_run(bundle, target_paths=target)
    root = bundle.expanduser().resolve(strict=True)
    verified = verify_bundle(root)
    manifest = verified["manifest"]
    if manifest.get("bundle_status") != "complete":
        raise StateBundleError("Refusing to apply a bundle that is not complete")
    items = _install_plan(root, target, manifest)
    collisions = [(label, destination) for label, _, destination in items if destination.exists()]
    if collisions and not replace_existing:
        raise StateBundleError(
            "TARGET_STATE_EXISTS: " + ", ".join(str(destination) for _, destination in collisions)
        )
    parents = {destination.parent.resolve(strict=False) for _, _, destination in items}
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".viettoxic-import-staging-", dir=target.data_dir.parent))
    rollback_root = stage_root / "rollback"
    prepared: list[tuple[str, Path, Path]] = []
    promoted: list[tuple[str, Path, Path | None]] = []
    backup_root: Path | None = None
    try:
        for index, (label, source, destination) in enumerate(items):
            staged = stage_root / "prepared" / f"{index:03d}-{label}"
            _copy_install_source(source, staged, label=label)
            prepared.append((label, staged, destination))
        staged_feedback = next(staged for label, staged, _ in prepared if label == "application-feedback") / "feedback.db"
        _application_analysis(staged_feedback)
        staged_models = next(staged for label, staged, _ in prepared if label == "models-options")
        invalid_models = [item["model"] for item in _verify_model_contracts(staged_models) if not item["valid"]]
        if invalid_models:
            raise StateBundleError(f"Staged model contract validation failed: {', '.join(invalid_models)}")

        if collisions:
            backup_root = (
                backup_dir.expanduser().resolve()
                if backup_dir
                else target.data_dir.parent / ".state_bundle_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
            if backup_root.exists():
                raise StateBundleError(f"Import backup already exists: {backup_root}")
            for label, destination in collisions:
                _backup_existing(destination, backup_root / label, label=label)

        for index, (label, staged, destination) in enumerate(prepared):
            original: Path | None = None
            if destination.exists():
                original = rollback_root / f"{index:03d}-{label}"
                original.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, original)
            try:
                os.replace(staged, destination)
            except Exception:
                if original is not None and original.exists() and not destination.exists():
                    os.replace(original, destination)
                raise
            promoted.append((label, destination, original))
            if _fail_after_promotions is not None and len(promoted) >= _fail_after_promotions:
                raise StateBundleError("Injected import promotion failure")

        target_report = verify_target(root, target_paths=target)
        if not target_report["valid"]:
            raise StateBundleError("Target verification failed after staged promotion")
        warnings = target_report.get("warnings", [])
        return {
            "mode": "apply",
            "installed": [str(destination) for _, destination, _ in promoted],
            "backup": str(backup_root) if backup_root else None,
            "rollback_available": bool(backup_root),
            "verification": target_report,
            "services_started": False,
            "warnings": warnings,
        }
    except Exception:
        for _, destination, original in reversed(promoted):
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            if original is not None and original.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(original, destination)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def verify_target(bundle: Path, *, target_paths: TargetPaths | None = None) -> dict[str, Any]:
    target = target_paths or resolve_target_paths()
    root = bundle.expanduser().resolve(strict=True)
    manifest = verify_bundle(root)["manifest"]
    application = _application_analysis(target.feedback_db)
    expected_counts = manifest["statistics"]["application_state_counts"]
    count_matches = application["state_counts"] == expected_counts
    references = _iter_path_references(target.feedback_db, roots=target.artifact_roots)
    stale_managed = _stale_managed_references(references)
    missing_required: list[dict[str, Any]] = []
    for item in references:
        if item["persistence"] in {"persistent-required", "model-component"} and item["exists"] is False:
            missing_required.append(item)
    model_contracts = _verify_model_contracts(target.model_options_dir)
    models_valid = bool(model_contracts) and all(item["valid"] for item in model_contracts)
    active_status = manifest["components"]["active_mlflow"]["status"]
    if active_status == "not_initialized":
        active_valid = not (target.mlflow_dir / "mlflow.db").exists()
        active_analysis: dict[str, Any] = {"status": "not_initialized"}
    else:
        active_analysis = _active_mlflow_analysis(
            target.mlflow_dir / "mlflow.db", target.mlflow_dir / "artifacts"
        )
        active_valid = active_analysis["status"] == "ready"
    legacy_present = all((target.mlflow_dir / "evidence" / name).is_file() for name in LEGACY_EVIDENCE_FILES)
    warnings: list[str] = []
    if application["automation_enabled"]:
        warnings.append("Imported state contains enabled automation. Review target environment before starting services.")
    valid = (
        application["integrity"] == "ok"
        and not application["missing_required_tables"]
        and count_matches
        and not stale_managed
        and not missing_required
        and models_valid
        and active_valid
        and legacy_present
    )
    safe_stale_managed = _safe_reference_records(stale_managed)
    safe_missing_required = _safe_reference_records(missing_required)
    return {
        "valid": valid,
        "feedback_db_integrity": application["integrity"],
        "required_tables_readable": not application["missing_required_tables"],
        "state_counts_match": count_matches,
        "expected_state_counts": expected_counts,
        "actual_state_counts": application["state_counts"],
        "portable_reference_count": sum(1 for item in references if item["classification"] == "portable"),
        "stale_managed_references": safe_stale_managed,
        "missing_required_artifacts": safe_missing_required,
        "model_contracts": model_contracts,
        "models_valid": models_valid,
        "active_mlflow": active_analysis,
        "legacy_evidence_present": legacy_present,
        "warnings": warnings,
    }


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-data-dir", type=Path)
    parser.add_argument("--target-runtime-dir", type=Path)
    parser.add_argument("--target-model-options-dir", type=Path)


def _target_from_args(args: argparse.Namespace) -> TargetPaths:
    return resolve_target_paths(
        data_dir=args.target_data_dir,
        runtime_dir=args.target_runtime_dir,
        model_options_dir=args.target_model_options_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="Inspect or create a private state bundle")
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--dry-run", action="store_true", help="Read-only source inventory (default without --output)")
    export_parser.add_argument("--allow-partial", action="store_true")
    inspect_parser = commands.add_parser("inspect", help="Inspect and verify a bundle")
    inspect_parser.add_argument("bundle", type=Path)
    verify_parser = commands.add_parser("verify", help="Verify bundle integrity")
    verify_parser.add_argument("bundle", type=Path)
    import_parser = commands.add_parser("import", help="Dry-run or explicitly apply a bundle")
    import_parser.add_argument("bundle", type=Path)
    mode = import_parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Explicit default")
    mode.add_argument("--apply", action="store_true")
    import_parser.add_argument("--replace-existing", action="store_true")
    import_parser.add_argument("--backup-dir", type=Path)
    _add_target_options(import_parser)
    target_parser = commands.add_parser("verify-target", help="Verify imported state without starting services")
    target_parser.add_argument("bundle", type=Path)
    _add_target_options(target_parser)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            dry_run = bool(args.dry_run or args.output is None)
            _print_json(export_bundle(output=args.output, dry_run=dry_run, allow_partial=args.allow_partial))
        elif args.command == "inspect":
            _print_json(inspect_bundle(args.bundle))
        elif args.command == "verify":
            result = verify_bundle(args.bundle)
            result.pop("manifest", None)
            _print_json(result)
        elif args.command == "import":
            _print_json(
                import_bundle(
                    args.bundle,
                    target_paths=_target_from_args(args),
                    apply=args.apply,
                    replace_existing=args.replace_existing,
                    backup_dir=args.backup_dir,
                )
            )
        elif args.command == "verify-target":
            _print_json(verify_target(args.bundle, target_paths=_target_from_args(args)))
        return 0
    except StateBundleError as exc:
        print(f"STATE_BUNDLE_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
