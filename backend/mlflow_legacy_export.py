"""Export deterministic, read-only evidence from the legacy root MLflow DB."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from backend.runtime_paths import get_legacy_mlflow_db_path, get_mlflow_evidence_dir


INVENTORY_FILENAME = "legacy_mlflow_inventory.json"
ARTIFACTS_FILENAME = "legacy_mlflow_artifacts.json"
CHECKSUMS_FILENAME = "legacy_mlflow_checksums.json"
LEGACY_RUN_CLASSIFICATION = "UNCERTAIN"
LEGACY_RUN_CLASSIFICATION_REASON = (
    "Legacy metadata contains no metrics or recoverable run artifacts; repository evidence "
    "does not prove thesis-evidence status."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_deterministic_json(path: Path, payload: Any) -> bytes:
    encoded = _json_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return encoded


def _iso_timestamp(milliseconds: Any) -> Optional[str]:
    if milliseconds is None:
        return None
    try:
        value = int(milliseconds)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    return value.hex() if isinstance(value, bytes) else value


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    if not _table_exists(connection, table):
        return []
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _records(
    connection: sqlite3.Connection,
    table: str,
    *,
    columns: Optional[Iterable[str]] = None,
    order_by: Iterable[str] = (),
) -> list[dict[str, Any]]:
    available = _table_columns(connection, table)
    if not available:
        return []
    selected = [column for column in (columns or available) if column in available]
    ordering = [column for column in order_by if column in available]
    sql = "SELECT " + ", ".join(f'"{column}"' for column in selected) + f' FROM "{table}"'
    if ordering:
        sql += " ORDER BY " + ", ".join(f'"{column}"' for column in ordering)
    result: list[dict[str, Any]] = []
    for row in connection.execute(sql):
        record = {column: _json_value(row[column]) for column in selected}
        for column in selected:
            if column.endswith(("_time", "_timestamp")) and isinstance(record[column], (int, float)):
                record[f"{column}_iso"] = _iso_timestamp(record[column])
        result.append(record)
    return result


def open_legacy_database(path: Path) -> sqlite3.Connection:
    """Open the legacy DB with SQLite enforcing a read-only immutable connection."""
    resolved = path.expanduser().resolve(strict=True)
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _path_from_file_uri(stored_uri: str) -> Optional[str]:
    parsed = urlparse(stored_uri)
    if parsed.scheme.lower() != "file":
        return None
    if parsed.netloc and len(parsed.netloc) == 2 and parsed.netloc[1] == ":":
        return url2pathname(f"{parsed.netloc}{unquote(parsed.path)}")
    if parsed.netloc:
        return url2pathname(f"//{parsed.netloc}{unquote(parsed.path)}")
    return url2pathname(unquote(parsed.path))


def classify_artifact_reference(stored_uri: Optional[str], base_dir: Path) -> dict[str, Any]:
    raw = str(stored_uri or "").strip()
    record: dict[str, Any] = {
        "classification": "missing_reference",
        "exists": False,
        "resolved_path_if_applicable": None,
        "status": "missing",
        "stored_uri": raw or None,
    }
    if not raw:
        return record

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme and scheme != "file" and not (len(scheme) == 1 and PureWindowsPath(raw).is_absolute()):
        record.update(
            classification="mlflow_managed" if scheme == "mlflow-artifacts" else "external",
            exists=None,
            status="not_locally_verifiable",
        )
        return record

    candidate_text = _path_from_file_uri(raw) if scheme == "file" else raw
    if candidate_text is None:
        return record

    windows_absolute = PureWindowsPath(candidate_text).is_absolute()
    posix_absolute = candidate_text.startswith("/")
    if windows_absolute:
        classification = "machine_specific_windows"
        native = os.name == "nt"
    elif candidate_text.startswith("/Users/"):
        classification = "machine_specific_macos"
        native = os.name != "nt"
    elif posix_absolute:
        classification = "machine_specific_posix"
        native = os.name != "nt"
    else:
        classification = "relative"
        native = True

    resolved: Optional[Path] = None
    if native:
        candidate = Path(candidate_text).expanduser()
        resolved = (candidate if candidate.is_absolute() else base_dir / candidate).resolve()
    exists = bool(resolved and resolved.exists())
    record.update(
        classification=classification,
        exists=exists,
        resolved_path_if_applicable=str(resolved) if resolved else None,
        status="available" if exists else "missing",
    )
    return record


def _build_inventory(connection: sqlite3.Connection, db_path: Path, db_sha256: str) -> dict[str, Any]:
    experiments = _records(
        connection,
        "experiments",
        columns=(
            "experiment_id",
            "name",
            "artifact_location",
            "lifecycle_stage",
            "creation_time",
            "last_update_time",
        ),
        order_by=("experiment_id",),
    )
    runs = _records(
        connection,
        "runs",
        columns=(
            "run_uuid",
            "name",
            "experiment_id",
            "status",
            "artifact_uri",
            "lifecycle_stage",
            "start_time",
            "end_time",
        ),
        order_by=("experiment_id", "start_time", "run_uuid"),
    )
    params = _records(connection, "params", order_by=("run_uuid", "key"))
    metrics = _records(connection, "metrics", order_by=("run_uuid", "key", "step", "timestamp"))
    latest_metrics = _records(connection, "latest_metrics", order_by=("run_uuid", "key"))
    tags = _records(connection, "tags", order_by=("run_uuid", "key"))

    def grouped(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            run_id = str(record.pop("run_uuid", ""))
            result.setdefault(run_id, []).append(record)
        return result

    params_by_run = grouped(params)
    metrics_by_run = grouped(metrics)
    latest_by_run = grouped(latest_metrics)
    tags_by_run = grouped(tags)
    for run in runs:
        run_id = str(run.get("run_uuid", ""))
        run["classification"] = LEGACY_RUN_CLASSIFICATION
        run["classification_reason"] = LEGACY_RUN_CLASSIFICATION_REASON
        run["params"] = params_by_run.get(run_id, [])
        run["metrics"] = metrics_by_run.get(run_id, [])
        run["latest_metrics"] = latest_by_run.get(run_id, [])
        run["tags"] = tags_by_run.get(run_id, [])

    alembic_revision = None
    if _table_exists(connection, "alembic_version"):
        row = connection.execute("SELECT version_num FROM alembic_version ORDER BY version_num LIMIT 1").fetchone()
        alembic_revision = str(row[0]) if row else None

    model_registry = {
        "registered_models": _records(connection, "registered_models", order_by=("name",)),
        "model_versions": _records(connection, "model_versions", order_by=("name", "version")),
        "registered_model_tags": _records(connection, "registered_model_tags", order_by=("name", "key")),
        "model_version_tags": _records(connection, "model_version_tags", order_by=("name", "version", "key")),
        "registered_model_aliases": _records(
            connection, "registered_model_aliases", order_by=("name", "alias")
        ),
    }
    return {
        "database": {
            "alembic_revision": alembic_revision,
            "integrity_result": str(connection.execute("PRAGMA quick_check").fetchone()[0]),
            "read_only": True,
            "sha256": db_sha256,
            "size_bytes": db_path.stat().st_size,
            "source_ref": "repository://mlflow.db",
        },
        "experiments": experiments,
        "export_policy": {
            "artifact_substitution": False,
            "legacy_database_role": "immutable_historical_evidence",
            "legacy_run_default_classification": LEGACY_RUN_CLASSIFICATION,
            "reconstruct_runs": False,
        },
        "mlflow_version": importlib.metadata.version("mlflow"),
        "model_registry": model_registry,
        "runs": runs,
    }


def _build_artifact_inventory(inventory: dict[str, Any], db_path: Path) -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    for experiment in inventory["experiments"]:
        reference = classify_artifact_reference(experiment.get("artifact_location"), db_path.parent)
        reference.update(kind="experiment", experiment_id=experiment.get("experiment_id"), run_id=None)
        references.append(reference)
    for run in inventory["runs"]:
        reference = classify_artifact_reference(run.get("artifact_uri"), db_path.parent)
        reference.update(kind="run", experiment_id=run.get("experiment_id"), run_id=run.get("run_uuid"))
        references.append(reference)
    references.sort(key=lambda item: (str(item["kind"]), str(item["experiment_id"]), str(item["run_id"])))
    return {
        "artifact_recovery_attempted": False,
        "artifact_substitution_attempted": False,
        "references": references,
        "source_database_sha256": inventory["database"]["sha256"],
    }


def export_legacy_evidence(
    *, db_path: Optional[Path] = None, output_dir: Optional[Path] = None
) -> dict[str, Any]:
    source = (db_path or get_legacy_mlflow_db_path()).expanduser().resolve(strict=True)
    destination = (output_dir or get_mlflow_evidence_dir()).expanduser().resolve()
    before_sha256 = sha256_file(source)
    with open_legacy_database(source) as connection:
        inventory = _build_inventory(connection, source, before_sha256)
    after_sha256 = sha256_file(source)
    if after_sha256 != before_sha256:
        raise RuntimeError("Legacy MLflow database changed during read-only export")

    artifacts = _build_artifact_inventory(inventory, source)
    destination.mkdir(parents=True, exist_ok=True)
    inventory_path = destination / INVENTORY_FILENAME
    artifacts_path = destination / ARTIFACTS_FILENAME
    checksums_path = destination / CHECKSUMS_FILENAME
    inventory_bytes = _write_deterministic_json(inventory_path, inventory)
    artifacts_bytes = _write_deterministic_json(artifacts_path, artifacts)
    checksums = {
        "files": {
            ARTIFACTS_FILENAME: hashlib.sha256(artifacts_bytes).hexdigest(),
            INVENTORY_FILENAME: hashlib.sha256(inventory_bytes).hexdigest(),
            "repository://mlflow.db": before_sha256,
        },
        "source_database_unchanged": True,
    }
    _write_deterministic_json(checksums_path, checksums)
    return {
        "artifact_references": len(artifacts["references"]),
        "experiments": len(inventory["experiments"]),
        "metrics": sum(len(run["metrics"]) for run in inventory["runs"]),
        "output_files": [str(inventory_path), str(artifacts_path), str(checksums_path)],
        "runs": len(inventory["runs"]),
        "source_database": str(source),
        "source_database_sha256": before_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Evidence output directory")
    args = parser.parse_args()
    print(json.dumps(export_legacy_evidence(output_dir=args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
