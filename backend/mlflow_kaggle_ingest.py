"""Validate and idempotently ingest portable Kaggle evidence through MLflow APIs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.mlflow_client_config import (
    MlflowClientConfigurationError,
    prepare_mlflow_client,
)
from kaggle.mlflow_evidence import (
    ARTIFACT_STATUSES,
    EVIDENCE_FILENAME,
    SCHEMA_VERSION,
    TRACKING_STATUSES,
    TRAINING_STATUSES,
    safe_relative_path,
    sha256_bytes,
)


ALLOWED_MODEL_FAMILIES = frozenset({"phobert", "tfidf_lr"})
ALLOWED_TRAINING_MODES = frozenset({"retrain", "finetune"})
MAX_BUNDLE_FILES = 5000
MAX_BUNDLE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|private[_-]?key|secret|token)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class KaggleEvidenceError(RuntimeError):
    """Base class for evidence validation and ingestion failures."""


class KaggleEvidenceNotFound(KaggleEvidenceError):
    pass


class KaggleEvidenceValidationError(KaggleEvidenceError):
    pass


class KaggleEvidenceConflictError(KaggleEvidenceError):
    pass


class KaggleEvidenceIngestionUnavailable(KaggleEvidenceError):
    pass


@dataclass(frozen=True)
class ValidatedArtifact:
    relative_path: str
    role: str
    required: bool
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ValidatedKaggleEvidence:
    archive_path: Path
    manifest: dict[str, Any]
    evidence_bytes: bytes
    evidence_sha256: str
    artifacts: tuple[ValidatedArtifact, ...]

    @property
    def source_job_id(self) -> str:
        return str(self.manifest["source_job_id"])

    @property
    def source_run_id(self) -> str:
        return str(self.manifest["source_run_id"])


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise KaggleEvidenceValidationError(f"{key} must be an object")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KaggleEvidenceValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _looks_machine_specific(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    return bool(
        _WINDOWS_ABSOLUTE_RE.match(value.strip())
        or normalized.startswith("/")
        or "://" in normalized
    )


def _validate_safe_scalars(
    payload: Mapping[str, Any],
    label: str,
    *,
    numeric_only: bool = False,
) -> None:
    for key, value in payload.items():
        key_text = str(key).strip()
        if not key_text:
            raise KaggleEvidenceValidationError(f"{label} contains an empty key")
        if _SENSITIVE_KEY_RE.search(key_text):
            raise KaggleEvidenceValidationError(f"{label} contains a prohibited sensitive key")
        if numeric_only:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KaggleEvidenceValidationError(f"metric {key_text!r} must be numeric")
            if isinstance(value, float) and not math.isfinite(value):
                raise KaggleEvidenceValidationError(f"metric {key_text!r} must be finite")
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise KaggleEvidenceValidationError(f"{label}.{key_text} must be a scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise KaggleEvidenceValidationError(f"{label}.{key_text} must be finite")
        if isinstance(value, str) and _looks_machine_specific(value):
            raise KaggleEvidenceValidationError(
                f"{label}.{key_text} contains a machine-specific absolute path"
            )


def _read_evidence_member(archive: zipfile.ZipFile) -> tuple[bytes, zipfile.ZipInfo]:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and info.filename == EVIDENCE_FILENAME
    ]
    if not matches:
        raise KaggleEvidenceNotFound(f"{EVIDENCE_FILENAME} is missing from the Kaggle artifact")
    if len(matches) != 1:
        raise KaggleEvidenceValidationError("the evidence manifest must occur exactly once")
    info = matches[0]
    if info.file_size > 2 * 1024 * 1024:
        raise KaggleEvidenceValidationError("the evidence manifest exceeds 2 MiB")
    return archive.read(info), info


def _sha256_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_kaggle_evidence(
    archive_path: Path,
    *,
    expected_source_job_id: str | None = None,
    expected_source_run_id: str | None = None,
) -> ValidatedKaggleEvidence:
    """Validate an untrusted Kaggle ZIP completely before any MLflow call."""
    resolved_archive = archive_path.resolve()
    try:
        with zipfile.ZipFile(resolved_archive, "r") as archive:
            members = archive.infolist()
            if len(members) > MAX_BUNDLE_FILES:
                raise KaggleEvidenceValidationError("Kaggle artifact contains too many files")
            if sum(max(0, int(info.file_size)) for info in members) > MAX_BUNDLE_UNCOMPRESSED_BYTES:
                raise KaggleEvidenceValidationError("Kaggle artifact uncompressed size exceeds 2 GiB")
            for info in members:
                if info.is_dir():
                    continue
                try:
                    safe_relative_path(info.filename)
                except ValueError as exc:
                    raise KaggleEvidenceValidationError(str(exc)) from exc
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise KaggleEvidenceValidationError(
                        f"symbolic-link ZIP members are not allowed: {info.filename}"
                    )
            evidence_bytes, _ = _read_evidence_member(archive)
            try:
                manifest = json.loads(evidence_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KaggleEvidenceValidationError("evidence manifest is not valid UTF-8 JSON") from exc
            if not isinstance(manifest, dict):
                raise KaggleEvidenceValidationError("evidence manifest must be a JSON object")
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise KaggleEvidenceValidationError(
                    f"unsupported evidence schema version: {manifest.get('schema_version')!r}"
                )
            if manifest.get("execution_origin") != "kaggle":
                raise KaggleEvidenceValidationError("execution_origin must be 'kaggle'")

            source_job_id = _required_text(manifest, "source_job_id")
            source_run_id = _required_text(manifest, "source_run_id")
            identity_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
            if not identity_re.fullmatch(source_job_id) or not identity_re.fullmatch(source_run_id):
                raise KaggleEvidenceValidationError("source job/run IDs contain unsupported characters")
            _required_text(manifest, "experiment_name")
            _required_text(manifest, "run_name")
            if expected_source_job_id and source_job_id != expected_source_job_id:
                raise KaggleEvidenceValidationError("source_job_id does not match the backend job")
            if expected_source_run_id and source_run_id != expected_source_run_id:
                raise KaggleEvidenceValidationError("source_run_id does not match the backend run")

            statuses = _required_mapping(manifest, "status")
            training_status = statuses.get("training_status")
            if training_status not in TRAINING_STATUSES:
                raise KaggleEvidenceValidationError("invalid training_status")
            if statuses.get("tracking_status") not in TRACKING_STATUSES:
                raise KaggleEvidenceValidationError("invalid tracking_status")
            if statuses.get("artifact_status") not in ARTIFACT_STATUSES:
                raise KaggleEvidenceValidationError("invalid artifact_status")

            training = _required_mapping(manifest, "training")
            model_family = _required_text(training, "model_family")
            training_mode = _required_text(training, "training_mode")
            _required_text(training, "dataset")
            _required_text(training, "script")
            _required_text(training, "base_model")
            initialization_mode = _required_text(training, "initialization_mode")
            if model_family not in ALLOWED_MODEL_FAMILIES:
                raise KaggleEvidenceValidationError(f"unsupported model family: {model_family}")
            if training_mode not in ALLOWED_TRAINING_MODES:
                raise KaggleEvidenceValidationError(f"unsupported training mode: {training_mode}")
            if training_mode == "finetune":
                parent_model_id = _required_text(training, "parent_model_id")
                allowed_initializations = (
                    {"existing_model_artifact"}
                    if training_status == "success"
                    else {"existing_model_artifact", "unconfirmed"}
                )
                if initialization_mode not in allowed_initializations or not parent_model_id:
                    raise KaggleEvidenceValidationError(
                        "finetune evidence requires valid initialization provenance and parent_model_id"
                    )
            _validate_safe_scalars(training, "training")

            params = _required_mapping(manifest, "params")
            metrics = _required_mapping(manifest, "metrics")
            tags = _required_mapping(manifest, "tags")
            timestamps = _required_mapping(manifest, "timestamps")
            provenance = _required_mapping(manifest, "provenance")
            _validate_safe_scalars(params, "params")
            _validate_safe_scalars(metrics, "metrics", numeric_only=True)
            _validate_safe_scalars(tags, "tags")
            _validate_safe_scalars(timestamps, "timestamps")
            _validate_safe_scalars(provenance, "provenance")

            raw_artifacts = manifest.get("artifacts")
            if not isinstance(raw_artifacts, list):
                raise KaggleEvidenceValidationError("artifacts must be an array")
            member_infos = [
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename != EVIDENCE_FILENAME
            ]
            if len({info.filename for info in member_infos}) != len(member_infos):
                raise KaggleEvidenceValidationError("duplicate ZIP member names are not allowed")
            archive_members = {info.filename: info for info in member_infos}
            validated: list[ValidatedArtifact] = []
            listed_paths: set[str] = set()
            for index, raw_artifact in enumerate(raw_artifacts):
                if not isinstance(raw_artifact, dict):
                    raise KaggleEvidenceValidationError(f"artifacts[{index}] must be an object")
                try:
                    relative_path = safe_relative_path(_required_text(raw_artifact, "relative_path"))
                except ValueError as exc:
                    raise KaggleEvidenceValidationError(str(exc)) from exc
                if relative_path in listed_paths:
                    raise KaggleEvidenceValidationError(f"duplicate artifact path: {relative_path}")
                listed_paths.add(relative_path)
                if _required_text(raw_artifact, "name") != relative_path:
                    raise KaggleEvidenceValidationError("artifact name must equal its relative bundle path")
                role = _required_text(raw_artifact, "role")
                size_bytes = raw_artifact.get("size_bytes")
                expected_sha = str(raw_artifact.get("sha256") or "").strip().lower()
                required = raw_artifact.get("required")
                if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
                    raise KaggleEvidenceValidationError(f"invalid artifact size: {relative_path}")
                if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                    raise KaggleEvidenceValidationError(f"invalid artifact SHA-256: {relative_path}")
                if not isinstance(required, bool):
                    raise KaggleEvidenceValidationError(f"artifact required flag must be boolean: {relative_path}")
                info = archive_members.get(relative_path)
                if info is None:
                    raise KaggleEvidenceValidationError(f"artifact is missing from the bundle: {relative_path}")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise KaggleEvidenceValidationError(f"symbolic-link artifacts are not allowed: {relative_path}")
                if info.file_size != size_bytes:
                    raise KaggleEvidenceValidationError(f"artifact size mismatch: {relative_path}")
                if _sha256_zip_member(archive, info) != expected_sha:
                    raise KaggleEvidenceValidationError(f"artifact checksum mismatch: {relative_path}")
                validated.append(
                    ValidatedArtifact(relative_path, role, required, size_bytes, expected_sha)
                )

            unknown_members = sorted(set(archive_members) - listed_paths)
            if unknown_members:
                raise KaggleEvidenceValidationError(
                    f"bundle contains unknown unmanifested artifact: {unknown_members[0]}"
                )
            artifact_status = statuses["artifact_status"]
            if artifact_status == "missing" and validated:
                raise KaggleEvidenceValidationError("artifact_status=missing requires an empty artifact list")
            if artifact_status == "complete":
                if not validated or not any(item.required for item in validated):
                    raise KaggleEvidenceValidationError(
                        "artifact_status=complete requires at least one required artifact"
                    )
                if statuses["training_status"] == "success" and not any(
                    item.role == "model" and item.required for item in validated
                ):
                    raise KaggleEvidenceValidationError(
                        "successful training requires a required model artifact"
                    )
    except zipfile.BadZipFile as exc:
        raise KaggleEvidenceValidationError("Kaggle artifact is not a valid ZIP") from exc

    return ValidatedKaggleEvidence(
        archive_path=resolved_archive,
        manifest=manifest,
        evidence_bytes=evidence_bytes,
        evidence_sha256=sha256_bytes(evidence_bytes),
        artifacts=tuple(validated),
    )


def ensure_kaggle_ingestion_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_kaggle_ingestion (
                source_job_id TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                ingestion_key TEXT NOT NULL UNIQUE,
                canonical_mlflow_run_id TEXT,
                experiment_name TEXT NOT NULL,
                ingestion_status TEXT NOT NULL,
                retriable INTEGER NOT NULL DEFAULT 0,
                tracking_status TEXT NOT NULL,
                artifact_status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ingested_at TEXT,
                PRIMARY KEY (source_job_id, source_run_id)
            )
            """
        )
        connection.commit()


def _row_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_kaggle_ingestion_record(
    db_path: Path, *, source_job_id: str, source_run_id: str
) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM mlflow_kaggle_ingestion
                WHERE source_job_id = ? AND source_run_id = ?
                """,
                (source_job_id, source_run_id),
            ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    return _row_payload(row)


def _ingestion_key(evidence: ValidatedKaggleEvidence) -> str:
    identity = "\0".join(
        ("kaggle", evidence.source_job_id, evidence.source_run_id, evidence.evidence_sha256)
    )
    return sha256_bytes(identity.encode("utf-8"))


def _reserve_ingestion(db_path: Path, evidence: ValidatedKaggleEvidence) -> dict[str, Any]:
    ensure_kaggle_ingestion_schema(db_path)
    now = datetime.now(timezone.utc).isoformat()
    key = _ingestion_key(evidence)
    experiment_name = f"viettoxic-kaggle-{evidence.manifest['training']['model_family'].replace('_', '-')}"
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT * FROM mlflow_kaggle_ingestion
            WHERE source_job_id = ? AND source_run_id = ?
            """,
            (evidence.source_job_id, evidence.source_run_id),
        ).fetchone()
        if existing is not None:
            existing_payload = dict(existing)
            if existing_payload["evidence_sha256"] != evidence.evidence_sha256:
                raise KaggleEvidenceConflictError(
                    "the source job/run identity already has different evidence; operator review is required"
                )
            if existing_payload["ingestion_status"] == "completed":
                connection.commit()
                return {"action": "existing", **existing_payload}
            if existing_payload["ingestion_status"] == "ingesting":
                connection.commit()
                return {"action": "in_progress", **existing_payload}
            connection.execute(
                """
                UPDATE mlflow_kaggle_ingestion
                SET ingestion_status = 'ingesting', retriable = 0, error_message = NULL, updated_at = ?
                WHERE source_job_id = ? AND source_run_id = ?
                """,
                (now, evidence.source_job_id, evidence.source_run_id),
            )
        else:
            connection.execute(
                """
                INSERT INTO mlflow_kaggle_ingestion (
                    source_job_id, source_run_id, evidence_sha256, ingestion_key,
                    experiment_name, ingestion_status, retriable, tracking_status,
                    artifact_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'ingesting', 0, ?, ?, ?, ?)
                """,
                (
                    evidence.source_job_id,
                    evidence.source_run_id,
                    evidence.evidence_sha256,
                    key,
                    experiment_name,
                    evidence.manifest["status"]["tracking_status"],
                    evidence.manifest["status"]["artifact_status"],
                    now,
                    now,
                ),
            )
        connection.commit()
    return {"action": "ingest", "ingestion_key": key, "experiment_name": experiment_name}


def _complete_ingestion(
    db_path: Path,
    evidence: ValidatedKaggleEvidence,
    *,
    canonical_run_id: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE mlflow_kaggle_ingestion
            SET canonical_mlflow_run_id = ?, ingestion_status = 'completed', retriable = 0,
                error_message = NULL, updated_at = ?, ingested_at = ?
            WHERE source_job_id = ? AND source_run_id = ?
            """,
            (canonical_run_id, now, now, evidence.source_job_id, evidence.source_run_id),
        )
        connection.commit()
    record = get_kaggle_ingestion_record(
        db_path, source_job_id=evidence.source_job_id, source_run_id=evidence.source_run_id
    )
    return {"action": "created", **(record or {})}


def _fail_ingestion(db_path: Path, evidence: ValidatedKaggleEvidence, error: Exception) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE mlflow_kaggle_ingestion
            SET ingestion_status = 'failed', retriable = 1, error_message = ?, updated_at = ?
            WHERE source_job_id = ? AND source_run_id = ?
            """,
            (f"{type(error).__name__}: canonical MLflow ingestion failed", now, evidence.source_job_id, evidence.source_run_id),
        )
        connection.commit()


def _conflict_ingestion(db_path: Path, evidence: ValidatedKaggleEvidence, detail: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE mlflow_kaggle_ingestion
            SET ingestion_status = 'conflict', retriable = 0, error_message = ?, updated_at = ?
            WHERE source_job_id = ? AND source_run_id = ?
            """,
            (detail[:1000], now, evidence.source_job_id, evidence.source_run_id),
        )
        connection.commit()


def _extract_validated_artifacts(evidence: ValidatedKaggleEvidence, target_root: Path) -> None:
    with zipfile.ZipFile(evidence.archive_path, "r") as archive:
        for artifact in evidence.artifacts:
            target = target_root.joinpath(*artifact.relative_path.split("/")).resolve()
            try:
                target.relative_to(target_root.resolve())
            except ValueError as exc:
                raise KaggleEvidenceValidationError("artifact extraction escaped the staging root") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(artifact.relative_path, "r") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    (target_root / EVIDENCE_FILENAME).write_bytes(evidence.evidence_bytes)


def ingest_kaggle_evidence(
    evidence: ValidatedKaggleEvidence,
    *,
    db_path: Path,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Create or recover exactly one canonical MLflow run for validated evidence."""
    reservation = _reserve_ingestion(db_path, evidence)
    if reservation["action"] == "existing":
        return reservation
    if reservation["action"] == "in_progress":
        return reservation

    ingestion_key = str(reservation["ingestion_key"])
    experiment_name = str(reservation["experiment_name"])
    local_run_id: str | None = None
    client = None
    try:
        mlflow, experiment = prepare_mlflow_client(
            experiment_name=experiment_name,
            timeout=timeout,
        )
        client = mlflow.tracking.MlflowClient()
        source_filter = (
            f"tags.`execution_origin` = 'kaggle' and "
            f"tags.`source_job_id` = '{evidence.source_job_id}' and "
            f"tags.`source_run_id` = '{evidence.source_run_id}'"
        )
        source_runs = client.search_runs(
            [experiment.experiment_id], filter_string=source_filter, max_results=20
        )
        for source_run in source_runs:
            existing_sha = source_run.data.tags.get("source_evidence_sha256")
            if existing_sha != evidence.evidence_sha256:
                raise KaggleEvidenceConflictError(
                    "canonical MLflow already contains conflicting evidence for this source identity"
                )
            if source_run.info.lifecycle_stage == "active" and source_run.info.status == "FINISHED":
                return _complete_ingestion(
                    db_path, evidence, canonical_run_id=source_run.info.run_id
                )
            if source_run.info.lifecycle_stage == "active":
                client.delete_run(source_run.info.run_id)

        reserved_tags = {
            "execution_origin": "kaggle",
            "ingestion_mode": "post_run",
            "source_job_id": evidence.source_job_id,
            "source_run_id": evidence.source_run_id,
            "source_evidence_sha256": evidence.evidence_sha256,
            "evidence_schema_version": str(SCHEMA_VERSION),
            "ingestion_key": ingestion_key,
            "source_training_status": evidence.manifest["status"]["training_status"],
            "source_tracking_status": evidence.manifest["status"]["tracking_status"],
            "source_artifact_status": evidence.manifest["status"]["artifact_status"],
        }
        source_tags = {str(key): str(value) for key, value in evidence.manifest["tags"].items()}
        source_tags.update(reserved_tags)
        with mlflow.start_run(run_name=evidence.manifest["run_name"]) as active_run:
            local_run_id = active_run.info.run_id
            mlflow.set_tags(source_tags)
            if evidence.manifest["params"]:
                mlflow.log_params(evidence.manifest["params"])
            if evidence.manifest["metrics"]:
                mlflow.log_metrics(evidence.manifest["metrics"])
            with tempfile.TemporaryDirectory(prefix="viettoxic-kaggle-ingest-") as temp_dir:
                staging_root = Path(temp_dir)
                _extract_validated_artifacts(evidence, staging_root)
                mlflow.log_artifacts(str(staging_root))
        return _complete_ingestion(db_path, evidence, canonical_run_id=local_run_id)
    except KaggleEvidenceConflictError as exc:
        _conflict_ingestion(db_path, evidence, str(exc))
        raise
    except Exception as exc:
        if local_run_id and client is not None:
            try:
                client.delete_run(local_run_id)
            except Exception:
                pass
        _fail_ingestion(db_path, evidence, exc)
        if isinstance(exc, MlflowClientConfigurationError):
            raise KaggleEvidenceIngestionUnavailable(str(exc)) from exc
        raise KaggleEvidenceIngestionUnavailable(
            f"canonical MLflow ingestion failed and is retriable: {type(exc).__name__}: {exc}"
        ) from exc
