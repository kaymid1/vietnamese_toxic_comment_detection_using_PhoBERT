"""Portable Kaggle-to-MLflow evidence contract.

This module intentionally uses only the Python standard library so the webhook
receiver can embed it into an isolated Kaggle job without credentials or local
runtime-path assumptions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
EVIDENCE_FILENAME = "mlflow_run_evidence.json"
TRAINING_STATUSES = frozenset({"success", "failed"})
TRACKING_STATUSES = frozenset({"complete", "partial", "failed", "disabled"})
ARTIFACT_STATUSES = frozenset({"complete", "partial", "missing"})
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower())
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _looks_machine_specific(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    lowered = normalized.lower()
    return bool(
        _WINDOWS_ABSOLUTE_RE.match(value.strip())
        or normalized.startswith("/")
        or "://" in lowered
    )


def portable_scalar_mapping(
    values: Mapping[str, Any] | None,
    *,
    numeric_only: bool = False,
) -> dict[str, Any]:
    """Keep safe scalar evidence and omit secrets or machine paths."""
    output: dict[str, Any] = {}
    for raw_key, value in (values or {}).items():
        key = str(raw_key).strip()
        if not key or _is_sensitive_key(key) or value is None:
            continue
        if numeric_only:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            output[key] = value
            continue
        if not isinstance(value, (str, int, float, bool)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, str) and _looks_machine_specific(value):
            continue
        output[key] = value
    return output


def safe_relative_path(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or "\\" in raw or _WINDOWS_ABSOLUTE_RE.match(raw):
        raise ValueError(f"artifact path must be portable and relative: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"artifact path must stay inside the bundle: {raw!r}")
    return path.as_posix()


def _artifact_role(relative_path: str) -> tuple[str, bool]:
    name = PurePosixPath(relative_path).name.lower()
    if name in {
        "model.safetensors",
        "pytorch_model.bin",
        "model_lr.joblib",
        "model_lr.pkl",
        "vectorizer.joblib",
        "vectorizer.pkl",
    }:
        return "model", True
    if name == "metrics.json":
        return "metrics", True
    if name in {"training_manifest.json", "run_config.json", "training_evidence.json"}:
        return "provenance", False
    if name.endswith("config.json") or name.endswith("tokenizer.json"):
        return "model_config", False
    return "supporting", False


def build_directory_artifacts(root: Path, paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    candidates = list(paths) if paths is not None else list(resolved_root.rglob("*"))
    artifacts: list[dict[str, Any]] = []
    for candidate in sorted((Path(path) for path in candidates), key=lambda item: str(item)):
        resolved = candidate.resolve()
        if not resolved.is_file() or resolved.name == EVIDENCE_FILENAME:
            continue
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"artifact is outside the bundle root: {candidate}") from exc
        relative = safe_relative_path(relative)
        role, required = _artifact_role(relative)
        artifacts.append(
            {
                "name": relative,
                "relative_path": relative,
                "role": role,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
                "required": required,
            }
        )
    return artifacts


def build_zip_artifacts(archive_path: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir() or PurePosixPath(info.filename).name == EVIDENCE_FILENAME:
                continue
            relative = safe_relative_path(info.filename)
            digest = hashlib.sha256()
            with archive.open(info, "r") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            role, required = _artifact_role(relative)
            artifacts.append(
                {
                    "name": relative,
                    "relative_path": relative,
                    "role": role,
                    "size_bytes": info.file_size,
                    "sha256": digest.hexdigest(),
                    "required": required,
                }
            )
    return artifacts


def build_evidence_manifest(
    *,
    source_job_id: str,
    source_run_id: str,
    experiment_name: str,
    run_name: str,
    training: Mapping[str, Any],
    training_status: str,
    tracking_status: str,
    artifact_status: str,
    params: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
    tags: Mapping[str, Any] | None,
    artifacts: Iterable[Mapping[str, Any]],
    timestamps: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if training_status not in TRAINING_STATUSES:
        raise ValueError(f"unsupported training status: {training_status}")
    if tracking_status not in TRACKING_STATUSES:
        raise ValueError(f"unsupported tracking status: {tracking_status}")
    if artifact_status not in ARTIFACT_STATUSES:
        raise ValueError(f"unsupported artifact status: {artifact_status}")

    portable_training = portable_scalar_mapping(training)
    portable_timestamps = portable_scalar_mapping(timestamps)
    portable_provenance = portable_scalar_mapping(provenance)
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_origin": "kaggle",
        "source_job_id": str(source_job_id).strip(),
        "source_run_id": str(source_run_id).strip(),
        "experiment_name": str(experiment_name).strip(),
        "run_name": str(run_name).strip(),
        "training": portable_training,
        "status": {
            "training_status": training_status,
            "tracking_status": tracking_status,
            "artifact_status": artifact_status,
        },
        "params": portable_scalar_mapping(params),
        "metrics": portable_scalar_mapping(metrics, numeric_only=True),
        "tags": portable_scalar_mapping(tags),
        "artifacts": [dict(item) for item in artifacts],
        "timestamps": portable_timestamps,
        "provenance": portable_provenance,
    }


def write_evidence_file(path: Path, manifest: Mapping[str, Any]) -> str:
    content = canonical_json_bytes(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_bytes(content)


def write_evidence_to_zip(archive_path: Path, manifest: Mapping[str, Any]) -> str:
    """Atomically add or replace the stable evidence member in a Kaggle ZIP."""
    content = canonical_json_bytes(manifest)
    temp_path = archive_path.with_name(f".{archive_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(archive_path, "r") as source, zipfile.ZipFile(
            temp_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for info in source.infolist():
                if info.is_dir() or PurePosixPath(info.filename).name == EVIDENCE_FILENAME:
                    continue
                safe_relative_path(info.filename)
                with source.open(info, "r") as source_member, target.open(info, "w") as target_member:
                    shutil.copyfileobj(source_member, target_member, length=1024 * 1024)
            target.writestr(EVIDENCE_FILENAME, content)
        os.replace(temp_path, archive_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return sha256_bytes(content)
