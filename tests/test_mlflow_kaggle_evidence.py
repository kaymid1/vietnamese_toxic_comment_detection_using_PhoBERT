import json
import zipfile
from pathlib import Path

import pytest

from backend.mlflow_kaggle_ingest import (
    KaggleEvidenceValidationError,
    validate_kaggle_evidence,
)
from kaggle.mlflow_evidence import (
    EVIDENCE_FILENAME,
    build_directory_artifacts,
    build_evidence_manifest,
    canonical_json_bytes,
    portable_scalar_mapping,
    write_evidence_file,
)


def _manifest(
    artifacts,
    *,
    family="tfidf_lr",
    mode="retrain",
    training_status="success",
    tracking_status="complete",
    artifact_status="complete",
):
    parent = "phobert/parent-v7" if mode == "finetune" else None
    return build_evidence_manifest(
        source_job_id="real_job_123",
        source_run_id="kaggle_run_123",
        experiment_name=f"viettoxic-kaggle-{family.replace('_', '-')}",
        run_name="kaggle_run_123",
        training={
            "model_family": family,
            "training_mode": mode,
            "dataset": "clean_victsd_gold",
            "script": "train_phobert.py" if family == "phobert" else "viettoxic_mlflow_retrain.py",
            "base_model": parent or ("vinai/phobert-base-v2" if family == "phobert" else "sklearn.LogisticRegression"),
            "parent_model_id": parent,
            "initialization_mode": "existing_model_artifact" if mode == "finetune" else "pretrained_base",
            "training_config_id": "kaggle_run_123",
        },
        training_status=training_status,
        tracking_status=tracking_status,
        artifact_status=artifact_status,
        params={"epochs": 3, "batch_size": 8},
        metrics={"macro_f1": 0.76, "toxic_f1": 0.71},
        tags={"model_family": family, "training_mode": mode},
        artifacts=artifacts,
        timestamps={"finished_at": "2026-08-17T00:00:00+00:00"},
        provenance={"notebook_sha256": "a" * 64, "parent_source_run_id": "parent-run" if parent else None},
    )


def _build_valid_zip(tmp_path: Path, *, family="tfidf_lr", mode="retrain"):
    root = tmp_path / f"payload-{family}-{mode}"
    root.mkdir()
    if family == "tfidf_lr":
        (root / "model_lr.joblib").write_bytes(b"model")
        (root / "vectorizer.joblib").write_bytes(b"vectorizer")
    else:
        (root / "config.json").write_text('{"model_type":"roberta"}', encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")
    (root / "metrics.json").write_text('{"macro_f1":0.76}', encoding="utf-8")
    manifest = _manifest(build_directory_artifacts(root), family=family, mode=mode)
    write_evidence_file(root / EVIDENCE_FILENAME, manifest)
    archive_path = tmp_path / f"{family}-{mode}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root).as_posix())
    return archive_path, manifest


def _rewrite_manifest(archive_path: Path, mutate):
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
    manifest = json.loads(members[EVIDENCE_FILENAME])
    mutate(manifest)
    members[EVIDENCE_FILENAME] = canonical_json_bytes(manifest)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_successful_retrain_produces_valid_portable_evidence(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    evidence = validate_kaggle_evidence(
        archive_path,
        expected_source_job_id="real_job_123",
        expected_source_run_id="kaggle_run_123",
    )
    assert evidence.manifest["status"] == {
        "training_status": "success",
        "tracking_status": "complete",
        "artifact_status": "complete",
    }
    assert all(not Path(item.relative_path).is_absolute() for item in evidence.artifacts)


def test_successful_finetune_retains_proven_parent(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path, family="phobert", mode="finetune")
    evidence = validate_kaggle_evidence(archive_path)
    assert evidence.manifest["training"]["parent_model_id"] == "phobert/parent-v7"
    assert evidence.manifest["training"]["initialization_mode"] == "existing_model_artifact"


def test_training_failure_and_missing_artifact_are_explicit(tmp_path):
    manifest = _manifest(
        [],
        training_status="failed",
        tracking_status="failed",
        artifact_status="missing",
    )
    archive_path = tmp_path / "failure.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(EVIDENCE_FILENAME, canonical_json_bytes(manifest))
    evidence = validate_kaggle_evidence(archive_path)
    assert evidence.manifest["status"]["training_status"] == "failed"
    assert evidence.manifest["status"]["artifact_status"] == "missing"
    assert evidence.artifacts == ()


def test_failed_finetune_does_not_claim_completed_initialization(tmp_path):
    manifest = _manifest(
        [],
        family="phobert",
        mode="finetune",
        training_status="failed",
        tracking_status="failed",
        artifact_status="missing",
    )
    manifest["training"]["initialization_mode"] = "unconfirmed"
    archive_path = tmp_path / "failed-finetune.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(EVIDENCE_FILENAME, canonical_json_bytes(manifest))

    evidence = validate_kaggle_evidence(archive_path)
    assert evidence.manifest["training"]["parent_model_id"] == "phobert/parent-v7"
    assert evidence.manifest["training"]["initialization_mode"] == "unconfirmed"


def test_partial_tracking_does_not_erase_successful_training(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    _rewrite_manifest(
        archive_path,
        lambda manifest: manifest["status"].update({"tracking_status": "partial"}),
    )
    evidence = validate_kaggle_evidence(archive_path)
    assert evidence.manifest["status"]["training_status"] == "success"
    assert evidence.manifest["status"]["tracking_status"] == "partial"


def test_manifest_generation_filters_secrets_and_machine_paths(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    _rewrite_manifest(
        archive_path,
        lambda manifest: manifest["tags"].update({"safe_tag": "kept"}),
    )
    evidence = validate_kaggle_evidence(archive_path)
    assert evidence.manifest["tags"] == {
        "model_family": "tfidf_lr",
        "training_mode": "retrain",
        "safe_tag": "kept",
    }

    filtered = portable_scalar_mapping(
        {"api_token": "do-not-export", "local_path": r"D:\private\model"}
    )
    assert filtered == {}


def test_sensitive_manifest_key_is_rejected(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    _rewrite_manifest(
        archive_path,
        lambda manifest: manifest["tags"].update({"api_token": "unexpected"}),
    )
    with pytest.raises(KaggleEvidenceValidationError, match="prohibited sensitive key"):
        validate_kaggle_evidence(archive_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"schema_version": 999}), "unsupported evidence schema"),
        (lambda value: value.pop("source_job_id"), "source_job_id"),
        (lambda value: value["status"].update({"tracking_status": "unknown"}), "invalid tracking_status"),
        (lambda value: value["metrics"].update({"macro_f1": "0.76"}), "must be numeric"),
        (lambda value: value["training"].pop("parent_model_id", None), "parent_model_id"),
    ],
)
def test_invalid_contract_fields_are_rejected_before_ingestion(tmp_path, mutate, message):
    archive_path, _ = _build_valid_zip(tmp_path, family="phobert", mode="finetune")
    _rewrite_manifest(archive_path, mutate)
    with pytest.raises(KaggleEvidenceValidationError, match=message):
        validate_kaggle_evidence(archive_path)


@pytest.mark.parametrize("unsafe_path", ["../../outside", r"..\..\outside", "/absolute/model.bin", r"D:\absolute\model.bin"])
def test_path_traversal_and_absolute_paths_are_rejected(tmp_path, unsafe_path):
    archive_path, _ = _build_valid_zip(tmp_path)

    def mutate(manifest):
        manifest["artifacts"][0]["name"] = unsafe_path
        manifest["artifacts"][0]["relative_path"] = unsafe_path

    _rewrite_manifest(archive_path, mutate)
    with pytest.raises(KaggleEvidenceValidationError, match="artifact path"):
        validate_kaggle_evidence(archive_path)


def test_artifact_checksum_mismatch_is_rejected(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    _rewrite_manifest(
        archive_path,
        lambda manifest: manifest["artifacts"][0].update({"sha256": "0" * 64}),
    )
    with pytest.raises(KaggleEvidenceValidationError, match="checksum mismatch"):
        validate_kaggle_evidence(archive_path)


def test_artifact_size_mismatch_is_rejected(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    _rewrite_manifest(
        archive_path,
        lambda manifest: manifest["artifacts"][0].update(
            {"size_bytes": manifest["artifacts"][0]["size_bytes"] + 1}
        ),
    )
    with pytest.raises(KaggleEvidenceValidationError, match="size mismatch"):
        validate_kaggle_evidence(archive_path)


def test_manifested_artifact_missing_from_bundle_is_rejected(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir() and info.filename != "model_lr.joblib"
        }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    with pytest.raises(KaggleEvidenceValidationError, match="artifact is missing"):
        validate_kaggle_evidence(archive_path)


def test_unknown_unmanifested_artifact_is_rejected(tmp_path):
    archive_path, _ = _build_valid_zip(tmp_path)
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("rogue.bin", b"not declared")
    with pytest.raises(KaggleEvidenceValidationError, match="unknown unmanifested artifact"):
        validate_kaggle_evidence(archive_path)
