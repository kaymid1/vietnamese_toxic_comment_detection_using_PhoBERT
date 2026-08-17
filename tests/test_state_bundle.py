import json
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from backend import state_bundle
from backend.artifact_refs import resolve_artifact_ref


WINDOWS_RUNTIME = r"D:\Code\Thesis\Thesis\.runtime\model_registry\abc\artifact.zip"
WINDOWS_MODEL = r"D:\Code\Thesis\Thesis\models\options\phobert\v2"


def _write_models(root: Path) -> None:
    phobert = root / "phobert" / "v2"
    phobert.mkdir(parents=True)
    (phobert / "config.json").write_text("{}\n", encoding="utf-8")
    (phobert / "model.safetensors").write_bytes(b"small-phobert")
    tfidf = root / "tfidf_lr" / "baseline"
    tfidf.mkdir(parents=True)
    (tfidf / "vectorizer.pkl").write_bytes(b"vectorizer")
    (tfidf / "model_lr.pkl").write_bytes(b"model")


def _write_legacy_evidence(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in state_bundle.LEGACY_EVIDENCE_FILES:
        (root / name).write_text(json.dumps({"fixture": name}) + "\n", encoding="utf-8")


def _insert_portable_fixture(db: Path) -> None:
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO mlflow_training_artifact (run_name, artifact_path, notes, created_at) VALUES (?, ?, ?, ?)",
            ("fixture", WINDOWS_RUNTIME, "required rollback artifact", "2026-08-17T00:00:00Z"),
        )
        connection.execute(
            """
            INSERT INTO mlflow_model_version (
                model_family, model_id, source_run_id, artifact_path, artifact_checksum, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("phobert", "fixture-phobert", "fixture-run", WINDOWS_MODEL, "a" * 64, "production", "2026-08-17T00:00:00Z"),
        )
        connection.execute(
            """
            INSERT INTO mlflow_production_slot (
                model_family, active_model_id, active_run_id, artifact_checksum, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("phobert", "fixture-phobert", "fixture-run", "a" * 64, "2026-08-17T00:00:00Z"),
        )
        connection.commit()


@pytest.fixture()
def source_state(qa_env: dict, monkeypatch: pytest.MonkeyPatch) -> state_bundle.SourcePaths:
    base = qa_env["base_dir"]
    data = base / "data"
    runtime = base / ".runtime"
    models = base / "models" / "options"
    artifact = runtime / "model_registry" / "abc" / "artifact.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"rollback-artifact")
    _write_models(models)
    evidence = data / "mlflow" / "evidence"
    _write_legacy_evidence(evidence)
    _insert_portable_fixture(qa_env["feedback_db"])
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    monkeypatch.setenv("APP_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(models))
    return state_bundle.SourcePaths(
        data_dir=data,
        runtime_dir=runtime,
        model_options_dir=models,
        feedback_db=qa_env["feedback_db"],
        active_mlflow_db=data / "mlflow" / "mlflow.db",
        active_mlflow_artifacts=data / "mlflow" / "artifacts",
        legacy_evidence_dir=evidence,
    )


def _target(tmp_path: Path) -> state_bundle.TargetPaths:
    mac = tmp_path / "Users" / "test"
    return state_bundle.TargetPaths(
        data_dir=mac / "VietToxicData",
        runtime_dir=mac / "VietToxicRuntime",
        model_options_dir=mac / "VietToxicModels",
    )


def _read_refs(db: Path) -> tuple[str, str]:
    with sqlite3.connect(db) as connection:
        runtime = connection.execute(
            "SELECT artifact_path FROM mlflow_training_artifact WHERE run_name='fixture'"
        ).fetchone()[0]
        model = connection.execute(
            "SELECT artifact_path FROM mlflow_model_version WHERE model_id='fixture-phobert'"
        ).fetchone()[0]
    return str(runtime), str(model)


def _export(source: state_bundle.SourcePaths, tmp_path: Path, name: str = "bundle") -> Path:
    bundle = tmp_path / name
    result = state_bundle.export_bundle(output=bundle, dry_run=False, source_paths=source)
    assert result["source_feedback_unchanged"] is True
    return bundle


def _refresh_integrity_metadata(bundle: Path) -> None:
    manifest_path = bundle / state_bundle.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = bundle.joinpath(*Path(record["path"]).parts)
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = state_bundle._sha256_file(path)
    manifest["content_identity"]["value"] = state_bundle._content_identity(
        manifest["files"], manifest["statistics"]["application_state_counts"]
    )
    state_bundle._write_json(manifest_path, manifest)
    state_bundle._write_checksums(bundle)


def test_windows_snapshot_to_macos_like_target_preserves_source(
    source_state, qa_env, monkeypatch, tmp_path
):
    source_before = source_state.feedback_db.read_bytes()
    bundle = _export(source_state, tmp_path)
    assert source_state.feedback_db.read_bytes() == source_before
    assert _read_refs(source_state.feedback_db) == (WINDOWS_RUNTIME, WINDOWS_MODEL)
    assert _read_refs(bundle / "application" / "feedback.db") == (
        "runtime://model_registry/abc/artifact.zip",
        "model://phobert/v2",
    )

    target = _target(tmp_path)
    dry_run = state_bundle.import_bundle(bundle, target_paths=target)
    assert dry_run["zero_target_writes"] is True
    assert not target.data_dir.exists()
    applied = state_bundle.import_bundle(bundle, target_paths=target, apply=True)
    assert applied["services_started"] is False
    assert applied["verification"]["valid"] is True
    assert _read_refs(target.feedback_db) == (
        "runtime://model_registry/abc/artifact.zip",
        "model://phobert/v2",
    )
    assert resolve_artifact_ref(
        "runtime://model_registry/abc/artifact.zip", roots=target.artifact_roots
    ) == target.runtime_dir / "model_registry" / "abc" / "artifact.zip"
    assert resolve_artifact_ref("model://phobert/v2", roots=target.artifact_roots) == target.model_options_dir / "phobert" / "v2"
    assert (target.runtime_dir / "model_registry" / "abc" / "artifact.zip").read_bytes() == b"rollback-artifact"
    before_upgrade_counts = state_bundle._table_counts(target.feedback_db)
    monkeypatch.setattr(qa_env["app_module"], "FEEDBACK_DIR", target.feedback_db.parent)
    monkeypatch.setattr(qa_env["app_module"], "FEEDBACK_DB_PATH", target.feedback_db)
    qa_env["app_module"].init_feedback_db()
    assert state_bundle._sqlite_integrity(target.feedback_db) == "ok"
    assert state_bundle._table_counts(target.feedback_db) == before_upgrade_counts


def test_deterministic_content_identity_excludes_export_timestamp(source_state, tmp_path):
    first = _export(source_state, tmp_path, "first")
    second = _export(source_state, tmp_path, "second")
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["content_identity"] == second_manifest["content_identity"]


def test_missing_required_and_external_references_are_reported(source_state, tmp_path):
    (source_state.runtime_dir / "model_registry" / "abc" / "artifact.zip").unlink()
    with sqlite3.connect(source_state.feedback_db) as connection:
        connection.execute(
            "INSERT INTO mlflow_training_artifact (run_name, artifact_path, created_at) VALUES (?, ?, ?)",
            ("external", r"D:\Downloads\unknown.zip", "2026-08-17T00:00:00Z"),
        )
        connection.commit()
    report = state_bundle.export_bundle(dry_run=True, source_paths=source_state)
    codes = {item["error_code"] for item in report["persistent_references"]}
    assert "MISSING_REFERENCED_ARTIFACT" in codes
    assert "EXTERNAL_REFERENCE_REQUIRES_REVIEW" in codes
    with pytest.raises(state_bundle.StateBundleError, match="invalid"):
        state_bundle.export_bundle(output=tmp_path / "refused", dry_run=False, source_paths=source_state)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape", r"..\escape", "/absolute/member", r"C:\absolute\member", "C:drive-relative", "//server/share"],
)
def test_checksum_inventory_rejects_unsafe_paths(source_state, tmp_path, unsafe_path):
    bundle = _export(source_state, tmp_path)
    (bundle / "checksums.sha256").write_text(f"{'0' * 64}  {unsafe_path}\n", encoding="utf-8")
    with pytest.raises(state_bundle.StateBundleError, match="Unsafe bundle-relative path"):
        state_bundle.verify_bundle(bundle)


def test_tampering_missing_members_and_unsupported_schema_fail(source_state, tmp_path):
    bundle = _export(source_state, tmp_path)
    manifest = bundle / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(state_bundle.StateBundleError, match="SHA-256 mismatch"):
        state_bundle.verify_bundle(bundle)

    bundle = _export(source_state, tmp_path, "missing")
    (bundle / "models" / "options" / "tfidf_lr" / "baseline" / "model_lr.pkl").unlink()
    with pytest.raises(state_bundle.StateBundleError, match="Bundle member mismatch"):
        state_bundle.verify_bundle(bundle)

    bundle = _export(source_state, tmp_path, "schema")
    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    state_bundle._write_json(bundle / "manifest.json", payload)
    state_bundle._write_checksums(bundle)
    with pytest.raises(state_bundle.StateBundleError, match="Unsupported bundle schema"):
        state_bundle.verify_bundle(bundle)


def test_size_mismatch_unexpected_component_and_corrupt_db_fail(source_state, tmp_path):
    bundle = _export(source_state, tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["size_bytes"] += 1
    state_bundle._write_json(manifest_path, manifest)
    state_bundle._write_checksums(bundle)
    with pytest.raises(state_bundle.StateBundleError, match="Size mismatch"):
        state_bundle.verify_bundle(bundle)

    bundle = _export(source_state, tmp_path, "component")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["components"]["unexpected"] = {"status": "present"}
    state_bundle._write_json(bundle / "manifest.json", manifest)
    state_bundle._write_checksums(bundle)
    with pytest.raises(state_bundle.StateBundleError, match="Unexpected bundle component"):
        state_bundle.verify_bundle(bundle)

    bundle = _export(source_state, tmp_path, "corrupt")
    (bundle / "application" / "feedback.db").write_bytes(b"not sqlite")
    _refresh_integrity_metadata(bundle)
    with pytest.raises(state_bundle.StateBundleError, match="SQLite integrity check failed"):
        state_bundle.verify_bundle(bundle)


def test_target_collision_refuses_and_replace_creates_sqlite_backup(source_state, tmp_path):
    bundle = _export(source_state, tmp_path)
    target = _target(tmp_path)
    target.feedback_db.parent.mkdir(parents=True)
    with closing(sqlite3.connect(target.feedback_db)) as connection:
        connection.execute("CREATE TABLE original_marker (value TEXT)")
        connection.execute("INSERT INTO original_marker VALUES ('keep')")
        connection.commit()
    target.model_options_dir.mkdir(parents=True)
    (target.model_options_dir / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(state_bundle.StateBundleError, match="TARGET_STATE_EXISTS"):
        state_bundle.import_bundle(bundle, target_paths=target, apply=True)
    backup = tmp_path / "target-backup"
    result = state_bundle.import_bundle(
        bundle,
        target_paths=target,
        apply=True,
        replace_existing=True,
        backup_dir=backup,
    )
    assert result["verification"]["valid"] is True
    with sqlite3.connect(backup / "application-feedback" / "feedback.db") as connection:
        assert connection.execute("SELECT value FROM original_marker").fetchone()[0] == "keep"
    assert (backup / "models-options" / "old.txt").read_text(encoding="utf-8") == "old"


def test_staging_and_promotion_failures_preserve_original_target(source_state, tmp_path, monkeypatch):
    bundle = _export(source_state, tmp_path)
    target = _target(tmp_path)
    target.feedback_db.parent.mkdir(parents=True)
    with closing(sqlite3.connect(target.feedback_db)) as connection:
        connection.execute("CREATE TABLE original_marker (value TEXT)")
        connection.execute("INSERT INTO original_marker VALUES ('keep')")
        connection.commit()
    before = target.feedback_db.read_bytes()
    original_copy = state_bundle._copy_install_source

    def fail_copy(source, staging, *, label):
        if label == "models-options":
            raise OSError("injected staging failure")
        return original_copy(source, staging, label=label)

    monkeypatch.setattr(state_bundle, "_copy_install_source", fail_copy)
    with pytest.raises(OSError, match="staging failure"):
        state_bundle.import_bundle(
            bundle, target_paths=target, apply=True, replace_existing=True, backup_dir=tmp_path / "backup-stage"
        )
    assert target.feedback_db.read_bytes() == before

    monkeypatch.setattr(state_bundle, "_copy_install_source", original_copy)
    with pytest.raises(state_bundle.StateBundleError, match="promotion failure"):
        state_bundle.import_bundle(
            bundle,
            target_paths=target,
            apply=True,
            replace_existing=True,
            backup_dir=tmp_path / "backup-promote",
            _fail_after_promotions=1,
        )
    assert target.feedback_db.read_bytes() == before


def test_symlink_bundle_member_is_rejected(source_state, tmp_path):
    bundle = _export(source_state, tmp_path)
    link = bundle / "models" / "options" / "symlink.bin"
    try:
        link.symlink_to(bundle / "manifest.json")
    except OSError:
        pytest.skip("Windows symlink privilege is unavailable")
    with pytest.raises(state_bundle.StateBundleError, match="Symlinks/junctions are not allowed"):
        state_bundle.verify_bundle(bundle)


def test_windows_junction_escape_is_rejected(tmp_path):
    if os.name != "nt" or not hasattr(Path, "is_junction"):
        pytest.skip("NTFS junction test is Windows-specific")
    target = tmp_path / "outside"
    target.mkdir()
    junction = tmp_path / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("NTFS junction creation is unavailable")
    assert junction.is_junction()
    with pytest.raises(state_bundle.StateBundleError, match="Symlinks/junctions are not allowed"):
        state_bundle._reject_symlink(junction, label="untrusted bundle")


def test_malformed_manifest_duplicate_checksum_and_duplicate_destination_fail(source_state, tmp_path):
    bundle = _export(source_state, tmp_path)
    (bundle / "manifest.json").write_text("{broken", encoding="utf-8")
    state_bundle._write_checksums(bundle)
    with pytest.raises(state_bundle.StateBundleError, match="Manifest is missing or malformed"):
        state_bundle.verify_bundle(bundle)

    bundle = _export(source_state, tmp_path, "duplicate-checksum")
    checksums = (bundle / "checksums.sha256").read_text(encoding="utf-8")
    first = checksums.splitlines()[0]
    (bundle / "checksums.sha256").write_text(checksums + first + "\n", encoding="utf-8")
    with pytest.raises(state_bundle.StateBundleError, match="Duplicate or recursive checksum"):
        state_bundle.verify_bundle(bundle)

    bundle = _export(source_state, tmp_path, "duplicate-destination")
    data = tmp_path / "target-data"
    target = state_bundle.TargetPaths(
        data_dir=data,
        runtime_dir=tmp_path / "target-runtime",
        model_options_dir=data / "processed" / "feedback",
    )
    with pytest.raises(state_bundle.StateBundleError, match="Duplicate import destination"):
        state_bundle.import_dry_run(bundle, target_paths=target)


def test_active_mlflow_analysis_rejects_running_and_machine_specific_metadata(tmp_path):
    db = tmp_path / "mlflow.db"
    with closing(sqlite3.connect(db)) as connection:
        connection.execute("CREATE TABLE runs (run_uuid TEXT, status TEXT, artifact_uri TEXT)")
        connection.execute(
            "CREATE TABLE experiments (experiment_id INTEGER, artifact_location TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES ('run-1', 'RUNNING', ?)",
            (r"file:///D:/old/mlruns/run-1",),
        )
        connection.execute(
            "INSERT INTO experiments VALUES (1, '/Users/old/mlruns/1')"
        )
        connection.commit()
    analysis = state_bundle._active_mlflow_analysis(db, tmp_path / "artifacts")
    assert analysis["status"] == "unsafe"
    assert analysis["running_run_ids"] == ["run-1"]
    assert {item["kind"] for item in analysis["unsafe_references"]} == {"run", "experiment"}


def test_metadata_never_contains_secret_values_and_automation_warns(source_state, tmp_path):
    secret = "never-print-this-secret-value"
    with closing(sqlite3.connect(source_state.feedback_db)) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO system_setting (key, value, updated_at) VALUES (?, ?, ?)",
            ("GEMINI_API_KEY", secret, "2026-08-17T00:00:00Z"),
        )
        connection.execute(
            "INSERT OR REPLACE INTO system_setting (key, value, updated_at) VALUES (?, ?, ?)",
            ("MLFLOW_AUTOMATION_ENABLED", "true", "2026-08-17T00:00:00Z"),
        )
        connection.commit()
    bundle = _export(source_state, tmp_path)
    metadata_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (bundle / "metadata").iterdir()
    ) + (bundle / "manifest.json").read_text(encoding="utf-8")
    assert secret not in metadata_text
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sensitivity"]["contains_sensitive_application_state"] is True
    report = state_bundle.import_dry_run(bundle, target_paths=_target(tmp_path))
    assert any("enabled automation" in warning for warning in report["warnings"])
