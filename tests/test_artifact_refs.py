import json
import sqlite3
from pathlib import Path

import pytest

from backend import artifact_refs, portability_migration


def test_portable_schemes_resolve_against_runtime_roots(monkeypatch, tmp_path):
    data = tmp_path / "data"
    runtime = tmp_path / "runtime"
    models = tmp_path / "models"
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    monkeypatch.setenv("APP_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(models))

    assert artifact_refs.resolve_artifact_ref("data://artifacts/run.zip") == data / "artifacts" / "run.zip"
    assert artifact_refs.resolve_artifact_ref("runtime://model_registry/a/artifact.zip") == runtime / "model_registry" / "a" / "artifact.zip"
    assert artifact_refs.resolve_artifact_ref("model://phobert/v2") == models / "phobert" / "v2"
    assert artifact_refs.encode_artifact_ref(runtime / "model_registry" / "a" / "artifact.zip") == "runtime://model_registry/a/artifact.zip"


def test_legacy_paths_and_urls_are_conservative(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("APP_RUNTIME_DIR", str(runtime))
    old = r"D:\Code\Thesis\Thesis\.runtime\model_registry\abc\artifact.zip"
    assert artifact_refs.encode_artifact_ref(old) == "runtime://model_registry/abc/artifact.zip"
    assert artifact_refs.resolve_artifact_ref(old) == Path(old).resolve()
    assert artifact_refs.inspect_artifact_ref("/Users/old/external.zip").classification == "external_absolute"
    unrelated = r"D:\Downloads\data\processed\random-model.zip"
    assert artifact_refs.encode_artifact_ref(unrelated) == unrelated
    assert artifact_refs.inspect_artifact_ref(unrelated).classification == "external_absolute"
    for url in ("https://example.test/a.zip", "http://example.test/a.zip", "kaggle://dataset/a", "file:///D:/old/a.zip"):
        assert artifact_refs.encode_artifact_ref(url) == url


def _create_db(path: Path, *, fail_trigger: bool = False) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE mlflow_training_artifact (id INTEGER PRIMARY KEY, artifact_path TEXT)")
        conn.execute("CREATE TABLE mlflow_model_version (id INTEGER PRIMARY KEY, artifact_path TEXT, artifact_uri TEXT, bundle_path TEXT)")
        conn.execute("CREATE TABLE mlflow_do_run (run_id TEXT PRIMARY KEY, artifact_uri TEXT, bundle_path TEXT, bundle_url TEXT, logs_json TEXT)")
        conn.execute("INSERT INTO mlflow_training_artifact VALUES (1, ?)", (r"D:\Code\Thesis\Thesis\.runtime\model_registry\abc\artifact.zip",))
        conn.execute("INSERT INTO mlflow_training_artifact VALUES (2, ?)", (r"D:\Downloads\external.zip",))
        conn.execute("INSERT INTO mlflow_model_version VALUES (1, ?, ?, ?)", (r"D:\Code\Thesis\Thesis\.runtime\model_registry\abc\artifact.zip", "https://example.test/a.zip", r"data\processed\bundle.zip"))
        conn.execute("INSERT INTO mlflow_do_run VALUES (?, ?, ?, ?, ?)", ("run-1", r"D:\Code\Thesis\Thesis\.runtime\kaggle_real_jobs\x\output\a.zip", r"data\processed\bundle.zip", "https://example.test/bundle", json.dumps([{"message": "D:\\leave text untouched", "artifact_path": r"D:\Code\Thesis\Thesis\.runtime\model_registry\abc\artifact.zip"}])))
        if fail_trigger:
            conn.execute("CREATE TRIGGER fail_model BEFORE UPDATE ON mlflow_model_version BEGIN SELECT RAISE(ABORT, 'forced'); END")


def test_dry_run_is_read_only_and_apply_backups_and_converts(tmp_path):
    db = tmp_path / "feedback.db"
    _create_db(db)
    before = db.read_bytes()
    plan = portability_migration.plan_migration(db)
    assert db.read_bytes() == before
    assert any(item["new"] == "runtime://model_registry/abc/artifact.zip" for item in plan)
    assert any(item["classification"] == "protected_uri" for item in plan)

    backup, _ = portability_migration.apply_migration(db)
    assert backup.exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT artifact_path FROM mlflow_training_artifact WHERE id=1").fetchone()[0] == "runtime://model_registry/abc/artifact.zip"
        assert conn.execute("SELECT artifact_path FROM mlflow_training_artifact WHERE id=2").fetchone()[0] == r"D:\Downloads\external.zip"
        assert conn.execute("SELECT artifact_uri, bundle_path, bundle_url FROM mlflow_do_run").fetchone() == ("runtime://kaggle_real_jobs/x/output/a.zip", "data://processed/bundle.zip", "https://example.test/bundle")
        logs = json.loads(conn.execute("SELECT logs_json FROM mlflow_do_run").fetchone()[0])
        assert logs[0]["message"] == "D:\\leave text untouched"
        assert logs[0]["artifact_path"] == "runtime://model_registry/abc/artifact.zip"


def test_apply_rolls_back_when_any_update_fails(tmp_path):
    db = tmp_path / "rollback.db"
    _create_db(db, fail_trigger=True)
    with pytest.raises(sqlite3.DatabaseError):
        portability_migration.apply_migration(db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT artifact_path FROM mlflow_training_artifact WHERE id=1").fetchone()[0].startswith("D:")
