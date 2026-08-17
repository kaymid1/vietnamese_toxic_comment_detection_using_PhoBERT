import hashlib
import sqlite3
from pathlib import Path

from backend.portability_migration import (
    apply_kaggle_registry_migration,
    plan_kaggle_registry_migration,
)


def _migration_db(path: Path, *, run_id: str, artifact_checksum: str, registry_checksum: str, registry_uri: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE mlflow_do_run (
                run_id TEXT PRIMARY KEY, artifact_uri TEXT, artifact_checksum TEXT
            );
            CREATE TABLE mlflow_training_artifact (
                id INTEGER PRIMARY KEY AUTOINCREMENT, source_run_id TEXT, artifact_path TEXT
            );
            CREATE TABLE mlflow_model_version (
                model_family TEXT, model_id TEXT, source_run_id TEXT, artifact_uri TEXT, artifact_checksum TEXT
            );
            """
        )
        old_uri = f"file://D:/Code/Thesis/Thesis/.runtime/kaggle_real_jobs/real_job/output/viettoxic/{run_id}.zip"
        conn.execute("INSERT INTO mlflow_do_run VALUES (?, ?, ?)", (run_id, old_uri, artifact_checksum))
        conn.execute("INSERT INTO mlflow_training_artifact (source_run_id, artifact_path) VALUES (?, ?)", (run_id, old_uri))
        conn.execute(
            "INSERT INTO mlflow_model_version VALUES (?, ?, ?, ?, ?)",
            ("tfidf_lr", f"tfidf_lr/{run_id}", run_id, registry_uri, registry_checksum),
        )
        conn.commit()


def test_kaggle_registry_migration_is_checksum_gated_and_idempotent(tmp_path: Path):
    db = tmp_path / "feedback.db"
    registry_root = tmp_path / "runtime" / "model_registry"
    run_id = "kaggle_migration_ok"
    artifact = registry_root / "tfidf_lr" / run_id / "artifact.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified artifact")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    uri = f"runtime://model_registry/tfidf_lr/{run_id}/artifact.zip"
    _migration_db(db, run_id=run_id, artifact_checksum=checksum, registry_checksum=checksum, registry_uri=uri)

    plan = plan_kaggle_registry_migration(db, registry_root=registry_root)
    assert len(plan) == 2
    assert all(item["eligible"] and item["checksum_verified"] for item in plan)

    backup, applied = apply_kaggle_registry_migration(db, registry_root=registry_root)
    assert backup is not None and backup.is_file()
    assert all(item["eligible"] for item in applied)

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT artifact_uri FROM mlflow_do_run WHERE run_id = ?", (run_id,)).fetchone()[0] == uri
        assert conn.execute("SELECT artifact_path FROM mlflow_training_artifact WHERE source_run_id = ?", (run_id,)).fetchone()[0] == uri

    second_backup, second_plan = apply_kaggle_registry_migration(db, registry_root=registry_root)
    assert second_backup is None
    assert second_plan == []


def test_kaggle_registry_migration_skips_checksum_mismatch(tmp_path: Path):
    db = tmp_path / "feedback.db"
    registry_root = tmp_path / "runtime" / "model_registry"
    run_id = "kaggle_migration_bad_checksum"
    artifact = registry_root / "tfidf_lr" / run_id / "artifact.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"real artifact")
    wrong_checksum = "f" * 64
    uri = f"runtime://model_registry/tfidf_lr/{run_id}/artifact.zip"
    _migration_db(db, run_id=run_id, artifact_checksum=wrong_checksum, registry_checksum=wrong_checksum, registry_uri=uri)

    plan = plan_kaggle_registry_migration(db, registry_root=registry_root)
    assert len(plan) == 2
    assert all(not item["eligible"] for item in plan)
    assert all("mismatch" in item["reason"] for item in plan)
