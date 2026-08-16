import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from backend.mlflow_legacy_export import (
    ARTIFACTS_FILENAME,
    CHECKSUMS_FILENAME,
    INVENTORY_FILENAME,
    classify_artifact_reference,
    export_legacy_evidence,
    open_legacy_database,
    sha256_file,
)


def _create_legacy_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY);
            INSERT INTO alembic_version VALUES ('test-revision');
            CREATE TABLE experiments (
                experiment_id INTEGER PRIMARY KEY,
                name TEXT,
                artifact_location TEXT,
                lifecycle_stage TEXT,
                creation_time INTEGER,
                last_update_time INTEGER
            );
            CREATE TABLE runs (
                run_uuid TEXT PRIMARY KEY,
                name TEXT,
                experiment_id INTEGER,
                status TEXT,
                artifact_uri TEXT,
                lifecycle_stage TEXT,
                start_time INTEGER,
                end_time INTEGER
            );
            CREATE TABLE params (key TEXT, value TEXT, run_uuid TEXT);
            CREATE TABLE metrics (
                key TEXT, value REAL, timestamp INTEGER, run_uuid TEXT, step INTEGER, is_nan INTEGER
            );
            CREATE TABLE latest_metrics (
                key TEXT, value REAL, timestamp INTEGER, run_uuid TEXT, step INTEGER, is_nan INTEGER
            );
            CREATE TABLE tags (key TEXT, value TEXT, run_uuid TEXT);
            CREATE TABLE registered_models (
                name TEXT PRIMARY KEY, creation_timestamp INTEGER, last_updated_timestamp INTEGER, description TEXT
            );
            CREATE TABLE model_versions (
                name TEXT, version INTEGER, creation_timestamp INTEGER, last_updated_timestamp INTEGER,
                description TEXT, user_id TEXT, current_stage TEXT, source TEXT, run_id TEXT, status TEXT,
                status_message TEXT, run_link TEXT, storage_location TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
            (1, "Legacy experiment", "/Users/mac/git/Thesis/mlruns/1", "active", 1000, 2000),
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-run",
                "legacy",
                1,
                "FINISHED",
                "/Users/mac/git/Thesis/mlruns/1/legacy-run/artifacts",
                "active",
                3000,
                4000,
            ),
        )
        connection.execute("INSERT INTO params VALUES ('epochs', '4', 'legacy-run')")
        connection.execute("INSERT INTO metrics VALUES ('f1', 0.75, 3500, 'legacy-run', 1, 0)")
        connection.execute("INSERT INTO latest_metrics VALUES ('f1', 0.75, 3500, 'legacy-run', 1, 0)")
        connection.execute("INSERT INTO tags VALUES ('mlflow.runName', 'legacy', 'legacy-run')")
        connection.execute(
            "INSERT INTO registered_models VALUES ('legacy-model', 1000, 2000, 'historical')"
        )
        connection.execute(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-model",
                1,
                1000,
                2000,
                "historical",
                None,
                "None",
                "missing",
                "legacy-run",
                "READY",
                None,
                None,
                None,
            ),
        )


def test_legacy_export_is_read_only_deterministic_and_complete(tmp_path: Path):
    source = tmp_path / "mlflow.db"
    output = tmp_path / "evidence"
    _create_legacy_db(source)
    before_hash = sha256_file(source)
    before_mtime = source.stat().st_mtime_ns

    first = export_legacy_evidence(db_path=source, output_dir=output)
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = export_legacy_evidence(db_path=source, output_dir=output)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first == second
    assert first_bytes == second_bytes
    assert sha256_file(source) == before_hash
    assert source.stat().st_mtime_ns == before_mtime
    assert set(first_bytes) == {INVENTORY_FILENAME, ARTIFACTS_FILENAME, CHECKSUMS_FILENAME}

    inventory = json.loads(first_bytes[INVENTORY_FILENAME])
    assert inventory["database"]["integrity_result"] == "ok"
    assert inventory["database"]["sha256"] == before_hash
    assert inventory["database"]["alembic_revision"] == "test-revision"
    assert inventory["experiments"][0]["experiment_id"] == 1
    assert inventory["runs"][0]["run_uuid"] == "legacy-run"
    assert inventory["runs"][0]["classification"] == "UNCERTAIN"
    assert inventory["runs"][0]["metrics"][0]["key"] == "f1"
    assert inventory["runs"][0]["start_time_iso"] == "1970-01-01T00:00:03Z"
    assert inventory["model_registry"]["registered_models"][0]["name"] == "legacy-model"

    artifacts = json.loads(first_bytes[ARTIFACTS_FILENAME])
    assert len(artifacts["references"]) == 2
    assert all(item["classification"] == "machine_specific_macos" for item in artifacts["references"])
    assert all(item["exists"] is False and item["status"] == "missing" for item in artifacts["references"])

    checksums = json.loads(first_bytes[CHECKSUMS_FILENAME])
    assert checksums["source_database_unchanged"] is True
    assert checksums["files"]["repository://mlflow.db"] == before_hash
    assert checksums["files"][INVENTORY_FILENAME] == hashlib.sha256(
        first_bytes[INVENTORY_FILENAME]
    ).hexdigest()


def test_legacy_database_connection_refuses_writes(tmp_path: Path):
    source = tmp_path / "mlflow.db"
    _create_legacy_db(source)

    with open_legacy_database(source) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO experiments (experiment_id) VALUES (99)")


def test_artifact_inventory_distinguishes_path_styles(tmp_path: Path):
    relative_artifact = tmp_path / "artifacts" / "model.txt"
    relative_artifact.parent.mkdir()
    relative_artifact.write_text("model", encoding="utf-8")

    windows = classify_artifact_reference(r"C:\Users\legacy\mlruns\1", tmp_path)
    relative = classify_artifact_reference("artifacts/model.txt", tmp_path)
    external = classify_artifact_reference("s3://legacy-bucket/run", tmp_path)

    assert windows["classification"] == "machine_specific_windows"
    assert relative == {
        "classification": "relative",
        "exists": True,
        "resolved_path_if_applicable": str(relative_artifact.resolve()),
        "status": "available",
        "stored_uri": "artifacts/model.txt",
    }
    assert external["classification"] == "external"
    assert external["exists"] is None
    assert external["status"] == "not_locally_verifiable"
