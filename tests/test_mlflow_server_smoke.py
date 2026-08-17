import json
import os
import signal
import socket
import sqlite3
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from backend.mlflow_client_config import configure_mlflow_client
from backend.mlflow_kaggle_ingest import (
    KaggleEvidenceConflictError,
    KaggleEvidenceIngestionUnavailable,
    get_kaggle_ingestion_record,
    ingest_kaggle_evidence,
    validate_kaggle_evidence,
)
from backend.mlflow_legacy_export import sha256_file
from backend.mlflow_server import (
    build_mlflow_server_command,
    resolve_mlflow_server_config,
    validate_mlflow_server_config,
)
from kaggle.mlflow_evidence import (
    EVIDENCE_FILENAME,
    build_directory_artifacts,
    build_evidence_manifest,
    write_evidence_file,
)


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_server(command: list[str], cwd: Path) -> tuple[subprocess.Popen[str], Path]:
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = cwd / f"mlflow-smoke-{time.time_ns()}.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
        )
    health_url = f"http://127.0.0.1:{command[command.index('--port') + 1]}/health"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = log_path.read_text(encoding="utf-8", errors="replace")
            raise AssertionError(f"MLflow server exited before becoming ready:\n{output}")
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return process, log_path
        except (OSError, URLError):
            time.sleep(0.2)
    _stop_server(process)
    output = log_path.read_text(encoding="utf-8", errors="replace")
    raise AssertionError(f"MLflow server did not become healthy within 180 seconds:\n{output[-4000:]}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_fresh_server_logs_and_recovers_portable_artifacts(monkeypatch, tmp_path: Path):
    port = _available_local_port()
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "portable-app-data"))
    monkeypatch.setenv("MLFLOW_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MLFLOW_SERVER_PORT", str(port))
    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    config = validate_mlflow_server_config(resolve_mlflow_server_config())
    legacy_hash = sha256_file(config.legacy_db_path)
    command = build_mlflow_server_command(config)
    project_root = config.legacy_db_path.parent
    previous_tracking_uri = mlflow.get_tracking_uri()

    first_server, _ = _start_server(command, config.backend_db_path.parent)
    try:
        configured = configure_mlflow_client(
            enabled=True,
            experiment_name="portable-local-client-smoke",
            run_name="phase-2b2a-smoke",
            tags={"viettoxic.execution": "local"},
            timeout=5.0,
        )
        assert configured is mlflow
        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id
        experiment_id = active_run.info.experiment_id
        try:
            mlflow.log_param("portable_param", "ok")
            mlflow.log_metric("portable_metric", 0.75)
            mlflow.log_text("portable artifact\n", "smoke.txt")
        finally:
            mlflow.end_run()

        client = MlflowClient(tracking_uri=config.client_tracking_uri)
        run = client.get_run(run_id)
        experiment = client.get_experiment(experiment_id)
        assert run.data.params["portable_param"] == "ok"
        assert run.data.metrics["portable_metric"] == 0.75
        assert run.info.artifact_uri.startswith("mlflow-artifacts:/")
        assert experiment.artifact_location.startswith("mlflow-artifacts:/")
        assert str(project_root).lower() not in run.info.artifact_uri.lower()
        downloaded = Path(client.download_artifacts(run_id, "smoke.txt", str(tmp_path / "download")))
        assert downloaded.read_text(encoding="utf-8") == "portable artifact\n"
    finally:
        _stop_server(first_server)

    second_server, _ = _start_server(command, config.backend_db_path.parent)
    try:
        restarted_client = MlflowClient(tracking_uri=config.client_tracking_uri)
        persisted = restarted_client.get_run(run_id)
        assert persisted.data.params["portable_param"] == "ok"
        assert persisted.data.metrics["portable_metric"] == 0.75
        assert Path(
            restarted_client.download_artifacts(run_id, "smoke.txt", str(tmp_path / "restart-download"))
        ).read_text(encoding="utf-8") == "portable artifact\n"
    finally:
        _stop_server(second_server)
        mlflow.set_tracking_uri(previous_tracking_uri)

    assert config.backend_db_path.is_file()
    assert config.backend_db_path != config.legacy_db_path
    assert sha256_file(config.legacy_db_path) == legacy_hash


def _build_ingestion_archive(tmp_path: Path, *, macro_f1: float) -> Path:
    root = tmp_path / f"ingestion-source-{str(macro_f1).replace('.', '-')}"
    root.mkdir()
    (root / "model_lr.joblib").write_bytes(b"portable-model")
    (root / "vectorizer.joblib").write_bytes(b"portable-vectorizer")
    (root / "metrics.json").write_text(json.dumps({"macro_f1": macro_f1}), encoding="utf-8")
    manifest = build_evidence_manifest(
        source_job_id="real_job_ingestion",
        source_run_id="kaggle_run_ingestion",
        experiment_name="viettoxic-kaggle-tfidf-lr",
        run_name="kaggle_run_ingestion",
        training={
            "model_family": "tfidf_lr",
            "training_mode": "retrain",
            "dataset": "clean_victsd_gold",
            "script": "viettoxic_mlflow_retrain.py",
            "base_model": "sklearn.LogisticRegression",
            "initialization_mode": "fresh_estimator",
            "training_config_id": "kaggle_run_ingestion",
        },
        training_status="success",
        tracking_status="complete",
        artifact_status="complete",
        params={"seed": 42, "size_train": 100},
        metrics={"macro_f1": macro_f1, "toxic_f1": 0.71},
        tags={"model_family": "tfidf_lr", "training_mode": "retrain"},
        artifacts=build_directory_artifacts(root),
        timestamps={"finished_at": "2026-08-17T00:00:00+00:00"},
        provenance={"notebook_sha256": "a" * 64},
    )
    write_evidence_file(root / EVIDENCE_FILENAME, manifest)
    archive_path = tmp_path / f"ingestion-{str(macro_f1).replace('.', '-')}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.iterdir()):
            archive.write(path, arcname=path.name)
    return archive_path


def test_kaggle_evidence_ingestion_is_retriable_idempotent_and_conflict_safe(
    monkeypatch, tmp_path: Path
):
    port = _available_local_port()
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "kaggle-ingestion-app-data"))
    monkeypatch.setenv("MLFLOW_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MLFLOW_SERVER_PORT", str(port))
    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    config = validate_mlflow_server_config(resolve_mlflow_server_config())
    command = build_mlflow_server_command(config)
    legacy_hash = sha256_file(config.legacy_db_path)
    mapping_db = tmp_path / "operational" / "feedback.db"
    archive_path = _build_ingestion_archive(tmp_path, macro_f1=0.76)
    evidence = validate_kaggle_evidence(archive_path)

    with pytest.raises(KaggleEvidenceIngestionUnavailable):
        ingest_kaggle_evidence(evidence, db_path=mapping_db, timeout=0.2)
    failed_record = get_kaggle_ingestion_record(
        mapping_db,
        source_job_id=evidence.source_job_id,
        source_run_id=evidence.source_run_id,
    )
    assert failed_record["ingestion_status"] == "failed"
    assert failed_record["retriable"] == 1
    assert archive_path.is_file()

    previous_tracking_uri = mlflow.get_tracking_uri()
    server, _ = _start_server(command, config.backend_db_path.parent)
    try:
        first = ingest_kaggle_evidence(evidence, db_path=mapping_db, timeout=5.0)
        second = ingest_kaggle_evidence(evidence, db_path=mapping_db, timeout=5.0)
        canonical_run_id = first["canonical_mlflow_run_id"]
        assert first["ingestion_status"] == "completed"
        assert second["canonical_mlflow_run_id"] == canonical_run_id
        assert second["action"] == "existing"

        client = MlflowClient(tracking_uri=config.client_tracking_uri)
        run = client.get_run(canonical_run_id)
        experiment = client.get_experiment(run.info.experiment_id)
        assert run.data.params["seed"] == "42"
        assert run.data.metrics["macro_f1"] == 0.76
        assert run.data.tags["execution_origin"] == "kaggle"
        assert run.data.tags["ingestion_mode"] == "post_run"
        assert run.data.tags["source_job_id"] == "real_job_ingestion"
        assert run.data.tags["source_run_id"] == "kaggle_run_ingestion"
        assert run.info.artifact_uri.startswith("mlflow-artifacts:/")
        assert experiment.artifact_location.startswith("mlflow-artifacts:/")
        assert str(tmp_path).lower() not in run.info.artifact_uri.lower()
        artifact_names = {item.path for item in client.list_artifacts(canonical_run_id)}
        assert {EVIDENCE_FILENAME, "metrics.json", "model_lr.joblib", "vectorizer.joblib"}.issubset(
            artifact_names
        )
        matching_runs = client.search_runs(
            [run.info.experiment_id],
            filter_string="tags.`source_job_id` = 'real_job_ingestion'",
            max_results=10,
        )
        assert [item.info.run_id for item in matching_runs] == [canonical_run_id]

        conflicting = validate_kaggle_evidence(_build_ingestion_archive(tmp_path, macro_f1=0.77))
        with pytest.raises(KaggleEvidenceConflictError):
            ingest_kaggle_evidence(conflicting, db_path=mapping_db, timeout=5.0)
        assert len(
            client.search_runs(
                [run.info.experiment_id],
                filter_string="tags.`source_job_id` = 'real_job_ingestion'",
                max_results=10,
            )
        ) == 1
    finally:
        _stop_server(server)
        mlflow.set_tracking_uri(previous_tracking_uri)

    with sqlite3.connect(mapping_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM mlflow_kaggle_ingestion").fetchone()[0] == 1
    assert sha256_file(config.legacy_db_path) == legacy_hash
