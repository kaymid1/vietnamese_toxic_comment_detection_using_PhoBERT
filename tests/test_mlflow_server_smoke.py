import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
from mlflow.tracking import MlflowClient

from backend.mlflow_client_config import configure_mlflow_client
from backend.mlflow_legacy_export import sha256_file
from backend.mlflow_server import (
    build_mlflow_server_command,
    resolve_mlflow_server_config,
    validate_mlflow_server_config,
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
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
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
