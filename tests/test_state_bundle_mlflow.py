import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
from mlflow.tracking import MlflowClient

from backend.mlflow_legacy_export import sha256_file
from backend.mlflow_server import (
    build_mlflow_server_command,
    resolve_mlflow_server_config,
    validate_mlflow_server_config,
)
from backend.state_bundle import (
    LEGACY_EVIDENCE_FILES,
    SourcePaths,
    TargetPaths,
    export_bundle,
    import_bundle,
)


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_server(command: list[str], cwd: Path) -> subprocess.Popen[str]:
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = cwd / f"state-bundle-mlflow-{time.time_ns()}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
        )
    health_url = f"http://127.0.0.1:{command[command.index('--port') + 1]}/health"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(log_path.read_text(encoding="utf-8", errors="replace"))
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return process
        except (OSError, URLError):
            time.sleep(0.2)
    _stop_server(process)
    raise AssertionError(log_path.read_text(encoding="utf-8", errors="replace"))


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


def _write_models(root: Path) -> None:
    model = root / "tfidf_lr" / "fixture"
    model.mkdir(parents=True)
    (model / "vectorizer.pkl").write_bytes(b"vectorizer")
    (model / "model_lr.pkl").write_bytes(b"model")


def _write_evidence(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_EVIDENCE_FILES:
        (root / name).write_text(json.dumps({"fixture": name}) + "\n", encoding="utf-8")


def test_active_mlflow_survives_export_import_and_restart(qa_env, monkeypatch, tmp_path):
    source_data = qa_env["base_dir"] / "source-data"
    source_runtime = qa_env["base_dir"] / "source-runtime"
    source_models = qa_env["base_dir"] / "source-models"
    _write_models(source_models)
    monkeypatch.setenv("APP_DATA_DIR", str(source_data))
    monkeypatch.setenv("APP_RUNTIME_DIR", str(source_runtime))
    monkeypatch.setenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(source_models))
    monkeypatch.setenv("MLFLOW_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MLFLOW_SERVER_PORT", str(_available_port()))
    monkeypatch.setenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    source_config = validate_mlflow_server_config(resolve_mlflow_server_config())
    _write_evidence(source_config.evidence_dir)
    legacy_hash = sha256_file(source_config.legacy_db_path)
    previous_tracking_uri = mlflow.get_tracking_uri()

    server = _start_server(build_mlflow_server_command(source_config), source_config.backend_db_path.parent)
    try:
        mlflow.set_tracking_uri(source_config.client_tracking_uri)
        mlflow.set_experiment("state-bundle-active-rehearsal")
        with mlflow.start_run(run_name="windows-source") as run:
            run_id = run.info.run_id
            mlflow.log_metric("macro_f1", 0.8123)
            mlflow.log_text("portable artifact\n", "evidence.txt")
    finally:
        _stop_server(server)

    source_paths = SourcePaths(
        data_dir=source_data,
        runtime_dir=source_runtime,
        model_options_dir=source_models,
        feedback_db=qa_env["feedback_db"],
        active_mlflow_db=source_config.backend_db_path,
        active_mlflow_artifacts=source_config.artifact_root,
        legacy_evidence_dir=source_config.evidence_dir,
    )
    bundle = tmp_path / "mlflow-bundle"
    exported = export_bundle(output=bundle, dry_run=False, source_paths=source_paths)
    assert exported["bundle_status"] == "complete"

    target = TargetPaths(
        data_dir=tmp_path / "Users" / "test" / "VietToxicData",
        runtime_dir=tmp_path / "Users" / "test" / "VietToxicRuntime",
        model_options_dir=tmp_path / "Users" / "test" / "VietToxicModels",
    )
    imported = import_bundle(bundle, target_paths=target, apply=True)
    assert imported["verification"]["valid"] is True

    monkeypatch.setenv("APP_DATA_DIR", str(target.data_dir))
    monkeypatch.setenv("APP_RUNTIME_DIR", str(target.runtime_dir))
    monkeypatch.setenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(target.model_options_dir))
    monkeypatch.setenv("MLFLOW_SERVER_PORT", str(_available_port()))
    target_config = validate_mlflow_server_config(resolve_mlflow_server_config())
    restarted = _start_server(build_mlflow_server_command(target_config), target_config.backend_db_path.parent)
    try:
        client = MlflowClient(tracking_uri=target_config.client_tracking_uri)
        recovered = client.get_run(run_id)
        assert recovered.data.metrics["macro_f1"] == 0.8123
        assert recovered.info.artifact_uri.startswith("mlflow-artifacts:/")
        assert str(source_data).lower() not in recovered.info.artifact_uri.lower()
        downloaded = Path(client.download_artifacts(run_id, "evidence.txt", str(tmp_path / "download")))
        assert downloaded.read_text(encoding="utf-8") == "portable artifact\n"
    finally:
        _stop_server(restarted)
        mlflow.set_tracking_uri(previous_tracking_uri)

    assert sha256_file(source_config.legacy_db_path) == legacy_hash
