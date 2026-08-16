from pathlib import Path

from backend import runtime_paths


def test_defaults_preserve_current_repository_layout(monkeypatch):
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("APP_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("VIETTOXIC_MODEL_OPTIONS_DIR", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("MLFLOW_SERVER_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_SERVER_PORT", raising=False)

    root = runtime_paths.get_project_root()
    assert runtime_paths.get_data_dir() == root / "data"
    assert runtime_paths.get_runtime_dir() == root / ".runtime"
    assert runtime_paths.get_feedback_db_path() == root / "data" / "processed" / "feedback" / "feedback.db"
    assert runtime_paths.get_model_options_dir() == root / "models" / "options"
    assert runtime_paths.get_legacy_mlflow_db_path() == root / "mlflow.db"
    assert runtime_paths.get_mlflow_dir() == root / "data" / "mlflow"
    assert runtime_paths.get_mlflow_db_path() == root / "data" / "mlflow" / "mlflow.db"
    assert runtime_paths.get_mlflow_artifact_root() == root / "data" / "mlflow" / "artifacts"
    assert runtime_paths.get_mlflow_evidence_dir() == root / "data" / "mlflow" / "evidence"
    assert runtime_paths.get_mlflow_server_host() == "127.0.0.1"
    assert runtime_paths.get_mlflow_server_port() == 5000
    assert runtime_paths.get_mlflow_tracking_uri() == "http://127.0.0.1:5000"
    assert runtime_paths.get_mlflow_db_path() != runtime_paths.get_legacy_mlflow_db_path()


def test_data_and_runtime_overrides_are_normalized(monkeypatch):
    data_dir = runtime_paths.get_project_root() / ".path-test-data"
    runtime_dir = runtime_paths.get_project_root() / ".path-test-runtime"
    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_RUNTIME_DIR", str(runtime_dir))

    assert runtime_paths.get_data_dir() == data_dir.resolve()
    assert runtime_paths.get_feedback_db_path() == data_dir.resolve() / "processed" / "feedback" / "feedback.db"
    assert runtime_paths.get_mlflow_db_path() == data_dir.resolve() / "mlflow" / "mlflow.db"
    assert runtime_paths.get_mlflow_evidence_dir() == data_dir.resolve() / "mlflow" / "evidence"
    assert runtime_paths.get_runtime_dir() == runtime_dir.resolve()
    assert runtime_paths.get_model_registry_dir() == runtime_dir.resolve() / "model_registry"
    assert runtime_paths.get_kaggle_runtime_dir() == runtime_dir.resolve() / "kaggle_real_jobs"


def test_model_and_mlflow_overrides(monkeypatch):
    model_dir = runtime_paths.get_project_root() / ".path-test-models"
    artifact_dir = runtime_paths.get_project_root() / ".path-test-mlflow-artifacts"
    monkeypatch.setenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(model_dir))
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(artifact_dir))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")

    assert runtime_paths.get_model_options_dir() == model_dir.resolve()
    assert runtime_paths.get_mlflow_artifact_root() == artifact_dir.resolve()
    assert runtime_paths.get_mlflow_tracking_uri() == "https://mlflow.example.test"


def test_posix_and_windows_style_overrides_are_pathlib_normalized(monkeypatch):
    posix_root = "/Users/example/viettoxic-data"
    monkeypatch.setenv("APP_DATA_DIR", posix_root)
    assert runtime_paths.get_data_dir() == Path(posix_root).expanduser().resolve()

    windows_root = r"C:\\viettoxic\\runtime"
    monkeypatch.setenv("APP_RUNTIME_DIR", windows_root)
    assert runtime_paths.get_runtime_dir() == Path(windows_root).expanduser().resolve()


def test_diagnostic_contains_only_effective_paths(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    diagnostic = runtime_paths.get_effective_paths()

    assert diagnostic["project_root"] == str(runtime_paths.get_project_root())
    assert diagnostic["mlflow_tracking_uri"] == "sqlite:///custom.db"
    assert {
        "feedback_db",
        "legacy_mlflow_db",
        "mlflow_db",
        "mlflow_evidence_dir",
        "mlflow_server_host",
        "mlflow_server_port",
        "model_options_dir",
        "model_registry_dir",
        "kaggle_runtime_dir",
    }.issubset(diagnostic)


def test_diagnostic_redacts_tracking_uri_credentials(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", "https://user:secret@mlflow.example.test/path?token=hidden#fragment"
    )

    diagnostic = runtime_paths.get_effective_paths()

    assert diagnostic["mlflow_tracking_uri"] == "https://mlflow.example.test/path"
    assert "secret" not in str(diagnostic)
    assert "hidden" not in str(diagnostic)


def test_server_host_and_port_drive_default_client_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_SERVER_HOST", "localhost")
    monkeypatch.setenv("MLFLOW_SERVER_PORT", "5055")

    assert runtime_paths.get_mlflow_server_tracking_uri() == "http://localhost:5055"
    assert runtime_paths.get_mlflow_tracking_uri() == "http://localhost:5055"


def test_invalid_server_port_is_rejected(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_PORT", "70000")

    try:
        runtime_paths.get_mlflow_server_port()
    except ValueError as exc:
        assert "between 1 and 65535" in str(exc)
    else:
        raise AssertionError("invalid MLflow server port was accepted")
