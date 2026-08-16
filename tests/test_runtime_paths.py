from pathlib import Path

from backend import runtime_paths


def test_defaults_preserve_current_repository_layout(monkeypatch):
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("APP_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("VIETTOXIC_MODEL_OPTIONS_DIR", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)

    root = runtime_paths.get_project_root()
    assert runtime_paths.get_data_dir() == root / "data"
    assert runtime_paths.get_runtime_dir() == root / ".runtime"
    assert runtime_paths.get_feedback_db_path() == root / "data" / "processed" / "feedback" / "feedback.db"
    assert runtime_paths.get_model_options_dir() == root / "models" / "options"
    assert runtime_paths.get_mlflow_db_path() == root / "mlflow.db"
    assert runtime_paths.get_mlflow_tracking_uri() == f"sqlite:///{(root / 'mlflow.db').as_posix()}"


def test_data_and_runtime_overrides_are_normalized(monkeypatch):
    data_dir = runtime_paths.get_project_root() / ".path-test-data"
    runtime_dir = runtime_paths.get_project_root() / ".path-test-runtime"
    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_RUNTIME_DIR", str(runtime_dir))

    assert runtime_paths.get_data_dir() == data_dir.resolve()
    assert runtime_paths.get_feedback_db_path() == data_dir.resolve() / "processed" / "feedback" / "feedback.db"
    assert runtime_paths.get_mlflow_db_path() == data_dir.resolve() / "mlflow" / "mlflow.db"
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
    assert {"feedback_db", "model_options_dir", "model_registry_dir", "kaggle_runtime_dir"}.issubset(diagnostic)
