"""Central, repository-relative runtime path contract.

The defaults deliberately retain the established project layout.  Environment
overrides are resolved once through this module so backend services agree on
the same locations without depending on the process working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(name: str, default: Path) -> Path:
    """Return an absolute normalized filesystem path for an optional setting."""
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_dir() -> Path:
    return _configured_path("APP_DATA_DIR", PROJECT_ROOT / "data")


def get_runtime_dir() -> Path:
    return _configured_path("APP_RUNTIME_DIR", PROJECT_ROOT / ".runtime")


def get_feedback_db_path() -> Path:
    return get_data_dir() / "processed" / "feedback" / "feedback.db"


def get_model_options_dir() -> Path:
    return _configured_path("VIETTOXIC_MODEL_OPTIONS_DIR", PROJECT_ROOT / "models" / "options")


def get_mlflow_db_path() -> Path:
    return get_data_dir() / "mlflow" / "mlflow.db" if os.getenv("APP_DATA_DIR", "").strip() else PROJECT_ROOT / "mlflow.db"


def get_mlflow_tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    return configured or f"sqlite:///{get_mlflow_db_path().as_posix()}"


def get_mlflow_artifact_root() -> Path:
    return _configured_path("MLFLOW_ARTIFACT_ROOT", get_data_dir() / "mlflow" / "artifacts")


def get_model_registry_dir() -> Path:
    return get_runtime_dir() / "model_registry"


def get_kaggle_runtime_dir() -> Path:
    return get_runtime_dir() / "kaggle_real_jobs"


def get_effective_paths() -> Dict[str, str]:
    """Safe diagnostic payload: filesystem locations and URI only, never secrets."""
    return {
        "project_root": str(get_project_root()),
        "data_dir": str(get_data_dir()),
        "runtime_dir": str(get_runtime_dir()),
        "feedback_db": str(get_feedback_db_path()),
        "model_options_dir": str(get_model_options_dir()),
        "mlflow_db": str(get_mlflow_db_path()),
        "mlflow_tracking_uri": get_mlflow_tracking_uri(),
        "mlflow_artifact_root": str(get_mlflow_artifact_root()),
        "model_registry_dir": str(get_model_registry_dir()),
        "kaggle_runtime_dir": str(get_kaggle_runtime_dir()),
    }


if __name__ == "__main__":
    for label, value in get_effective_paths().items():
        print(f"{label}: {value}")
