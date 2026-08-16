"""Central cross-platform runtime path contract.

Established application-state defaults remain repository-relative. MLflow is
explicitly split into an immutable root database and a fresh writable store
under the data directory. Environment overrides are normalized here so backend
services never depend on the process working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from urllib.parse import urlsplit, urlunsplit


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


def get_legacy_mlflow_db_path() -> Path:
    """Return the immutable repository-root MLflow history database."""
    return (PROJECT_ROOT / "mlflow.db").resolve()


def get_mlflow_dir() -> Path:
    """Return the writable directory for the fresh portable MLflow store."""
    return (get_data_dir() / "mlflow").resolve()


def get_mlflow_db_path() -> Path:
    """Return the new writable MLflow DB, never the legacy root database."""
    return get_mlflow_dir() / "mlflow.db"


def get_mlflow_server_host() -> str:
    return os.getenv("MLFLOW_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"


def get_mlflow_server_port() -> int:
    raw = os.getenv("MLFLOW_SERVER_PORT", "5000").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError("MLFLOW_SERVER_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("MLFLOW_SERVER_PORT must be between 1 and 65535")
    return port


def get_mlflow_server_tracking_uri() -> str:
    host = get_mlflow_server_host()
    uri_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{uri_host}:{get_mlflow_server_port()}"


def get_mlflow_tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    return configured or get_mlflow_server_tracking_uri()


def get_safe_mlflow_tracking_uri() -> str:
    """Return the tracking URI without credentials, query values, or fragments."""
    raw = get_mlflow_tracking_uri()
    parsed = urlsplit(raw)
    if not parsed.netloc:
        return raw.split("#", 1)[0].split("?", 1)[0]
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def get_mlflow_artifact_root() -> Path:
    return _configured_path("MLFLOW_ARTIFACT_ROOT", get_mlflow_dir() / "artifacts")


def get_mlflow_evidence_dir() -> Path:
    return get_mlflow_dir() / "evidence"


def get_mlflow_backend_store_uri() -> str:
    return f"sqlite:///{get_mlflow_db_path().as_posix()}"


def get_mlflow_artifact_destination_uri() -> str:
    return get_mlflow_artifact_root().as_uri()


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
        "legacy_mlflow_db": str(get_legacy_mlflow_db_path()),
        "mlflow_dir": str(get_mlflow_dir()),
        "mlflow_db": str(get_mlflow_db_path()),
        "mlflow_artifact_root": str(get_mlflow_artifact_root()),
        "mlflow_evidence_dir": str(get_mlflow_evidence_dir()),
        "mlflow_server_host": get_mlflow_server_host(),
        "mlflow_server_port": str(get_mlflow_server_port()),
        "mlflow_tracking_uri": get_safe_mlflow_tracking_uri(),
        "model_registry_dir": str(get_model_registry_dir()),
        "kaggle_runtime_dir": str(get_kaggle_runtime_dir()),
    }


if __name__ == "__main__":
    for label, value in get_effective_paths().items():
        print(f"{label}: {value}")
