"""Shared MLflow client contract for local VietToxic training runs.

The server-side SQLite and artifact paths belong to ``backend.mlflow_server``.
Local training clients use only the HTTP tracking URI resolved by
``backend.runtime_paths`` and never fall back to a local ``mlruns`` store.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from backend.runtime_paths import get_mlflow_tracking_uri, get_safe_mlflow_tracking_uri


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})

LOCAL_EXPERIMENT_NAMES = {
    "phobert_v1_macro_f1": "viettoxic-local-phobert-v1-full-finetune-macro-f1",
    "phobert_v2_f1_toxic": "viettoxic-local-phobert-v2-full-finetune-f1-toxic",
    "phobert_v2_macro_f1": "viettoxic-local-phobert-v2-full-finetune-macro-f1",
    "phobert_v2_adaptive_macro_f1": "viettoxic-local-phobert-v2-adaptive-macro-f1",
}


class MlflowClientConfigurationError(RuntimeError):
    """Raised when enabled local tracking cannot safely start."""


def is_mlflow_enabled(value: str | None = None) -> bool:
    """Resolve the explicit MLflow switch; disabled is the safe default."""
    raw = os.getenv("MLFLOW_ENABLED", "false") if value is None else value
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise MlflowClientConfigurationError(
        "MLFLOW_ENABLED must be one of true/false, 1/0, yes/no, or on/off"
    )


def get_effective_mlflow_tracking_uri() -> str:
    """Return the central HTTP(S) client URI, rejecting filesystem backends."""
    tracking_uri = get_mlflow_tracking_uri().strip()
    parsed = urlsplit(tracking_uri)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise MlflowClientConfigurationError(
            "Local training requires an HTTP(S) MLFLOW_TRACKING_URI served by MLflow; "
            "file, mlruns, and direct SQLite tracking backends are not supported."
        )
    return tracking_uri.rstrip("/")


def get_local_experiment_name(workflow: str) -> str:
    """Resolve a deterministic workflow experiment with one optional override."""
    try:
        default = LOCAL_EXPERIMENT_NAMES[workflow]
    except KeyError as exc:
        raise ValueError(f"Unknown local MLflow workflow: {workflow}") from exc
    return os.getenv("MLFLOW_EXPERIMENT_NAME", default).strip() or default


def _health_url(tracking_uri: str) -> str:
    parsed = urlsplit(tracking_uri)
    path = parsed.path.rstrip("/") + "/health"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _unreachable_message(detail: str) -> str:
    return (
        "MLflow tracking is enabled but the configured server is unreachable.\n\n"
        f"Tracking URI:\n{get_safe_mlflow_tracking_uri()}\n\n"
        "Start the server with:\npython -m backend.mlflow_server\n\n"
        "Or explicitly disable MLflow for this training run with MLFLOW_ENABLED=false.\n\n"
        f"Details: {detail}"
    )


def _client_initialization_message(detail: str) -> str:
    return (
        "The MLflow server is reachable, but the tracking client could not initialize.\n\n"
        f"Tracking URI:\n{get_safe_mlflow_tracking_uri()}\n\n"
        "Verify the MLflow client dependency and server permissions, or explicitly disable "
        "tracking with MLFLOW_ENABLED=false.\n\n"
        f"Details: {detail}"
    )


def check_mlflow_reachable(tracking_uri: str | None = None, *, timeout: float = 3.0) -> str:
    """Require the configured MLflow HTTP health endpoint to return HTTP 200."""
    effective_uri = tracking_uri or get_effective_mlflow_tracking_uri()
    parsed = urlsplit(effective_uri)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise MlflowClientConfigurationError(
            "Local training requires an HTTP(S) MLflow tracking server; no local backend fallback is allowed."
        )
    try:
        with urlopen(Request(_health_url(effective_uri), method="GET"), timeout=timeout) as response:
            if response.status != 200:
                raise MlflowClientConfigurationError(
                    _unreachable_message(f"health endpoint returned HTTP {response.status}")
                )
    except MlflowClientConfigurationError:
        raise
    except HTTPError as exc:
        raise MlflowClientConfigurationError(
            _unreachable_message(f"health endpoint returned HTTP {exc.code}")
        ) from exc
    except (OSError, URLError) as exc:
        reason = getattr(exc, "reason", None)
        detail = f"{type(exc).__name__}: {reason if reason is not None else 'connection failed'}"
        raise MlflowClientConfigurationError(_unreachable_message(detail)) from exc
    return effective_uri


def build_local_training_tags(
    *,
    training_mode: str,
    dataset: str,
    script: str,
    run_config_id: str,
    base_model: str,
    parent_model: str | None = None,
) -> dict[str, str]:
    """Build truthful, cross-platform provenance tags for a local run."""
    tags = {
        "viettoxic.model_family": "phobert",
        "viettoxic.training_mode": str(training_mode),
        "viettoxic.dataset": str(dataset),
        "viettoxic.script": str(script),
        "viettoxic.run_config_id": str(run_config_id),
        "viettoxic.base_model": str(base_model),
        "viettoxic.execution": "local",
        "viettoxic.platform": platform.system().lower() or "unknown",
    }
    if parent_model:
        tags["viettoxic.parent_model"] = str(parent_model)
    return tags


def configure_mlflow_client(
    *,
    enabled: bool | None = None,
    experiment_name: str,
    run_name: str,
    tags: Mapping[str, Any] | None = None,
    timeout: float = 3.0,
) -> Any | None:
    """Preflight and start one local MLflow run, or do nothing when disabled."""
    should_enable = is_mlflow_enabled() if enabled is None else bool(enabled)
    if not should_enable:
        return None

    tracking_uri = get_effective_mlflow_tracking_uri()
    check_mlflow_reachable(tracking_uri, timeout=timeout)
    try:
        mlflow = importlib.import_module("mlflow")
        os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags({str(key): str(value) for key, value in tags.items()})
    except Exception as exc:
        try:
            if "mlflow" in locals() and mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
        except Exception:
            pass
        raise MlflowClientConfigurationError(
            _client_initialization_message(f"MLflow client initialization failed: {type(exc).__name__}")
        ) from exc
    return mlflow
