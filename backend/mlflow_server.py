"""Safe launcher for the fresh local-only MLflow tracking server."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from backend.env_loader import load_env_files

load_env_files()

from backend.runtime_paths import (
    get_legacy_mlflow_db_path,
    get_mlflow_artifact_destination_uri,
    get_mlflow_artifact_root,
    get_mlflow_backend_store_uri,
    get_mlflow_db_path,
    get_mlflow_evidence_dir,
    get_mlflow_server_host,
    get_mlflow_server_port,
    get_mlflow_server_tracking_uri,
)


@dataclass(frozen=True)
class MlflowServerConfig:
    legacy_db_path: Path
    backend_db_path: Path
    backend_store_uri: str
    artifact_root: Path
    artifact_destination_uri: str
    evidence_dir: Path
    host: str
    port: int
    client_tracking_uri: str


def resolve_mlflow_server_config() -> MlflowServerConfig:
    return MlflowServerConfig(
        legacy_db_path=get_legacy_mlflow_db_path(),
        backend_db_path=get_mlflow_db_path(),
        backend_store_uri=get_mlflow_backend_store_uri(),
        artifact_root=get_mlflow_artifact_root(),
        artifact_destination_uri=get_mlflow_artifact_destination_uri(),
        evidence_dir=get_mlflow_evidence_dir(),
        host=get_mlflow_server_host(),
        port=get_mlflow_server_port(),
        client_tracking_uri=get_mlflow_server_tracking_uri(),
    )


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first.expanduser().resolve())) == os.path.normcase(
        str(second.expanduser().resolve())
    )


def _validate_host(host: str) -> None:
    candidate = host.strip()
    if not candidate or any(char.isspace() for char in candidate):
        raise ValueError("MLFLOW_SERVER_HOST must be a non-empty host without whitespace")
    if "://" in candidate or "/" in candidate or "\\" in candidate:
        raise ValueError("MLFLOW_SERVER_HOST must contain only a host, not a URL or path")
    address_candidate = candidate[1:-1] if candidate.startswith("[") and candidate.endswith("]") else candidate
    try:
        ipaddress.ip_address(address_candidate)
        return
    except ValueError:
        pass
    if len(candidate) > 253 or not all(
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
        for label in candidate.rstrip(".").split(".")
    ):
        raise ValueError("MLFLOW_SERVER_HOST is not a valid IP address or hostname")


def _ensure_writable_directory(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"{label} is not a directory: {path}")
    try:
        with tempfile.NamedTemporaryFile(prefix=".viettoxic-preflight-", dir=path, delete=True):
            pass
    except OSError as exc:
        raise RuntimeError(f"{label} is not writable: {path}") from exc


def validate_mlflow_server_config(
    config: MlflowServerConfig, *, create_directories: bool = True
) -> MlflowServerConfig:
    if _same_path(config.backend_db_path, config.legacy_db_path):
        raise RuntimeError("Refusing to use the immutable legacy root mlflow.db as the writable backend")
    if not 1 <= int(config.port) <= 65535:
        raise ValueError("MLflow server port must be between 1 and 65535")
    _validate_host(config.host)
    if config.backend_db_path.exists() and not config.backend_db_path.is_file():
        raise RuntimeError(f"MLflow backend DB path is not a file: {config.backend_db_path}")
    if _same_path(config.artifact_root, config.legacy_db_path):
        raise RuntimeError("MLflow artifact root cannot be the legacy database path")
    if _same_path(config.evidence_dir, config.legacy_db_path):
        raise RuntimeError("MLflow evidence directory cannot be the legacy database path")
    if create_directories:
        _ensure_writable_directory(config.backend_db_path.parent, "MLflow directory")
        _ensure_writable_directory(config.artifact_root, "MLflow artifact root")
        _ensure_writable_directory(config.evidence_dir, "MLflow evidence directory")
    return config


def build_mlflow_server_command(config: MlflowServerConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        config.backend_store_uri,
        "--serve-artifacts",
        "--artifacts-destination",
        config.artifact_destination_uri,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--workers",
        "1",
    ]


def safe_diagnostic(config: MlflowServerConfig) -> dict[str, object]:
    return {
        "artifact_destination_uri": config.artifact_destination_uri,
        "artifact_root": str(config.artifact_root),
        "backend_db_path": str(config.backend_db_path),
        "backend_store_uri": config.backend_store_uri,
        "client_tracking_uri": config.client_tracking_uri,
        "evidence_dir": str(config.evidence_dir),
        "host": config.host,
        "legacy_db_path": str(config.legacy_db_path),
        "port": config.port,
        "serve_artifacts": True,
    }


def run_mlflow_server(command: Sequence[str], *, working_directory: Path) -> int:
    return subprocess.call(list(command), cwd=working_directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run preflight and print the safe resolved contract without starting MLflow",
    )
    args = parser.parse_args()
    config = validate_mlflow_server_config(resolve_mlflow_server_config())
    if args.check:
        print(json.dumps(safe_diagnostic(config), indent=2, sort_keys=True))
        return 0
    return run_mlflow_server(
        build_mlflow_server_command(config), working_directory=config.backend_db_path.parent
    )


if __name__ == "__main__":
    raise SystemExit(main())
