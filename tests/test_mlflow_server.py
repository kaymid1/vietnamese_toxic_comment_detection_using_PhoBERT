from dataclasses import replace
from pathlib import Path

import pytest

from backend import runtime_paths
from backend.mlflow_legacy_export import sha256_file
from backend.mlflow_server import (
    build_mlflow_server_command,
    resolve_mlflow_server_config,
    validate_mlflow_server_config,
)


def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_SERVER_HOST", raising=False)
    monkeypatch.delenv("MLFLOW_SERVER_PORT", raising=False)
    return resolve_mlflow_server_config()


def test_server_config_is_local_and_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = _isolated_config(monkeypatch, tmp_path)
    legacy_hash = sha256_file(config.legacy_db_path)

    validate_mlflow_server_config(config)

    assert config.backend_db_path == tmp_path / "app-data" / "mlflow" / "mlflow.db"
    assert config.backend_db_path != config.legacy_db_path
    assert config.artifact_root == tmp_path / "app-data" / "mlflow" / "artifacts"
    assert config.evidence_dir == tmp_path / "app-data" / "mlflow" / "evidence"
    assert config.host == "127.0.0.1"
    assert config.port == 5000
    assert config.client_tracking_uri == "http://127.0.0.1:5000"
    assert config.backend_db_path.parent.is_dir()
    assert config.artifact_root.is_dir()
    assert config.evidence_dir.is_dir()
    assert sha256_file(config.legacy_db_path) == legacy_hash


def test_server_command_uses_served_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = _isolated_config(monkeypatch, tmp_path)
    command = build_mlflow_server_command(config)

    assert command[:4] == [command[0], "-m", "mlflow", "server"]
    assert command[command.index("--backend-store-uri") + 1] == config.backend_store_uri
    assert "--serve-artifacts" in command
    assert command[command.index("--artifacts-destination") + 1] == config.artifact_destination_uri
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "5000"
    assert command[command.index("--workers") + 1] == "1"
    assert str(config.legacy_db_path) not in command


def test_preflight_refuses_legacy_database_as_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = _isolated_config(monkeypatch, tmp_path)
    unsafe = replace(
        config,
        backend_db_path=config.legacy_db_path,
        backend_store_uri=f"sqlite:///{config.legacy_db_path.as_posix()}",
    )

    with pytest.raises(RuntimeError, match="immutable legacy root mlflow.db"):
        validate_mlflow_server_config(unsafe)


@pytest.mark.parametrize("host", ["", "http://127.0.0.1", "bad host", "../mlflow"])
def test_preflight_rejects_invalid_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str
):
    config = _isolated_config(monkeypatch, tmp_path)
    invalid = replace(config, host=host)

    with pytest.raises(ValueError, match="MLFLOW_SERVER_HOST"):
        validate_mlflow_server_config(invalid, create_directories=False)
