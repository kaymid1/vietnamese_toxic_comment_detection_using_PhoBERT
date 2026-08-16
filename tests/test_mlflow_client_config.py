from pathlib import Path
from urllib.error import URLError

import pytest

from backend import mlflow_client_config


class _HealthyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeMlflow:
    def __init__(self):
        self.calls = []
        self._active_run = None

    def set_tracking_uri(self, value):
        self.calls.append(("tracking_uri", value))

    def set_experiment(self, value):
        self.calls.append(("experiment", value))

    def start_run(self, *, run_name):
        self.calls.append(("run", run_name))
        self._active_run = object()

    def set_tags(self, value):
        self.calls.append(("tags", value))

    def active_run(self):
        return self._active_run

    def end_run(self, *, status):
        self.calls.append(("end", status))
        self._active_run = None


def test_disabled_does_not_initialize_mlflow_or_create_mlruns(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_ENABLED", "false")

    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(mlflow_client_config.importlib, "import_module", fail_import)

    assert mlflow_client_config.configure_mlflow_client(
        experiment_name="unused", run_name="unused"
    ) is None
    assert not (tmp_path / "mlruns").exists()


def test_server_host_and_port_drive_default_uri_and_explicit_uri_wins(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_SERVER_HOST", "localhost")
    monkeypatch.setenv("MLFLOW_SERVER_PORT", "5055")
    assert mlflow_client_config.get_effective_mlflow_tracking_uri() == "http://localhost:5055"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.test/root/")
    assert (
        mlflow_client_config.get_effective_mlflow_tracking_uri()
        == "https://tracking.example.test/root"
    )


def test_experiment_names_are_deterministic_with_one_explicit_override(monkeypatch):
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    assert mlflow_client_config.get_local_experiment_name("phobert_v2_macro_f1") == (
        "viettoxic-local-phobert-v2-full-finetune-macro-f1"
    )

    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "approved-custom-experiment")
    assert (
        mlflow_client_config.get_local_experiment_name("phobert_v2_macro_f1")
        == "approved-custom-experiment"
    )


@pytest.mark.parametrize("tracking_uri", ["mlruns/", "file:///tmp/mlruns", "sqlite:///mlflow.db"])
def test_local_filesystem_tracking_backends_are_rejected(monkeypatch, tracking_uri):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    with pytest.raises(mlflow_client_config.MlflowClientConfigurationError) as exc_info:
        mlflow_client_config.get_effective_mlflow_tracking_uri()
    assert "HTTP(S)" in str(exc_info.value)
    assert "not supported" in str(exc_info.value)


def test_enabled_unreachable_fails_before_mlflow_import_and_redacts_uri(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", "http://user:secret@127.0.0.1:1/root?token=hidden"
    )
    imported = []

    def unavailable(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(mlflow_client_config, "urlopen", unavailable)
    monkeypatch.setattr(
        mlflow_client_config.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(mlflow_client_config.MlflowClientConfigurationError) as exc_info:
        mlflow_client_config.configure_mlflow_client(
            enabled=True, experiment_name="exp", run_name="run"
        )

    message = str(exc_info.value)
    assert "configured server is unreachable" in message
    assert "python -m backend.mlflow_server" in message
    assert "MLFLOW_ENABLED=false" in message
    assert "http://127.0.0.1:1/root" in message
    assert "secret" not in message
    assert "hidden" not in message
    assert imported == []


def test_enabled_reachable_configures_http_experiment_run_and_tags(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5055")
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(mlflow_client_config, "urlopen", lambda request, timeout: _HealthyResponse())
    monkeypatch.setattr(
        mlflow_client_config.importlib,
        "import_module",
        lambda name: fake_mlflow,
    )

    result = mlflow_client_config.configure_mlflow_client(
        enabled=True,
        experiment_name="viettoxic-local-test",
        run_name="run-123",
        tags={"viettoxic.execution": "local"},
    )

    assert result is fake_mlflow
    assert fake_mlflow.calls == [
        ("tracking_uri", "http://127.0.0.1:5055"),
        ("experiment", "viettoxic-local-test"),
        ("run", "run-123"),
        ("tags", {"viettoxic.execution": "local"}),
    ]


def test_provenance_tags_include_only_proven_parent_model():
    tags = mlflow_client_config.build_local_training_tags(
        training_mode="finetune",
        dataset="victsd_gold",
        script="train.py",
        run_config_id="run-123",
        base_model="bundled/base_model",
        parent_model="phobert/parent-v7",
    )

    assert tags["viettoxic.model_family"] == "phobert"
    assert tags["viettoxic.training_mode"] == "finetune"
    assert tags["viettoxic.dataset"] == "victsd_gold"
    assert tags["viettoxic.script"] == "train.py"
    assert tags["viettoxic.run_config_id"] == "run-123"
    assert tags["viettoxic.base_model"] == "bundled/base_model"
    assert tags["viettoxic.parent_model"] == "phobert/parent-v7"
    assert tags["viettoxic.execution"] == "local"
    assert tags["viettoxic.platform"]


def test_migrated_scripts_have_no_mlruns_fallback():
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        "05_train_phobert_macro_f1.py",
        "06_train_phobert_lora.py",
        "06_train_phobert_lora_macro_f1.py",
        "06_train_phobert_lora_macro_f1_finetune.py",
    ]
    for name in scripts:
        source = (project_root / "scripts" / name).read_text(encoding="utf-8")
        assert '"mlruns/"' not in source
        assert "configure_mlflow_client(" in source
