import json
import subprocess
import sys
from pathlib import Path


def test_backend_app_loads_env_before_import_time_defaults(tmp_path):
    project = tmp_path / "project"
    backend_dir = project / "backend"
    backend_dir.mkdir(parents=True)

    data_dir = tmp_path / "canonical-data"
    runtime_dir = tmp_path / "canonical-runtime"
    models_dir = tmp_path / "canonical-models"
    (project / ".env").write_text(
        "APP_DATA_DIR=from-root-env\n"
        "APP_RUNTIME_DIR=from-root-env\n"
        "VIETTOXIC_MODEL_OPTIONS_DIR=from-root-env\n"
        "MLFLOW_ACCEPT_THRESHOLD=0.11\n",
        encoding="utf-8",
    )
    (project / ".env.local").write_text(
        "APP_DATA_DIR=from-root-local\n"
        "APP_RUNTIME_DIR=from-root-local\n"
        "VIETTOXIC_MODEL_OPTIONS_DIR=from-root-local\n"
        "MLFLOW_ACCEPT_THRESHOLD=0.22\n",
        encoding="utf-8",
    )
    (backend_dir / ".env").write_text(
        "APP_DATA_DIR=from-backend-env\n"
        "APP_RUNTIME_DIR=from-backend-env\n"
        "VIETTOXIC_MODEL_OPTIONS_DIR=from-backend-env\n"
        "MLFLOW_ACCEPT_THRESHOLD=0.33\n",
        encoding="utf-8",
    )
    (backend_dir / ".env.local").write_text(
        f"APP_DATA_DIR={data_dir}\n"
        f"APP_RUNTIME_DIR={runtime_dir}\n"
        f"VIETTOXIC_MODEL_OPTIONS_DIR={models_dir}\n"
        "MLFLOW_ACCEPT_THRESHOLD=0.44\n",
        encoding="utf-8",
    )

    script = """
import json
import sys
from pathlib import Path

import backend.runtime_paths as runtime_paths

runtime_paths.get_project_root = lambda: Path(sys.argv[1])

import backend.app as app
import backend.system_settings as system_settings
import infer_crawled_local

print(json.dumps({
    "app_data": str(app.APP_DATA_DIR),
    "runtime": str(runtime_paths.get_runtime_dir()),
    "models": str(app.MODEL_OPTIONS_DIR),
    "feedback_db": str(app.FEEDBACK_DB_PATH),
    "settings_db": str(system_settings.DEFAULT_SETTINGS_DB_PATH),
    "inference_models": str(infer_crawled_local.MODEL_OPTIONS_DIR),
    "threshold": app.MLFLOW_ACCEPT_THRESHOLD,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(project)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["app_data"] == str(data_dir.resolve())
    assert payload["runtime"] == str(runtime_dir.resolve())
    assert payload["models"] == str(models_dir.resolve())
    assert payload["feedback_db"] == str((data_dir / "processed" / "feedback" / "feedback.db").resolve())
    assert payload["settings_db"] == payload["feedback_db"]
    assert Path(payload["inference_models"]).resolve() == models_dir.resolve()
    assert payload["threshold"] == 0.44
