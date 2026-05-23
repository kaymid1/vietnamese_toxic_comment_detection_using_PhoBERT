import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app as app_module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture()
def qa_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    base_dir = tmp_path / "qa_repo"
    base_dir.mkdir(parents=True, exist_ok=True)

    feedback_dir = base_dir / "data" / "processed" / "feedback"
    feedback_db = feedback_dir / "feedback.db"
    processed_dir = base_dir / "data" / "processed"
    victsd_gold_dir = processed_dir / "victsd_gold"
    models_dir = base_dir / "models" / "options"
    kaggle_root = base_dir / ".runtime" / "kaggle_real_jobs"

    for split in ("train", "validation", "test"):
        _write_jsonl(
            victsd_gold_dir / f"{split}.jsonl",
            [
                {
                    "text": f"sample {split} text",
                    "toxicity": 0,
                    "meta": {"source": "victsd"},
                }
            ],
        )

    monkeypatch.setattr(app_module, "BASE_DIR", base_dir)
    monkeypatch.setattr(app_module, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(app_module, "FEEDBACK_DB_PATH", feedback_db)
    monkeypatch.setattr(app_module, "MODEL_OPTIONS_DIR", models_dir)
    monkeypatch.setattr(
        app_module,
        "DATASET_VERSION_DIRS",
        {
            "victsd_gold": victsd_gold_dir,
        },
    )
    monkeypatch.setattr(app_module, "KAGGLE_ARTIFACT_ROOT", kaggle_root)
    monkeypatch.setenv("VIETTOXIC_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("VIETTOXIC_ADMIN_PASSWORD", "admin-password")
    monkeypatch.setenv("VIETTOXIC_ADMIN_SESSION_SECRET", "test-admin-session-secret")

    app_module.init_feedback_db()

    return {
        "base_dir": base_dir,
        "feedback_db": feedback_db,
        "processed_dir": processed_dir,
        "kaggle_root": kaggle_root,
        "app_module": app_module,
    }


@pytest.fixture()
def client(qa_env: dict) -> TestClient:
    return TestClient(qa_env["app_module"].app)


@pytest.fixture()
def admin_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/admin/login",
        json={"username": "admin", "password": "admin-password"},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    return {"Authorization": f"Bearer {token}"}
