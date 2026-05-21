import io
import json
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path


def _insert_kaggle_run(
    feedback_db: Path,
    run_id: str,
    *,
    status: str,
    artifact_uri: str | None = None,
    current_stage: str = "complete",
) -> None:
    now = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(feedback_db) as conn:
        conn.execute(
            """
            INSERT INTO mlflow_do_run (
                run_id, batch_id, provider, gpu_profile, status, current_stage, logs_json,
                created_at, updated_at, droplet_id, artifact_uri, artifact_checksum,
                spaces_bucket, spaces_key, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                "batch_kaggle",
                "kaggle",
                "kaggle-gpu",
                status,
                current_stage,
                "[]",
                now,
                now,
                None,
                artifact_uri,
                None,
                None,
                None,
                None,
            ),
        )
        conn.commit()


def _build_metrics_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": "kaggle_local_run",
        "test": {
            "f1_toxic": 0.71,
            "macro_f1": 0.76,
            "accuracy": 0.88,
            "precision_toxic": 0.69,
            "recall_toxic": 0.73,
        },
    }
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results/metrics.json", json.dumps(payload))
        zf.writestr("model.bin", b"fake-model-bytes")


def test_kaggle_trigger_dry_run(client):
    response = client.post(
        "/api/mlflow/kaggle/trigger",
        json={
            "dry_run": True,
            "training_mode": "finetune",
            "training_scope": "light_only",
            "provider": "kaggle",
            "compute_mode": "kaggle",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "dry_run"
    assert payload["provider"] == "kaggle"
    assert payload["dry_run"] is True
    assert payload.get("run_id")


def test_kaggle_status_missing_run(client):
    response = client.get("/api/mlflow/kaggle/status", params={"run_id": "run_missing"})
    assert response.status_code == 404
    assert "Kaggle run not found" in str(response.json().get("detail"))


def test_kaggle_mock_artifact_rejected(client, qa_env):
    run_id = "run_mock_artifact"
    _insert_kaggle_run(
        qa_env["feedback_db"],
        run_id,
        status="completed",
        artifact_uri="mock://artifact.zip",
    )
    response = client.get("/api/mlflow/kaggle/artifact/download", params={"run_id": run_id})
    assert response.status_code == 400
    assert "Mock Kaggle artifacts are not downloadable" in str(response.json().get("detail"))


def test_kaggle_real_local_artifact_download_and_metrics_parsing(client, qa_env):
    run_id = "run_real_artifact"
    artifact_path = qa_env["kaggle_root"] / run_id / "output" / "artifact.zip"
    _build_metrics_zip(artifact_path)
    _insert_kaggle_run(
        qa_env["feedback_db"],
        run_id,
        status="completed",
        artifact_uri=str(artifact_path),
    )

    status_response = client.get("/api/mlflow/kaggle/status", params={"run_id": run_id})
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["artifact_kind"] == "real"
    assert status_payload["metrics"]["f1_toxic"] == 0.71
    assert status_payload["metrics"]["macro_f1"] == 0.76
    assert status_payload["metrics"]["source_member"] == "results/metrics.json"

    download_response = client.get("/api/mlflow/kaggle/artifact/download", params={"run_id": run_id})
    assert download_response.status_code == 200
    assert download_response.headers.get("content-type") == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(download_response.content))
    assert "results/metrics.json" in archive.namelist()
