import sqlite3
from datetime import datetime
from pathlib import Path


def _insert_status_run(feedback_db: Path, run_id: str) -> None:
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
                "batch_contract",
                "kaggle",
                "kaggle-gpu",
                "completed",
                "complete",
                "[]",
                now,
                now,
                "mock_job_1",
                "mock://contract.zip",
                None,
                None,
                None,
                None,
            ),
        )
        conn.commit()


def test_mlflow_overview_contract_shape(client, admin_headers):
    response = client.get("/api/mlflow/overview", headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()

    expected_top_keys = {
        "active_batch_id",
        "model_name",
        "status",
        "source_job_id",
        "last_run_at",
        "pipeline_counts",
        "has_data",
    }
    assert expected_top_keys.issubset(payload.keys())

    expected_pipeline_keys = {"crawled", "inferred", "accepted", "candidate", "discarded"}
    assert expected_pipeline_keys.issubset((payload.get("pipeline_counts") or {}).keys())


def test_mlflow_kaggle_status_contract_shape(client, qa_env, admin_headers):
    run_id = "run_contract"
    _insert_status_run(qa_env["feedback_db"], run_id)

    response = client.get("/api/mlflow/kaggle/status", params={"run_id": run_id}, headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()

    expected_keys = {
        "run_id",
        "batch_id",
        "provider",
        "gpu_profile",
        "compute_mode",
        "training_mode",
        "base_model",
        "status",
        "current_stage",
        "logs",
        "log_events",
        "stages",
        "artifact_uri",
        "artifact_kind",
        "artifact_download_url",
        "artifact_checksum",
        "metrics",
        "error_message",
        "run_mode",
        "status_source",
        "stage_timestamps",
        "created_at",
        "updated_at",
        "job_id",
    }
    assert expected_keys.issubset(payload.keys())
