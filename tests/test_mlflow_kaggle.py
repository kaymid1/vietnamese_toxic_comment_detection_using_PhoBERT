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


def test_kaggle_trigger_dry_run(client, admin_headers):
    response = client.post(
        "/api/mlflow/kaggle/trigger",
        headers=admin_headers,
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


def test_kaggle_preflight_real_mode_requires_credentials(client, admin_headers, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9000/kaggle/trigger",
            },
            "clear": ["KAGGLE_USERNAME", "KAGGLE_KEY"],
        },
    )
    assert response.status_code == 200

    preflight = client.get("/api/mlflow/kaggle/preflight", headers=admin_headers)
    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["ready"] is False
    assert payload["checks"]["KAGGLE_NOTEBOOK_URL"] is True
    assert payload["checks"]["KAGGLE_WEBHOOK_URL"] is True
    assert payload["checks"]["KAGGLE_USERNAME"] is False
    assert payload["checks"]["KAGGLE_KEY"] is False
    assert set(payload["missing"]) == {"KAGGLE_USERNAME", "KAGGLE_KEY"}
    assert payload["config"]["webhook_mode"] == "real"


def test_kaggle_trigger_real_mode_requires_credentials(client, admin_headers, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9000/kaggle/trigger",
            },
            "clear": ["KAGGLE_USERNAME", "KAGGLE_KEY"],
        },
    )
    assert response.status_code == 200

    trigger = client.post(
        "/api/mlflow/kaggle/trigger",
        headers=admin_headers,
        json={
            "dry_run": False,
            "training_mode": "finetune",
            "training_scope": "light_only",
            "provider": "kaggle",
            "compute_mode": "kaggle",
        },
    )
    assert trigger.status_code == 400
    assert "requires KAGGLE_USERNAME and KAGGLE_KEY" in str(trigger.json().get("detail"))


def test_kaggle_trigger_real_mode_rejects_mock_job_id(client, admin_headers, qa_env, monkeypatch):
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9000/kaggle/trigger",
                "KAGGLE_USERNAME": "kaymi",
                "KAGGLE_KEY": "secret-key",
            }
        },
    )
    assert response.status_code == 200

    app_module = qa_env["app_module"]
    monkeypatch.setattr(app_module, "_kaggle_http_json", lambda method, url, payload=None: {"job_id": "mock_abc123"})

    trigger = client.post(
        "/api/mlflow/kaggle/trigger",
        headers=admin_headers,
        json={
            "dry_run": False,
            "training_mode": "finetune",
            "training_scope": "light_only",
            "provider": "kaggle",
            "compute_mode": "kaggle",
        },
    )
    assert trigger.status_code == 502
    assert "webhook returned a mock job_id" in str(trigger.json().get("detail"))


def test_kaggle_trigger_webhook_failed_status_returns_502(client, admin_headers, qa_env, monkeypatch):
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9000/kaggle/trigger",
                "KAGGLE_USERNAME": "kaymi",
                "KAGGLE_KEY": "secret-key",
            }
        },
    )
    assert response.status_code == 200

    app_module = qa_env["app_module"]
    monkeypatch.setattr(
        app_module,
        "_kaggle_http_json",
        lambda method, url, payload=None: {"accepted": True, "status": "failed", "job_id": "real_deadbeef", "message": "Auth failed"},
    )

    trigger = client.post(
        "/api/mlflow/kaggle/trigger",
        headers=admin_headers,
        json={
            "dry_run": False,
            "training_mode": "finetune",
            "training_scope": "light_only",
            "provider": "kaggle",
            "compute_mode": "kaggle",
        },
    )
    assert trigger.status_code == 502
    assert "Auth failed" in str(trigger.json().get("detail"))


def test_kaggle_webhook_receiver_reads_db_settings_at_request_time(client, qa_env, admin_headers, monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    monkeypatch.setattr(receiver, "DEFAULT_SETTINGS_DB_PATH", qa_env["feedback_db"])
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_KERNEL_OWNER": "db-owner",
                "KAGGLE_KERNEL_SLUG": "db-slug",
                "KAGGLE_USERNAME": "db-user",
                "KAGGLE_KEY": "db-key",
            }
        },
    )
    assert response.status_code == 200

    assert receiver._webhook_mode() == "real"
    assert receiver._resolve_owner_slug(None) == ("db-owner", "db-slug")
    subprocess_env = receiver._build_subprocess_env()
    assert subprocess_env["KAGGLE_USERNAME"] == "db-user"
    assert subprocess_env["KAGGLE_KEY"] == "db-key"

    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={"settings": {"KAGGLE_WEBHOOK_MODE": "mock"}},
    )
    assert response.status_code == 200
    assert receiver._webhook_mode() == "mock"


def test_kaggle_status_missing_run(client, admin_headers):
    response = client.get("/api/mlflow/kaggle/status", params={"run_id": "run_missing"}, headers=admin_headers)
    assert response.status_code == 404
    assert "Kaggle run not found" in str(response.json().get("detail"))


def test_kaggle_mock_artifact_rejected(client, qa_env, admin_headers):
    run_id = "run_mock_artifact"
    _insert_kaggle_run(
        qa_env["feedback_db"],
        run_id,
        status="completed",
        artifact_uri="mock://artifact.zip",
    )
    response = client.get("/api/mlflow/kaggle/artifact/download", params={"run_id": run_id}, headers=admin_headers)
    assert response.status_code == 400
    assert "Mock Kaggle artifacts are not downloadable" in str(response.json().get("detail"))


def test_kaggle_real_local_artifact_download_and_metrics_parsing(client, qa_env, admin_headers):
    run_id = "run_real_artifact"
    artifact_path = qa_env["kaggle_root"] / run_id / "output" / "artifact.zip"
    _build_metrics_zip(artifact_path)
    _insert_kaggle_run(
        qa_env["feedback_db"],
        run_id,
        status="completed",
        artifact_uri=str(artifact_path),
    )

    status_response = client.get("/api/mlflow/kaggle/status", params={"run_id": run_id}, headers=admin_headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["artifact_kind"] == "real"
    assert status_payload["metrics"]["f1_toxic"] == 0.71
    assert status_payload["metrics"]["macro_f1"] == 0.76
    assert status_payload["metrics"]["source_member"] == "results/metrics.json"

    download_response = client.get("/api/mlflow/kaggle/artifact/download", params={"run_id": run_id}, headers=admin_headers)
    assert download_response.status_code == 200
    assert download_response.headers.get("content-type") == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(download_response.content))
    assert "results/metrics.json" in archive.namelist()
