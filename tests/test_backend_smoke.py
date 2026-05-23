import sqlite3
from datetime import datetime
from pathlib import Path


def _seed_mlflow_batch(feedback_db: Path, batch_id: str = "batch_smoke") -> None:
    now = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(feedback_db) as conn:
        conn.execute(
            """
            INSERT INTO mlflow_crawl_batch (batch_id, model_id, status, source_job_id, created_at, completed_at, options_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (batch_id, "phobert/v2", "completed", "job_1", now, now, "{}"),
        )
        conn.execute(
            """
            INSERT INTO mlflow_comment_item (
                batch_id, job_id, url, url_hash, domain_category, segment_id, text, score,
                pseudo_label, gate_bucket, verification_status, segment_hash, context_segment_hash,
                html_tag, seg_threshold_used, label_source, label_confidence, created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                "job_1",
                "https://example.com/a",
                "hash_a",
                "news",
                "seg_1",
                "accepted sample",
                0.95,
                1,
                "accepted",
                "accepted",
                "seg_hash_1",
                "ctx_hash_1",
                "p",
                0.4,
                "auto_gate",
                "high",
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO mlflow_comment_item (
                batch_id, job_id, url, url_hash, domain_category, segment_id, text, score,
                pseudo_label, gate_bucket, verification_status, segment_hash, context_segment_hash,
                html_tag, seg_threshold_used, label_source, label_confidence, created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                "job_1",
                "https://example.com/b",
                "hash_b",
                "news",
                "seg_2",
                "candidate sample",
                0.5,
                0,
                "candidate",
                "unverified",
                "seg_hash_2",
                "ctx_hash_2",
                "span",
                0.4,
                "auto_gate",
                "medium",
                now,
                None,
            ),
        )
        conn.commit()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert "default" in payload
    assert isinstance(payload["models"], list)


def test_analyze_validation_rejects_blank_urls(client):
    response = client.post("/api/analyze", json={"urls": ["   "]})
    assert response.status_code == 400
    assert "No valid URLs provided" in str(response.json().get("detail"))


def test_mlflow_overview_requires_admin(client):
    response = client.get("/api/mlflow/overview")
    assert response.status_code == 401


def test_admin_login_rejects_bad_password(client):
    response = client.post(
        "/api/admin/login",
        json={"username": "admin", "password": "wrong-password"},
    )
    assert response.status_code == 401


def test_admin_login_success_returns_session_token(client):
    response = client.post(
        "/api/admin/login",
        json={"username": "admin", "password": "admin-password"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("token"), str) and payload["token"]
    assert isinstance(payload.get("expires_at"), str) and payload["expires_at"]
    assert payload.get("username") == "admin"


def test_models_endpoint_contract_for_homepage(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert isinstance(payload["models"], list)
    assert "default" in payload


def test_mlflow_import_zip_requires_admin(client):
    response = client.post("/api/models/import-zip")
    assert response.status_code == 401


def test_system_settings_require_admin(client):
    assert client.get("/api/admin/system-settings").status_code == 401
    assert client.patch("/api/admin/system-settings", json={"settings": {}}).status_code == 401
    assert (
        client.post("/api/admin/system-settings/reveal-secret", json={"key": "KAGGLE_KEY"}).status_code
        == 401
    )


def test_system_settings_validate_mask_and_reveal_secret(client, admin_headers):
    invalid_key = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={"settings": {"UNKNOWN_SETTING": "x"}},
    )
    assert invalid_key.status_code == 400

    invalid_int = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={"settings": {"KAGGLE_WEBHOOK_TIMEOUT_SEC": 5}},
    )
    assert invalid_int.status_code == 400

    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_KEY": "secret-token",
                "KAGGLE_WEBHOOK_TIMEOUT_SEC": 181,
                "KAGGLE_KERNEL_PRIVATE": False,
            }
        },
    )
    assert response.status_code == 200
    groups = response.json()["groups"]
    settings = {
        item["key"]: item
        for group in groups
        for item in group["settings"]
    }
    assert settings["KAGGLE_KEY"]["secret"] is True
    assert settings["KAGGLE_KEY"]["value"] is None
    assert settings["KAGGLE_KEY"]["has_value"] is True
    assert "secret-token" not in str(settings["KAGGLE_KEY"]["masked_value"])
    assert settings["KAGGLE_WEBHOOK_TIMEOUT_SEC"]["value"] == "181"
    assert settings["KAGGLE_KERNEL_PRIVATE"]["value"] == "false"

    reveal = client.post(
        "/api/admin/system-settings/reveal-secret",
        headers=admin_headers,
        json={"key": "KAGGLE_KEY"},
    )
    assert reveal.status_code == 200
    assert reveal.json()["value"] == "secret-token"


def test_system_settings_db_override_env_for_gemini(client, admin_headers, monkeypatch, qa_env):
    app_module = qa_env["app_module"]
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    assert app_module.get_gemini_model_candidates()[0] == "env-model"

    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={"settings": {"GEMINI_MODEL": "db-model", "GEMINI_FALLBACK_MODELS": "fallback-a,fallback-b"}},
    )
    assert response.status_code == 200
    assert app_module.get_gemini_model_candidates() == ["db-model", "fallback-a", "fallback-b"]


def test_system_settings_drive_kaggle_preflight(client, admin_headers):
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9000/kaggle/trigger",
            }
        },
    )
    assert response.status_code == 200

    preflight = client.get("/api/mlflow/kaggle/preflight", headers=admin_headers)
    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["ready"] is True
    assert payload["missing"] == []
    assert payload["checks"]["KAGGLE_NOTEBOOK_URL"] is True
    assert payload["checks"]["KAGGLE_WEBHOOK_URL"] is True


def test_admin_system_settings_endpoint_contract(client, admin_headers):
    response = client.get("/api/admin/system-settings", headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()
    assert "groups" in payload
    assert isinstance(payload["groups"], list)
    group_ids = {group.get("id") for group in payload["groups"] if isinstance(group, dict)}
    assert {"kaggle_account", "kaggle_kernel", "kaggle_webhook", "gemini", "video_asr"}.issubset(group_ids)


def test_mlflow_review_history_all_batches_empty_contract(client, admin_headers):
    response = client.get(
        "/api/mlflow/review-history",
        headers=admin_headers,
        params={"scope": "all_batches", "decision": "all", "page": 1, "page_size": 25},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "all_batches"
    assert payload["decision"] == "all"
    assert payload["page"] == 1
    assert payload["page_size"] == 25
    assert isinstance(payload["items"], list)


def test_mlflow_overview_without_batch(client, admin_headers):
    response = client.get("/api/mlflow/overview", headers=admin_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "empty"
    assert payload["has_data"] is False
    assert payload["pipeline_counts"] == {
        "crawled": 0,
        "inferred": 0,
        "accepted": 0,
        "candidate": 0,
        "discarded": 0,
    }


def test_dataset_export_uses_temp_dataset(client, qa_env):
    response = client.post(
        "/api/dataset/export",
        json={
            "dataset_version": "victsd_gold",
            "model_version": "model-test",
            "policy_version": "policy-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 3
    export_path = qa_env["base_dir"] / payload["path"]
    assert export_path.exists()
    assert export_path.is_file()


def test_dataset_preview_rejects_legacy_v1_dataset(client):
    for legacy_version in ("v1", "victsd_v1"):
        response = client.get(
            "/api/dataset/preview",
            params={"dataset_version": legacy_version},
        )
        assert response.status_code == 400


def test_protocol_summary_endpoint_removed(client):
    response = client.get("/api/protocols/summary")
    assert response.status_code == 404


def test_mlflow_manual_export_bundle_isolated_temp_db(client, qa_env, admin_headers):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_export")
    response = client.post(
        "/api/mlflow/manual/export-bundle",
        headers=admin_headers,
        json={
            "scope": "batch",
            "batch_id": "batch_export",
            "bundle_profile": "clean_victsd_gold",
            "dataset_version": "victsd_gold",
            "model_version": "model-test",
            "policy_version": "policy-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "batch"
    assert payload["batch_id"] == "batch_export"
    assert payload["count"] == 1
    assert "train.jsonl" in payload["required_zip_contents"]

    bundle_path = qa_env["base_dir"] / payload["bundle_path"]
    assert bundle_path.exists()
    assert bundle_path.suffix == ".zip"
