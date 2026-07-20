import json
import sqlite3
import zipfile
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


def test_system_settings_drive_kaggle_preflight(client, admin_headers, qa_env, monkeypatch):
    app_module = qa_env["app_module"]
    monkeypatch.setattr(app_module, "_kaggle_webhook_reachability", lambda webhook_url: (True, None))
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
    assert payload["checks"]["KAGGLE_WEBHOOK_REACHABLE"] is True


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


def test_mlflow_training_preview_and_full_bundle_include_phobert_assets(client, qa_env, admin_headers):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_phobert")
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            """
            UPDATE mlflow_comment_item
            SET constructiveness_score = 0.82,
                constructiveness_label = 1,
                constructiveness_confidence = 'high'
            WHERE text = 'accepted sample'
            """
        )
        conn.execute(
            """
            INSERT INTO mlflow_comment_item (
                batch_id, job_id, url, url_hash, domain_category, segment_id, text, score,
                pseudo_label, constructiveness_score, constructiveness_label, constructiveness_confidence,
                gate_bucket, verification_status, segment_hash, context_segment_hash,
                html_tag, seg_threshold_used, label_source, label_confidence, created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "batch_phobert",
                "job_1",
                "https://example.com/c",
                "hash_c",
                "news",
                "seg_3",
                "clean sample",
                0.05,
                0,
                0.12,
                0,
                "high",
                "accepted",
                "auto_accepted",
                "seg_hash_3",
                "ctx_hash_3",
                "p",
                0.4,
                "auto_gate",
                "high",
                datetime.utcnow().isoformat() + "Z",
                None,
            ),
        )
        conn.commit()

    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_phobert", "strict_batch": "true"},
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["counts"]["selected_toxic"] == 1
    assert preview_payload["counts"]["selected_clean"] == 1
    assert preview_payload["constructiveness"]["included"] == 2

    response = client.post(
        "/api/mlflow/manual/export-bundle",
        headers=admin_headers,
        json={
            "scope": "batch",
            "batch_id": "batch_phobert",
            "bundle_profile": "full_bundle",
            "model_kind": "phobert",
            "training_mode": "finetune",
            "balance_strategy": "balanced_50_50",
            "dataset_version": "victsd_gold",
            "model_version": "model-test",
            "policy_version": "policy-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["balance_stats"]["selected_toxic"] == 1
    assert payload["balance_stats"]["selected_clean"] == 1
    assert payload["constructiveness"]["included"] == 2

    bundle_path = qa_env["base_dir"] / payload["bundle_path"]
    with zipfile.ZipFile(bundle_path, "r") as zf:
        names = set(zf.namelist())
        assert "pseudo/accepted.jsonl" in names
        assert "pseudo/manifest.json" in names
        assert "scripts/train_phobert.py" in names
        pseudo_rows = [
            json.loads(line)
            for line in zf.read("pseudo/accepted.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
    assert {row["toxicity"] for row in pseudo_rows} == {0, 1}
    assert all(row.get("constructiveness") in {0, 1} for row in pseudo_rows)


def test_mlflow_lock_prevents_accidental_remove_in_training_preview(client, qa_env, admin_headers):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_lock_preview")
    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_lock_preview", "strict_batch": "true"},
    )
    assert preview.status_code == 200
    payload = preview.json()
    target = next(item for item in payload["items"] if item["text"] == "accepted sample")
    target_id = int(target["id"])

    lock_resp = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={"updates": [{"id": target_id, "lock_state": True}]},
    )
    assert lock_resp.status_code == 200
    assert lock_resp.json()["updated"] >= 1

    remove_resp = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={"updates": [{"id": target_id, "selected_for_training": False}]},
    )
    assert remove_resp.status_code == 200
    assert remove_resp.json()["skipped_locked"] == 1

    preview_after = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_lock_preview", "strict_batch": "true"},
    )
    assert preview_after.status_code == 200
    target_after = next(item for item in preview_after.json()["items"] if int(item["id"]) == target_id)
    assert int(target_after["is_locked"]) == 1
    assert int(target_after["selected_for_training"]) == 1

    unlock_and_remove = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={"updates": [{"id": target_id, "lock_state": False, "selected_for_training": False}]},
    )
    assert unlock_and_remove.status_code == 200
    assert unlock_and_remove.json()["updated"] >= 1


def test_mlflow_gemini_review_requires_admin(client, qa_env):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_gemini_auth")
    response = client.post("/api/mlflow/training-preview/gemini-review", json={"ids": [1]})
    assert response.status_code == 401


def test_mlflow_gemini_review_missing_api_key(client, qa_env, admin_headers):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_gemini_missing_key")
    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_gemini_missing_key", "strict_batch": "true"},
    )
    target_id = preview.json()["items"][0]["id"]

    response = client.post(
        "/api/mlflow/training-preview/gemini-review",
        headers=admin_headers,
        json={"ids": [target_id]},
    )
    assert response.status_code == 400
    assert "Missing GEMINI_API_KEY" in response.json()["detail"]


def test_mlflow_gemini_review_parses_mock_response_and_apply_metadata(client, qa_env, admin_headers, monkeypatch):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_gemini_review")
    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_gemini_review", "strict_batch": "true"},
    )
    assert preview.status_code == 200
    target_id = preview.json()["items"][0]["id"]

    def fake_call_gemini(prompt: str) -> str:
        assert "toxicity_label" in prompt
        return json.dumps(
            [
                {
                    "id": target_id,
                    "toxicity_label": 0,
                    "constructiveness_label": None,
                    "confidence": "high",
                    "reason": "Khong thay cong kich hoac ngon tu doc hai.",
                    "action": "apply",
                }
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(qa_env["app_module"], "call_gemini", fake_call_gemini)
    response = client.post(
        "/api/mlflow/training-preview/gemini-review",
        headers=admin_headers,
        json={"ids": [target_id]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "gemini"
    assert payload["reviewed"] == 1
    suggestion = payload["suggestions"][0]
    assert suggestion["id"] == target_id
    assert suggestion["toxicity_label"] == 0
    assert suggestion["constructiveness_label"] is None
    assert suggestion["confidence"] == "high"

    apply_response = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={
            "updates": [
                {
                    "id": target_id,
                    "pseudo_label": suggestion["toxicity_label"],
                    "clear_constructiveness": True,
                    "label_source": "gemini_assist",
                    "label_confidence": suggestion["confidence"],
                }
            ]
        },
    )
    assert apply_response.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        row = conn.execute(
            "SELECT pseudo_label, constructiveness_label, label_source, label_confidence FROM mlflow_comment_item WHERE id = ?",
            (target_id,),
        ).fetchone()
    assert row == (0, None, "gemini_assist", "high")


def test_mlflow_gemini_review_malformed_response_returns_502(client, qa_env, admin_headers, monkeypatch):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_gemini_bad")
    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_gemini_bad", "strict_batch": "true"},
    )
    target_id = preview.json()["items"][0]["id"]

    monkeypatch.setattr(qa_env["app_module"], "call_gemini", lambda _prompt: "not json")
    response = client.post(
        "/api/mlflow/training-preview/gemini-review",
        headers=admin_headers,
        json={"ids": [target_id]},
    )
    assert response.status_code == 502
    assert "valid review suggestions" in response.json()["detail"]


def test_mlflow_lock_prevents_drop_in_candidate_review(client, qa_env, admin_headers):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_lock_candidates")
    candidates = client.get(
        "/api/mlflow/candidates",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_lock_candidates", "strict_batch": "true"},
    )
    assert candidates.status_code == 200
    target = next(item for item in candidates.json()["items"] if item["text"] == "candidate sample")
    target_id = int(target["id"])

    lock_resp = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={"updates": [{"id": target_id, "lock_state": True}]},
    )
    assert lock_resp.status_code == 200
    assert lock_resp.json()["locked_updated"] >= 1

    drop_resp = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={"updates": [{"id": target_id, "action": "drop"}]},
    )
    assert drop_resp.status_code == 200
    assert drop_resp.json()["skipped_locked"] == 1


def test_mlflow_collection_dedupes_existing_url_and_duplicate_segment(qa_env):
    app_module = qa_env["app_module"]
    now = datetime.utcnow().isoformat() + "Z"

    def response_result(url: str, url_hash: str, context_hash: str) -> list[dict]:
        return [
            {
                "status": "ok",
                "url": url,
                "url_hash": url_hash,
                "domain_category": "news",
                "toxicity": {
                    "by_segment": [
                        {
                            "segment_id": f"{url_hash}:0",
                            "text": "Một bình luận mới cần admin review",
                            "score": 0.91,
                            "html_tags": ["comment"],
                            "segment_hash": "segment_hash_shared",
                            "context_segment_hash": context_hash,
                            "seg_threshold_used": 0.4,
                        }
                    ]
                },
            }
        ]

    first_rows = app_module.build_mlflow_comment_rows(
        response_result("https://example.com/a", "hash_a", "ctx_hash_a"),
        "batch_auto_1",
        "job_auto_1",
        0.8,
        0.2,
        now,
    )
    first = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_1",
        model_id="phobert/v2",
        source_job_id="job_auto_1",
        rows=first_rows,
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert first["batch_id"] == "batch_auto_1"
    assert first["inserted"] == 1

    same_url_rows = app_module.build_mlflow_comment_rows(
        response_result("https://example.com/a", "hash_a", "ctx_hash_a_new"),
        "batch_auto_2",
        "job_auto_2",
        0.8,
        0.2,
        now,
    )
    same_url = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_2",
        model_id="phobert/v2",
        source_job_id="job_auto_2",
        rows=same_url_rows,
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert same_url["batch_id"] is None
    assert same_url["inserted"] == 0
    assert same_url["skipped_existing_url"] == 1

    duplicate_segment_rows = app_module.build_mlflow_comment_rows(
        response_result("https://example.com/b", "hash_b", "ctx_hash_a"),
        "batch_auto_3",
        "job_auto_3",
        0.8,
        0.2,
        now,
    )
    duplicate_segment = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_3",
        model_id="phobert/v2",
        source_job_id="job_auto_3",
        rows=duplicate_segment_rows,
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert duplicate_segment["batch_id"] is None
    assert duplicate_segment["inserted"] == 0
    assert duplicate_segment["skipped_duplicate_item"] == 1

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        item_count = conn.execute("SELECT COUNT(1) FROM mlflow_comment_item").fetchone()[0]
        batch_count = conn.execute("SELECT COUNT(1) FROM mlflow_crawl_batch").fetchone()[0]
    assert item_count == 1
    assert batch_count == 1
