import json
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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


def test_synthetic_admin_endpoints_require_admin(client, admin_headers):
    assert client.get("/api/dataset/synthetic/preview").status_code == 401
    assert client.post("/api/dataset/synthetic/gemini-review", json={"ids": [1]}).status_code == 401
    assert (
        client.get(
            "/api/dataset/synthetic/preview",
            headers=admin_headers,
            params={"reviewed": "true"},
        ).status_code
        == 400
    )
    assert client.get("/api/dataset/synthetic/training-preview-summary").status_code == 401
    assert (
        client.post(
            "/api/dataset/synthetic/transfer-to-training-preview",
            json={"ids": [1]},
        ).status_code
        == 401
    )


def test_synthetic_gemini_review_persists_provenance_and_transfers(client, qa_env, admin_headers, monkeypatch):
    created_at = "2026-08-08T09:00:00+00:00"
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            """
            INSERT INTO synthetic_generation_batch (
                batch_id, domain, style, target_label, requested_count, generated_count,
                generator_model, prompt_version, created_at
            ) VALUES ('synthetic_gemini_batch', 'news', 'informal', 1, 1, 1, 'gemini-test', 'v1', ?)
            """,
            (created_at,),
        )
        cursor = conn.execute(
            """
            INSERT INTO synthetic_dataset_row (
                batch_id, text, label, constructiveness, domain, style,
                is_accepted, created_at, reviewed_at
            ) VALUES ('synthetic_gemini_batch', 'synthetic pending review', 1, NULL, 'news', 'informal', 0, ?, NULL)
            """,
            (created_at,),
        )
        row_id = int(cursor.lastrowid)
        conn.commit()

    def fake_call_gemini(prompt: str) -> str:
        assert "synthetic pending review" in prompt
        return json.dumps(
            [
                {
                    "id": row_id,
                    "toxicity_label": 0,
                    "constructiveness_label": 1,
                    "confidence": "high",
                    "reason": "Không có nội dung công kích.",
                    "action": "apply",
                }
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(qa_env["app_module"], "call_gemini", fake_call_gemini)
    suggestion_response = client.post(
        "/api/dataset/synthetic/gemini-review",
        headers=admin_headers,
        json={"ids": [row_id]},
    )
    assert suggestion_response.status_code == 200
    suggestion = suggestion_response.json()["suggestions"][0]
    assert suggestion["toxicity_label"] == 0
    assert suggestion["constructiveness_label"] == 1
    assert suggestion["provider"] == "gemini"
    assert suggestion["model"] == "gemini-2.5-flash"

    apply_response = client.post(
        "/api/dataset/synthetic/review",
        headers=admin_headers,
        json={
            "updates": [
                {
                    "id": row_id,
                    "is_accepted": True,
                    "label": suggestion["toxicity_label"],
                    "constructiveness": suggestion["constructiveness_label"],
                    "review_method": "gemini_assisted",
                    "label_confidence": suggestion["confidence"],
                    "review_provider": suggestion["provider"],
                    "review_model_name": suggestion["model"],
                }
            ]
        },
    )
    assert apply_response.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        reviewed = conn.execute(
            """
            SELECT label, constructiveness, is_accepted, review_method, label_confidence, reviewed_by,
                   review_provider, review_model_name
            FROM synthetic_dataset_row WHERE id = ?
            """,
            (row_id,),
        ).fetchone()
    assert reviewed == (0, 1, 1, "gemini_assisted", "high", "admin", "gemini", "gemini-2.5-flash")

    transfer = client.post(
        "/api/dataset/synthetic/transfer-to-training-preview",
        headers=admin_headers,
        json={"ids": [row_id]},
    )
    assert transfer.json()["transferred"] == 1
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        mlflow_row = conn.execute(
            """
            SELECT pseudo_label, constructiveness_label, training_review_status, label_source, label_confidence,
                   review_provider, review_model_name
            FROM mlflow_comment_item WHERE source_type = 'synthetic' AND source_row_id = ?
            """,
            (row_id,),
        ).fetchone()
    assert mlflow_row == (0, 1, "manual_gemini", "gemini_assist", "high", "gemini", "gemini-2.5-flash")


def test_admin_confirms_synthetic_transfer_to_training_preview(client, qa_env, admin_headers):
    created_at = "2026-08-08T10:00:00+00:00"
    reviewed_at = "2026-08-08T10:05:00+00:00"
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            """
            INSERT INTO synthetic_generation_batch (
                batch_id, domain, style, target_label, requested_count, generated_count,
                generator_model, prompt_version, created_at
            ) VALUES ('synthetic_test_batch', 'education', 'formal', 1, 3, 3, 'gemini-test', 'v1', ?)
            """,
            (created_at,),
        )
        conn.executemany(
            """
            INSERT INTO synthetic_dataset_row (
                batch_id, text, label, constructiveness, domain, style,
                is_accepted, created_at, reviewed_at
            ) VALUES ('synthetic_test_batch', ?, ?, ?, 'education', 'formal', ?, ?, ?)
            """,
            [
                ("synthetic toxic sample", 1, 0, 1, created_at, reviewed_at),
                ("synthetic clean sample", 0, 1, 1, created_at, reviewed_at),
                ("synthetic rejected sample", 1, None, 0, created_at, reviewed_at),
            ],
        )
        conn.commit()

    summary = client.get(
        "/api/dataset/synthetic/training-preview-summary",
        headers=admin_headers,
    )
    assert summary.status_code == 200
    summary_payload = summary.json()
    assert summary_payload["eligible"] == 2
    assert summary_payload["toxic"] == 1
    assert summary_payload["clean"] == 1
    assert summary_payload["constructive"] == 1
    assert summary_payload["non_constructive"] == 1

    transfer = client.post(
        "/api/dataset/synthetic/transfer-to-training-preview",
        headers=admin_headers,
        json={"ids": summary_payload["ids"]},
    )
    assert transfer.status_code == 200
    assert transfer.json() == {
        "transferred": 2,
        "toxic": 1,
        "clean": 1,
        "skipped": 0,
        "automation_scheduled_for": [],
    }

    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "all_batches"},
    )
    assert preview.status_code == 200
    synthetic_items = [item for item in preview.json()["items"] if item["source_type"] == "synthetic"]
    assert len(synthetic_items) == 2
    assert {item["pseudo_label"] for item in synthetic_items} == {0, 1}
    assert {item["label_source"] for item in synthetic_items} == {"synthetic_review"}
    assert all(item["source_row_id"] in summary_payload["ids"] for item in synthetic_items)

    summary_after = client.get(
        "/api/dataset/synthetic/training-preview-summary",
        headers=admin_headers,
    ).json()
    assert summary_after["eligible"] == 0
    assert summary_after["already_transferred"] == 2

    repeated = client.post(
        "/api/dataset/synthetic/transfer-to-training-preview",
        headers=admin_headers,
        json={"ids": summary_payload["ids"]},
    )
    assert repeated.json() == {
        "transferred": 0,
        "toxic": 0,
        "clean": 0,
        "skipped": 2,
        "automation_scheduled_for": [],
    }


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


def test_system_settings_override_mlflow_bundle_minimum_at_runtime(client, admin_headers):
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={"settings": {"MLFLOW_THRESHOLD_TARGET_MAX": 3}},
    )
    assert response.status_code == 200

    groups = {group["id"]: group for group in response.json()["groups"]}
    mlflow_setting = next(
        item for item in groups["mlflow_dataset"]["settings"] if item["key"] == "MLFLOW_THRESHOLD_TARGET_MAX"
    )
    assert mlflow_setting["value"] == "3"

    threshold = client.get("/api/mlflow/threshold-status", headers=admin_headers)
    assert threshold.status_code == 200
    assert threshold.json()["target_max_test_stage"] == 3
    assert threshold.json()["remaining_to_target"] == 3


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
    assert {"kaggle_account", "kaggle_kernel", "kaggle_webhook", "mlflow_dataset", "gemini", "video_asr"}.issubset(group_ids)


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
    assert preview_payload["constructiveness"]["constructive"] + preview_payload["constructiveness"]["non_constructive"] == 2

    candidates = client.get(
        "/api/mlflow/candidates",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_phobert", "strict_batch": "true"},
    )
    assert candidates.status_code == 200
    candidate_payload = candidates.json()
    preview_ids = {int(item["id"]) for item in preview_payload["items"]}
    candidate_ids = {int(item["id"]) for item in candidate_payload["items"]}
    assert preview_ids.isdisjoint(candidate_ids)
    assert {item["gate_bucket"] for item in preview_payload["items"]} == {"accepted"}
    assert {item["gate_bucket"] for item in candidate_payload["items"]} == {"candidate"}
    assert {item["verification_status"] for item in candidate_payload["items"]} == {"unverified"}

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
    preview_removed = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_lock_preview", "strict_batch": "true"},
    )
    assert target_id not in {int(item["id"]) for item in preview_removed.json()["items"]}


def test_mlflow_training_plan_and_manual_toxicity_override(client, qa_env, admin_headers):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_training_plan")
    candidates = client.get(
        "/api/mlflow/candidates",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_training_plan", "strict_batch": "true"},
    )
    candidate_id = int(candidates.json()["items"][0]["id"])
    include_clean = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={"updates": [{"id": candidate_id, "action": "include_clean"}]},
    )
    assert include_clean.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            "UPDATE mlflow_comment_item SET text = ? WHERE id = ?",
            ("sample train text", candidate_id),
        )
        accepted_id = int(
            conn.execute(
                "SELECT id FROM mlflow_comment_item WHERE batch_id = ? AND id <> ?",
                ("batch_training_plan", candidate_id),
            ).fetchone()[0]
        )
        conn.commit()

    plan = client.get(
        "/api/mlflow/training-plan",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_training_plan", "strict_batch": "true", "balance_strategy": "balanced_50_50"},
    )
    assert plan.status_code == 200
    plan_payload = plan.json()
    assert plan_payload["summary"]["gold_train"] == 1
    assert plan_payload["summary"]["eligible_mlflow"] == 2
    assert plan_payload["summary"]["after_balance"] == 2
    assert plan_payload["summary"]["duplicates_skipped"] == 1
    assert plan_payload["summary"]["mlflow_added"] == 1
    assert plan_payload["summary"]["final_train"] == 2
    assert plan_payload["row_statuses"][str(accepted_id)]["will_finetune"] is True
    assert plan_payload["row_statuses"][str(candidate_id)]["reason_code"] == "duplicate"

    manual_override = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={
            "updates": [
                {
                    "id": accepted_id,
                    "pseudo_label": 0,
                    "label_source": "manual_override",
                    "label_confidence": "high",
                }
            ]
        },
    )
    assert manual_override.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        row = conn.execute(
            "SELECT pseudo_label, verification_status, training_review_status, label_source FROM mlflow_comment_item WHERE id = ?",
            (accepted_id,),
        ).fetchone()
    assert row == (0, "manual_accepted", "manual_approved", "manual_override")

    deselect = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={"updates": [{"id": accepted_id, "selected_for_training": False}]},
    )
    assert deselect.status_code == 200
    plan_after = client.get(
        "/api/mlflow/training-plan",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_training_plan", "strict_batch": "true", "balance_strategy": "all"},
    )
    assert plan_after.json()["row_statuses"][str(accepted_id)]["reason_code"] == "not_selected"


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
                    "reviewed_by_gemini": True,
                    "review_provider": suggestion["provider"],
                    "review_model_name": suggestion["model"],
                }
            ]
        },
    )
    assert apply_response.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        row = conn.execute(
            "SELECT pseudo_label, constructiveness_label, label_source, label_confidence, review_provider, review_model_name FROM mlflow_comment_item WHERE id = ?",
            (target_id,),
        ).fetchone()
    assert row == (0, None, "gemini_assist", "high", "gemini", "gemini-2.5-flash")


def test_mlflow_bulk_gemini_review_and_apply_preserves_review_origin(client, qa_env, admin_headers, monkeypatch):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_gemini_bulk")
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            """
            UPDATE mlflow_comment_item
            SET gate_bucket = 'accepted', verification_status = 'manual_accepted', training_review_status = 'manual_approved'
            WHERE batch_id = ? AND text = 'candidate sample'
            """,
            ("batch_gemini_bulk",),
        )
        rows = conn.execute(
            "SELECT id, training_review_status FROM mlflow_comment_item WHERE batch_id = ? ORDER BY id",
            ("batch_gemini_bulk",),
        ).fetchall()
        conn.commit()

    ids = [int(row[0]) for row in rows]
    assert [row[1] for row in rows] == ["auto", "manual_approved"]

    def fake_call_gemini(prompt: str) -> str:
        assert "đúng 2 object" in prompt
        return json.dumps(
            [
                {
                    "id": item_id,
                    "toxicity_label": index % 2,
                    "constructiveness_label": None,
                    "confidence": "high",
                    "reason": f"Gợi ý {index + 1}",
                    "action": "apply",
                }
                for index, item_id in enumerate(ids)
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(qa_env["app_module"], "call_gemini", fake_call_gemini)
    review_response = client.post(
        "/api/mlflow/training-preview/gemini-review",
        headers=admin_headers,
        json={"ids": ids},
    )
    assert review_response.status_code == 200
    suggestions = review_response.json()["suggestions"]
    assert review_response.json()["reviewed"] == 2
    assert {int(item["id"]) for item in suggestions} == set(ids)

    apply_response = client.post(
        "/api/mlflow/training-preview/review",
        headers=admin_headers,
        json={
            "updates": [
                {
                    "id": item["id"],
                    "pseudo_label": item["toxicity_label"],
                    "clear_constructiveness": True,
                    "label_source": "gemini_assist",
                    "label_confidence": item["confidence"],
                    "reviewed_by_gemini": True,
                    "review_provider": item["provider"],
                    "review_model_name": item["model"],
                }
                for item in suggestions
            ]
        },
    )
    assert apply_response.status_code == 200
    assert apply_response.json()["updated"] == 2

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        applied_rows = conn.execute(
            "SELECT training_review_status, label_source, review_provider, review_model_name FROM mlflow_comment_item WHERE id IN (?, ?) ORDER BY id",
            tuple(ids),
        ).fetchall()
    assert applied_rows == [
        ("auto_gemini", "gemini_assist", "gemini", "gemini-2.5-flash"),
        ("manual_gemini", "gemini_assist", "gemini", "gemini-2.5-flash"),
    ]


def test_gemini_review_uses_actual_model_and_enforces_rate_window(qa_env, monkeypatch):
    app_module = qa_env["app_module"]
    rows = [
        {
            "id": 501,
            "text": "comment provenance",
            "score": 0.5,
            "pseudo_label": 0,
            "constructiveness_score": None,
            "constructiveness_label": None,
            "gate_bucket": "accepted",
            "domain_category": "news",
            "url": "https://example.com/provenance",
        }
    ]

    def fake_call_gemini(_prompt: str) -> str:
        payload = json.dumps(
            [{"id": 501, "toxicity_label": 1, "constructiveness_label": None, "confidence": "high", "reason": "reviewed", "action": "apply"}]
        )
        return app_module.GeminiTextResponse(payload, model="gemini-fallback-test")

    monkeypatch.setattr(app_module, "call_gemini", fake_call_gemini)
    suggestion = app_module.run_mlflow_gemini_review(rows)[0]
    assert suggestion["provider"] == "gemini"
    assert suggestion["model"] == "gemini-fallback-test"

    with pytest.raises(app_module.HTTPException) as exc_info:
        app_module.validate_gemini_review_item_limit(list(range(10)))
    assert exc_info.value.status_code == 422
    assert "at most 9 comments" in str(exc_info.value.detail)

    clock = [100.0]
    sleeps: list[float] = []
    monkeypatch.setattr(
        app_module,
        "get_int_setting",
        lambda key, default, min_value=0: 13 if key == "GEMINI_MIN_REQUEST_INTERVAL_SECONDS" else default,
    )
    monkeypatch.setattr(app_module.time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)
    app_module.GEMINI_NEXT_REQUEST_AT = 0.0
    app_module.wait_for_gemini_request_slot()
    app_module.wait_for_gemini_request_slot()
    assert sleeps == [13.0]


def test_mlflow_gemini_review_splits_more_than_three_comments_into_complete_batches(qa_env, monkeypatch):
    app_module = qa_env["app_module"]
    rows = [
        {
            "id": row_id,
            "text": f"comment {row_id}",
            "score": 0.5,
            "pseudo_label": 0,
            "constructiveness_score": None,
            "constructiveness_label": None,
            "gate_bucket": "candidate",
            "domain_category": "news",
            "url": f"https://example.com/{row_id}",
        }
        for row_id in range(1, 8)
    ]
    request_sizes = []

    def fake_call_gemini(prompt: str) -> str:
        payload_start = prompt.rfind("\n[")
        assert payload_start >= 0
        payload = json.loads(prompt[payload_start + 1 :])
        request_sizes.append(len(payload))
        return json.dumps(
            [
                {
                    "id": item["id"],
                    "toxicity_label": item["id"] % 2,
                    "constructiveness_label": None,
                    "confidence": "high",
                    "reason": "Đã review",
                    "action": "apply",
                }
                for item in payload
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(app_module, "call_gemini", fake_call_gemini)
    suggestions = app_module.run_mlflow_gemini_review(rows)

    assert request_sizes == [3, 3, 1]
    assert [item["id"] for item in suggestions] == list(range(1, 8))


def test_mlflow_gemini_review_keeps_valid_rows_when_one_row_never_returns_json(qa_env, monkeypatch):
    app_module = qa_env["app_module"]
    rows = [
        {
            "id": row_id,
            "text": f"comment {row_id}",
            "score": 0.5,
            "pseudo_label": 0,
            "constructiveness_score": None,
            "constructiveness_label": None,
            "gate_bucket": "accepted",
            "domain_category": "news",
            "url": f"https://example.com/{row_id}",
        }
        for row_id in range(1, 4)
    ]
    calls_for_bad_row = 0

    def fake_call_gemini(prompt: str) -> str:
        nonlocal calls_for_bad_row
        payload_start = prompt.rfind("\n[")
        payload = json.loads(prompt[payload_start + 1 :])
        if any(item["id"] == 3 for item in payload):
            calls_for_bad_row += 1
            return "not json"
        return json.dumps(
            [
                {
                    "id": item["id"],
                    "toxicity_label": 0,
                    "constructiveness_label": None,
                    "confidence": "high",
                    "reason": "JSON hợp lệ",
                    "action": "apply",
                }
                for item in payload
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(app_module, "call_gemini", fake_call_gemini)
    suggestions = app_module.run_mlflow_gemini_review(rows)

    assert [item["id"] for item in suggestions] == [1, 2]
    assert calls_for_bad_row == app_module.GEMINI_REVIEW_JSON_ATTEMPTS * 2


def test_mlflow_manual_verify_gemini_review_and_apply(client, qa_env, admin_headers, monkeypatch):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_manual_gemini")
    candidates = client.get(
        "/api/mlflow/candidates",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_manual_gemini", "strict_batch": "true"},
    )
    assert candidates.status_code == 200
    candidate_id = int(candidates.json()["items"][0]["id"])
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            "UPDATE mlflow_comment_item SET constructiveness_label = 1 WHERE id = ?",
            (candidate_id,),
        )
        conn.commit()

    def fake_call_gemini(prompt: str) -> str:
        assert "candidate sample" in prompt
        return json.dumps(
            [
                {
                    "id": candidate_id,
                    "toxicity_label": 1,
                    "constructiveness_label": None,
                    "confidence": "high",
                    "reason": "Có công kích trực tiếp.",
                    "action": "apply",
                }
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr(qa_env["app_module"], "call_gemini", fake_call_gemini)
    review_response = client.post(
        "/api/mlflow/candidates/gemini-review",
        headers=admin_headers,
        json={"ids": [candidate_id]},
    )
    assert review_response.status_code == 200
    suggestion = review_response.json()["suggestions"][0]
    assert suggestion["toxicity_label"] == 1
    assert suggestion["constructiveness_label"] is None

    apply_response = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={
            "updates": [
                {
                    "id": candidate_id,
                    "action": "include_toxic",
                    "decision": "accept",
                    "pseudo_label": 1,
                    "clear_constructiveness": True,
                    "label_source": "gemini_assist",
                    "label_confidence": "high",
                    "reviewed_by_gemini": True,
                    "review_provider": suggestion["provider"],
                    "review_model_name": suggestion["model"],
                }
            ]
        },
    )
    assert apply_response.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        row = conn.execute(
            """
            SELECT gate_bucket, verification_status, pseudo_label, constructiveness_label,
                   training_review_status, label_source, label_confidence, review_provider, review_model_name
            FROM mlflow_comment_item WHERE id = ?
            """,
            (candidate_id,),
        ).fetchone()
    assert row == (
        "accepted", "manual_accepted", 1, None, "manual_gemini", "gemini_assist", "high",
        "gemini", "gemini-2.5-flash",
    )

    remaining = client.get("/api/mlflow/candidates", headers=admin_headers, params={"scope": "all_batches"})
    preview = client.get("/api/mlflow/training-preview", headers=admin_headers, params={"scope": "all_batches"})
    assert candidate_id not in {int(item["id"]) for item in remaining.json()["items"]}
    preview_by_id = {int(item["id"]): item for item in preview.json()["items"]}
    assert candidate_id in preview_by_id
    assert preview_by_id[candidate_id]["training_review_status"] == "manual_gemini"
    assert preview_by_id[candidate_id]["label_source"] == "gemini_assist"


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


def _prediction_response_result(
    url: str,
    url_hash: str,
    context_hash: str,
    score: float,
    predicted_label: int,
) -> list[dict]:
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
                            "score": score,
                            "toxic_prob_adjusted": score,
                            "toxic_label": predicted_label,
                            "html_tags": ["comment"],
                            "segment_hash": "segment_hash_shared",
                            "context_segment_hash": context_hash,
                            "seg_threshold_used": 0.4,
                        }
                    ]
                },
            }
    ]


def _persist_prediction(
    app_module,
    *,
    batch_id: str,
    model_id: str,
    context_hash: str,
    score: float,
    predicted_label: int,
    url: str = "https://example.com/a",
    url_hash: str = "hash_a",
) -> dict:
    now = datetime.utcnow().isoformat() + "Z"
    return app_module.insert_mlflow_comment_rows(
        batch_id=batch_id,
        model_id=model_id,
        source_job_id=f"job_{batch_id}",
        rows=app_module.build_mlflow_comment_rows(
            _prediction_response_result(url, url_hash, context_hash, score, predicted_label),
            batch_id,
            f"job_{batch_id}",
            0.8,
            0.2,
            now,
        ),
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )


def test_mlflow_collection_keeps_one_sample_with_multi_model_predictions(qa_env, client, admin_headers):
    app_module = qa_env["app_module"]
    now = datetime.utcnow().isoformat() + "Z"

    first_rows = app_module.build_mlflow_comment_rows(
        _prediction_response_result("https://example.com/a", "hash_a", "ctx_hash_a", 0.55, 0),
        "batch_auto_1",
        "job_auto_1",
        0.8,
        0.2,
        now,
    )
    first = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_1",
        model_id="tfidf_lr/v1",
        source_job_id="job_auto_1",
        rows=first_rows,
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert first["batch_id"] == "batch_auto_1"
    assert first["inserted"] == 1
    assert first["predictions_inserted"] == 1

    phobert_rows = app_module.build_mlflow_comment_rows(
        _prediction_response_result("https://example.com/a", "hash_a", "ctx_hash_a", 0.94, 1),
        "batch_auto_2",
        "job_auto_2",
        0.8,
        0.2,
        now,
    )
    phobert = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_2",
        model_id="phobert/v1",
        source_job_id="job_auto_2",
        rows=phobert_rows,
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert phobert["batch_id"] == "batch_auto_2"
    assert phobert["inserted"] == 0
    assert phobert["samples_reused"] == 1
    assert phobert["predictions_inserted"] == 1
    assert phobert["counts"]["candidate"] == 1

    duplicate_model = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_3",
        model_id="phobert/v1",
        source_job_id="job_auto_3",
        rows=app_module.build_mlflow_comment_rows(
            _prediction_response_result("https://example.com/a", "hash_a", "ctx_hash_a", 0.94, 1),
            "batch_auto_3",
            "job_auto_3",
            0.8,
            0.2,
            now,
        ),
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert duplicate_model["batch_id"] is None
    assert duplicate_model["predictions_inserted"] == 0
    assert duplicate_model["skipped_duplicate_item"] == 1

    changed_content = app_module.insert_mlflow_comment_rows(
        batch_id="batch_auto_4",
        model_id="phobert/v1",
        source_job_id="job_auto_4",
        rows=app_module.build_mlflow_comment_rows(
            _prediction_response_result("https://example.com/a", "hash_a", "ctx_hash_changed", 0.91, 1),
            "batch_auto_4",
            "job_auto_4",
            0.8,
            0.2,
            now,
        ),
        options_json='{"source":"user_analyze"}',
        created_at=now,
    )
    assert changed_content["samples_inserted"] == 1
    assert changed_content["predictions_inserted"] == 1

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        item_count = conn.execute("SELECT COUNT(1) FROM mlflow_comment_item").fetchone()[0]
        prediction_count = conn.execute("SELECT COUNT(1) FROM mlflow_comment_prediction").fetchone()[0]
        pending_count = conn.execute(
            "SELECT COUNT(1) FROM mlflow_comment_item WHERE gate_bucket = 'candidate' AND verification_status = 'unverified'"
        ).fetchone()[0]
        batch_count = conn.execute("SELECT COUNT(1) FROM mlflow_crawl_batch").fetchone()[0]
    assert item_count == 2
    assert prediction_count == 3
    assert pending_count == 1
    assert batch_count == 3

    batch_overview = client.get(
        "/api/mlflow/overview",
        headers=admin_headers,
        params={"batch_id": "batch_auto_2", "strict_batch": "true"},
    )
    assert batch_overview.status_code == 200
    assert batch_overview.json()["pipeline_counts"]["inferred"] == 1
    crawl_history = client.get("/api/mlflow/crawl-history", headers=admin_headers)
    assert crawl_history.status_code == 200
    assert any(item["batch_id"] == "batch_auto_2" and item["segment_count"] == 1 for item in crawl_history.json()["items"])


def test_mlflow_reviewed_sample_keeps_ground_truth_and_prediction_agreement(qa_env, client, admin_headers):
    app_module = qa_env["app_module"]
    first = _persist_prediction(
        app_module,
        batch_id="batch_review_1",
        model_id="tfidf_lr/v1",
        context_hash="ctx_reviewed",
        score=0.55,
        predicted_label=0,
    )
    assert first["samples_inserted"] == 1

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        sample_id = int(conn.execute("SELECT id FROM mlflow_comment_item").fetchone()[0])

    reviewed = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={"updates": [{"id": sample_id, "action": "include_toxic"}]},
    )
    assert reviewed.status_code == 200

    agreeing = _persist_prediction(
        app_module,
        batch_id="batch_review_2",
        model_id="phobert/v1",
        context_hash="ctx_reviewed",
        score=0.94,
        predicted_label=1,
    )
    disagreeing = _persist_prediction(
        app_module,
        batch_id="batch_review_3",
        model_id="phobert/v2",
        context_hash="ctx_reviewed",
        score=0.12,
        predicted_label=0,
    )
    assert agreeing["predictions_inserted"] == 1
    assert disagreeing["predictions_inserted"] == 1

    candidates = client.get("/api/mlflow/candidates", headers=admin_headers, params={"scope": "all_batches"})
    assert candidates.status_code == 200
    assert candidates.json()["total"] == 0

    history = client.get("/api/mlflow/review-history", headers=admin_headers, params={"scope": "all_batches"})
    assert history.status_code == 200
    item = next(row for row in history.json()["items"] if int(row["id"]) == sample_id)
    assert item["human_label"] == 1
    assert len(item["prediction_history"]) == 3
    agreement_by_model = {
        prediction["model_id"]: prediction["agreement_with_human"]
        for prediction in item["prediction_history"]
    }
    assert agreement_by_model["tfidf_lr/v1"] is False
    assert agreement_by_model["phobert/v1"] is True
    assert agreement_by_model["phobert/v2"] is False

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.row_factory = sqlite3.Row
        sample = conn.execute(
            "SELECT verification_status, pseudo_label FROM mlflow_comment_item WHERE id = ?",
            (sample_id,),
        ).fetchone()
        training_rows = app_module.select_mlflow_training_rows(conn, None, "all")[0]
    assert tuple(sample) == ("manual_accepted", 1)
    assert len(training_rows) == 1
    assert int(training_rows[0]["pseudo_label"]) == 1


def test_mlflow_active_candidate_prediction_reopens_only_unreviewed_auto_sample(qa_env):
    app_module = qa_env["app_module"]
    accepted = _persist_prediction(
        app_module,
        batch_id="batch_auto_accept_1",
        model_id="phobert/v1",
        context_hash="ctx_auto_accept",
        score=0.95,
        predicted_label=1,
    )
    assert accepted["counts"]["accepted"] == 1

    candidate = _persist_prediction(
        app_module,
        batch_id="batch_auto_accept_2",
        model_id="tfidf_lr/v1",
        context_hash="ctx_auto_accept",
        score=0.55,
        predicted_label=0,
    )
    assert candidate["predictions_inserted"] == 1
    assert candidate["counts"]["candidate"] == 1
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        state = conn.execute(
            "SELECT gate_bucket, verification_status, selected_for_training FROM mlflow_comment_item"
        ).fetchone()
    assert state == ("candidate", "unverified", 0)


def test_mlflow_model_reevaluation_agreement_conflict_uncertain_and_human_resolution(
    qa_env, client, admin_headers, monkeypatch
):
    app_module = qa_env["app_module"]
    for suffix in ("agree", "conflict", "uncertain"):
        persisted = _persist_prediction(
            app_module,
            batch_id=f"batch_reeval_{suffix}",
            model_id="tfidf_lr/v1",
            context_hash=f"ctx_reeval_{suffix}",
            score=0.95,
            predicted_label=1,
        )
        assert persisted["counts"]["accepted"] == 1

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        ids = {
            row[1].rsplit("_", 1)[-1]: int(row[0])
            for row in conn.execute("SELECT id, context_segment_hash FROM mlflow_comment_item")
        }

    monkeypatch.setattr(
        app_module,
        "resolve_model_path",
        lambda _root, _model_id: ("phobert", "v2", qa_env["base_dir"] / "fake-model"),
    )

    def fake_re_evaluate(rows, model_id):
        by_id = {
            ids["agree"]: (0.94, 1),
            ids["conflict"]: (0.08, 0),
            ids["uncertain"]: (0.55, 0),
        }
        return model_id, {
            int(row["id"]): {
                "raw_toxicity_score": by_id[int(row["id"])][0],
                "adjusted_toxicity_score": by_id[int(row["id"])][0],
                "predicted_label": by_id[int(row["id"])][1],
                "constructiveness_score": None,
                "constructiveness_label": None,
                "constructiveness_confidence": "missing",
                "seg_threshold_used": 0.4,
            }
            for row in rows
        }

    monkeypatch.setattr(app_module, "run_mlflow_model_re_evaluation", fake_re_evaluate)
    response = client.post(
        "/api/mlflow/re-evaluate",
        headers=admin_headers,
        json={
            "model_id": "phobert/v2",
            "selection": "selected",
            "sample_ids": list(ids.values()),
        },
    )
    assert response.status_code == 200
    assert response.json()["summary"] == {
        "requested": 3,
        "evaluated": 3,
        "agreement": 1,
        "conflict": 1,
        "uncertain": 1,
        "needs_review": 2,
        "skipped": 0,
        "failed": 0,
    }

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.row_factory = sqlite3.Row
        states = {
            int(row[0]): tuple(row[1:])
            for row in conn.execute(
                "SELECT id, pseudo_label, gate_bucket, verification_status, selected_for_training, training_review_status, review_reason FROM mlflow_comment_item"
            )
        }
        origins = conn.execute(
            "SELECT sample_item_id, record_origin FROM mlflow_comment_prediction WHERE model_id = 'phobert/v2'"
        ).fetchall()
        training_ids = {int(row["id"]) for row in app_module.select_mlflow_training_rows(conn, None, "all")[0]}
    assert states[ids["agree"]] == (1, "accepted", "auto_accepted", 1, "auto", None)
    assert states[ids["conflict"]] == (1, "candidate", "unverified", 0, "pending", "model_conflict")
    assert states[ids["uncertain"]] == (1, "candidate", "unverified", 0, "pending", "model_uncertain")
    assert {tuple(row) for row in origins} == {
        (ids["agree"], "model_re_evaluation"),
        (ids["conflict"], "model_re_evaluation"),
        (ids["uncertain"], "model_re_evaluation"),
    }
    assert training_ids == {ids["agree"]}

    preview = client.get("/api/mlflow/training-preview", headers=admin_headers, params={"scope": "all_batches"})
    assert preview.status_code == 200
    assert preview.json()["counts"]["selected"] == 1
    assert preview.json()["counts"]["requires_human_review"] == 2
    assert preview.json()["counts"]["model_conflicts"] == 1
    assert {int(item["id"]) for item in preview.json()["items"]} == set(ids.values())

    reviewed = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={"updates": [{"id": ids["conflict"], "action": "include_toxic"}]},
    )
    assert reviewed.status_code == 200
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        state = conn.execute(
            "SELECT pseudo_label, gate_bucket, verification_status, selected_for_training, review_reason FROM mlflow_comment_item WHERE id = ?",
            (ids["conflict"],),
        ).fetchone()
        history_count = conn.execute(
            "SELECT COUNT(1) FROM mlflow_comment_prediction WHERE sample_item_id = ?",
            (ids["conflict"],),
        ).fetchone()[0]
    assert state == (1, "accepted", "manual_accepted", 1, "model_conflict_resolved")
    assert history_count == 2

    duplicate = client.post(
        "/api/mlflow/re-evaluate",
        headers=admin_headers,
        json={"model_id": "phobert/v2", "selection": "selected", "sample_ids": [ids["conflict"]]},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["summary"]["skipped"] == 1
    assert duplicate.json()["results"][0]["message"] == "Already evaluated with this model version"


def test_mlflow_model_reevaluation_preserves_human_authority_and_bulk_excludes_human(
    qa_env, client, admin_headers, monkeypatch
):
    app_module = qa_env["app_module"]
    manual_seed = _persist_prediction(
        app_module,
        batch_id="batch_reeval_manual",
        model_id="tfidf_lr/v1",
        context_hash="ctx_reeval_manual",
        score=0.55,
        predicted_label=0,
    )
    assert manual_seed["counts"]["candidate"] == 1
    auto_seed = _persist_prediction(
        app_module,
        batch_id="batch_reeval_auto",
        model_id="tfidf_lr/v1",
        context_hash="ctx_reeval_auto",
        score=0.95,
        predicted_label=1,
    )
    assert auto_seed["counts"]["accepted"] == 1
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        manual_id = int(conn.execute("SELECT id FROM mlflow_comment_item WHERE context_segment_hash = 'ctx_reeval_manual'").fetchone()[0])
        auto_id = int(conn.execute("SELECT id FROM mlflow_comment_item WHERE context_segment_hash = 'ctx_reeval_auto'").fetchone()[0])

    reviewed = client.post(
        "/api/mlflow/candidates/review",
        headers=admin_headers,
        json={"updates": [{"id": manual_id, "action": "include_toxic"}]},
    )
    assert reviewed.status_code == 200
    monkeypatch.setattr(
        app_module,
        "resolve_model_path",
        lambda _root, _model_id: ("phobert", "v2", qa_env["base_dir"] / "fake-model"),
    )

    seen_ids = []

    def fake_re_evaluate(rows, model_id):
        seen_ids.extend(int(row["id"]) for row in rows)
        return model_id, {
            int(row["id"]): {
                "raw_toxicity_score": 0.08,
                "adjusted_toxicity_score": 0.08,
                "predicted_label": 0,
                "constructiveness_score": None,
                "constructiveness_label": None,
                "constructiveness_confidence": "missing",
                "seg_threshold_used": 0.4,
            }
            for row in rows
        }

    monkeypatch.setattr(app_module, "run_mlflow_model_re_evaluation", fake_re_evaluate)
    explicit = client.post(
        "/api/mlflow/re-evaluate",
        headers=admin_headers,
        json={"model_id": "phobert/v2", "selection": "selected", "sample_ids": [manual_id]},
    )
    assert explicit.status_code == 200
    assert explicit.json()["results"][0]["status"] == "human_disagreement"
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        state = conn.execute(
            "SELECT pseudo_label, verification_status, selected_for_training FROM mlflow_comment_item WHERE id = ?",
            (manual_id,),
        ).fetchone()
    assert state == (1, "manual_accepted", 1)

    seen_ids.clear()
    bulk = client.post(
        "/api/mlflow/re-evaluate",
        headers=admin_headers,
        json={"model_id": "phobert/v2", "selection": "all_auto_eligible"},
    )
    assert bulk.status_code == 200
    assert seen_ids == [auto_id]
    assert bulk.json()["summary"]["conflict"] == 1
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        manual_state = conn.execute(
            "SELECT pseudo_label, verification_status, selected_for_training FROM mlflow_comment_item WHERE id = ?",
            (manual_id,),
        ).fetchone()
    assert manual_state == (1, "manual_accepted", 1)


def test_mlflow_batch_scoped_preview_and_export_include_reused_canonical_sample(qa_env, client, admin_headers):
    app_module = qa_env["app_module"]
    first = _persist_prediction(
        app_module,
        batch_id="batch_scope_original",
        model_id="tfidf_lr/v1",
        context_hash="ctx_scope_reused",
        score=0.95,
        predicted_label=1,
    )
    second = _persist_prediction(
        app_module,
        batch_id="batch_scope_latest",
        model_id="phobert/v2",
        context_hash="ctx_scope_reused",
        score=0.94,
        predicted_label=1,
    )
    assert first["samples_inserted"] == 1
    assert second["samples_reused"] == 1

    preview = client.get(
        "/api/mlflow/training-preview",
        headers=admin_headers,
        params={"scope": "batch", "batch_id": "batch_scope_latest", "strict_batch": "true"},
    )
    assert preview.status_code == 200
    assert preview.json()["counts"]["selected"] == 1
    assert len(preview.json()["items"][0]["prediction_history"]) == 2

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.row_factory = sqlite3.Row
        scoped_rows = app_module.select_mlflow_training_rows(conn, "batch_scope_latest", "all")[0]
    assert len(scoped_rows) == 1


def test_feedback_schema_is_initialized_during_fastapi_startup(monkeypatch, tmp_path):
    import backend.app as app_module

    feedback_dir = tmp_path / "startup_repo" / "data" / "processed" / "feedback"
    feedback_db = feedback_dir / "feedback.db"
    monkeypatch.setattr(app_module, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(app_module, "FEEDBACK_DB_PATH", feedback_db)

    assert not feedback_db.exists()

    with TestClient(app_module.app):
        assert feedback_db.exists()
        with sqlite3.connect(feedback_db) as conn:
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
        assert "mlflow_comment_prediction" in tables
        assert "mlflow_comment_item" in tables
        assert "feedback_page" in tables

        _seed_mlflow_batch(feedback_db, batch_id="startup_legacy_batch")
        app_module.init_feedback_db()
        with sqlite3.connect(feedback_db) as conn:
            first_prediction_count = conn.execute(
                "SELECT COUNT(1) FROM mlflow_comment_prediction WHERE batch_id = ?",
                ("startup_legacy_batch",),
            ).fetchone()[0]
            sample_count = conn.execute(
                "SELECT COUNT(1) FROM mlflow_comment_item WHERE batch_id = ?",
                ("startup_legacy_batch",),
            ).fetchone()[0]

        app_module.init_feedback_db()
        with sqlite3.connect(feedback_db) as conn:
            repeated_prediction_count = conn.execute(
                "SELECT COUNT(1) FROM mlflow_comment_prediction WHERE batch_id = ?",
                ("startup_legacy_batch",),
            ).fetchone()[0]

        assert first_prediction_count == sample_count
        assert repeated_prediction_count == first_prediction_count


def test_mlflow_legacy_backfill_preserves_review_state_without_fabricating_label(qa_env):
    app_module = qa_env["app_module"]
    now = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            """
            INSERT INTO mlflow_crawl_batch (batch_id, model_id, status, source_job_id, created_at, completed_at, options_json)
            VALUES ('legacy_batch', 'phobert/legacy', 'completed', 'legacy_job', ?, ?, '{}')
            """,
            (now, now),
        )
        conn.execute(
            """
            INSERT INTO mlflow_comment_item (
                batch_id, job_id, url, url_hash, text, score, pseudo_label,
                gate_bucket, verification_status, segment_hash, context_segment_hash,
                dedupe_key, html_tag, seg_threshold_used, label_source, label_confidence,
                selected_for_training, training_review_status, created_at, reviewed_at
            ) VALUES (
                'legacy_batch', 'legacy_job', 'https://example.com/legacy', 'legacy_hash', 'legacy reviewed',
                0.55, 1, 'accepted', 'manual_accepted', 'legacy_segment', 'legacy_context',
                'comment_only_v3:legacy_context:comment', 'comment', 0.4, 'manual_override', 'high',
                1, 'manual_approved', ?, ?
            )
            """,
            (now, now),
        )
        conn.execute(
            """
            INSERT INTO mlflow_comment_item (
                batch_id, job_id, url, url_hash, text, score, pseudo_label,
                gate_bucket, verification_status, segment_hash, context_segment_hash,
                dedupe_key, html_tag, seg_threshold_used, label_source, label_confidence,
                selected_for_training, training_review_status, created_at
            ) VALUES (
                'legacy_batch', 'legacy_job', 'https://example.com/legacy-auto', 'legacy_auto_hash', 'legacy auto',
                0.55, 1, 'accepted', 'auto_accepted', 'legacy_auto_segment', 'legacy_auto_context',
                'comment_only_v3:legacy_auto_context:comment', 'comment', 0.4, 'auto_gate', 'high',
                1, 'auto', ?
            )
            """,
            (now,),
        )
        conn.commit()

    app_module.init_feedback_db()
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        prediction = conn.execute(
            """
            SELECT model_id, raw_toxicity_score, predicted_label, record_origin
            FROM mlflow_comment_prediction
            WHERE sample_item_id = (
                SELECT id FROM mlflow_comment_item WHERE text = 'legacy reviewed'
            )
            """
        ).fetchone()
        state = conn.execute(
            "SELECT verification_status, pseudo_label FROM mlflow_comment_item WHERE text = 'legacy reviewed'"
        ).fetchone()
        auto_state = conn.execute(
            "SELECT gate_bucket, verification_status FROM mlflow_comment_item WHERE text = 'legacy auto'"
        ).fetchone()
    assert prediction == ("phobert/legacy", 0.55, None, "legacy_backfill")
    assert state == ("manual_accepted", 1)
    assert auto_state == ("accepted", "auto_accepted")


def test_mlflow_clear_batch_reassigns_shared_sample_without_orphans(qa_env, client, admin_headers):
    app_module = qa_env["app_module"]
    _persist_prediction(
        app_module,
        batch_id="batch_clear_1",
        model_id="tfidf_lr/v1",
        context_hash="ctx_clear",
        score=0.55,
        predicted_label=0,
    )
    _persist_prediction(
        app_module,
        batch_id="batch_clear_2",
        model_id="phobert/v1",
        context_hash="ctx_clear",
        score=0.94,
        predicted_label=1,
    )

    cleared = client.post(
        "/api/mlflow/clear-batch",
        headers=admin_headers,
        json={"batch_id": "batch_clear_1"},
    )
    assert cleared.status_code == 200
    assert cleared.json()["deleted_rows"]["mlflow_comment_prediction"] == 1
    assert cleared.json()["deleted_rows"]["mlflow_comment_item_reassigned"] == 1
    assert cleared.json()["deleted_rows"]["mlflow_crawl_batch"] == 1

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT batch_id FROM mlflow_comment_item").fetchone()[0] == "batch_clear_2"
        assert conn.execute("SELECT COUNT(1) FROM mlflow_comment_prediction").fetchone()[0] == 1
