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


def test_mlflow_overview_without_batch(client):
    response = client.get("/api/mlflow/overview")
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


def test_mlflow_manual_export_bundle_isolated_temp_db(client, qa_env):
    _seed_mlflow_batch(qa_env["feedback_db"], batch_id="batch_export")
    response = client.post(
        "/api/mlflow/manual/export-bundle",
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
