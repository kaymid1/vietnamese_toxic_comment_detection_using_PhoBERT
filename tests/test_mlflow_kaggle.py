import ast
import hashlib
import io
import json
import sqlite3
import typing
import urllib.parse
import zipfile
from datetime import datetime
from pathlib import Path

import pytest


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


def _build_metrics_zip(path: Path, *, f1_toxic: float = 0.71, macro_f1: float = 0.76) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": "kaggle_local_run",
        "test": {
            "f1_toxic": f1_toxic,
            "macro_f1": macro_f1,
            "accuracy": 0.88,
            "precision_toxic": 0.69,
            "recall_toxic": 0.73,
        },
        "sizes": {"train": 4000, "validation": 1200, "test": 984},
        "dataset_evidence": {
            "raw_train": 6956,
            "used_train": 4000,
            "used_gold": 3990,
            "expected_mlflow_count": 10,
            "included_mlflow_count": 10,
            "included_all_expected_mlflow": True,
            "included_mlflow_ids": list(range(1, 11)),
            "bundle_sha256": "a" * 64,
            "duration_seconds": 12.5,
        },
        "confusion_matrix": {
            "test": {"tn": 800, "fp": 70, "fn": 40, "tp": 74},
        },
    }
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results/metrics.json", json.dumps(payload))
        zf.writestr("model.bin", b"fake-model-bytes")


def _build_family_serving_zip(
    path: Path,
    *,
    model_family: str,
    f1_toxic: float = 0.71,
    macro_f1: float = 0.76,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": path.stem,
        "test": {
            "f1_toxic": f1_toxic,
            "macro_f1": macro_f1,
            "accuracy": 0.88,
            "precision_toxic": 0.69,
            "recall_toxic": 0.73,
        },
        "sizes": {"train": 4000, "validation": 1, "test": 1},
    }
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metrics.json", json.dumps(payload))
        if model_family == "tfidf_lr":
            import joblib
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            vectorizer = TfidfVectorizer().fit(["bình luận sạch", "bình luận độc hại"])
            model = LogisticRegression(random_state=42).fit(
                vectorizer.transform(["bình luận sạch", "bình luận độc hại"]),
                [0, 1],
            )
            model_buffer = io.BytesIO()
            vectorizer_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            joblib.dump(vectorizer, vectorizer_buffer)
            zf.writestr("model_lr.joblib", model_buffer.getvalue())
            zf.writestr("vectorizer.joblib", vectorizer_buffer.getvalue())
        else:
            import torch
            from safetensors.torch import save

            zf.writestr("export/config.json", json.dumps({"model_type": "roberta"}))
            zf.writestr("export/model.safetensors", save({"test.weight": torch.zeros(1)}))


def _prepare_family_production_flow(qa_env: dict, *, model_family: str) -> tuple[str, str]:
    app_module = qa_env["app_module"]
    run_id = f"run_promote_{model_family}"
    model_kind = "lr_smoke" if model_family == "tfidf_lr" else "phobert"
    baseline_name = "baseline_tfidf" if model_family == "tfidf_lr" else "baseline"
    baseline_id = f"{model_family}/{baseline_name}"
    baseline_dir = app_module.MODEL_OPTIONS_DIR / model_family / baseline_name
    baseline_dir.mkdir(parents=True, exist_ok=True)
    if model_family == "tfidf_lr":
        (baseline_dir / "model_lr.pkl").write_bytes(b"baseline-model")
        (baseline_dir / "vectorizer.pkl").write_bytes(b"baseline-vectorizer")
    else:
        (baseline_dir / "config.json").write_text(json.dumps({"model_type": "roberta"}), encoding="utf-8")
        (baseline_dir / "model.safetensors").write_bytes(b"baseline-weights")
    (baseline_dir / "metrics.json").write_text(
        json.dumps(
            {
                "f1_toxic": 0.60,
                "macro_f1": 0.70,
                "accuracy": 0.82,
                "precision": 0.58,
                "recall": 0.62,
            }
        ),
        encoding="utf-8",
    )

    artifact_path = qa_env["kaggle_root"] / run_id / "output" / f"{run_id}.zip"
    _build_family_serving_zip(artifact_path, model_family=model_family)
    artifact_checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    bundle_path = qa_env["processed_dir"] / f"bundle_{run_id}.zip"
    test_bytes = (app_module.DATASET_VERSION_DIRS["victsd_gold"] / "test.jsonl").read_bytes()
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test.jsonl", test_bytes)
    bundle_checksum = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    _insert_kaggle_run(
        qa_env["feedback_db"],
        run_id,
        status="completed",
        artifact_uri=str(artifact_path),
    )
    logs = [
        {
            "ts": "2026-01-01T00:00:00Z",
            "message": f"[META] model_kind={model_kind} training_mode=finetune base_model=default",
            "stage": "prepare_bundle",
            "source": "backend",
        }
    ]
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            """
            UPDATE mlflow_do_run
            SET logs_json = ?, artifact_checksum = ?, bundle_path = ?, bundle_checksum = ?
            WHERE run_id = ?
            """,
            (json.dumps(logs), artifact_checksum, str(bundle_path), bundle_checksum, run_id),
        )
        conn.commit()
    return run_id, baseline_id


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
    assert payload.get("bundle_path")
    assert len(payload.get("bundle_checksum") or "") == 64
    assert payload["training_plan"]["run_id"] == payload["run_id"]
    assert len(payload["training_plan"]["included_mlflow_ids_sha256"]) == 64
    assert len(payload["training_plan"]["feedback_snapshot_sha256"]) == 64

    status = client.get(
        "/api/mlflow/kaggle/status",
        headers=admin_headers,
        params={"run_id": payload["run_id"]},
    )
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["bundle_path"] == payload["bundle_path"]
    assert status_payload["bundle_checksum"] == payload["bundle_checksum"]
    bundle_url = urllib.parse.urlparse(status_payload["bundle_url"])
    bundle_download = client.get(f"{bundle_url.path}?{bundle_url.query}")
    assert bundle_download.status_code == 200
    assert bundle_download.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(bundle_download.content)) as zf:
        build_report = json.loads(zf.read("build_report.json"))
    assert build_report["lineage_run_id"] == payload["run_id"]
    assert build_report["included_mlflow_ids"] == payload["training_plan"]["included_mlflow_ids"]

    query = urllib.parse.parse_qs(bundle_url.query)
    invalid_download = client.get(
        bundle_url.path,
        params={"run_id": query["run_id"][0], "token": "0" * 32},
    )
    assert invalid_download.status_code == 403


def test_kaggle_preflight_real_mode_requires_credentials(client, admin_headers, qa_env, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    app_module = qa_env["app_module"]
    monkeypatch.setattr(app_module, "_kaggle_webhook_reachability", lambda webhook_url: (True, None))
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
    assert payload["checks"]["KAGGLE_WEBHOOK_REACHABLE"] is True
    assert payload["checks"]["KAGGLE_USERNAME"] is False
    assert payload["checks"]["KAGGLE_KEY"] is False
    assert set(payload["missing"]) == {"KAGGLE_USERNAME", "KAGGLE_KEY"}
    assert payload["config"]["webhook_mode"] == "real"


def test_kaggle_preflight_reports_unreachable_webhook(client, admin_headers, qa_env, monkeypatch):
    app_module = qa_env["app_module"]
    monkeypatch.setattr(
        app_module,
        "_kaggle_webhook_reachability",
        lambda webhook_url: (False, "Webhook receiver unreachable (http://127.0.0.1:9000/health): refused"),
    )
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "mock",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9000/kaggle/trigger",
            }
        },
    )
    assert response.status_code == 200

    preflight = client.get("/api/mlflow/kaggle/preflight", headers=admin_headers)
    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["ready"] is False
    assert payload["missing"] == []
    assert payload["checks"]["KAGGLE_WEBHOOK_URL"] is True
    assert payload["checks"]["KAGGLE_WEBHOOK_REACHABLE"] is False
    assert any("Webhook receiver unreachable" in warning for warning in payload["warnings"])


def test_kaggle_preflight_reports_offline_public_bundle_tunnel(client, admin_headers, qa_env, monkeypatch):
    app_module = qa_env["app_module"]
    monkeypatch.setattr(app_module, "_kaggle_webhook_reachability", lambda webhook_url: (True, None))
    monkeypatch.setattr(
        app_module,
        "_kaggle_public_bundle_reachability",
        lambda public_base_url: (False, "Public bundle tunnel health returned HTTP 404"),
    )
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "http://127.0.0.1:9001/kaggle/trigger",
                "KAGGLE_BUNDLE_PUBLIC_BASE_URL": "https://offline.example.test",
                "KAGGLE_USERNAME": "kaymi",
                "KAGGLE_KEY": "secret-key",
            }
        },
    )
    assert response.status_code == 200

    preflight = client.get("/api/mlflow/kaggle/preflight", headers=admin_headers)
    payload = preflight.json()

    assert preflight.status_code == 200
    assert payload["ready"] is False
    assert payload["checks"]["KAGGLE_BUNDLE_PUBLIC_REACHABLE"] is False
    assert any("Public bundle tunnel" in warning for warning in payload["warnings"])


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
    captured_payload = {}

    def fake_kaggle_call(method, url, payload=None):
        captured_payload.update(payload or {})
        return {"job_id": "mock_abc123"}

    monkeypatch.setattr(app_module, "_kaggle_http_json", fake_kaggle_call)

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
    assert captured_payload["bundle_url"].startswith("http://testserver/api/mlflow/kaggle/bundle?")
    assert len(captured_payload["bundle_checksum"]) == 64


def test_kaggle_trigger_uses_public_webhook_origin_for_bundle(client, admin_headers, qa_env, monkeypatch):
    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={
            "settings": {
                "KAGGLE_WEBHOOK_MODE": "real",
                "KAGGLE_NOTEBOOK_URL": "https://www.kaggle.com/code/kaymid/thesis-phobert/edit",
                "KAGGLE_WEBHOOK_URL": "https://public.example.test/kaggle/trigger",
                "KAGGLE_USERNAME": "kaymi",
                "KAGGLE_KEY": "secret-key",
            }
        },
    )
    assert response.status_code == 200
    app_module = qa_env["app_module"]
    captured_payload = {}

    def fake_kaggle_call(method, url, payload=None):
        captured_payload.update(payload or {})
        return {"job_id": "real_public123", "status": "running"}

    monkeypatch.setattr(app_module, "_kaggle_http_json", fake_kaggle_call)
    trigger = client.post(
        "/api/mlflow/kaggle/trigger",
        headers=admin_headers,
        json={
            "dry_run": False,
            "model_kind": "lr_smoke",
            "training_mode": "finetune",
            "training_scope": "light_only",
            "provider": "kaggle",
            "compute_mode": "kaggle",
        },
    )
    assert trigger.status_code == 200
    assert captured_payload["bundle_url"].startswith(
        "https://public.example.test/api/mlflow/kaggle/bundle?"
    )
    assert "127.0.0.1" not in captured_payload["bundle_url"]


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
                "KAGGLE_REAL_BUNDLE_URL_TEMPLATE": "https://example.test/bundles/{batch_id}/{run_id}.zip",
            }
        },
    )
    assert response.status_code == 200

    assert receiver._webhook_mode() == "real"
    assert receiver._resolve_owner_slug(None) == ("db-owner", "db-slug")
    subprocess_env = receiver._build_subprocess_env()
    assert subprocess_env["KAGGLE_USERNAME"] == "db-user"
    assert subprocess_env["KAGGLE_KEY"] == "db-key"
    assert receiver._resolve_bundle_url("batch-1", "run-1") == "https://example.test/bundles/batch-1/run-1.zip"

    response = client.patch(
        "/api/admin/system-settings",
        headers=admin_headers,
        json={"settings": {"KAGGLE_WEBHOOK_MODE": "mock"}},
    )
    assert response.status_code == 200
    assert receiver._webhook_mode() == "mock"


def test_kaggle_webhook_receiver_strips_notebook_source_bom(tmp_path, monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    source = tmp_path / "source_with_bom.py"
    source.write_text("\ufeff# %% [markdown]\n# title\n\ufeffprint('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        receiver,
        "_setting",
        lambda key, default="": str(source) if key == "KAGGLE_REAL_NOTEBOOK_SOURCE" else default,
    )

    script = receiver._build_real_script_content(receiver.TriggerRequest(run_id="run_bom"))

    assert "\ufeff" not in script
    compile(script, "<kaggle-script>", "exec")


def test_kaggle_webhook_receiver_reports_inaccessible_kernel_owner(tmp_path, monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    monkeypatch.setattr(
        receiver,
        "_run_cmd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("403 Client Error: Permission 'kernels.get' was denied")
        ),
    )
    monkeypatch.setattr(
        receiver,
        "_setting",
        lambda key, default="": "actual-user" if key == "KAGGLE_USERNAME" else default,
    )

    try:
        receiver._fetch_existing_kernel_metadata("typo-user/train-kernel", tmp_path)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected inaccessible Kaggle kernel to raise RuntimeError")

    assert "Cannot access Kaggle kernel 'typo-user/train-kernel'" in message
    assert "KAGGLE_KERNEL_OWNER" in message
    assert "KAGGLE_USERNAME is 'actual-user'" in message


def test_kaggle_webhook_receiver_attaches_phobert_train_script(tmp_path):
    import backend.kaggle_webhook_receiver as receiver

    attached = receiver._attach_phobert_train_script(tmp_path)

    assert attached == tmp_path / "scripts" / "train_phobert.py"
    assert attached.exists()
    assert "PhoBERT Fine-tune" in attached.read_text(encoding="utf-8")


def test_kaggle_webhook_receiver_embeds_phobert_train_script(monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    monkeypatch.setattr(receiver, "_setting", lambda key, default="": default)

    script = receiver._build_real_script_content(
        receiver.TriggerRequest(run_id="run_embed", model_kind="phobert", training_mode="finetune")
    )

    assert "/kaggle/working/viettoxic/scripts/train_phobert.py" in script
    assert "PhoBERT Fine-tune" in script
    assert "cudaErrorNoKernelImageForDevice" in script
    assert "build_training_arguments" in script
    assert "_train_script_path.write_text" in script
    compile(script, "<kaggle-script>", "exec")


def test_kaggle_webhook_receiver_prefers_run_bundle_url(monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    monkeypatch.setattr(receiver, "_setting", lambda key, default="": default)
    bundle_url = "https://api.example.test/api/mlflow/kaggle/bundle?run_id=run_exact&token=secret"
    script = receiver._build_real_script_content(
        receiver.TriggerRequest(
            run_id="run_exact",
            model_kind="lr_smoke",
            training_mode="finetune",
            bundle_url=bundle_url,
            bundle_checksum="a" * 64,
        )
    )

    assert bundle_url in script
    assert '"VIETTOXIC_BUNDLE_CHECKSUM": "' + ("a" * 64) + '"' in script
    assert '"VIETTOXIC_BUNDLE_DOWNLOAD_REQUIRED": "true"' in script
    compile(script, "<kaggle-script>", "exec")


def test_kaggle_webhook_receiver_proxies_bundle_to_local_backend(monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    class FakeHeaders:
        def get_content_type(self):
            return "application/zip"

        def get(self, key, default=None):
            if key.lower() == "content-disposition":
                return 'attachment; filename="bundle.zip"'
            return default

    class FakeUpstream:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"zip-bytes"

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeUpstream()

    monkeypatch.setattr(receiver, "urlopen", fake_urlopen)
    response = receiver.kaggle_bundle_proxy("run_public", "a" * 32)
    assert response.body == b"zip-bytes"
    assert response.media_type == "application/zip"
    assert captured["url"].startswith("http://127.0.0.1:8000/api/mlflow/kaggle/bundle?")
    assert "run_id=run_public" in captured["url"]
    assert "token=" + ("a" * 32) in captured["url"]


def test_kaggle_webhook_receiver_prefers_model_zip_artifact(tmp_path):
    import backend.kaggle_webhook_receiver as receiver

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    results_zip = output_dir / "results_full_victsd_gold.zip"
    model_zip = output_dir / "best_model_full_victsd_gold.zip"
    with zipfile.ZipFile(results_zip, "w") as archive:
        archive.writestr("metrics.json", "{}")
    with zipfile.ZipFile(model_zip, "w") as archive:
        archive.writestr("metrics.json", "{}")
        archive.writestr("model/config.json", "{}")
        archive.writestr("model/model.safetensors", b"model")

    assert receiver._pick_artifact_file(output_dir) == model_zip


def test_kaggle_webhook_receiver_selects_lr_result_not_input_bundle(tmp_path):
    import backend.kaggle_webhook_receiver as receiver

    output_dir = tmp_path / "output" / "viettoxic"
    output_dir.mkdir(parents=True)
    input_bundle = output_dir / "mlflow_bundle.zip"
    result_zip = output_dir / "kaggle_run_123.zip"
    with zipfile.ZipFile(input_bundle, "w") as archive:
        archive.writestr("dataset.jsonl", '{"text":"new row"}\n')
    with zipfile.ZipFile(result_zip, "w") as archive:
        archive.writestr("metrics.json", "{}")
        archive.writestr("model_lr.joblib", b"model")
        archive.writestr("vectorizer.joblib", b"vectorizer")

    assert receiver._pick_artifact_file(tmp_path / "output") == result_zip


def test_kaggle_webhook_receiver_rejects_input_bundle_as_artifact(tmp_path):
    import backend.kaggle_webhook_receiver as receiver

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with zipfile.ZipFile(output_dir / "mlflow_bundle.zip", "w") as archive:
        archive.writestr("dataset.jsonl", '{"text":"new row"}\n')

    assert receiver._pick_artifact_file(output_dir) is None


def test_kaggle_webhook_receiver_repairs_completed_bundle_artifact(tmp_path):
    import backend.kaggle_webhook_receiver as receiver

    output_dir = tmp_path / "output" / "viettoxic"
    output_dir.mkdir(parents=True)
    input_bundle = output_dir / "mlflow_bundle.zip"
    result_zip = output_dir / "kaggle_run_repaired.zip"
    with zipfile.ZipFile(input_bundle, "w") as archive:
        archive.writestr("dataset.jsonl", '{"text":"new row"}\n')
    with zipfile.ZipFile(result_zip, "w") as archive:
        archive.writestr("metrics.json", "{}")
        archive.writestr("model_lr.joblib", b"model")
        archive.writestr("vectorizer.joblib", b"vectorizer")

    job = {
        "mode": "real",
        "job_id": "real_repair",
        "run_id": "kaggle_repair",
        "status": "completed",
        "current_stage": "complete",
        "kernel_ref": "owner/kernel",
        "work_dir": str(tmp_path),
        "artifact_uri": receiver._resolve_artifact_uri(input_bundle, {}),
        "artifact_checksum": receiver._sha256_file(input_bundle),
    }
    jobs = {"real_repair": job}

    payload = receiver._status_real("real_repair", job, jobs)

    assert payload["status"] == "completed"
    assert payload["artifact_uri"] == receiver._resolve_artifact_uri(result_zip, job)
    assert payload["artifact_checksum"] == receiver._sha256_file(result_zip)


def test_kaggle_webhook_receiver_lr_smoke_uses_sqlite_mlflow(monkeypatch):
    import backend.kaggle_webhook_receiver as receiver

    monkeypatch.setattr(receiver, "_setting", lambda key, default="": default)

    script = receiver._build_real_script_content(
        receiver.TriggerRequest(run_id="run_smoke", model_kind="lr_smoke", training_mode="retrain")
    )

    assert "MLFLOW_ALLOW_FILE_STORE" in script
    assert "sqlite:///" in script
    assert "MLflow logging skipped" in script
    assert "smoke_retrain_tfidf_lr_multitask" in script
    assert "ngrok-skip-browser-warning" in script
    compile(script, "<kaggle-script>", "exec")


def test_lr_smoke_keeps_all_mlflow_rows_when_downsampling_gold():
    source_path = Path(__file__).parents[1] / "kaggle" / "notebooks" / "mlflow_retrain" / "viettoxic_mlflow_retrain.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"downsample_rows", "_mlflow_comment_id", "select_smoke_train_rows", "verify_smoke_mlflow_coverage"}
    selected_nodes = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted]
    namespace = {"Any": typing.Any, "hashlib": hashlib, "SEED": 42}
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(source_path), "exec"), namespace)

    rows = [{"text": f"gold-{idx}", "toxicity": idx % 2} for idx in range(6)]
    rows.extend(
        [
            {"text": "new-101", "toxicity": 1, "meta": {"source": "MLFlowAccepted", "mlflow_comment_id": 101}},
            {"text": "new-102", "toxicity": 0, "meta": {"source": "MLFlowAccepted", "mlflow_comment_id": 102}},
            {"text": "synthetic-103", "toxicity": 1, "meta": {"source": "SyntheticReviewed", "mlflow_comment_id": 103}},
        ]
    )

    chosen, evidence = namespace["select_smoke_train_rows"](rows, 4)
    assert len(chosen) == 4
    assert evidence["used_gold"] == 1
    assert evidence["used_mlflow"] == 3
    assert evidence["mlflow_comment_ids"] == [101, 102, 103]
    coverage = namespace["verify_smoke_mlflow_coverage"]([101, 102, 103], [101, 102, 103])
    assert coverage["included_all_expected_mlflow"] is True
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        namespace["verify_smoke_mlflow_coverage"]([101], [101, 102])


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
    assert status_payload["metrics"]["dataset_evidence"]["included_mlflow_count"] == 10
    assert status_payload["metrics"]["dataset_evidence"]["included_all_expected_mlflow"] is True
    assert status_payload["metrics"]["confusion_matrix"]["test"]["tp"] == 74
    assert status_payload["metrics"]["sizes"]["train"] == 4000

    download_response = client.get("/api/mlflow/kaggle/artifact/download", params={"run_id": run_id}, headers=admin_headers)
    assert download_response.status_code == 200
    assert download_response.headers.get("content-type") == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(download_response.content))
    assert "results/metrics.json" in archive.namelist()


def test_kaggle_status_includes_previous_completed_run_of_same_model_kind(client, qa_env, admin_headers):
    previous_run_id = "run_previous_artifact"
    current_run_id = "run_current_artifact"
    previous_artifact = qa_env["kaggle_root"] / previous_run_id / "output" / "artifact.zip"
    current_artifact = qa_env["kaggle_root"] / current_run_id / "output" / "artifact.zip"
    _build_metrics_zip(previous_artifact, f1_toxic=0.65, macro_f1=0.72)
    _build_metrics_zip(current_artifact, f1_toxic=0.71, macro_f1=0.76)
    _insert_kaggle_run(
        qa_env["feedback_db"],
        previous_run_id,
        status="completed",
        artifact_uri=str(previous_artifact),
    )
    _insert_kaggle_run(
        qa_env["feedback_db"],
        current_run_id,
        status="completed",
        artifact_uri=str(current_artifact),
    )
    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        conn.execute(
            "UPDATE mlflow_do_run SET created_at = ? WHERE run_id = ?",
            ("2026-01-01T00:00:00Z", previous_run_id),
        )
        conn.execute(
            "UPDATE mlflow_do_run SET created_at = ? WHERE run_id = ?",
            ("2026-01-02T00:00:00Z", current_run_id),
        )
        conn.commit()

    response = client.get(
        "/api/mlflow/kaggle/status",
        params={"run_id": current_run_id},
        headers=admin_headers,
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["previous_run"]["run_id"] == previous_run_id
    assert payload["previous_run"]["metrics"]["f1_toxic"] == 0.65
    assert payload["previous_run"]["metrics"]["macro_f1"] == 0.72


@pytest.mark.parametrize("model_family", ["tfidf_lr", "phobert"])
def test_family_candidate_compare_promote_and_rollback_end_to_end(
    client,
    qa_env,
    admin_headers,
    model_family,
):
    run_id, baseline_id = _prepare_family_production_flow(qa_env, model_family=model_family)

    comparison_response = client.get(
        "/api/mlflow/compare/latest",
        params={"run_id": run_id},
        headers=admin_headers,
    )
    assert comparison_response.status_code == 200
    comparison = comparison_response.json()
    assert comparison["model_family"] == model_family
    assert comparison["current"]["model"] == baseline_id
    assert comparison["candidate"]["source_run_id"] == run_id
    assert comparison["candidate"]["artifact_verified"] is True
    assert comparison["test_comparability_verified"] is True
    assert comparison["promotion_enabled"] is True
    assert all(check["passed"] for check in comparison["gate_checks"])

    promote_response = client.post(
        "/api/mlflow/promote",
        headers=admin_headers,
        json={
            "run_id": run_id,
            "artifact_checksum": comparison["candidate"]["artifact_checksum"],
            "expected_current_version": baseline_id,
        },
    )
    assert promote_response.status_code == 200
    promoted = promote_response.json()
    assert promoted["status"] == "promoted"
    assert promoted["model_family"] == model_family
    assert promoted["previous_model"] == baseline_id
    promoted_model = promoted["candidate_model"]

    models_response = client.get("/api/models")
    assert models_response.status_code == 200
    models_payload = models_response.json()
    assert promoted_model in models_payload["models"]
    assert models_payload["production_slots"][model_family] == promoted_model

    installed_dir = qa_env["app_module"].MODEL_OPTIONS_DIR / model_family / run_id
    assert (installed_dir / "production_manifest.json").is_file()
    if model_family == "tfidf_lr":
        assert (installed_dir / "model_lr.pkl").is_file()
        assert (installed_dir / "vectorizer.pkl").is_file()
    else:
        assert (installed_dir / "config.json").is_file()
        assert (installed_dir / "model.safetensors").is_file()

    rollback_response = client.post(
        "/api/mlflow/rollback",
        headers=admin_headers,
        json={
            "model_family": model_family,
            "expected_current_version": promoted_model,
        },
    )
    assert rollback_response.status_code == 200
    rolled_back = rollback_response.json()
    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["active_model"] == baseline_id

    models_after_rollback = client.get("/api/models").json()
    assert models_after_rollback["production_slots"][model_family] == baseline_id


def test_lr_smoke_training_mode_is_always_retrain(qa_env):
    request = qa_env["app_module"].MlflowDOTriggerRequest(
        model_kind="lr_smoke",
        training_mode="finetune",
    )
    assert qa_env["app_module"]._do_resolve_training_mode(request) == "retrain"
    assert qa_env["app_module"]._do_resolve_base_model(request) is None


def test_automation_is_disabled_by_default_and_does_not_start_a_cycle(client, admin_headers):
    status_response = client.get("/api/mlflow/automation/status", headers=admin_headers)

    assert status_response.status_code == 200
    tfidf = next(item for item in status_response.json()["families"] if item["model_family"] == "tfidf_lr")
    assert tfidf["policy"]["enabled"] is False
    assert tfidf["ready"] is False
    assert tfidf["blocked_reason"] == "global_disabled"

    cycle_response = client.post(
        "/api/mlflow/automation/cycle",
        headers=admin_headers,
        json={"model_family": "tfidf_lr"},
    )

    assert cycle_response.status_code == 200
    result = cycle_response.json()["results"][0]
    assert result["started"] is False
    assert result["blocked_reason"] == "global_disabled"


def test_full_auto_promotes_only_an_automation_created_candidate(qa_env):
    app_module = qa_env["app_module"]
    run_id, baseline_id = _prepare_family_production_flow(qa_env, model_family="tfidf_lr")
    app_module.update_system_settings(
        qa_env["feedback_db"],
        {
            "MLFLOW_AUTOMATION_ENABLED": True,
            "MLFLOW_AUTOMATION_TFIDF_LR_MODE": "full_auto",
            "MLFLOW_AUTOMATION_DRY_RUN": False,
        },
    )
    app_module._automation_record_event("tfidf_lr", "train_started", "running", source_run_id=run_id)

    app_module._automation_handle_terminal_run(run_id)

    with sqlite3.connect(qa_env["feedback_db"]) as conn:
        slot = conn.execute(
            "SELECT active_model_id FROM mlflow_production_slot WHERE model_family = 'tfidf_lr'"
        ).fetchone()
        event = conn.execute(
            """
            SELECT status FROM mlflow_automation_event
            WHERE source_run_id = ? AND action = 'auto_promote'
            ORDER BY id DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()

    assert slot[0] == f"tfidf_lr/{run_id}"
    assert slot[0] != baseline_id
    assert event[0] == "promoted"


def test_gemini_evaluate_persists_a_compact_vietnamese_assessment(
    client,
    qa_env,
    admin_headers,
    monkeypatch,
):
    app_module = qa_env["app_module"]
    run_id, _ = _prepare_family_production_flow(qa_env, model_family="tfidf_lr")
    calls: list[str] = []

    def fake_gemini(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            {
                "summary": "Candidate có cải thiện nhỏ; cần xem trade-off recall trước khi quyết định.",
                "verdict": "review",
                "recommendation": "Admin kiểm tra false negative và metric theo domain.",
                "strengths": ["Artifact và metrics đầy đủ."],
                "risks": ["Recall có thể giảm."],
                "metric_observations": ["So sánh dùng cùng test fingerprint."],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(app_module, "call_gemini", fake_gemini)

    response = client.post(
        "/api/mlflow/kaggle/evaluate",
        headers=admin_headers,
        json={"run_id": run_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "evaluated"
    assert payload["evaluation"]["verdict"] == "review"
    assert "Candidate" in payload["evaluation"]["summary"]
    assert len(calls) == 1
    assert "Không tự promote model" in calls[0]

    cached = client.post(
        "/api/mlflow/kaggle/evaluate",
        headers=admin_headers,
        json={"run_id": run_id},
    )
    assert cached.status_code == 200
    assert cached.json()["status"] == "cached"
    assert len(calls) == 1

    status = client.get("/api/mlflow/kaggle/status", headers=admin_headers, params={"run_id": run_id})
    assert status.status_code == 200
    assert status.json()["gemini_evaluation"]["evaluation"]["verdict"] == "review"


def test_system_settings_expose_vietnamese_help_and_ai_instructions(client, admin_headers):
    response = client.get("/api/admin/system-settings", headers=admin_headers)

    assert response.status_code == 200
    groups = {group["id"]: group for group in response.json()["groups"]}
    assert groups["kaggle_account"]["label"] == "Tài khoản Kaggle"
    gemini_review = next(item for item in groups["ai_instructions"]["settings"] if item["key"] == "GEMINI_REVIEW_INSTRUCTION")
    assert "Prompt" in gemini_review["description"]
    assert gemini_review["value"]
