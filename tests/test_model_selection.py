import json
import logging
import sqlite3
from pathlib import Path

import pytest

import backend.app as app_module
import infer_crawled_local as inference


V2_ID = "phobert/phobert_v2_finetuned"
V2_LEGACY_ID = "phobert/phobert_lora_4.7"
V1_ID = "phobert/baseline"
TFIDF_ID = "tfidf_lr/baseline_tfidf"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_phobert(
    model_root: Path,
    name: str,
    base_model: str,
    *,
    with_tokenizer: bool = True,
) -> Path:
    model_dir = model_root / "phobert" / name
    _write_json(model_dir / "config.json", {"model_type": "roberta"})
    (model_dir / "model.safetensors").write_bytes(b"checkpoint")
    _write_json(
        model_dir / "training_manifest.json",
        {"hyperparams": {"base_model": base_model}},
    )
    _write_json(
        model_dir / "run_config.json",
        {"hyperparameters": {"MODEL_NAME": base_model}},
    )
    if with_tokenizer:
        _write_json(model_dir / "tokenizer_config.json", {"tokenizer_class": "PhobertTokenizer"})
        (model_dir / "vocab.txt").write_text("token", encoding="utf-8")
        (model_dir / "bpe.codes").write_text("token", encoding="utf-8")
    return model_dir


def _create_tfidf(model_root: Path) -> Path:
    model_dir = model_root / "tfidf_lr" / "baseline_tfidf"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "vectorizer.pkl").write_bytes(b"vectorizer")
    (model_dir / "model_lr.pkl").write_bytes(b"classifier")
    return model_dir


def _create_runtime_catalog(model_root: Path) -> dict[str, Path]:
    return {
        V2_ID: _create_phobert(
            model_root,
            "phobert_lora_4.7",
            "vinai/phobert-base-v2",
        ),
        V1_ID: _create_phobert(
            model_root,
            "baseline",
            "vinai/phobert-base",
        ),
        TFIDF_ID: _create_tfidf(model_root),
    }


def test_verified_v2_is_default_in_backend_and_local_inference(tmp_path: Path):
    model_root = tmp_path / "models"
    _create_runtime_catalog(model_root)

    assert app_module.get_default_model_id(model_root) == V2_ID
    assert inference.get_default_model_id(model_root) == V2_ID


def test_default_fallback_order_skips_missing_or_incompatible_models(tmp_path: Path):
    model_root = tmp_path / "models"
    _create_tfidf(model_root)
    v1_dir = _create_phobert(model_root, "baseline", "vinai/phobert-base")
    other_dir = _create_phobert(model_root, "candidate", "vinai/phobert-base-v2")
    bad_v2_dir = model_root / "phobert" / "phobert_lora_4.7"
    bad_v2_dir.mkdir(parents=True)
    _write_json(
        bad_v2_dir / "training_manifest.json",
        {"hyperparams": {"base_model": "vinai/phobert-base"}},
    )

    assert app_module.get_default_model_id(model_root) == "phobert/candidate"

    (other_dir / "model.safetensors").unlink()
    assert app_module.get_default_model_id(model_root) == V1_ID

    (v1_dir / "model.safetensors").unlink()
    assert app_module.get_default_model_id(model_root) == TFIDF_ID


@pytest.mark.parametrize(
    ("requested_model", "expected_model"),
    [
        (None, V2_ID),
        (TFIDF_ID, TFIDF_ID),
        (V1_ID, V1_ID),
        (V2_ID, V2_ID),
        (V2_LEGACY_ID, V2_ID),
    ],
)
def test_analyze_default_and_explicit_selection_are_visible(
    client,
    qa_env,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    requested_model: str | None,
    expected_model: str,
):
    _create_runtime_catalog(app_module.MODEL_OPTIONS_DIR)
    monkeypatch.setattr(app_module, "crawl_urls", lambda *args, **kwargs: [])

    options = {"collect_for_mlflow": False}
    if requested_model:
        options["model_name"] = requested_model

    with caplog.at_level(logging.INFO, logger="viet-toxic-backend"):
        response = client.post(
            "/api/analyze",
            json={"urls": ["https://example.com/comments"], "options": options},
        )

    assert response.status_code == 200
    assert response.json()["model_name"] == expected_model
    assert any(expected_model in record.getMessage() for record in caplog.records)


def test_models_api_exposes_accurate_labels(client):
    _create_runtime_catalog(app_module.MODEL_OPTIONS_DIR)

    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default"] == V2_ID
    assert set(payload["models"]) == {TFIDF_ID, V1_ID, V2_ID}
    assert payload["labels"] == {
        TFIDF_ID: "TF-IDF + Logistic Regression",
        V1_ID: "PhoBERT v1 Baseline",
        V2_ID: "PhoBERT v2 Fine-tuned",
    }
    assert payload["finetune_base_models"] == [V1_ID, V2_ID]


def test_models_api_excludes_incomplete_checkpoint_from_finetune_dropdown(client):
    _create_runtime_catalog(app_module.MODEL_OPTIONS_DIR)
    incomplete_id = "phobert/incomplete"
    _create_phobert(
        app_module.MODEL_OPTIONS_DIR,
        "incomplete",
        "vinai/phobert-base-v2",
        with_tokenizer=False,
    )

    payload = client.get("/api/models").json()

    assert incomplete_id in payload["models"]
    assert incomplete_id not in payload["finetune_base_models"]


def test_legacy_v2_identifier_resolves_to_canonical_id(tmp_path: Path):
    model_root = tmp_path / "models"
    catalog = _create_runtime_catalog(model_root)

    backend_type, backend_name, backend_path = app_module.resolve_model_path(
        model_root,
        V2_LEGACY_ID,
    )
    inference_type, inference_name, inference_path = inference.resolve_model_path(
        model_root,
        V2_LEGACY_ID,
    )

    assert (backend_type, backend_name, backend_path) == (
        "phobert",
        "phobert_v2_finetuned",
        catalog[V2_ID],
    )
    assert (inference_type, inference_name, inference_path) == (
        "phobert",
        "phobert_v2_finetuned",
        catalog[V2_ID],
    )


def test_tokenizer_source_matches_selected_v1_and_v2_checkpoint(tmp_path: Path):
    model_root = tmp_path / "models"
    catalog = _create_runtime_catalog(model_root)

    v1_source = inference.resolve_phobert_tokenizer_source(catalog[V1_ID])
    v2_source = inference.resolve_phobert_tokenizer_source(catalog[V2_ID])

    assert Path(v1_source).resolve() == catalog[V1_ID].resolve()
    assert Path(v2_source).resolve() == catalog[V2_ID].resolve()
    assert v1_source != v2_source
    assert inference.get_phobert_base_model(catalog[V1_ID]) == "vinai/phobert-base"
    assert inference.get_phobert_base_model(catalog[V2_ID]) == "vinai/phobert-base-v2"


def test_tokenizer_falls_back_to_checkpoint_base_metadata(tmp_path: Path):
    model_dir = _create_phobert(
        tmp_path / "models",
        "v2_without_local_tokenizer",
        "vinai/phobert-base-v2",
        with_tokenizer=False,
    )

    assert (
        inference.resolve_phobert_tokenizer_source(model_dir)
        == "vinai/phobert-base-v2"
    )


def test_training_tracker_migrates_legacy_lora_descriptions():
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE training_tracker_phase (id TEXT, title TEXT, updated_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE training_tracker_group (id TEXT, title TEXT, updated_at TEXT)"
        )
        conn.execute(
            """
            CREATE TABLE training_tracker_task (
                id TEXT, label TEXT, param TEXT, updated_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO training_tracker_phase VALUES ('phase_5', 'LoRA config', '')"
        )
        conn.execute(
            "INSERT INTO training_tracker_group VALUES ('p1_group_14', 'Learning rate LoRA', '')"
        )
        conn.execute(
            """
            INSERT INTO training_tracker_task
            VALUES ('p5_task_1', 'Test r=8', 'LORA_R=8', '')
            """
        )

        app_module.migrate_training_tracker_lora_terminology(conn)

        assert conn.execute(
            "SELECT title FROM training_tracker_phase WHERE id = 'phase_5'"
        ).fetchone()[0] == "Giai đoạn 5 — Full fine-tuning config"
        assert conn.execute(
            "SELECT title FROM training_tracker_group WHERE id = 'p1_group_14'"
        ).fetchone()[0] == "1.4 Learning rate (full fine-tuning)"
        assert conn.execute(
            "SELECT label, param FROM training_tracker_task WHERE id = 'p5_task_1'"
        ).fetchone() == ("Test LR=1e-5", "LEARNING_RATE=1e-5")
