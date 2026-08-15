import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from infer_crawled_local import infer_crawled


def _write_tfidf_models(model_dir: Path, *, include_constructiveness: bool) -> None:
    texts = [
        "Bình luận có lập luận và giải pháp rõ ràng",
        "Đồ ngu",
        "Tôi đề xuất công khai kế hoạch thực hiện",
        "Vớ vẩn quá",
    ]
    vectorizer = TfidfVectorizer().fit(texts)
    features = vectorizer.transform(texts)
    toxicity_model = LogisticRegression(random_state=0).fit(features, [0, 1, 0, 1])

    model_dir.mkdir(parents=True)
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    joblib.dump(toxicity_model, model_dir / "model_lr.pkl")
    if include_constructiveness:
        constructiveness_model = LogisticRegression(random_state=0).fit(features, [1, 0, 1, 0])
        joblib.dump(constructiveness_model, model_dir / "model_lr_constructiveness.pkl")


def _write_crawl_input(data_dir: Path) -> None:
    page_dir = data_dir / "page-1"
    page_dir.mkdir(parents=True)
    (page_dir / "meta.json").write_text(
        json.dumps({"url": "https://example.test/article", "status": "ok"}),
        encoding="utf-8",
    )
    (page_dir / "segments.jsonl").write_text(
        json.dumps({"text": "Bình luận có lập luận và giải pháp rõ ràng"}) + "\n",
        encoding="utf-8",
    )


def test_tfidf_constructiveness_model_is_emitted_when_present(tmp_path: Path):
    model_dir = tmp_path / "models" / "tfidf_lr" / "with_constructiveness"
    data_dir = tmp_path / "crawl"
    output_dir = tmp_path / "output"
    _write_tfidf_models(model_dir, include_constructiveness=True)
    _write_crawl_input(data_dir)

    result = infer_crawled(
        data_dir=str(data_dir),
        out_dir=str(output_dir),
        model_path=str(model_dir),
        model_type="tfidf_lr",
        quiet=True,
    )

    segment = result["segment_results"][0]
    assert 0.0 <= segment["constructiveness_prob"] <= 1.0
    assert segment["constructiveness_label"] in {0, 1}
    assert result["page_results"][0]["avg_constructiveness_prob"] == segment["constructiveness_prob"]


def test_tfidf_without_constructiveness_model_remains_supported(tmp_path: Path):
    model_dir = tmp_path / "models" / "tfidf_lr" / "toxicity_only"
    data_dir = tmp_path / "crawl"
    output_dir = tmp_path / "output"
    _write_tfidf_models(model_dir, include_constructiveness=False)
    _write_crawl_input(data_dir)

    result = infer_crawled(
        data_dir=str(data_dir),
        out_dir=str(output_dir),
        model_path=str(model_dir),
        model_type="tfidf_lr",
        quiet=True,
    )

    segment = result["segment_results"][0]
    assert "constructiveness_prob" not in segment
    assert "avg_constructiveness_prob" not in result["page_results"][0]
