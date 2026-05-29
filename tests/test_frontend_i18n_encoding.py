from pathlib import Path


FRONTEND_SRC = Path(__file__).resolve().parents[1] / "comprehensive_ui" / "src"

MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "Ä",
    "Æ",
    "Î",
    "Ï",
    "Å",
    "áº",
    "á»",
    "â€",
    "â†",
    "âš",
    "âœ",
    "ï¸",
    "ðŸ",
)


def _frontend_text_files() -> list[Path]:
    suffixes = {".ts", ".tsx", ".js", ".jsx", ".css", ".html"}
    return [
        path
        for path in FRONTEND_SRC.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("._")
    ]


def test_frontend_source_has_no_common_utf8_mojibake_markers():
    offenders: list[str] = []
    for path in _frontend_text_files():
        text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in MOJIBAKE_MARKERS):
                rel = path.relative_to(FRONTEND_SRC.parents[1])
                offenders.append(f"{rel}:{line_number}: {line.strip()[:120]}")

    assert offenders == []


def test_kaggle_pipeline_vietnamese_copy_is_utf8_clean():
    text = (FRONTEND_SRC / "app" / "components" / "MLFlowPage.tsx").read_text(encoding="utf-8-sig")

    expected_copy = [
        "Pipeline tự động Google Kaggle (API trực tiếp)",
        "Kích hoạt Kaggle Pipeline",
        "Flow tự động hiện chạy qua Google Kaggle (GPU runtime).",
        "Để trống để dùng base model mặc định của script finetune.",
        "Retrain phù hợp khi refresh dataset lớn; Finetune phù hợp khi thêm ít data/pseudo mới để giảm tài nguyên.",
    ]

    for copy in expected_copy:
        assert copy in text
