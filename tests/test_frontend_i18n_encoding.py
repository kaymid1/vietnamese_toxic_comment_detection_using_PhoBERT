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
        "Tạo bundle & kích hoạt Kaggle",
        "Flow tự động hiện chạy qua Google Kaggle (GPU runtime).",
        "Để trống để dùng base model mặc định của script finetune.",
        "Retrain phù hợp khi refresh dataset lớn; Finetune phù hợp khi thêm ít data/pseudo mới để giảm tài nguyên.",
    ]

    for copy in expected_copy:
        assert copy in text


def test_kaggle_ui_session_is_restored_until_explicit_clear():
    hook_text = (FRONTEND_SRC / "hooks" / "useMlflowStore.ts").read_text(encoding="utf-8-sig")
    page_text = (FRONTEND_SRC / "app" / "components" / "MLFlowPage.tsx").read_text(encoding="utf-8-sig")

    assert 'DO_RUN_STORAGE_KEY = "viettoxic:mlflow:kaggleRunId"' in hook_text
    assert "useState<string | null>(readPersistedDORunId)" in hook_text
    assert "refreshDOStatus(doRunId)" in hook_text
    assert "window.localStorage.removeItem(DO_RUN_STORAGE_KEY)" in hook_text
    assert 'MLFLOW_ACTIVE_TAB_KEY = "viettoxic:mlflow:activeTab"' in page_text


def test_kaggle_trigger_button_is_locked_while_request_or_run_is_active():
    page_text = (FRONTEND_SRC / "app" / "components" / "MLFlowPage.tsx").read_text(encoding="utf-8-sig")

    assert 'const KAGGLE_TERMINAL_STATUSES = new Set(["completed", "failed", "dry_run", "placeholder"])' in page_text
    assert "const [kaggleTriggerPending, setKaggleTriggerPending] = useState(false);" in page_text
    assert "const kaggleTriggerPendingRef = useRef(false);" in page_text
    assert "kaggleTriggerPendingRef.current = true;" in page_text
    assert "kaggleTriggerPendingRef.current = false;" in page_text
    assert "kaggleTriggerPending || loading || doHasActiveRun" in page_text
