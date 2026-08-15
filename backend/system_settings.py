import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_DB_PATH = BASE_DIR / "data" / "processed" / "feedback" / "feedback.db"


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    label: str
    group: str
    group_label: str
    value_type: str
    default: Optional[str] = None
    required: bool = False
    secret: bool = False
    min_value: Optional[int] = None
    options: Optional[tuple[str, ...]] = None
    multiline: bool = False


GROUPS: tuple[tuple[str, str], ...] = (
    ("kaggle_account", "Tài khoản Kaggle"),
    ("kaggle_kernel", "Kaggle Kernel"),
    ("kaggle_webhook", "Kaggle Webhook"),
    ("mlflow_automation", "Tự động hóa MLflow"),
    ("gemini", "Cấu hình Gemini"),
    ("ai_instructions", "AI Instructions"),
    ("video_asr", "Video/ASR"),
)

GROUPS = GROUPS[:3] + (("mlflow_dataset", "MLflow Dataset"),) + GROUPS[3:]


DEFAULT_GEMINI_REVIEW_INSTRUCTION = (
    "Bạn là reviewer dữ liệu tiếng Việt cho bài toán toxic-content detection. "
    "Ưu tiên chất lượng nhãn training; chỉ đánh giá từ nội dung và ngữ cảnh được cung cấp. "
    "Khi không đủ chắc chắn, yêu cầu review thêm thay vì suy đoán."
)
DEFAULT_GEMINI_EVALUATE_INSTRUCTION = (
    "Bạn là trợ lý đánh giá thí nghiệm MLOps tiếng Việt. Phân tích khách quan candidate mới "
    "so với production và run trước từ các số liệu được cung cấp. Nêu cải thiện, đánh đổi, "
    "rủi ro và khuyến nghị rõ ràng. Không bịa số liệu, không khẳng định deployment thành công, "
    "và không thay thế production gate hay quyết định cuối cùng của admin."
)


SETTING_DEFINITIONS: tuple[SettingDefinition, ...] = (
    SettingDefinition("KAGGLE_USERNAME", "Username", "kaggle_account", "Kaggle Account", "string"),
    SettingDefinition("KAGGLE_KEY", "API Key", "kaggle_account", "Kaggle Account", "string", secret=True),
    SettingDefinition("KAGGLE_NOTEBOOK_URL", "Notebook URL", "kaggle_account", "Kaggle Account", "string", required=True),
    SettingDefinition("KAGGLE_KERNEL_OWNER", "Kernel owner", "kaggle_kernel", "Kaggle Kernel", "string"),
    SettingDefinition("KAGGLE_KERNEL_SLUG", "Kernel slug", "kaggle_kernel", "Kaggle Kernel", "string"),
    SettingDefinition("KAGGLE_KERNEL_TITLE", "Kernel title", "kaggle_kernel", "Kaggle Kernel", "string", default="thesis-phobert"),
    SettingDefinition(
        "KAGGLE_KERNEL_ACCELERATOR",
        "Accelerator",
        "kaggle_kernel",
        "Kaggle Kernel",
        "string",
        default="NvidiaTeslaT4",
    ),
    SettingDefinition("KAGGLE_KERNEL_PRIVATE", "Private kernel", "kaggle_kernel", "Kaggle Kernel", "bool", default="true"),
    SettingDefinition(
        "KAGGLE_KERNEL_DATASET_SOURCES",
        "Dataset sources",
        "kaggle_kernel",
        "Kaggle Kernel",
        "string",
        multiline=True,
    ),
    SettingDefinition("KAGGLE_WEBHOOK_URL", "Trigger webhook URL", "kaggle_webhook", "Kaggle Webhook", "string", required=True),
    SettingDefinition("KAGGLE_STATUS_WEBHOOK_URL", "Status webhook URL", "kaggle_webhook", "Kaggle Webhook", "string"),
    SettingDefinition("KAGGLE_REAL_BUNDLE_URL", "Real run bundle URL", "kaggle_webhook", "Kaggle Webhook", "string"),
    SettingDefinition("KAGGLE_BUNDLE_PUBLIC_BASE_URL", "Backend public URL for run bundles", "kaggle_webhook", "Kaggle Webhook", "string"),
    SettingDefinition(
        "KAGGLE_REAL_BUNDLE_URL_TEMPLATE",
        "Real run bundle URL template",
        "kaggle_webhook",
        "Kaggle Webhook",
        "string",
    ),
    SettingDefinition(
        "KAGGLE_WEBHOOK_TIMEOUT_SEC",
        "Webhook timeout seconds",
        "kaggle_webhook",
        "Kaggle Webhook",
        "int",
        default="180",
        min_value=10,
    ),
    SettingDefinition(
        "KAGGLE_WEBHOOK_MODE",
        "Webhook mode",
        "kaggle_webhook",
        "Kaggle Webhook",
        "enum",
        default="mock",
        options=("mock", "real"),
    ),
    SettingDefinition(
        "KAGGLE_REAL_TEST_MODE",
        "Real test mode",
        "kaggle_webhook",
        "Kaggle Webhook",
        "string",
        default="phobert",
    ),
    SettingDefinition(
        "KAGGLE_PUSH_RETRY_ATTEMPTS",
        "Push retry attempts",
        "kaggle_webhook",
        "Kaggle Webhook",
        "int",
        default="4",
        min_value=1,
    ),
    SettingDefinition(
        "KAGGLE_PUSH_RETRY_DELAY_SEC",
        "Push retry delay seconds",
        "kaggle_webhook",
        "Kaggle Webhook",
        "int",
        default="5",
        min_value=1,
    ),
    SettingDefinition(
        "MLFLOW_THRESHOLD_TARGET_MAX",
        "Minimum MLflow rows for training",
        "mlflow_dataset",
        "MLflow Dataset",
        "int",
        default="10",
        min_value=1,
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_ENABLED",
        "Global automation switch",
        "mlflow_automation",
        "MLflow Automation",
        "bool",
        default="false",
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_TFIDF_LR_MODE",
        "TF-IDF/LR mode",
        "mlflow_automation",
        "MLflow Automation",
        "enum",
        default="disabled",
        options=("disabled", "train_only", "full_auto"),
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_PHOBERT_MODE",
        "PhoBERT mode",
        "mlflow_automation",
        "MLflow Automation",
        "enum",
        default="disabled",
        options=("disabled", "train_only", "full_auto"),
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_MIN_NEW_ROWS",
        "Minimum new eligible rows",
        "mlflow_automation",
        "MLflow Automation",
        "int",
        default="50",
        min_value=1,
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_COOLDOWN_MINUTES",
        "Cooldown minutes",
        "mlflow_automation",
        "MLflow Automation",
        "int",
        default="1440",
        min_value=0,
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_DRY_RUN",
        "Automation dry run",
        "mlflow_automation",
        "MLflow Automation",
        "bool",
        default="true",
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_POLL_SECONDS",
        "Status poll seconds",
        "mlflow_automation",
        "MLflow Automation",
        "int",
        default="30",
        min_value=10,
    ),
    SettingDefinition(
        "MLFLOW_AUTOMATION_MAX_POLL_MINUTES",
        "Maximum watcher minutes",
        "mlflow_automation",
        "MLflow Automation",
        "int",
        default="180",
        min_value=1,
    ),
    SettingDefinition("GEMINI_API_KEY", "API key", "gemini", "Gemini", "string", required=True, secret=True),
    SettingDefinition("GEMINI_MODEL", "Primary model", "gemini", "Gemini", "string", default="gemini-1.5-flash-latest", required=True),
    SettingDefinition("GEMINI_FALLBACK_MODELS", "Fallback models", "gemini", "Gemini", "string", multiline=True),
    SettingDefinition("GEMINI_API_VERSION", "API version", "gemini", "Gemini", "string", default="v1beta"),
    SettingDefinition("GEMINI_MAX_TOKENS", "Max tokens", "gemini", "Gemini", "int", default="1024", min_value=1),
    SettingDefinition(
        "GEMINI_MIN_REQUEST_INTERVAL_SECONDS",
        "Minimum request interval",
        "gemini",
        "Gemini",
        "int",
        default="13",
        min_value=0,
    ),
    SettingDefinition(
        "GEMINI_RETRY_ATTEMPTS",
        "Transient retry attempts",
        "gemini",
        "Gemini",
        "int",
        default="2",
        min_value=1,
    ),
    SettingDefinition(
        "GEMINI_REVIEW_MAX_ITEMS",
        "Maximum comments per review",
        "gemini",
        "Gemini",
        "int",
        default="9",
        min_value=1,
    ),
    SettingDefinition(
        "GEMINI_REVIEW_INSTRUCTION",
        "Gemini Review instruction",
        "ai_instructions",
        "AI Instructions",
        "string",
        default=DEFAULT_GEMINI_REVIEW_INSTRUCTION,
        multiline=True,
    ),
    SettingDefinition(
        "GEMINI_EVALUATE_INSTRUCTION",
        "Gemini Evaluate instruction",
        "ai_instructions",
        "AI Instructions",
        "string",
        default=DEFAULT_GEMINI_EVALUATE_INSTRUCTION,
        multiline=True,
    ),
    SettingDefinition("VIDEO_LONG_SECONDS", "Long video seconds", "video_asr", "Video/ASR", "int", default="180", min_value=0),
    SettingDefinition(
        "VIDEO_TRANSCRIPT_LIMIT_SECONDS",
        "Transcript limit seconds",
        "video_asr",
        "Video/ASR",
        "int",
        default="180",
        min_value=0,
    ),
    SettingDefinition("ASR_TRIM_SECONDS", "ASR trim seconds", "video_asr", "Video/ASR", "int", default="180", min_value=0),
)


DEFINITIONS_BY_KEY: Dict[str, SettingDefinition] = {definition.key: definition for definition in SETTING_DEFINITIONS}


SETTING_VI_METADATA: Dict[str, tuple[str, str]] = {
    "KAGGLE_USERNAME": ("Tên người dùng", "Tài khoản Kaggle dùng để xác thực khi chạy notebook thật."),
    "KAGGLE_KEY": ("API key", "Khóa bí mật của Kaggle. Chỉ dùng cho backend; giao diện luôn che giá trị."),
    "KAGGLE_NOTEBOOK_URL": ("Đường dẫn notebook", "URL notebook Kaggle mà webhook sẽ kích hoạt."),
    "KAGGLE_KERNEL_OWNER": ("Chủ sở hữu kernel", "Tên tài khoản Kaggle sở hữu kernel/notebook."),
    "KAGGLE_KERNEL_SLUG": ("Kernel slug", "Định danh ngắn của Kaggle kernel, thường là phần cuối URL notebook."),
    "KAGGLE_KERNEL_TITLE": ("Tên hiển thị kernel", "Tên hiển thị khi tạo hoặc đồng bộ metadata của Kaggle kernel."),
    "KAGGLE_KERNEL_ACCELERATOR": ("Bộ tăng tốc", "Loại accelerator Kaggle yêu cầu, ví dụ NvidiaTeslaT4."),
    "KAGGLE_KERNEL_PRIVATE": ("Kernel riêng tư", "Bật để không công khai notebook và artifact trên Kaggle."),
    "KAGGLE_KERNEL_DATASET_SOURCES": ("Nguồn dataset", "Danh sách dataset Kaggle gắn thêm cho kernel, mỗi dòng một nguồn."),
    "KAGGLE_WEBHOOK_URL": ("URL kích hoạt", "Dịch vụ nhận yêu cầu tạo Kaggle run từ backend."),
    "KAGGLE_STATUS_WEBHOOK_URL": ("URL trạng thái", "Dịch vụ để backend hỏi tiến độ, artifact URI và checksum của Kaggle run."),
    "KAGGLE_REAL_BUNDLE_URL": ("Bundle URL cố định", "URL bundle dùng cho real-run legacy; ưu tiên bundle snapshot theo từng run."),
    "KAGGLE_BUNDLE_PUBLIC_BASE_URL": ("Public URL bundle", "URL public để Kaggle tải bundle snapshot. Bắt buộc với real automation."),
    "KAGGLE_REAL_BUNDLE_URL_TEMPLATE": ("Mẫu URL bundle", "Mẫu tạo URL bundle real-run khi dùng biến run id."),
    "KAGGLE_WEBHOOK_TIMEOUT_SEC": ("Timeout webhook (giây)", "Thời gian backend chờ webhook phản hồi trước khi coi trigger thất bại."),
    "KAGGLE_WEBHOOK_MODE": ("Chế độ webhook", "mock chỉ mô phỏng; real gọi webhook Kaggle thật."),
    "KAGGLE_REAL_TEST_MODE": ("Chế độ test real", "Profile real-run được webhook dùng khi tạo thử nghiệm."),
    "KAGGLE_PUSH_RETRY_ATTEMPTS": ("Số lần thử lại", "Số lần retry khi đẩy/cập nhật Kaggle kernel gặp lỗi tạm thời."),
    "KAGGLE_PUSH_RETRY_DELAY_SEC": ("Chờ giữa các lần retry", "Số giây chờ giữa hai lần retry Kaggle."),
    "MLFLOW_AUTOMATION_ENABLED": ("Công tắc tự động hóa toàn cục", "Tắt là chặn mọi cycle tự động của cả TF-IDF và PhoBERT."),
    "MLFLOW_AUTOMATION_TFIDF_LR_MODE": ("Chế độ TF-IDF/LR", "disabled: không chạy; train_only: train rồi chờ admin; full_auto: đạt gate thì tự promote."),
    "MLFLOW_AUTOMATION_PHOBERT_MODE": ("Chế độ PhoBERT", "disabled: không chạy; train_only: train rồi chờ admin; full_auto: đạt gate thì tự promote."),
    "MLFLOW_AUTOMATION_MIN_NEW_ROWS": ("Tối thiểu mẫu mới", "Số dòng eligible mới cần có kể từ cycle gần nhất trước khi cho phép train."),
    "MLFLOW_AUTOMATION_COOLDOWN_MINUTES": ("Thời gian chờ (phút)", "Khoảng cách tối thiểu giữa hai automation cycle của cùng model family."),
    "MLFLOW_AUTOMATION_DRY_RUN": ("Chạy thử không gọi cloud", "Tạo và kiểm tra bundle nhưng không kích hoạt Kaggle cloud thật."),
    "MLFLOW_AUTOMATION_POLL_SECONDS": ("Chu kỳ hỏi trạng thái", "Số giây giữa các lần worker tự hỏi trạng thái Kaggle."),
    "MLFLOW_AUTOMATION_MAX_POLL_MINUTES": ("Giới hạn theo dõi (phút)", "Thời gian tối đa worker nội bộ theo dõi một run trước khi cần refresh lại."),
    "GEMINI_API_KEY": ("Gemini API key", "Khóa bí mật để gọi Gemini Review và Gemini Evaluate."),
    "GEMINI_MODEL": ("Model chính", "Model Gemini được dùng trước; fallback chỉ dùng khi model chính lỗi/quá tải."),
    "GEMINI_FALLBACK_MODELS": ("Model dự phòng", "Danh sách model Gemini fallback, phân tách bằng dấu phẩy."),
    "GEMINI_API_VERSION": ("Phiên bản API", "Phiên bản Google Generative Language API dùng để gọi model."),
    "GEMINI_MAX_TOKENS": ("Giới hạn token", "Số token đầu ra tối đa cho mỗi câu trả lời Gemini."),
    "GEMINI_MIN_REQUEST_INTERVAL_SECONDS": ("Khoảng cách request tối thiểu", "Số giây tối thiểu giữa hai lần gọi Gemini; mặc định 13 giây phù hợp mức 5 RPM."),
    "GEMINI_RETRY_ATTEMPTS": ("Số lần retry tạm thời", "Số lần thử cùng model khi gặp 429 theo phút hoặc lỗi 503; quota theo ngày chuyển fallback ngay."),
    "GEMINI_REVIEW_MAX_ITEMS": ("Giới hạn comment mỗi lần Review", "Số comment tối đa Admin có thể gửi trong một thao tác Gemini Review; backend vẫn chia chunk tối đa 3 comment."),
    "GEMINI_REVIEW_INSTRUCTION": ("Instruction Gemini Review", "Prompt chỉ dẫn Gemini đánh giá nhãn comment; được hiển thị để giải trình với hội đồng."),
    "GEMINI_EVALUATE_INSTRUCTION": ("Instruction Gemini Evaluate", "Prompt chỉ dẫn Gemini nhận định candidate train mới so với run trước/production; không tự thay gate."),
    "VIDEO_LONG_SECONDS": ("Ngưỡng video dài", "Số giây để phân loại video là dài trong luồng xử lý video."),
    "VIDEO_TRANSCRIPT_LIMIT_SECONDS": ("Giới hạn transcript", "Số giây audio/video tối đa được đưa vào transcript."),
    "ASR_TRIM_SECONDS": ("Cắt ASR (giây)", "Số giây audio tối đa gửi qua ASR để giới hạn chi phí và thời gian."),
}


SETTING_VI_METADATA["MLFLOW_THRESHOLD_TARGET_MAX"] = (
    "Số mẫu MLflow tối thiểu để train",
    "Số dòng MLflow thực tế được thêm vào bundle cần có để bundle đủ điều kiện train. Dùng giá trị thấp cho demo/test; không thay đổi gate 0.20/0.80.",
)


def ensure_system_settings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_setting (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
        """
    )


def _read_override(db_path: Path, key: str) -> Optional[str]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            ensure_system_settings_table(conn)
            row = conn.execute("SELECT value FROM system_setting WHERE key = ?", (key,)).fetchone()
            return str(row[0]) if row else None
    except sqlite3.Error:
        return None


def _env_or_default(definition: SettingDefinition) -> Optional[str]:
    value = os.getenv(definition.key)
    if value is not None:
        return value
    return definition.default


def get_setting(key: str, default: Optional[str] = None, *, db_path: Optional[Path] = None) -> Optional[str]:
    definition = DEFINITIONS_BY_KEY.get(key)
    resolved_db_path = db_path or DEFAULT_SETTINGS_DB_PATH
    override = _read_override(resolved_db_path, key)
    if override is not None:
        return override
    if definition is not None:
        value = _env_or_default(definition)
        return default if value is None else value
    value = os.getenv(key)
    return default if value is None else value


def get_int_setting(key: str, default: int, *, db_path: Optional[Path] = None, min_value: Optional[int] = None) -> int:
    raw = get_setting(key, str(default), db_path=db_path)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = default
    definition = DEFINITIONS_BY_KEY.get(key)
    floor = min_value if min_value is not None else definition.min_value if definition else None
    if floor is not None:
        value = max(floor, value)
    return value


def get_bool_setting(key: str, default: bool, *, db_path: Optional[Path] = None) -> bool:
    raw = get_setting(key, "true" if default else "false", db_path=db_path)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}{'*' * max(4, len(value) - 6)}{value[-4:]}"


def _coerce_value(definition: SettingDefinition, value: Any) -> str:
    if value is None:
        return ""
    if definition.value_type == "bool":
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return "true"
        if text in {"0", "false", "no", "off"}:
            return "false"
        raise ValueError(f"{definition.key} must be a boolean")
    if definition.value_type == "int":
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{definition.key} must be an integer") from exc
        if definition.min_value is not None and number < definition.min_value:
            raise ValueError(f"{definition.key} must be >= {definition.min_value}")
        return str(number)
    text = str(value).strip()
    if definition.value_type == "enum":
        allowed = definition.options or ()
        if text not in allowed:
            raise ValueError(f"{definition.key} must be one of: {', '.join(allowed)}")
    return text


def list_system_settings(db_path: Path) -> Dict[str, Any]:
    values: Dict[str, str] = {}
    with sqlite3.connect(db_path) as conn:
        ensure_system_settings_table(conn)
        rows = conn.execute("SELECT key, value FROM system_setting").fetchall()
        values = {str(row[0]): str(row[1]) for row in rows}

    grouped: Dict[str, List[Dict[str, Any]]] = {group_id: [] for group_id, _ in GROUPS}
    for definition in SETTING_DEFINITIONS:
        vietnamese_label, vietnamese_description = SETTING_VI_METADATA.get(
            definition.key,
            (definition.label, f"Cấu hình runtime cho {definition.key}."),
        )
        has_override = definition.key in values
        raw_value = values.get(definition.key)
        if raw_value is None:
            raw_value = _env_or_default(definition)
        has_value = raw_value is not None and str(raw_value).strip() != ""
        setting: Dict[str, Any] = {
            "key": definition.key,
            "label": vietnamese_label,
            "description": vietnamese_description,
            "type": definition.value_type,
            "required": definition.required,
            "secret": definition.secret,
            "has_value": has_value,
            "source": "db" if has_override else "env" if os.getenv(definition.key) is not None else "default",
            "value": None if definition.secret else raw_value,
            "masked_value": _mask_secret(raw_value) if definition.secret else None,
            "default": None if definition.secret else definition.default,
            "min": definition.min_value,
            "options": list(definition.options or []),
            "multiline": definition.multiline,
        }
        grouped.setdefault(definition.group, []).append(setting)

    return {
        "groups": [
            {"id": group_id, "label": group_label, "settings": grouped.get(group_id, [])}
            for group_id, group_label in GROUPS
        ]
    }


def update_system_settings(
    db_path: Path,
    updates: Dict[str, Any],
    *,
    clear: Optional[Iterable[str]] = None,
    updated_by: Optional[str] = None,
) -> Dict[str, Any]:
    unknown = sorted(set(updates.keys()) - set(DEFINITIONS_BY_KEY.keys()))
    clear_keys = list(clear or [])
    unknown.extend(sorted(set(clear_keys) - set(DEFINITIONS_BY_KEY.keys())))
    if unknown:
        raise ValueError(f"Unknown system setting key(s): {', '.join(sorted(set(unknown)))}")

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    coerced = {key: _coerce_value(DEFINITIONS_BY_KEY[key], value) for key, value in updates.items()}
    with sqlite3.connect(db_path) as conn:
        ensure_system_settings_table(conn)
        for key in clear_keys:
            conn.execute("DELETE FROM system_setting WHERE key = ?", (key,))
        for key, value in coerced.items():
            conn.execute(
                """
                INSERT INTO system_setting (key, value, updated_at, updated_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (key, value, now, updated_by),
            )
        conn.commit()
    return list_system_settings(db_path)


def reveal_system_setting(db_path: Path, key: str) -> Dict[str, Any]:
    definition = DEFINITIONS_BY_KEY.get(key)
    if definition is None:
        raise KeyError(key)
    if not definition.secret:
        raise ValueError(f"{key} is not a secret setting")
    value = get_setting(key, "", db_path=db_path) or ""
    return {"key": key, "value": value, "has_value": bool(value.strip())}
