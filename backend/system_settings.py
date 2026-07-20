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
    ("kaggle_account", "Kaggle Account"),
    ("kaggle_kernel", "Kaggle Kernel"),
    ("kaggle_webhook", "Kaggle Webhook"),
    ("gemini", "Gemini"),
    ("video_asr", "Video/ASR"),
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
    SettingDefinition("GEMINI_API_KEY", "API key", "gemini", "Gemini", "string", required=True, secret=True),
    SettingDefinition("GEMINI_MODEL", "Primary model", "gemini", "Gemini", "string", default="gemini-1.5-flash-latest", required=True),
    SettingDefinition("GEMINI_FALLBACK_MODELS", "Fallback models", "gemini", "Gemini", "string", multiline=True),
    SettingDefinition("GEMINI_API_VERSION", "API version", "gemini", "Gemini", "string", default="v1beta"),
    SettingDefinition("GEMINI_MAX_TOKENS", "Max tokens", "gemini", "Gemini", "int", default="1024", min_value=1),
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
        has_override = definition.key in values
        raw_value = values.get(definition.key)
        if raw_value is None:
            raw_value = _env_or_default(definition)
        has_value = raw_value is not None and str(raw_value).strip() != ""
        setting: Dict[str, Any] = {
            "key": definition.key,
            "label": definition.label,
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
