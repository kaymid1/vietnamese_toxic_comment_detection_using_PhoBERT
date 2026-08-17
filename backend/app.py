import base64
import csv
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shlex
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import subprocess
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.runtime_paths import get_project_root


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("viet-toxic-backend")


# Resolve the project root and load environment files before importing local
# modules that materialize environment-dependent defaults at import time.
BASE_DIR = get_project_root()

from backend.env_loader import load_env_files

if load_env_files():
    logger.info("Loaded environment variables from .env files")


from domain_classifier import CATEGORY_THRESHOLDS
from backend.crawl_adapter import crawl_urls
from backend.system_settings import (
    ensure_system_settings_table,
    get_bool_setting as get_runtime_bool_setting,
    get_int_setting as get_runtime_int_setting,
    get_setting as get_runtime_setting,
    list_system_settings,
    reveal_system_setting,
    update_system_settings,
)
from backend.runtime_paths import (
    get_data_dir,
    get_feedback_db_path,
    get_kaggle_runtime_dir,
    get_model_options_dir,
    get_model_registry_dir,
)
from backend.artifact_refs import encode_artifact_ref, resolve_artifact_ref
from backend.mlflow_kaggle_ingest import (
    KaggleEvidenceConflictError,
    KaggleEvidenceIngestionUnavailable,
    KaggleEvidenceNotFound,
    KaggleEvidenceValidationError,
    get_kaggle_ingestion_record,
    ingest_kaggle_evidence,
    validate_kaggle_evidence,
)
from infer_crawled_local import infer_crawled, build_segment_hash, build_context_segment_hash


APP_DATA_DIR = get_data_dir()
PROCESSED_DATA_DIR = APP_DATA_DIR / "processed"
DATA_DIR = APP_DATA_DIR / "raw" / "crawled_urls"
MODEL_OPTIONS_DIR = get_model_options_dir()
FEEDBACK_DB_PATH = get_feedback_db_path()
FEEDBACK_DIR = FEEDBACK_DB_PATH.parent
EXPERIMENT_REGISTRY_PATH = BASE_DIR / "experiments" / "registry.json"
EVAL_POLICY_PATH = BASE_DIR / "config" / "eval_policy.json"
ERROR_ANALYSIS_PATH = PROCESSED_DATA_DIR / "error_analysis.json"
HARD_CASES_PATH = PROCESSED_DATA_DIR / "hard_case_candidates.json"
LOCAL_M1_ARTIFACT_DIR = PROCESSED_DATA_DIR / "mlflow_local_artifacts"

DEFAULT_DATASET_VERSION = os.getenv("VIETTOXIC_DATASET_VERSION", "victsd_gold")
DEFAULT_MODEL_VERSION = os.getenv("VIETTOXIC_MODEL_VERSION", "unknown")
DEFAULT_POLICY_VERSION = os.getenv("VIETTOXIC_POLICY_VERSION", "policy-v1")
REQUIRED_VERSION_KEYS = ("dataset_version", "model_version", "policy_version")

MLFLOW_ACCEPT_THRESHOLD = float(os.getenv("MLFLOW_ACCEPT_THRESHOLD", "0.8"))
MLFLOW_DISCARD_THRESHOLD = float(os.getenv("MLFLOW_DISCARD_THRESHOLD", "0.2"))
MLFLOW_THRESHOLD_TARGET_MAX = max(1, int(os.getenv("MLFLOW_THRESHOLD_TARGET_MAX", "10")))
MLFLOW_CLEAR_ALL_CONFIRM_TOKEN = os.getenv("MLFLOW_CLEAR_ALL_CONFIRM_TOKEN", "DELETE_ALL_MLFLOW_DATA")
ANALYZE_COLLECT_FOR_MLFLOW_DEFAULT = os.getenv("ANALYZE_COLLECT_FOR_MLFLOW_DEFAULT", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
KAGGLE_WEBHOOK_TIMEOUT_SEC = max(10, int(os.getenv("KAGGLE_WEBHOOK_TIMEOUT_SEC", "180")))
AUTOMATION_WATCHER_LOCK = threading.Lock()
AUTOMATION_WATCH_RUN_IDS: set[str] = set()
GEMINI_REQUEST_SLOT_LOCK = threading.Lock()
GEMINI_NEXT_REQUEST_AT = 0.0

DO_API_BASE = "https://api.digitalocean.com/v2"
DO_DEFAULT_REGION = "sgp1"
DO_DEFAULT_IMAGE = "ubuntu-24-04-x64"
DO_DEFAULT_GPU_SIZE = "gpu-h100x1-80gb"
DO_DEFAULT_TAG_PREFIX = "viettoxic-mlflow"
DO_SSH_KEY_IDS: List[str] = []
DO_SSH_PRIVATE_KEY_PATH = ""
DO_SSH_USER = "root"
DO_BOOTSTRAP_TIMEOUT_SEC = 600
DO_SPACES_BUCKET = ""
DO_SPACES_REGION = "sgp1"
DO_SPACES_KEY = ""
DO_SPACES_SECRET = ""
DO_SPACES_ENDPOINT = "https://sgp1.digitaloceanspaces.com"
DO_STAGES = [
    "trigger_vm_gpu",
    "upload_data_and_train_files",
    "train",
    "save_artifact",
    "destroy_vm",
]
LOCAL_M1_STAGES = [
    "prepare_local_bundle",
    "train_local_m1",
    "save_artifact",
    "finalize_local_run",
]

DATASET_VERSION_ALIASES: Dict[str, str] = {
    "latest": "victsd_gold",
    "victsd_gold": "victsd_gold",
}
DATASET_VERSION_DIRS: Dict[str, Path] = {
    "victsd_gold": PROCESSED_DATA_DIR / "victsd_gold",
}

MODEL_TYPES = {
    "phobert": {
        "required": ("config.json",),
        "required_any": ("model.safetensors", "pytorch_model.bin"),
    },
    "tfidf_lr": {
        "required": ("vectorizer.pkl", "model_lr.pkl"),
        "required_any": (),
    },
}

PHOBERT_V2_FINETUNED_STORAGE_NAME = "phobert_lora_4.7"
PHOBERT_V2_FINETUNED_NAME = "phobert_v2_finetuned"
PHOBERT_V2_FINETUNED_ID = f"phobert/{PHOBERT_V2_FINETUNED_NAME}"
PHOBERT_V2_FINETUNED_LEGACY_ID = f"phobert/{PHOBERT_V2_FINETUNED_STORAGE_NAME}"
PHOBERT_V2_BASE_MODEL = "vinai/phobert-base-v2"
PHOBERT_V1_BASELINE_NAME = "baseline"
PHOBERT_V1_BASELINE_ID = f"phobert/{PHOBERT_V1_BASELINE_NAME}"
MODEL_DISPLAY_NAMES = {
    "tfidf_lr/baseline_tfidf": "TF-IDF + Logistic Regression",
    "tfidf_lr/baseline_tfidf_sol": "TF-IDF + Logistic Regression (Gold + Constructiveness)",
    PHOBERT_V1_BASELINE_ID: "PhoBERT v1 Baseline",
    PHOBERT_V2_FINETUNED_ID: "PhoBERT v2 Fine-tuned",
    PHOBERT_V2_FINETUNED_LEGACY_ID: "PhoBERT v2 Fine-tuned",
}

TRAINING_TRACKER_DEFAULT_PHASES: List[Dict[str, Any]] = [
    {
        "id": "phase_0",
        "title": "Giai đoạn 0 — Cố định baseline",
        "tasks": [
            {"id": "p0_task_1", "label": "Giữ nguyên dataset split hiện tại"},
            {"id": "p0_task_2", "label": "Giữ nguyên seed cố định"},
            {"id": "p0_task_3", "label": "Record metric: macro_f1"},
            {"id": "p0_task_4", "label": "Record metric: f1_toxic"},
            {"id": "p0_task_5", "label": "Record metric: precision_toxic"},
            {"id": "p0_task_6", "label": "Record metric: recall_toxic"},
            {"id": "p0_task_7", "label": "Record metric: confusion matrix"},
            {"id": "p0_task_8", "label": "Record metric: best threshold theo macro_f1"},
            {"id": "p0_task_9", "label": "Record metric: best threshold theo f1_toxic"},
            {"id": "p0_task_10", "label": "Save metrics.json"},
            {"id": "p0_task_11", "label": "Save threshold_sweep_validation_raw.json"},
            {"id": "p0_task_12", "label": "Save error_analysis.json"},
            {"id": "p0_task_13", "label": "Save training_manifest.json"},
        ],
    },
    {
        "id": "phase_1",
        "title": "Giai đoạn 1 — Thử nghiệm nhanh",
        "groups": [
            {
                "id": "p1_group_11",
                "title": "1.1 Threshold tuning",
                "tasks": [
                    {"id": "p1_11_task_1", "label": "Chạy với PRIMARY_THRESHOLD_OBJECTIVE=macro_f1", "param": "PRIMARY_THRESHOLD_OBJECTIVE=macro_f1"},
                    {"id": "p1_11_task_2", "label": "Chạy với PRIMARY_THRESHOLD_OBJECTIVE=f1_toxic", "param": "PRIMARY_THRESHOLD_OBJECTIVE=f1_toxic"},
                    {"id": "p1_11_task_3", "label": "So sánh macro_f1, f1_toxic, precision_toxic, recall_toxic"},
                ],
            },
            {
                "id": "p1_group_12",
                "title": "1.2 Toxic weight scale",
                "tasks": [
                    {"id": "p1_12_task_1", "label": "Test TOXIC_WEIGHT_SCALE=0.5", "param": "TOXIC_WEIGHT_SCALE=0.5"},
                    {"id": "p1_12_task_2", "label": "Test TOXIC_WEIGHT_SCALE=0.75", "param": "TOXIC_WEIGHT_SCALE=0.75"},
                    {"id": "p1_12_task_3", "label": "Test TOXIC_WEIGHT_SCALE=1.0", "param": "TOXIC_WEIGHT_SCALE=1.0"},
                ],
            },
            {
                "id": "p1_group_13",
                "title": "1.3 Focal gamma",
                "tasks": [
                    {"id": "p1_13_task_1", "label": "Test FOCAL_GAMMA=1.5", "param": "FOCAL_GAMMA=1.5"},
                    {"id": "p1_13_task_2", "label": "Test FOCAL_GAMMA=2.0", "param": "FOCAL_GAMMA=2.0"},
                    {"id": "p1_13_task_3", "label": "Test FOCAL_GAMMA=2.5", "param": "FOCAL_GAMMA=2.5"},
                ],
            },
            {
                "id": "p1_group_14",
                "title": "1.4 Learning rate (full fine-tuning)",
                "tasks": [
                    {"id": "p1_14_task_1", "label": "Test LR=2e-5", "param": "LEARNING_RATE=2e-5"},
                    {"id": "p1_14_task_2", "label": "Test LR=5e-5", "param": "LEARNING_RATE=5e-5"},
                    {"id": "p1_14_task_3", "label": "Test LR=1e-4", "param": "LEARNING_RATE=1e-4"},
                ],
            },
        ],
    },
    {
        "id": "phase_2",
        "title": "Giai đoạn 2 — Pseudo-label",
        "tasks": [
            {"id": "p2_task_1", "label": "Chạy seed model trên unlabeled data, lưu prob_toxic"},
            {"id": "p2_task_2", "label": "Chia mẫu: low confidence toxic (0.50–0.60)"},
            {"id": "p2_task_3", "label": "Chia mẫu: medium confidence toxic (0.60–0.75)"},
            {"id": "p2_task_4", "label": "Chia mẫu: upper-medium toxic (0.75–0.85)"},
            {"id": "p2_task_5", "label": "Chia mẫu: very high confidence toxic (>0.85)"},
            {"id": "p2_task_6", "label": "Spot-check thủ công một phần nhỏ"},
            {"id": "p2_task_7", "label": "Loại mẫu quá ngắn, spam, url-only, duplicate"},
            {"id": "p2_task_8", "label": "Test PSEUDO_LOSS_WEIGHT=0.3", "param": "PSEUDO_LOSS_WEIGHT=0.3"},
            {"id": "p2_task_9", "label": "Test PSEUDO_LOSS_WEIGHT=0.5", "param": "PSEUDO_LOSS_WEIGHT=0.5"},
            {"id": "p2_task_10", "label": "Test MAX_PSEUDO_RATIO=0.2", "param": "MAX_PSEUDO_RATIO=0.2"},
            {"id": "p2_task_11", "label": "Test MAX_PSEUDO_RATIO=0.3", "param": "MAX_PSEUDO_RATIO=0.3"},
            {"id": "p2_task_12", "label": "Test MAX_PSEUDO_RATIO=0.4", "param": "MAX_PSEUDO_RATIO=0.4"},
        ],
    },
    {
        "id": "phase_3",
        "title": "Giai đoạn 3 — Hard toxic mining",
        "tasks": [
            {"id": "p3_task_1", "label": "Lấy false negatives từ error_analysis.json"},
            {"id": "p3_task_2", "label": "Review thủ công false negatives"},
            {"id": "p3_task_3", "label": "Tag lỗi: implicit toxic"},
            {"id": "p3_task_4", "label": "Tag lỗi: sarcasm/irony"},
            {"id": "p3_task_5", "label": "Tag lỗi: harassment nhẹ"},
            {"id": "p3_task_6", "label": "Tag lỗi: profanity biến thể"},
            {"id": "p3_task_7", "label": "Tag lỗi: slang/teencode"},
            {"id": "p3_task_8", "label": "Tag lỗi: context-dependent toxic"},
            {"id": "p3_task_9", "label": "Oversample hard toxic subset hoặc gán sample_weight cao hơn"},
        ],
    },
    {
        "id": "phase_4",
        "title": "Giai đoạn 4 — Data augmentation",
        "tasks": [
            {"id": "p4_task_1", "label": "Chỉ augment class toxic"},
            {"id": "p4_task_2", "label": "Augment: slang substitution"},
            {"id": "p4_task_3", "label": "Augment: teencode normalization / denormalization"},
            {"id": "p4_task_4", "label": "Augment: typo injection nhẹ"},
            {"id": "p4_task_5", "label": "Augment: paraphrase nhẹ"},
            {"id": "p4_task_6", "label": "Test: baseline không augment"},
            {"id": "p4_task_7", "label": "Test: toxic augment x1"},
            {"id": "p4_task_8", "label": "Test: toxic augment x2"},
        ],
    },
    {
        "id": "phase_5",
        "title": "Giai đoạn 5 — Full fine-tuning config",
        "tasks": [
            {"id": "p5_task_1", "label": "Test LR=1e-5", "param": "LEARNING_RATE=1e-5"},
            {"id": "p5_task_2", "label": "Test LR=2e-5", "param": "LEARNING_RATE=2e-5"},
            {"id": "p5_task_3", "label": "Test LR=3e-5", "param": "LEARNING_RATE=3e-5"},
            {"id": "p5_task_4", "label": "Test weight_decay=0.01", "param": "WEIGHT_DECAY=0.01"},
            {"id": "p5_task_5", "label": "Test weight_decay=0.05", "param": "WEIGHT_DECAY=0.05"},
            {"id": "p5_task_6", "label": "Test warmup_ratio=0.08", "param": "WARMUP_RATIO=0.08"},
            {"id": "p5_task_7", "label": "Test head_dropout=0.05", "param": "HEAD_DROPOUT=0.05"},
            {"id": "p5_task_8", "label": "Test head_dropout=0.1", "param": "HEAD_DROPOUT=0.1"},
            {"id": "p5_task_9", "label": "Test gradient accumulation=1", "param": "GRAD_ACCUM=1"},
            {"id": "p5_task_10", "label": "Test gradient accumulation=2", "param": "GRAD_ACCUM=2"},
        ],
    },
]

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    return get_runtime_setting(key, default, db_path=FEEDBACK_DB_PATH)


def get_int_setting(key: str, default: int, min_value: Optional[int] = None) -> int:
    return get_runtime_int_setting(key, default, db_path=FEEDBACK_DB_PATH, min_value=min_value)


def get_bool_setting(key: str, default: bool = False) -> bool:
    return get_runtime_bool_setting(key, default, db_path=FEEDBACK_DB_PATH)


def get_mlflow_bundle_min_rows() -> int:
    return get_int_setting(
        "MLFLOW_THRESHOLD_TARGET_MAX",
        MLFLOW_THRESHOLD_TARGET_MAX,
        min_value=1,
    )


DO_API_BASE = os.getenv("DO_API_BASE", "https://api.digitalocean.com/v2").rstrip("/")
DO_DEFAULT_REGION = os.getenv("DO_DEFAULT_REGION", "sgp1")
DO_DEFAULT_IMAGE = os.getenv("DO_DEFAULT_IMAGE", "ubuntu-24-04-x64")
DO_DEFAULT_GPU_SIZE = os.getenv("DO_DEFAULT_GPU_SIZE", "gpu-h100x1-80gb")
DO_DEFAULT_TAG_PREFIX = os.getenv("DO_DEFAULT_TAG_PREFIX", "viettoxic-mlflow")
DO_SSH_KEY_IDS = [k.strip() for k in os.getenv("DO_SSH_KEY_IDS", "").split(",") if k.strip()]
DO_SSH_PRIVATE_KEY_PATH = os.getenv("DO_SSH_PRIVATE_KEY_PATH", "").strip()
DO_SSH_USER = os.getenv("DO_SSH_USER", "root").strip() or "root"
DO_BOOTSTRAP_TIMEOUT_SEC = max(60, int(os.getenv("DO_BOOTSTRAP_TIMEOUT_SEC", "600")))
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET", "")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION", "sgp1")
DO_SPACES_KEY = os.getenv("DO_SPACES_KEY", "").strip()
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET", "").strip()
DO_SPACES_ENDPOINT = os.getenv("DO_SPACES_ENDPOINT", f"https://{DO_SPACES_REGION}.digitaloceanspaces.com").rstrip("/")
DO_DEFAULT_CPU_SIZE = os.getenv("DO_DEFAULT_CPU_SIZE", "s-16vcpu-32gb").strip()
DO_CPU_MIN_VCPUS = max(1, int(os.getenv("DO_CPU_MIN_VCPUS", "8")))
DO_CPU_MIN_MEMORY_MB = max(1024, int(os.getenv("DO_CPU_MIN_MEMORY_MB", "16384")))
DO_TRAIN_BASE_MINUTES = max(1, int(os.getenv("DO_TRAIN_BASE_MINUTES", "10")))
DO_TRAIN_ROWS_PER_MIN_GPU = max(1, int(os.getenv("DO_TRAIN_ROWS_PER_MIN_GPU", "320")))
DO_TRAIN_ROWS_PER_MIN_CPU = max(1, int(os.getenv("DO_TRAIN_ROWS_PER_MIN_CPU", "180")))
DO_TELEMETRY_INTERVAL_SEC = max(10, int(os.getenv("DO_TELEMETRY_INTERVAL_SEC", "30")))


def build_job_meta(
    job_id: str,
    urls: List[str],
    url_hashes: List[str],
    model_ids: List[str],
    enable_video: bool,
    merged_used: bool,
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "urls": urls,
        "url_hashes": url_hashes,
        "model_ids": model_ids,
        "enable_video": enable_video,
        "merged_used": merged_used,
    }


def save_job_meta(out_dir: Path, meta: Dict[str, Any]) -> None:
    try:
        (out_dir / "job_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write job_meta.json for %s", out_dir)


def load_job_meta(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "job_meta.json"
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _admin_config() -> Tuple[str, str, str, int]:
    username = os.getenv("VIETTOXIC_ADMIN_USERNAME", "").strip()
    password = os.getenv("VIETTOXIC_ADMIN_PASSWORD", "")
    secret = os.getenv("VIETTOXIC_ADMIN_SESSION_SECRET", "").strip()
    try:
        ttl_seconds = int(os.getenv("VIETTOXIC_ADMIN_SESSION_TTL_SECONDS", str(8 * 60 * 60)))
    except ValueError:
        ttl_seconds = 8 * 60 * 60
    ttl_seconds = max(300, ttl_seconds)
    missing = []
    if not username:
        missing.append("VIETTOXIC_ADMIN_USERNAME")
    if not password:
        missing.append("VIETTOXIC_ADMIN_PASSWORD")
    if not secret:
        missing.append("VIETTOXIC_ADMIN_SESSION_SECRET")
    if missing:
        raise HTTPException(status_code=503, detail=f"Admin auth is not configured; missing: {', '.join(missing)}")
    return username, password, secret, ttl_seconds


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def _sign_admin_payload(payload_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(digest)


def _create_admin_token(username: str, secret: str, ttl_seconds: int) -> Tuple[str, str]:
    now = int(time.time())
    expires_at_ts = now + ttl_seconds
    payload = {"sub": username, "iat": now, "exp": expires_at_ts}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign_admin_payload(payload_b64, secret)
    expires_at = datetime.fromtimestamp(expires_at_ts, timezone.utc).isoformat().replace("+00:00", "Z")
    return f"{payload_b64}.{signature}", expires_at


def _verify_admin_token(token: str) -> Tuple[str, str]:
    username, _, secret, _ = _admin_config()
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid admin token") from exc

    expected_signature = _sign_admin_payload(payload_b64, secret)
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid admin token")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid admin token") from exc

    try:
        token_username = str(payload.get("sub") or "")
        expires_at_ts = int(payload.get("exp") or 0)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid admin token") from exc
    if token_username != username:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    if expires_at_ts <= int(time.time()):
        raise HTTPException(status_code=401, detail="Admin session expired")

    expires_at = datetime.fromtimestamp(expires_at_ts, timezone.utc).isoformat().replace("+00:00", "Z")
    return token_username, expires_at


def _admin_session_from_authorization(authorization: Optional[str]) -> Tuple[str, str]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Admin authorization required")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Admin bearer token required")
    return _verify_admin_token(token.strip())


def require_admin(authorization: Optional[str] = Header(default=None)) -> str:
    username, _ = _admin_session_from_authorization(authorization)
    return username


app = FastAPI(title="VietComment Analyzer Local API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.ngrok-free\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Request %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error while processing request")
        raise
    logger.info("Response %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "VietComment Analyzer API is running. Use POST /api/analyze.",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


class AnalyzeOptions(BaseModel):
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=256, ge=16)
    page_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    seg_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    crawl_timeout_sec: int = Field(default=90, ge=30, le=300)
    max_load_more_clicks: int = Field(default=4, ge=0, le=30)
    max_comments_per_url: int = Field(default=50, ge=0, le=5000)
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    enable_video: bool = False
    selenium_fallback_mode: Literal["auto", "ask"] = "auto"
    collect_for_mlflow: bool = ANALYZE_COLLECT_FOR_MLFLOW_DEFAULT
    mlflow_gate_accept_threshold: float = Field(default=MLFLOW_ACCEPT_THRESHOLD, ge=0.0, le=1.0)
    mlflow_gate_discard_threshold: float = Field(default=MLFLOW_DISCARD_THRESHOLD, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    urls: List[str] = Field(min_items=1)
    options: Optional[AnalyzeOptions] = None


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class AdminLoginResponse(BaseModel):
    token: str
    expires_at: str
    username: str


class SystemSettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any] = Field(default_factory=dict)
    clear: List[str] = Field(default_factory=list)


class SystemSettingRevealRequest(BaseModel):
    key: str = Field(min_length=1)


@app.post("/api/admin/login", response_model=AdminLoginResponse)
def admin_login(request: AdminLoginRequest) -> Dict[str, str]:
    username, password, secret, ttl_seconds = _admin_config()
    if not hmac.compare_digest(request.username.strip(), username) or not hmac.compare_digest(request.password, password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    token, expires_at = _create_admin_token(username, secret, ttl_seconds)
    return {"token": token, "expires_at": expires_at, "username": username}


@app.get("/api/admin/session")
def admin_session(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    username, expires_at = _admin_session_from_authorization(authorization)
    return {"authenticated": True, "username": username, "expires_at": expires_at}


@app.get("/api/admin/system-settings", dependencies=[Depends(require_admin)])
def admin_system_settings() -> Dict[str, Any]:
    init_feedback_db()
    return list_system_settings(FEEDBACK_DB_PATH)


@app.patch("/api/admin/system-settings")
def admin_update_system_settings(
    request: SystemSettingsUpdateRequest,
    admin_username: str = Depends(require_admin),
) -> Dict[str, Any]:
    init_feedback_db()
    try:
        was_enabled = get_bool_setting("MLFLOW_AUTOMATION_ENABLED", False)
        was_dry_run = get_bool_setting("MLFLOW_AUTOMATION_DRY_RUN", True)
        payload = update_system_settings(
            FEEDBACK_DB_PATH,
            request.settings,
            clear=request.clear,
            updated_by=admin_username,
        )
        enabled_now = get_bool_setting("MLFLOW_AUTOMATION_ENABLED", False)
        scheduled: List[str] = []
        dry_run_now = get_bool_setting("MLFLOW_AUTOMATION_DRY_RUN", True)
        if enabled_now and (not was_enabled or (was_dry_run and not dry_run_now)):
            for family in ("tfidf_lr", "phobert"):
                if _automation_mode(family) == "disabled":
                    continue
                threading.Thread(
                    target=_run_automation_cycle,
                    args=(family, "settings_enabled"),
                    name=f"mlflow-automation-enable-{family}",
                    daemon=True,
                ).start()
                scheduled.append(family)
        payload["automation_scheduled_for"] = scheduled
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/system-settings/reveal-secret")
def admin_reveal_system_setting_secret(
    request: SystemSettingRevealRequest,
    _: str = Depends(require_admin),
) -> Dict[str, Any]:
    init_feedback_db()
    try:
        return reveal_system_setting(FEEDBACK_DB_PATH, request.key.strip())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown system setting key: {request.key}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class FeedbackPageItem(BaseModel):
    url: str
    url_hash: str
    html_tag: str
    html_tag_override: Optional[str] = None
    seg_threshold_used: Optional[float] = None
    score_overall: Optional[float] = None
    label: str


class FeedbackRequest(BaseModel):
    job_id: str
    model_id: str
    items: List[FeedbackPageItem] = Field(min_items=1)


class SegmentFeedbackItem(BaseModel):
    url: str
    url_hash: str
    model_id: str
    html_tag: str
    html_tag_override: Optional[str] = None
    segment_id: str
    text: str
    score: Optional[float] = None
    seg_threshold_used: Optional[float] = None
    label: str
    context_segment_hash: Optional[str] = None


class SegmentFeedbackRequest(BaseModel):
    job_id: str
    items: List[SegmentFeedbackItem] = Field(min_items=1)






class DatasetExportRequest(BaseModel):
    source: Optional[List[str]] = None
    label: Optional[List[int]] = None
    split: Optional[List[str]] = None
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    policy_version: Optional[str] = None


class FeedbackDeleteRequest(BaseModel):
    ids: List[int] = Field(min_items=1)


SyntheticDomain = Literal["education", "news", "politic"]
SyntheticStyle = Literal["formal", "informal"]


class SyntheticGenerateRequest(BaseModel):
    domain: SyntheticDomain
    style: SyntheticStyle
    label: int = Field(ge=0, le=1)
    constructiveness: Optional[int] = Field(default=None, ge=0, le=1)
    count: int = Field(default=10, ge=1, le=200)
    model: Optional[str] = None


class SyntheticReviewItem(BaseModel):
    id: int
    is_accepted: bool
    text: Optional[str] = None
    label: Optional[int] = Field(default=None, ge=0, le=1)
    constructiveness: Optional[int] = Field(default=None, ge=0, le=1)
    review_method: Literal["manual", "gemini_assisted"] = "manual"
    label_confidence: Optional[Literal["low", "medium", "high"]] = None
    review_provider: Optional[str] = Field(default=None, max_length=32)
    review_model_name: Optional[str] = Field(default=None, max_length=160)


class SyntheticReviewRequest(BaseModel):
    updates: List[SyntheticReviewItem] = Field(min_items=1)


class SyntheticTrainingPreviewTransferRequest(BaseModel):
    ids: List[int] = Field(min_items=1)


class SyntheticDeleteRequest(BaseModel):
    ids: List[int] = Field(min_items=1)


class SyntheticExportRequest(BaseModel):
    batch_id: Optional[str] = None
    domain: Optional[SyntheticDomain] = None
    style: Optional[SyntheticStyle] = None
    label: Optional[int] = Field(default=None, ge=0, le=1)
    accepted_only: bool = True


class TrainingTrackerCreatePhaseRequest(BaseModel):
    title: str = Field(min_length=1)


class TrainingTrackerUpdatePhaseRequest(BaseModel):
    title: str = Field(min_length=1)


class TrainingTrackerReorderPhasesRequest(BaseModel):
    phase_ids: List[str] = Field(min_items=1)


class TrainingTrackerCreateGroupRequest(BaseModel):
    phase_id: str = Field(min_length=1)
    title: str = Field(min_length=1)


class TrainingTrackerUpdateGroupRequest(BaseModel):
    title: str = Field(min_length=1)


class TrainingTrackerReorderGroupsRequest(BaseModel):
    phase_id: str = Field(min_length=1)
    group_ids: List[str] = Field(min_items=1)


class TrainingTrackerCreateTaskRequest(BaseModel):
    phase_id: str = Field(min_length=1)
    group_id: Optional[str] = None
    label: str = Field(min_length=1)
    param: Optional[str] = None


class TrainingTrackerUpdateTaskRequest(BaseModel):
    label: str = Field(min_length=1)
    param: Optional[str] = None


class TrainingTrackerReorderTasksRequest(BaseModel):
    phase_id: str = Field(min_length=1)
    group_id: Optional[str] = None
    task_ids: List[str] = Field(min_items=1)


class TrainingTrackerTaskCheckRequest(BaseModel):
    checked: bool


class TrainingTrackerCreateResultRequest(BaseModel):
    scenario_name: str = Field(min_length=1)
    phase_id: Optional[str] = None
    macro_f1: float
    f1_toxic: float
    precision_toxic: float
    recall_toxic: float
    val_loss: Optional[float] = None
    best_threshold_macro_f1: Optional[float] = None
    best_threshold_f1_toxic: Optional[float] = None
    notes: Optional[str] = None


class AnalyzeCompareOptions(AnalyzeOptions):
    model_names: List[str] = Field(min_items=2)


class AnalyzeCompareRequest(BaseModel):
    urls: List[str] = Field(min_items=1)
    options: AnalyzeCompareOptions


class AnalyzeRerunRequest(BaseModel):
    job_id: str
    model_name: Optional[str] = None
    options: Optional[AnalyzeOptions] = None
    prefer_merged: bool = True


class AskAIRequest(BaseModel):
    url: str
    html_tag: Optional[str] = None
    overall: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    question: Optional[str] = None


class MlflowIngestOptions(BaseModel):
    model_name: Optional[str] = None
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=256, ge=16)
    page_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    seg_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    crawl_timeout_sec: int = Field(default=90, ge=30, le=300)
    max_load_more_clicks: int = Field(default=4, ge=0, le=30)
    max_comments_per_url: int = Field(default=50, ge=0, le=5000)
    gate_accept_threshold: float = Field(default=MLFLOW_ACCEPT_THRESHOLD, ge=0.0, le=1.0)
    gate_discard_threshold: float = Field(default=MLFLOW_DISCARD_THRESHOLD, ge=0.0, le=1.0)
    persist_unused: bool = True


class MlflowIngestRequest(BaseModel):
    urls: List[str] = Field(min_items=1)
    options: Optional[MlflowIngestOptions] = None


class MlflowCandidateReviewItem(BaseModel):
    id: int
    action: Optional[Literal["include_toxic", "include_clean", "drop"]] = None
    decision: Optional[Literal["accept", "reject"]] = None
    pseudo_label: Optional[int] = Field(default=None, ge=0, le=1)
    constructiveness_label: Optional[int] = Field(default=None, ge=0, le=1)
    clear_constructiveness: bool = False
    lock_state: Optional[bool] = None
    label_source: Optional[str] = Field(default=None, max_length=64)
    label_confidence: Optional[str] = Field(default=None, max_length=32)
    reviewed_by_gemini: bool = False
    review_provider: Optional[str] = Field(default=None, max_length=32)
    review_model_name: Optional[str] = Field(default=None, max_length=160)


class MlflowCandidateReviewRequest(BaseModel):
    updates: List[MlflowCandidateReviewItem] = Field(min_items=1)


class MlflowManualExportBundleRequest(BaseModel):
    batch_id: Optional[str] = None
    scope: Literal["all_batches", "batch"] = "all_batches"
    bundle_profile: Literal["clean_victsd_gold", "full_bundle"] = "clean_victsd_gold"
    model_kind: Literal["phobert", "lr_smoke"] = "phobert"
    training_mode: Literal["retrain", "finetune"] = "finetune"
    balance_strategy: Literal["balanced_50_50", "all"] = "balanced_50_50"
    include_base_model: bool = False
    base_model: Optional[str] = None
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    policy_version: Optional[str] = None
    include_unused: bool = False
    unused_scope: Literal["all", "auto_discarded", "manual_rejected"] = "all"
    lineage_run_id: Optional[str] = Field(default=None, max_length=80)


class MlflowTrainingPreviewReviewItem(BaseModel):
    id: int
    selected_for_training: Optional[bool] = None
    pseudo_label: Optional[int] = Field(default=None, ge=0, le=1)
    constructiveness_label: Optional[int] = Field(default=None, ge=0, le=1)
    clear_constructiveness: bool = False
    lock_state: Optional[bool] = None
    label_source: Optional[str] = Field(default=None, max_length=64)
    label_confidence: Optional[str] = Field(default=None, max_length=32)
    reviewed_by_gemini: bool = False
    review_provider: Optional[str] = Field(default=None, max_length=32)
    review_model_name: Optional[str] = Field(default=None, max_length=160)


class MlflowTrainingPreviewReviewRequest(BaseModel):
    updates: List[MlflowTrainingPreviewReviewItem] = Field(min_length=1)


class MlflowTrainingPreviewGeminiReviewRequest(BaseModel):
    ids: List[int] = Field(min_length=1, max_length=25)


class MlflowModelReEvaluationRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=240)
    selection: Literal["selected", "all_auto_eligible"] = "selected"
    sample_ids: List[int] = Field(default_factory=list, max_length=300)
    training_scope: Literal["all_batches", "batch"] = "all_batches"
    batch_id: Optional[str] = None


class MlflowManualImportArtifactRequest(BaseModel):
    run_name: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    notes: Optional[str] = None


class MlflowDOTriggerRequest(BaseModel):
    batch_id: Optional[str] = None
    provider: Literal["kaggle"] = "kaggle"
    compute_mode: Literal["kaggle"] = "kaggle"
    model_kind: Literal["phobert", "lr_smoke"] = "phobert"
    training_mode: Literal["retrain", "finetune"] = "retrain"
    training_scope: Literal["light_only"] = "light_only"
    base_model: Optional[str] = None
    balance_strategy: Literal["balanced_50_50", "all"] = "balanced_50_50"
    bundle_scope: Literal["all_batches", "batch"] = "all_batches"
    dry_run: bool = True


class MlflowAutomationCycleRequest(BaseModel):
    model_family: Optional[Literal["tfidf_lr", "phobert"]] = None


class MlflowGeminiEvaluateRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=120)
    force: bool = False


class MlflowPromoteRequest(BaseModel):
    run_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    candidate_model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    artifact_checksum: Optional[str] = Field(default=None, min_length=64, max_length=64)
    expected_current_version: Optional[str] = Field(default=None, max_length=160)


class MlflowRollbackRequest(BaseModel):
    model_family: Literal["tfidf_lr", "phobert"]
    expected_current_version: Optional[str] = Field(default=None, max_length=160)


class MlflowRegistryLifecycleRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=240)
    confirm: bool = False


class MlflowClearBatchRequest(BaseModel):
    batch_id: str = Field(min_length=1)


class MlflowClearAllRequest(BaseModel):
    confirm_token: str = Field(min_length=1)


def list_model_types(model_root: Path) -> List[str]:
    if not model_root.exists():
        return []
    if not model_root.is_dir():
        raise NotADirectoryError(f"Model root path must be a directory: {model_root}")
    try:
        return sorted(
            [
                p.name
                for p in model_root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
        )
    except PermissionError as exc:
        raise PermissionError(f"Permission denied while reading model root: {model_root}") from exc


def list_models_by_type(model_root: Path, model_type: str) -> List[str]:
    if model_type not in MODEL_TYPES:
        return []
    base_dir = model_root / model_type
    if not base_dir.exists():
        return []
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Model base path must be a directory: {base_dir}")
    try:
        return sorted(
            [
                p.name
                for p in base_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
        )
    except PermissionError as exc:
        raise PermissionError(f"Permission denied while reading model directory: {base_dir}") from exc


def list_all_models(model_root: Path) -> List[Dict[str, str]]:
    models: List[Dict[str, str]] = []
    for model_type in list_model_types(model_root):
        for name in list_models_by_type(model_root, model_type):
            public_name = (
                PHOBERT_V2_FINETUNED_NAME
                if model_type == "phobert" and name == PHOBERT_V2_FINETUNED_STORAGE_NAME
                else name
            )
            models.append({
                "id": f"{model_type}/{public_name}",
                "type": model_type,
                "name": public_name,
            })
    return models


def _is_deprecated_model_name(name: str) -> bool:
    return "deprecated" in name.lower()


def _load_model_json(model_dir: Path, filename: str) -> Dict[str, Any]:
    path = model_dir / filename
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def get_phobert_base_model(model_dir: Path) -> Optional[str]:
    manifest = _load_model_json(model_dir, "training_manifest.json")
    manifest_hyperparams = manifest.get("hyperparams")
    if isinstance(manifest_hyperparams, dict):
        base_model = manifest_hyperparams.get("base_model")
        if isinstance(base_model, str) and base_model.strip():
            return base_model.strip()

    run_config = _load_model_json(model_dir, "run_config.json")
    run_hyperparameters = run_config.get("hyperparameters")
    if isinstance(run_hyperparameters, dict):
        base_model = run_hyperparameters.get("MODEL_NAME")
        if isinstance(base_model, str) and base_model.strip():
            return base_model.strip()
    return None


def _is_compatible_model(model_type: str, model_dir: Path) -> bool:
    try:
        validate_model_artifacts(model_type, model_dir)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def _read_production_slot_model_id(model_family: str) -> Optional[str]:
    if model_family not in {"phobert", "tfidf_lr"} or not FEEDBACK_DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'mlflow_production_slot'"
            ).fetchone()
            if not table_exists:
                return None
            row = conn.execute(
                "SELECT active_model_id FROM mlflow_production_slot WHERE model_family = ?",
                (model_family,),
            ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]).strip() if row and row[0] else None


def _read_production_slot_state(model_family: str) -> Dict[str, Optional[str]]:
    state: Dict[str, Optional[str]] = {
        "active_model_id": None,
        "previous_model_id": None,
        "updated_at": None,
    }
    if model_family not in {"phobert", "tfidf_lr"} or not FEEDBACK_DB_PATH.exists():
        return state
    try:
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            row = conn.execute(
                "SELECT active_model_id, previous_model_id, updated_at "
                "FROM mlflow_production_slot WHERE model_family = ?",
                (model_family,),
            ).fetchone()
    except sqlite3.Error:
        return state
    if row:
        state["active_model_id"] = str(row[0]).strip() if row[0] else None
        state["previous_model_id"] = str(row[1]).strip() if row[1] else None
        state["updated_at"] = str(row[2]).strip() if row[2] else None
    return state


def _get_default_model_id_without_slot(model_root: Path) -> Optional[str]:
    phobert_models = list_models_by_type(model_root, "phobert")
    preferred_dir = model_root / "phobert" / PHOBERT_V2_FINETUNED_STORAGE_NAME
    if (
        PHOBERT_V2_FINETUNED_STORAGE_NAME in phobert_models
        and get_phobert_base_model(preferred_dir) == PHOBERT_V2_BASE_MODEL
        and _is_compatible_model("phobert", preferred_dir)
    ):
        return PHOBERT_V2_FINETUNED_ID

    for name in phobert_models:
        if name in {PHOBERT_V2_FINETUNED_STORAGE_NAME, PHOBERT_V1_BASELINE_NAME}:
            continue
        if _is_deprecated_model_name(name):
            continue
        if _is_compatible_model("phobert", model_root / "phobert" / name):
            return f"phobert/{name}"

    legacy_dir = model_root / "phobert" / PHOBERT_V1_BASELINE_NAME
    if (
        PHOBERT_V1_BASELINE_NAME in phobert_models
        and _is_compatible_model("phobert", legacy_dir)
    ):
        return PHOBERT_V1_BASELINE_ID

    for model in list_all_models(model_root):
        model_type = str(model.get("type") or "")
        name = str(model.get("name") or "")
        model_id = str(model.get("id") or "")
        if not model_type or not name or not model_id:
            continue
        if model_id == PHOBERT_V2_FINETUNED_ID:
            continue
        storage_name = (
            PHOBERT_V2_FINETUNED_STORAGE_NAME
            if model_id == PHOBERT_V2_FINETUNED_ID
            else name
        )
        if _is_compatible_model(model_type, model_root / model_type / storage_name):
            return model_id
    return None


def get_family_default_model_id(model_root: Path, model_family: str) -> Optional[str]:
    recorded = _read_production_slot_model_id(model_family)
    if recorded:
        try:
            resolved_type, resolved_name, _ = resolve_model_path(model_root, recorded)
            if resolved_type == model_family:
                return f"{resolved_type}/{resolved_name}"
        except (FileNotFoundError, OSError, ValueError):
            logger.warning("Ignoring invalid %s production slot model: %s", model_family, recorded)

    if model_family == "phobert":
        fallback = _get_default_model_id_without_slot(model_root)
        if fallback and fallback.startswith("phobert/"):
            return fallback
        return None

    if model_family == "tfidf_lr":
        preferred = "tfidf_lr/baseline_tfidf"
        try:
            resolved_type, resolved_name, _ = resolve_model_path(model_root, preferred)
            return f"{resolved_type}/{resolved_name}"
        except (FileNotFoundError, OSError, ValueError):
            pass
        for name in list_models_by_type(model_root, "tfidf_lr"):
            if _is_deprecated_model_name(name):
                continue
            if _is_compatible_model("tfidf_lr", model_root / "tfidf_lr" / name):
                return f"tfidf_lr/{name}"
    return None


def get_default_model_id(model_root: Path) -> Optional[str]:
    return get_family_default_model_id(model_root, "phobert") or _get_default_model_id_without_slot(model_root)


def validate_model_artifacts(model_type: str, model_dir: Path) -> None:
    requirements = MODEL_TYPES.get(model_type)
    if not requirements:
        raise ValueError(f"Unsupported model type: {model_type}")

    missing = [name for name in requirements["required"] if not (model_dir / name).exists()]
    required_any = requirements["required_any"]
    if required_any and not any((model_dir / name).exists() for name in required_any):
        missing.append(" or ".join(required_any))

    if missing:
        files = sorted([p.name for p in model_dir.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f"Checkpoint folder missing: {', '.join(missing)}. Files found: {files}"
        )


MODEL_IMPORT_MAX_ZIP_BYTES = int(os.getenv("MODEL_IMPORT_MAX_ZIP_BYTES", str(512 * 1024 * 1024)))
MODEL_IMPORT_MAX_TOTAL_UNCOMPRESSED_BYTES = int(
    os.getenv("MODEL_IMPORT_MAX_TOTAL_UNCOMPRESSED_BYTES", str(2 * 1024 * 1024 * 1024))
)
MODEL_IMPORT_MAX_FILES = int(os.getenv("MODEL_IMPORT_MAX_FILES", "5000"))


def _sanitize_import_model_name(raw_name: str) -> str:
    name = (raw_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="model_name is required")
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="model_name is too long (max 80)")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(status_code=400, detail="model_name contains invalid characters")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid model_name")
    return name


def _validate_model_import_zip(zf: zipfile.ZipFile) -> None:
    members = zf.infolist()
    if len(members) == 0:
        raise HTTPException(status_code=400, detail="ZIP is empty")
    if len(members) > MODEL_IMPORT_MAX_FILES:
        raise HTTPException(status_code=400, detail="ZIP has too many files")

    total_uncompressed = 0
    for info in members:
        filename = info.filename or ""
        pure = Path(filename)
        if not filename or pure.is_absolute() or ".." in pure.parts:
            raise HTTPException(status_code=400, detail="ZIP contains unsafe path")

        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise HTTPException(status_code=400, detail="ZIP symlinks are not allowed")

        total_uncompressed += max(0, int(info.file_size))
        if total_uncompressed > MODEL_IMPORT_MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise HTTPException(status_code=400, detail="ZIP uncompressed size exceeds limit")


def _find_imported_model_dir(extracted_root: Path) -> Path:
    candidates: List[Path] = [extracted_root]
    for child in extracted_root.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            candidates.append(child)

    for candidate in candidates:
        try:
            validate_model_artifacts("phobert", candidate)
            return candidate
        except Exception:
            continue

    raise HTTPException(status_code=400, detail="ZIP does not contain a valid PhoBERT model directory")


def parse_model_id(model_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not model_id:
        return None, None
    if any(x in model_id for x in ("..", "\\")):
        raise ValueError(f"Invalid model name: {model_id}")
    if "/" not in model_id:
        return "phobert", model_id
    model_type, name = model_id.split("/", 1)
    if not model_type or not name or any(x in name for x in ("..", "/", "\\")):
        raise ValueError(f"Invalid model name: {model_id}")
    return model_type, name


def resolve_model_path(model_root: Path, model_id: Optional[str]) -> Tuple[str, str, Path]:
    if not model_id:
        model_id = get_default_model_id(model_root)

    if model_id is None:
        raise ValueError("No default model available")

    model_type, name = parse_model_id(model_id)
    if not model_type or not name:
        raise ValueError(f"Invalid model name: {model_id}")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if any(x in name for x in ("..", "/", "\\")):
        raise ValueError(f"Invalid model name: {model_id}")

    base_dir = model_root / model_type
    models = list_models_by_type(model_root, model_type)
    if not models:
        raise ValueError(f"No models found under {base_dir}")
    storage_name = name
    public_name = name
    if model_type == "phobert" and name in {
        PHOBERT_V2_FINETUNED_NAME,
        PHOBERT_V2_FINETUNED_STORAGE_NAME,
    }:
        storage_name = PHOBERT_V2_FINETUNED_STORAGE_NAME
        public_name = PHOBERT_V2_FINETUNED_NAME
    if storage_name not in models:
        raise ValueError(f"Model '{model_id}' not found. Available: {models}")

    model_path = base_dir / storage_name
    if not model_path.is_dir():
        raise ValueError(f"Model '{model_id}' is not a directory under {base_dir}")

    validate_model_artifacts(model_type, model_path)
    return model_type, public_name, model_path


def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def normalize_input_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip().strip("\"'`")
    if not cleaned:
        return None
    # Trim trailing punctuation often copied from text/log blocks.
    cleaned = cleaned.rstrip("),.;")
    if not cleaned:
        return None

    parsed = urllib.parse.urlparse(cleaned)
    if not parsed.scheme:
        cleaned = f"https://{cleaned}"
        parsed = urllib.parse.urlparse(cleaned)

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return None

    host = (parsed.hostname or "").strip().lower()
    if not host or "." not in host:
        return None

    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{host}{path}{query}"


def normalize_input_urls(raw_urls: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in raw_urls:
        url = normalize_input_url(raw)
        if not url or url in seen:
            continue
        normalized.append(url)
        seen.add(url)
    return normalized


def to_relative(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(BASE_DIR)
        return str(rel)
    except Exception:
        return str(path)


def resolve_model_root() -> Path:
    return MODEL_OPTIONS_DIR


def map_results_to_response(
    crawl_results: List[Dict[str, Any]],
    page_by_hash: Dict[str, Any],
    page_by_url: Dict[str, Any],
    seg_by_hash: Dict[str, List[Dict[str, Any]]],
    seg_by_url: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    response_results: List[Dict[str, Any]] = []
    for crawl in crawl_results:
        url = crawl.get("url")
        url_hash = crawl.get("url_hash") or hash_url(url)
        status = crawl.get("status", "error")
        error = crawl.get("error")

        segments_path = crawl.get("segments_path")
        if status == "ok" and (not segments_path or not Path(segments_path).exists()):
            logger.warning(
                "segments missing for url=%s path=%s",
                url,
                segments_path,
            )
            status = "error"
            error = "segments.jsonl not found after crawl"

        page_info = None
        if status == "ok":
            page_info = page_by_hash.get(url_hash) or page_by_url.get(url)
            if not page_info:
                logger.warning("no inference result for url=%s hash=%s", url, url_hash)
                status = "error"
                error = "No inference result for this URL"

        overall = None
        overall_constructiveness = None
        if page_info:
            overall = normalize_score(page_info.get("avg_toxic_prob"))
            if overall is None:
                overall = normalize_score(page_info.get("toxic_ratio"))
            overall_constructiveness = normalize_score(page_info.get("avg_constructiveness_prob"))

        segment_entries = seg_by_hash.get(url_hash) or seg_by_url.get(url) or []
        by_segment = []
        by_segment_constructiveness = []
        constructiveness_present_count = 0
        constructiveness_label_present_count = 0
        constructiveness_positive_count = 0
        for idx, seg in enumerate(segment_entries):
            score = normalize_score(seg.get("toxic_prob"))
            constructiveness_score = normalize_score(seg.get("constructiveness_prob"))
            constructiveness_label = normalize_int(seg.get("constructiveness_label"))
            text = seg.get("text") or seg.get("text_preview") or ""
            if constructiveness_score is not None:
                constructiveness_present_count += 1
            if constructiveness_label is not None:
                constructiveness_label_present_count += 1
                if constructiveness_label == 1:
                    constructiveness_positive_count += 1
            by_segment.append(
                {
                    "segment_id": f"{url_hash}:{idx}",
                    "score": score if score is not None else 0.0,
                    "text_preview": text[:160],
                    "text": text,
                    "html_tags": seg.get("html_tags"),
                    "og_types": seg.get("og_types"),
                    "ai_learned": seg.get("ai_learned"),
                    "ai_learned_label": seg.get("ai_learned_label"),
                    "segment_hash": seg.get("segment_hash"),
                    "context_segment_hash": seg.get("context_segment_hash"),
                    "toxic_label": seg.get("toxic_label"),
                    "constructiveness_score": constructiveness_score,
                    "constructiveness_label": constructiveness_label,
                    "toxic_prob_adjusted": normalize_score(seg.get("toxic_prob_adjusted")),
                    "ai_learned_mode": seg.get("ai_learned_mode"),
                    "learned_support": seg.get("learned_support"),
                    "learned_agreement": normalize_score(seg.get("learned_agreement")),
                    "seg_threshold_used": normalize_score(seg.get("seg_threshold_used")),
                }
            )
            if constructiveness_score is not None:
                by_segment_constructiveness.append(
                    {
                        "segment_id": f"{url_hash}:{idx}",
                        "score": constructiveness_score,
                        "text_preview": text[:160],
                        "text": text,
                        "constructiveness_label": constructiveness_label,
                        "html_tags": seg.get("html_tags"),
                        "og_types": seg.get("og_types"),
                        "segment_hash": seg.get("segment_hash"),
                        "context_segment_hash": seg.get("context_segment_hash"),
                    }
                )

        constructiveness_available = (
            overall_constructiveness is not None or constructiveness_present_count > 0
        )
        if not segment_entries:
            constructiveness_missing_reason = "no_segments"
        elif constructiveness_available:
            constructiveness_missing_reason = None
        else:
            constructiveness_missing_reason = "constructiveness_not_emitted_by_inference"

        response_results.append(
            {
                "url": url,
                "url_hash": url_hash,
                "domain_category": page_info.get("domain_category") if page_info else None,
                "status": status,
                "crawl_status": crawl.get("crawl_status"),
                "error": error,
                "warnings": crawl.get("warnings") or [],
                "comment_cap_hit": bool(crawl.get("comment_cap_hit") or False),
                "max_comments_per_url": normalize_int(crawl.get("max_comments_per_url")),
                "crawled_comment_count": normalize_int(crawl.get("num_segments")),
                "crawl_output_dir": to_relative(crawl.get("output_dir")),
                "segments_path": to_relative(segments_path),
                "videos": [],
                "html_tags": page_info.get("html_tags") if page_info else None,
                "og_types": page_info.get("og_types") if page_info else None,
                "seg_threshold_used": normalize_score(page_info.get("seg_threshold_used")) if page_info else None,
                "page_toxic": normalize_int(page_info.get("page_toxic")) if page_info else None,
                "toxicity": {
                    "overall": overall,
                    "by_segment": by_segment,
                },
                "constructiveness": {
                    "overall": overall_constructiveness,
                    "by_segment": by_segment_constructiveness,
                    "meta": {
                        "available": constructiveness_available,
                        "threshold": 0.5,
                        "total_segments": len(segment_entries),
                        "segments_with_score": constructiveness_present_count,
                        "segments_without_score": max(0, len(segment_entries) - constructiveness_present_count),
                        "segments_with_label": constructiveness_label_present_count,
                        "constructive_segments": constructiveness_positive_count,
                        "non_constructive_segments": max(0, constructiveness_label_present_count - constructiveness_positive_count),
                        "missing_reason": constructiveness_missing_reason,
                    },
                },
            }
        )

    return response_results


def gemini_base_url(api_version: str, api_key: str, suffix: str = "") -> str:
    trimmed = suffix.lstrip("/")
    if trimmed:
        trimmed = f"/{trimmed}"
    return f"https://generativelanguage.googleapis.com/{api_version}{trimmed}?key={api_key}"


def normalize_gemini_model_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.startswith("models/"):
        cleaned = cleaned.split("/", 1)[1]
    return cleaned or None


class GeminiTextResponse(str):
    def __new__(
        cls,
        value: str,
        *,
        model: str,
        usage_metadata: Optional[Dict[str, Any]] = None,
    ) -> "GeminiTextResponse":
        instance = str.__new__(cls, value)
        instance.provider = "gemini"
        instance.model = model
        instance.usage_metadata = usage_metadata or {}
        return instance


class GeminiRequestFailure(Exception):
    def __init__(self, status_code: Optional[int], detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def get_gemini_model_candidates() -> List[str]:
    primary = normalize_gemini_model_name(get_setting("GEMINI_MODEL", "gemini-1.5-flash-latest"))
    fallback_raw = get_setting("GEMINI_FALLBACK_MODELS", "") or ""
    tokens = fallback_raw.replace(";", ",").replace("|", ",").split(",") if fallback_raw else []
    fallbacks = [normalize_gemini_model_name(token) for token in tokens if token.strip()]
    candidates: List[str] = []
    for name in [primary, *fallbacks]:
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def is_gemini_rate_limited(status_code: int, detail: str) -> bool:
    if status_code in {429, 503}:
        return True
    lowered = detail.lower()
    if (
        "resource_exhausted" in lowered
        or "rate limit" in lowered
        or "quota" in lowered
        or "status\": \"unavailable\"" in lowered
        or "high demand" in lowered
    ):
        return True
    return False


def is_gemini_daily_quota_exhausted(detail: str) -> bool:
    lowered = detail.lower()
    return any(
        marker in lowered
        for marker in (
            "per_day",
            "perday",
            "per day",
            "requests per day",
            "daily quota",
            "rpd",
        )
    )


def wait_for_gemini_request_slot() -> None:
    global GEMINI_NEXT_REQUEST_AT
    interval = get_int_setting("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", 13, min_value=0)
    if interval <= 0:
        return
    with GEMINI_REQUEST_SLOT_LOCK:
        now = time.monotonic()
        wait_seconds = max(0.0, GEMINI_NEXT_REQUEST_AT - now)
        if wait_seconds > 0:
            logger.info("Waiting %.1fs for the Gemini request rate window", wait_seconds)
            time.sleep(wait_seconds)
        GEMINI_NEXT_REQUEST_AT = time.monotonic() + float(interval)


def request_gemini_raw(req: urllib.request.Request, *, model: str, timeout: int) -> str:
    attempts = min(4, get_int_setting("GEMINI_RETRY_ATTEMPTS", 2, min_value=1))
    last_failure: Optional[GeminiRequestFailure] = None
    for attempt in range(1, attempts + 1):
        wait_for_gemini_request_slot()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp else str(exc)
            last_failure = GeminiRequestFailure(exc.code, detail)
            retryable = is_gemini_rate_limited(exc.code, detail) and not is_gemini_daily_quota_exhausted(detail)
            if retryable and attempt < attempts:
                logger.warning(
                    "Gemini transient error on %s; waiting for the next request slot (%s/%s)",
                    model,
                    attempt + 1,
                    attempts,
                )
                continue
            break
        except urllib.error.URLError as exc:
            last_failure = GeminiRequestFailure(None, str(exc))
            if attempt < attempts:
                logger.warning(
                    "Gemini network error on %s; waiting for the next request slot (%s/%s)",
                    model,
                    attempt + 1,
                    attempts,
                )
                continue
            break
    raise last_failure or GeminiRequestFailure(None, "Unknown Gemini request error")


def call_gemini(prompt: str) -> str:
    api_key = get_setting("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    api_version = get_setting("GEMINI_API_VERSION", "v1beta") or "v1beta"
    max_tokens = get_int_setting("GEMINI_MAX_TOKENS", 1024, min_value=1)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": max_tokens,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    candidates = get_gemini_model_candidates()
    if not candidates:
        raise HTTPException(status_code=400, detail="Missing GEMINI_MODEL")

    last_error: Optional[str] = None
    for idx, model in enumerate(candidates):
        url = gemini_base_url(api_version, api_key, f"models/{model}:generateContent")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = request_gemini_raw(req, model=model, timeout=30)
        except GeminiRequestFailure as exc:
            last_error = exc.detail
            if exc.status_code == 404 and idx < len(candidates) - 1:
                logger.warning("Gemini model not found: %s", model)
                continue
            if exc.status_code is not None and is_gemini_rate_limited(exc.status_code, exc.detail) and idx < len(candidates) - 1:
                logger.warning("Gemini rate limited on %s, trying fallback", model)
                continue
            if exc.status_code is None and idx < len(candidates) - 1:
                logger.warning("Gemini network error on %s, trying fallback", model)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini API error: {exc.detail}") from exc

        try:
            parsed = json.loads(raw)
            model_candidates = parsed.get("candidates") or []
            if not model_candidates:
                raise ValueError("No candidates returned")
            parts = model_candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No content parts returned")
            text = "\n".join([p.get("text", "") for p in parts if p.get("text")])
            return GeminiTextResponse(text, model=model, usage_metadata=parsed.get("usageMetadata"))
        except Exception as exc:
            last_error = str(exc)
            raise HTTPException(status_code=502, detail=f"Gemini response parse error: {exc}") from exc

    raise HTTPException(status_code=502, detail=f"Gemini API error: {last_error or 'Unknown error'}")


def list_gemini_models() -> Dict[str, Any]:
    api_key = get_setting("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    api_version = get_setting("GEMINI_API_VERSION", "v1beta") or "v1beta"
    url = gemini_base_url(api_version, api_key, "models")
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini response parse error: {exc}") from exc

    return {
        "api_version": api_version,
        "model": get_setting("GEMINI_MODEL", "gemini-1.5-flash-latest"),
        "fallback_models": get_gemini_model_candidates()[1:],
        "max_tokens": get_int_setting("GEMINI_MAX_TOKENS", 1024, min_value=1),
        "min_request_interval_seconds": get_int_setting("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", 13, min_value=0),
        "retry_attempts": min(4, get_int_setting("GEMINI_RETRY_ATTEMPTS", 2, min_value=1)),
        "review_max_items": min(25, get_int_setting("GEMINI_REVIEW_MAX_ITEMS", 9, min_value=1)),
        "models": parsed.get("models", []),
    }


def load_page_results(out_dir: Path) -> List[Dict[str, Any]]:
    # TODO: expand parser for additional output formats if infer changes its schema.
    json_path = out_dir / "page_level_results.json"
    csv_path = out_dir / "page_level_results.csv"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                return [row for row in reader]
        except Exception:
            return []
    return []


def load_page_results_map(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    results = load_page_results(out_dir)
    by_hash = {r.get("url_hash"): r for r in results if r.get("url_hash")}
    by_url = {r.get("url"): r for r in results if r.get("url")}
    return by_hash, by_url


def load_segment_results(out_dir: Path) -> List[Dict[str, Any]]:
    seg_path = out_dir / "crawled_predictions.jsonl"
    if not seg_path.exists():
        return []
    results: List[Dict[str, Any]] = []
    with seg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def normalize_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


SYNTHETIC_PROMPT_VERSION = "v1"
PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+\]|<[^>]+>|\{[^}]+\}")
SYNTHETIC_FALLBACK_MODEL = "gemini-1.5-flash-latest"
SYNTHETIC_MAX_RETRIES = 3
SYNTHETIC_LENGTH_BUCKET_ORDER = ["very_short", "short_medium", "medium_long", "long"]
SYNTHETIC_LENGTH_BUCKET_RATIOS: Dict[str, float] = {
    "very_short": 0.20,
    "short_medium": 0.40,
    "medium_long": 0.30,
    "long": 0.10,
}
SYNTHETIC_LENGTH_DEFAULT_BOUNDS = (8, 18, 32)
_SYNTHETIC_LENGTH_BOUNDS_CACHE: Optional[Tuple[int, int, int]] = None


def normalize_synthetic_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def synthetic_word_length(text: str) -> int:
    normalized = normalize_synthetic_text(text)
    if not normalized:
        return 0
    return len(normalized.split(" "))


def quantile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    pos = (len(sorted_values) - 1) * q
    lower_idx = int(math.floor(pos))
    upper_idx = int(math.ceil(pos))
    if lower_idx == upper_idx:
        return float(sorted_values[lower_idx])
    weight = pos - lower_idx
    return (1.0 - weight) * sorted_values[lower_idx] + weight * sorted_values[upper_idx]


def get_synthetic_length_bounds() -> Tuple[int, int, int]:
    global _SYNTHETIC_LENGTH_BOUNDS_CACHE
    if _SYNTHETIC_LENGTH_BOUNDS_CACHE is not None:
        return _SYNTHETIC_LENGTH_BOUNDS_CACHE

    lengths: List[int] = []
    source_files = [
        PROCESSED_DATA_DIR / "victsd_gold" / "train.jsonl",
        PROCESSED_DATA_DIR / "victsd_gold" / "validation.jsonl",
        PROCESSED_DATA_DIR / "victsd_gold" / "test.jsonl",
    ]

    for file_path in source_files:
        if not file_path.exists():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    text = str(row.get("text") or "")
                    length = synthetic_word_length(text)
                    if length > 0:
                        lengths.append(length)
        except Exception:
            logger.warning("Failed reading ViCTSD length source: %s", file_path)

    if not lengths:
        _SYNTHETIC_LENGTH_BOUNDS_CACHE = SYNTHETIC_LENGTH_DEFAULT_BOUNDS
        return _SYNTHETIC_LENGTH_BOUNDS_CACHE

    lengths.sort()
    q20 = int(round(quantile(lengths, 0.20)))
    q60 = int(round(quantile(lengths, 0.60)))
    q90 = int(round(quantile(lengths, 0.90)))

    b1 = max(1, q20)
    b2 = max(b1 + 1, q60)
    b3 = max(b2 + 1, q90)
    _SYNTHETIC_LENGTH_BOUNDS_CACHE = (b1, b2, b3)
    return _SYNTHETIC_LENGTH_BOUNDS_CACHE


def classify_synthetic_length_bucket(length_words: int, bounds: Tuple[int, int, int]) -> str:
    b1, b2, b3 = bounds
    if length_words <= b1:
        return "very_short"
    if length_words <= b2:
        return "short_medium"
    if length_words <= b3:
        return "medium_long"
    return "long"


def build_length_bucket_targets(total_count: int) -> Dict[str, int]:
    if total_count <= 0:
        return {key: 0 for key in SYNTHETIC_LENGTH_BUCKET_ORDER}

    targets: Dict[str, int] = {}
    fractions: List[Tuple[float, str]] = []
    assigned = 0
    for key in SYNTHETIC_LENGTH_BUCKET_ORDER:
        raw = total_count * SYNTHETIC_LENGTH_BUCKET_RATIOS[key]
        base = int(math.floor(raw))
        targets[key] = base
        assigned += base
        fractions.append((raw - base, key))

    remainder = total_count - assigned
    for _, key in sorted(fractions, key=lambda item: item[0], reverse=True):
        if remainder <= 0:
            break
        targets[key] += 1
        remainder -= 1

    return targets


def build_length_bucket_guidance(targets: Dict[str, int], bounds: Tuple[int, int, int]) -> str:
    b1, b2, b3 = bounds
    return (
        "Phân bổ độ dài bắt buộc theo số từ gần giống ViCTSD:\n"
        f"- very_short (<= {b1} từ): {targets.get('very_short', 0)} mẫu\n"
        f"- short_medium ({b1 + 1}-{b2} từ): {targets.get('short_medium', 0)} mẫu\n"
        f"- medium_long ({b2 + 1}-{b3} từ): {targets.get('medium_long', 0)} mẫu\n"
        f"- long (> {b3} từ): {targets.get('long', 0)} mẫu"
    )


def build_structure_fingerprint(text: str) -> str:
    normalized = normalize_synthetic_text(text).lower()
    skeleton = re.sub(r"\d+", "<num>", normalized)
    skeleton = re.sub(r"\b[a-zA-ZÀ-ỹ]{1,2}\b", "<w>", skeleton)
    skeleton = re.sub(r"[a-zA-ZÀ-ỹ]+", "<tok>", skeleton)
    skeleton = re.sub(r"\s+", " ", skeleton).strip()
    return hashlib.sha256(skeleton.encode("utf-8")).hexdigest()


def build_text_hash(text: str) -> str:
    normalized = normalize_synthetic_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_synthetic_meta(
    *,
    sample_id: int,
    batch_id: str,
    domain: str,
    style: str,
    model_name: str,
    created_at: str,
) -> Dict[str, Any]:
    return {
        "source": "synthetic_llm",
        "split": "synthetic",
        "is_augmented": True,
        "sample_id": sample_id,
        "batch_id": batch_id,
        "domain": domain,
        "style": style,
        "generator_model": model_name,
        "prompt_version": SYNTHETIC_PROMPT_VERSION,
        "created_at": created_at,
    }


def parse_json_array_from_llm(raw: str) -> List[Dict[str, Any]]:
    cleaned = (raw or "").strip()
    if not cleaned:
        return []

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    def extract_items(parsed: Any) -> List[Dict[str, Any]]:
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("items", "samples", "data", "rows", "results"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def extract_objects_from_text(text: str) -> List[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        items: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(text):
            start_obj = text.find("{", idx)
            if start_obj == -1:
                break
            try:
                parsed_obj, end_obj = decoder.raw_decode(text[start_obj:])
            except json.JSONDecodeError:
                idx = start_obj + 1
                continue
            if isinstance(parsed_obj, dict):
                items.append(parsed_obj)
            idx = start_obj + max(end_obj, 1)
        return items

    try:
        parsed_direct = json.loads(cleaned)
        direct_items = extract_items(parsed_direct)
        if direct_items:
            return direct_items
    except json.JSONDecodeError:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return extract_objects_from_text(cleaned)

    payload = cleaned[start : end + 1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return extract_objects_from_text(payload)

    return extract_items(parsed)


def build_mlflow_gemini_review_prompt(rows: List[sqlite3.Row]) -> str:
    instruction = (
        get_setting(
            "GEMINI_REVIEW_INSTRUCTION",
            "Bạn là reviewer dữ liệu tiếng Việt cho bài toán toxic-content detection.",
        )
        or ""
    ).strip()
    payload = []
    for row in rows:
        payload.append(
            {
                "id": int(row["id"]),
                "text": str(row["text"] or "")[:1200],
                "model_score": row["score"],
                "current_toxicity_label": normalize_int(row["pseudo_label"]),
                "constructiveness_score": row["constructiveness_score"],
                "current_constructiveness_label": normalize_int(row["constructiveness_label"]),
                "gate_bucket": row["gate_bucket"],
                "domain_category": row["domain_category"],
                "url": row["url"],
            }
        )
    return (
        f"{instruction}\n"
        "Hãy review từng comment và chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.\n"
        f"Phải trả về đúng {len(payload)} object, mỗi id đầu vào xuất hiện đúng một lần; không được bỏ sót comment.\n"
        "Schema bắt buộc: một JSON array, mỗi object gồm:\n"
        "- id: number, đúng id đầu vào\n"
        "- toxicity_label: 0 hoặc 1, với 1 = toxic, 0 = clean\n"
        "- constructiveness_label: 0, 1, hoặc null; dùng null nếu không đủ chắc chắn\n"
        "- confidence: low, medium, hoặc high\n"
        "- reason: chuỗi ngắn tối đa 160 ký tự bằng tiếng Việt\n"
        "- action: apply hoặc review_more; dùng review_more nếu mơ hồ/thiếu ngữ cảnh\n"
        "Ưu tiên chất lượng nhãn training hơn tốc độ. Không tự cân bằng toxic/clean.\n"
        "Dữ liệu cần review:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def normalize_gemini_review_suggestions(raw: str, expected_ids: set[int]) -> List[Dict[str, Any]]:
    parsed_items = parse_json_array_from_llm(raw)
    suggestions: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in parsed_items:
        row_id = normalize_int(item.get("id"))
        if row_id is None or row_id not in expected_ids or row_id in seen_ids:
            continue

        toxicity = normalize_int(item.get("toxicity_label"))
        if toxicity not in {0, 1}:
            toxicity = normalize_int(item.get("pseudo_label"))
        if toxicity not in {0, 1}:
            continue

        constructiveness = normalize_int(item.get("constructiveness_label"))
        if constructiveness not in {0, 1}:
            constructiveness = None

        confidence = str(item.get("confidence") or "").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        action = str(item.get("action") or "").strip().lower()
        if action not in {"apply", "review_more"}:
            action = "apply" if confidence == "high" else "review_more"

        reason = str(item.get("reason") or "").strip()
        if len(reason) > 180:
            reason = reason[:177].rstrip() + "..."

        suggestions.append(
            {
                "id": row_id,
                "toxicity_label": int(toxicity),
                "constructiveness_label": constructiveness,
                "confidence": confidence,
                "reason": reason,
                "action": action,
            }
        )
        seen_ids.add(row_id)

    if not suggestions:
        raise HTTPException(status_code=502, detail="Gemini response did not contain valid review suggestions")
    return suggestions


GEMINI_REVIEW_BATCH_SIZE = 3
GEMINI_REVIEW_JSON_ATTEMPTS = 2


def validate_gemini_review_item_limit(ids: List[int]) -> None:
    configured = get_int_setting("GEMINI_REVIEW_MAX_ITEMS", 9, min_value=1)
    maximum = min(25, configured)
    if len(ids) > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"Gemini Review accepts at most {maximum} comments per operation to protect the API rate window",
        )


def build_gemini_review_response(suggestions: List[Dict[str, Any]], requested: int) -> Dict[str, Any]:
    models = sorted({str(item.get("model") or "").strip() for item in suggestions if item.get("model")})
    return {
        "status": "ok",
        "provider": "gemini",
        "model": models[0] if len(models) == 1 else None,
        "models": models,
        "suggestions": suggestions,
        "requested": requested,
        "reviewed": len(suggestions),
    }


def request_mlflow_gemini_review_chunk(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    expected_ids = {int(row["id"]) for row in rows}
    suggestions: Optional[List[Dict[str, Any]]] = None
    for attempt in range(1, GEMINI_REVIEW_JSON_ATTEMPTS + 1):
        raw = call_gemini(build_mlflow_gemini_review_prompt(rows))
        try:
            suggestions = normalize_gemini_review_suggestions(raw, expected_ids)
            actual_model = getattr(raw, "model", None) or get_setting("GEMINI_MODEL", "gemini-1.5-flash-latest")
            for suggestion in suggestions:
                suggestion["provider"] = getattr(raw, "provider", "gemini")
                suggestion["model"] = actual_model
            break
        except HTTPException:
            if attempt < GEMINI_REVIEW_JSON_ATTEMPTS:
                logger.warning(
                    "Gemini returned an invalid review response for %s row(s); retrying (%s/%s)",
                    len(rows),
                    attempt + 1,
                    GEMINI_REVIEW_JSON_ATTEMPTS,
                )

    if suggestions is None:
        if len(rows) == 1:
            logger.warning("Gemini could not produce a valid review for row id=%s after %s attempts", rows[0]["id"], GEMINI_REVIEW_JSON_ATTEMPTS)
            return []
        logger.warning("Gemini returned an invalid review batch; retrying %s rows individually", len(rows))
        return [
            suggestion
            for row in rows
            for suggestion in request_mlflow_gemini_review_chunk([row])
        ]

    returned_ids = {int(item["id"]) for item in suggestions}
    missing_rows = [row for row in rows if int(row["id"]) not in returned_ids]
    if missing_rows:
        logger.warning(
            "Gemini omitted %s/%s review rows; retrying the missing rows individually",
            len(missing_rows),
            len(rows),
        )
        for row in missing_rows:
            suggestions.extend(request_mlflow_gemini_review_chunk([row]))
    return suggestions


def run_mlflow_gemini_review(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for start in range(0, len(rows), GEMINI_REVIEW_BATCH_SIZE):
        chunk = rows[start : start + GEMINI_REVIEW_BATCH_SIZE]
        suggestions.extend(request_mlflow_gemini_review_chunk(chunk))
    return suggestions


def build_synthetic_prompt(
    domain: str,
    style: str,
    label: int,
    count: int,
    constructiveness: Optional[int] = None,
    length_guidance: Optional[str] = None,
) -> str:
    toxicity = "toxic" if label == 1 else "clean"
    constructiveness_text = (
        "masked/unknown" if constructiveness is None else ("constructive" if constructiveness == 1 else "non-constructive")
    )
    guidance = f"\n7) {length_guidance}" if length_guidance else ""
    return (
        "Bạn là hệ thống tạo dữ liệu tiếng Việt cho phân loại toxic. "
        "Hãy tạo đúng số lượng mẫu theo yêu cầu và trả về JSON array hợp lệ, không có text ngoài JSON.\n"
        f"Yêu cầu: domain={domain}, style={style}, toxicity={label} ({toxicity}), số mẫu={count}.\n"
        "Mỗi phần tử bắt buộc có schema: {\"text\": string, \"label\": 0|1, \"toxicity\": 0|1, \"constructiveness\": 0|1|null, \"meta\": object}.\n"
        "Trả về JSON compact, không markdown, không giải thích.\n"
        "Ràng buộc bắt buộc:\n"
        "1) Không lặp cấu trúc câu giữa các mẫu.\n"
        "2) Không dùng placeholder dạng [tên], [trường], <name>, {city}.\n"
        "3) Phải dùng tên/tổ chức cụ thể giả định (vd: Trường THPT Nguyễn Trãi, GS. Nguyễn Văn A).\n"
        "4) Dữ liệu phải tự nhiên, đúng tiếng Việt.\n"
        "5) meta phải chứa source=\"synthetic_llm\", domain, style.\n"
        "6) label trong từng sample phải đúng bằng label yêu cầu."
        f"\nConstructiveness target: {constructiveness_text}. "
        "Each sample must include constructiveness as 0, 1, or null. "
        f"{'Use null when the target is masked/unknown.' if constructiveness is None else f'Every constructiveness value must equal {constructiveness}.'}"
        f"{guidance}"
    )


def call_gemini_with_model(prompt: str, model_name: Optional[str] = None) -> str:
    api_key = get_setting("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    api_version = get_setting("GEMINI_API_VERSION", "v1beta") or "v1beta"
    max_tokens = max(get_int_setting("GEMINI_MAX_TOKENS", 1024, min_value=1), 4096)

    requested = normalize_gemini_model_name(model_name)
    candidates: List[str] = []
    if requested:
        candidates.append(requested)
    for name in get_gemini_model_candidates():
        if name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates = [SYNTHETIC_FALLBACK_MODEL]

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")

    last_error: Optional[str] = None
    for idx, model in enumerate(candidates):
        url = gemini_base_url(api_version, api_key, f"models/{model}:generateContent")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            raw = request_gemini_raw(req, model=model, timeout=30)
        except GeminiRequestFailure as exc:
            last_error = exc.detail
            can_fallback = exc.status_code is None or exc.status_code == 404 or (
                exc.status_code is not None and is_gemini_rate_limited(exc.status_code, exc.detail)
            )
            if can_fallback and idx < len(candidates) - 1:
                logger.warning("Gemini failed on %s, trying fallback", model)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini API error: {exc.detail}") from exc

        try:
            parsed = json.loads(raw)
            model_candidates = parsed.get("candidates") or []
            if not model_candidates:
                raise ValueError("No candidates returned")
            parts = model_candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No content parts returned")
            text = "\n".join([p.get("text", "") for p in parts if p.get("text")])
            if text.strip():
                return GeminiTextResponse(text, model=model, usage_metadata=parsed.get("usageMetadata"))
            raise ValueError("Empty text returned")
        except Exception as exc:
            last_error = str(exc)
            if idx < len(candidates) - 1:
                logger.warning("Gemini parse/content error on %s, trying fallback", model)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini response parse error: {exc}") from exc

    raise HTTPException(status_code=502, detail=f"Gemini API error: {last_error or 'Unknown error'}")


def ensure_table_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def seed_training_tracker_default(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(1) FROM training_tracker_phase").fetchone()
    if row and int(row[0]) > 0:
        return

    for phase_index, phase in enumerate(TRAINING_TRACKER_DEFAULT_PHASES):
        phase_id = phase.get("id") or uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO training_tracker_phase (id, title, sort_order, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (phase_id, phase.get("title") or phase_id, phase_index),
        )

        direct_tasks = phase.get("tasks") or []
        for task_index, task in enumerate(direct_tasks):
            task_id = task.get("id") or uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO training_tracker_task (
                    id, phase_id, group_id, label, param, sort_order, checked, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    task_id,
                    phase_id,
                    task.get("label") or task_id,
                    task.get("param"),
                    task_index,
                ),
            )

        groups = phase.get("groups") or []
        for group_index, group in enumerate(groups):
            group_id = group.get("id") or uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO training_tracker_group (
                    id, phase_id, title, sort_order, created_at, updated_at
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (group_id, phase_id, group.get("title") or group_id, group_index),
            )
            for task_index, task in enumerate(group.get("tasks") or []):
                task_id = task.get("id") or uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO training_tracker_task (
                        id, phase_id, group_id, label, param, sort_order, checked, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        task_id,
                        phase_id,
                        group_id,
                        task.get("label") or task_id,
                        task.get("param"),
                        task_index,
                    ),
                )


def migrate_training_tracker_lora_terminology(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE training_tracker_group
        SET title = '1.4 Learning rate (full fine-tuning)', updated_at = CURRENT_TIMESTAMP
        WHERE id = 'p1_group_14' AND lower(title) LIKE '%lora%'
        """
    )
    conn.execute(
        """
        UPDATE training_tracker_phase
        SET title = 'Giai đoạn 5 — Full fine-tuning config', updated_at = CURRENT_TIMESTAMP
        WHERE id = 'phase_5' AND lower(title) LIKE '%lora%'
        """
    )
    replacements = {
        "p5_task_1": ("Test LR=1e-5", "LEARNING_RATE=1e-5"),
        "p5_task_2": ("Test LR=2e-5", "LEARNING_RATE=2e-5"),
        "p5_task_3": ("Test LR=3e-5", "LEARNING_RATE=3e-5"),
        "p5_task_4": ("Test weight_decay=0.01", "WEIGHT_DECAY=0.01"),
        "p5_task_5": ("Test weight_decay=0.05", "WEIGHT_DECAY=0.05"),
        "p5_task_6": ("Test warmup_ratio=0.08", "WARMUP_RATIO=0.08"),
        "p5_task_7": ("Test head_dropout=0.05", "HEAD_DROPOUT=0.05"),
        "p5_task_8": ("Test head_dropout=0.1", "HEAD_DROPOUT=0.1"),
        "p5_task_9": ("Test gradient accumulation=1", "GRAD_ACCUM=1"),
        "p5_task_10": ("Test gradient accumulation=2", "GRAD_ACCUM=2"),
    }
    for task_id, (label, param) in replacements.items():
        conn.execute(
            """
            UPDATE training_tracker_task
            SET label = ?, param = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND (
                upper(COALESCE(param, '')) LIKE 'LORA_%'
                OR lower(label) LIKE '%lora%'
                OR lower(label) LIKE 'test r=%'
            )
            """,
            (label, param, task_id),
        )


def init_feedback_db() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_system_settings_table(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_page (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                url TEXT NOT NULL,
                url_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                html_tag TEXT NOT NULL,
                html_tag_override TEXT,
                seg_threshold_used REAL,
                score_overall REAL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_segment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                url TEXT NOT NULL,
                url_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                html_tag TEXT NOT NULL,
                html_tag_override TEXT,
                segment_id TEXT NOT NULL,
                text TEXT NOT NULL,
                score REAL,
                seg_threshold_used REAL,
                label TEXT NOT NULL,
                segment_hash TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_generation_batch (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                style TEXT NOT NULL,
                target_label INTEGER NOT NULL,
                requested_count INTEGER NOT NULL,
                generated_count INTEGER NOT NULL,
                generator_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_dataset_row (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                text TEXT NOT NULL,
                label INTEGER NOT NULL,
                constructiveness INTEGER,
                domain TEXT NOT NULL,
                style TEXT NOT NULL,
                is_accepted INTEGER NOT NULL DEFAULT 1,
                structure_fingerprint TEXT,
                text_hash TEXT,
                validation_flags TEXT,
                meta_json TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                review_method TEXT NOT NULL DEFAULT 'manual',
                label_confidence TEXT,
                reviewed_by TEXT,
                review_provider TEXT,
                review_model_name TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_synth_row_batch ON synthetic_dataset_row(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_synth_row_accept ON synthetic_dataset_row(is_accepted)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_synth_row_dims ON synthetic_dataset_row(domain, style, label)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_crawl_batch (
                batch_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL,
                source_job_id TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                options_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_comment_item (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                job_id TEXT,
                url TEXT NOT NULL,
                url_hash TEXT NOT NULL,
                domain_category TEXT,
                segment_id TEXT,
                text TEXT NOT NULL,
                score REAL,
                pseudo_label INTEGER,
                constructiveness_score REAL,
                constructiveness_label INTEGER,
                constructiveness_confidence TEXT,
                selected_for_training INTEGER NOT NULL DEFAULT 1,
                training_review_status TEXT NOT NULL DEFAULT 'auto',
                is_locked INTEGER NOT NULL DEFAULT 0,
                gate_bucket TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                segment_hash TEXT,
                context_segment_hash TEXT,
                dedupe_key TEXT,
                html_tag TEXT,
                seg_threshold_used REAL,
                label_source TEXT,
                label_confidence TEXT,
                review_provider TEXT,
                review_model_name TEXT,
                review_reason TEXT,
                source_type TEXT NOT NULL DEFAULT 'crawl',
                source_row_id INTEGER,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                FOREIGN KEY(batch_id) REFERENCES mlflow_crawl_batch(batch_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_comment_prediction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_item_id INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                job_id TEXT,
                model_id TEXT NOT NULL,
                raw_toxicity_score REAL,
                adjusted_toxicity_score REAL,
                predicted_label INTEGER,
                constructiveness_score REAL,
                constructiveness_label INTEGER,
                constructiveness_confidence TEXT,
                seg_threshold_used REAL,
                record_origin TEXT NOT NULL DEFAULT 'inference',
                created_at TEXT NOT NULL,
                FOREIGN KEY(sample_item_id) REFERENCES mlflow_comment_item(id) ON DELETE CASCADE,
                FOREIGN KEY(batch_id) REFERENCES mlflow_crawl_batch(batch_id) ON DELETE CASCADE,
                UNIQUE(sample_item_id, model_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_training_artifact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_model_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_family TEXT NOT NULL,
                model_id TEXT NOT NULL UNIQUE,
                source_run_id TEXT NOT NULL UNIQUE,
                artifact_path TEXT NOT NULL,
                artifact_checksum TEXT NOT NULL,
                bundle_checksum TEXT,
                test_fingerprint TEXT,
                metrics_json TEXT,
                status TEXT NOT NULL DEFAULT 'candidate',
                created_at TEXT NOT NULL,
                promoted_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_production_slot (
                model_family TEXT PRIMARY KEY,
                active_model_id TEXT NOT NULL,
                active_run_id TEXT,
                artifact_checksum TEXT,
                previous_model_id TEXT,
                previous_run_id TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_promotion_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_family TEXT NOT NULL,
                action TEXT NOT NULL,
                source_run_id TEXT,
                from_model_id TEXT,
                to_model_id TEXT,
                artifact_checksum TEXT,
                status TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_automation_state (
                model_family TEXT PRIMARY KEY,
                last_triggered_eligible_count INTEGER NOT NULL DEFAULT 0,
                last_triggered_at TEXT,
                last_run_id TEXT,
                active_run_id TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_automation_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_family TEXT NOT NULL,
                action TEXT NOT NULL,
                source_run_id TEXT,
                status TEXT NOT NULL,
                eligible_count INTEGER,
                detail TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_gemini_evaluation (
                run_id TEXT PRIMARY KEY,
                model_family TEXT NOT NULL,
                previous_run_id TEXT,
                prompt_instruction TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                model_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mlflow_model_version_family_status
            ON mlflow_model_version(model_family, status, created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mlflow_promotion_event_family_created
            ON mlflow_promotion_event(model_family, created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mlflow_automation_event_family_created
            ON mlflow_automation_event(model_family, created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mlflow_gemini_evaluation_family_created
            ON mlflow_gemini_evaluation(model_family, created_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mlflow_do_run (
                run_id TEXT PRIMARY KEY,
                batch_id TEXT,
                provider TEXT NOT NULL,
                gpu_profile TEXT,
                status TEXT NOT NULL,
                current_stage TEXT,
                logs_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mlflow_item_batch ON mlflow_comment_item(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mlflow_item_bucket ON mlflow_comment_item(batch_id, gate_bucket)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mlflow_item_status ON mlflow_comment_item(batch_id, verification_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mlflow_item_hash ON mlflow_comment_item(context_segment_hash, segment_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mlflow_prediction_batch ON mlflow_comment_prediction(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mlflow_prediction_sample ON mlflow_comment_prediction(sample_item_id, created_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_tracker_phase (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_tracker_group (
                id TEXT PRIMARY KEY,
                phase_id TEXT NOT NULL,
                title TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(phase_id) REFERENCES training_tracker_phase(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_tracker_task (
                id TEXT PRIMARY KEY,
                phase_id TEXT NOT NULL,
                group_id TEXT,
                label TEXT NOT NULL,
                param TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                checked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(phase_id) REFERENCES training_tracker_phase(id) ON DELETE CASCADE,
                FOREIGN KEY(group_id) REFERENCES training_tracker_group(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_tracker_result (
                id TEXT PRIMARY KEY,
                scenario_name TEXT NOT NULL,
                phase_id TEXT,
                macro_f1 REAL NOT NULL,
                f1_toxic REAL NOT NULL,
                precision_toxic REAL NOT NULL,
                recall_toxic REAL NOT NULL,
                val_loss REAL,
                best_threshold_macro_f1 REAL,
                best_threshold_f1_toxic REAL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_phase_order ON training_tracker_phase(sort_order)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_group_phase_order ON training_tracker_group(phase_id, sort_order)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_task_phase_group_order ON training_tracker_task(phase_id, group_id, sort_order)")

        result_columns = conn.execute("PRAGMA table_info(training_tracker_result)").fetchall()
        phase_id_column = next((row for row in result_columns if row[1] == "phase_id"), None)
        if phase_id_column and int(phase_id_column[3]) == 1:
            conn.execute("ALTER TABLE training_tracker_result RENAME TO training_tracker_result_old")
            conn.execute(
                """
                CREATE TABLE training_tracker_result (
                    id TEXT PRIMARY KEY,
                    scenario_name TEXT NOT NULL,
                    phase_id TEXT,
                    macro_f1 REAL NOT NULL,
                    f1_toxic REAL NOT NULL,
                    precision_toxic REAL NOT NULL,
                    recall_toxic REAL NOT NULL,
                    val_loss REAL,
                    best_threshold_macro_f1 REAL,
                    best_threshold_f1_toxic REAL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO training_tracker_result (
                    id, scenario_name, phase_id, macro_f1, f1_toxic, precision_toxic, recall_toxic,
                    val_loss, best_threshold_macro_f1, best_threshold_f1_toxic, notes, created_at
                )
                SELECT id, scenario_name, phase_id, macro_f1, f1_toxic, precision_toxic, recall_toxic,
                       val_loss, best_threshold_macro_f1, best_threshold_f1_toxic, notes, created_at
                FROM training_tracker_result_old
                """
            )
            conn.execute("DROP TABLE training_tracker_result_old")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_result_created ON training_tracker_result(created_at DESC)")

        ensure_table_column(conn, "feedback_page", "html_tag", "TEXT")
        ensure_table_column(conn, "feedback_page", "html_tag_override", "TEXT")
        ensure_table_column(conn, "feedback_segment", "html_tag", "TEXT")
        ensure_table_column(conn, "feedback_segment", "html_tag_override", "TEXT")
        ensure_table_column(conn, "feedback_segment", "segment_hash", "TEXT")
        ensure_table_column(conn, "feedback_segment", "context_segment_hash", "TEXT")
        ensure_table_column(conn, "synthetic_generation_batch", "generator_model", "TEXT NOT NULL DEFAULT 'gemini-1.5-flash-latest'")
        ensure_table_column(conn, "synthetic_generation_batch", "prompt_version", "TEXT NOT NULL DEFAULT 'v1'")
        ensure_table_column(conn, "synthetic_dataset_row", "is_accepted", "INTEGER NOT NULL DEFAULT 1")
        ensure_table_column(conn, "synthetic_dataset_row", "structure_fingerprint", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "text_hash", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "validation_flags", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "meta_json", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "reviewed_at", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "constructiveness", "INTEGER")
        ensure_table_column(conn, "synthetic_dataset_row", "review_method", "TEXT NOT NULL DEFAULT 'manual'")
        ensure_table_column(conn, "synthetic_dataset_row", "label_confidence", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "reviewed_by", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "review_provider", "TEXT")
        ensure_table_column(conn, "synthetic_dataset_row", "review_model_name", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "label_source", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "label_confidence", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "review_provider", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "review_model_name", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "review_reason", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "domain_category", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "constructiveness_score", "REAL")
        ensure_table_column(conn, "mlflow_comment_item", "constructiveness_label", "INTEGER")
        ensure_table_column(conn, "mlflow_comment_item", "constructiveness_confidence", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "selected_for_training", "INTEGER NOT NULL DEFAULT 1")
        ensure_table_column(conn, "mlflow_comment_item", "training_review_status", "TEXT NOT NULL DEFAULT 'auto'")
        ensure_table_column(conn, "mlflow_comment_item", "is_locked", "INTEGER NOT NULL DEFAULT 0")
        ensure_table_column(conn, "mlflow_comment_item", "dedupe_key", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "source_type", "TEXT NOT NULL DEFAULT 'crawl'")
        ensure_table_column(conn, "mlflow_comment_item", "source_row_id", "INTEGER")
        ensure_table_column(conn, "mlflow_comment_prediction", "constructiveness_confidence", "TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mlflow_item_dedupe_key
            ON mlflow_comment_item(dedupe_key)
            WHERE dedupe_key IS NOT NULL AND dedupe_key <> ''
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mlflow_item_source_row
            ON mlflow_comment_item(source_type, source_row_id)
            WHERE source_type = 'synthetic' AND source_row_id IS NOT NULL
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO mlflow_comment_prediction (
                sample_item_id, batch_id, job_id, model_id,
                raw_toxicity_score, adjusted_toxicity_score, predicted_label,
                constructiveness_score, constructiveness_label, constructiveness_confidence, seg_threshold_used,
                record_origin, created_at
            )
            SELECT item.id, item.batch_id, item.job_id, batch.model_id,
                   item.score, NULL, NULL,
                   item.constructiveness_score, item.constructiveness_label, item.constructiveness_confidence, item.seg_threshold_used,
                   'legacy_backfill', item.created_at
            FROM mlflow_comment_item AS item
            JOIN mlflow_crawl_batch AS batch ON batch.batch_id = item.batch_id
            WHERE item.source_type = 'crawl'
            """
        )
        ensure_table_column(conn, "mlflow_do_run", "droplet_id", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "artifact_uri", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "artifact_checksum", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "spaces_bucket", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "spaces_key", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "error_message", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "bundle_path", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "bundle_url", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "bundle_checksum", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "bundle_token_hash", "TEXT")
        ensure_table_column(conn, "mlflow_do_run", "bundle_manifest_json", "TEXT")
        ensure_table_column(conn, "mlflow_training_artifact", "source_run_id", "TEXT")
        ensure_table_column(conn, "mlflow_training_artifact", "metrics_json", "TEXT")
        ensure_table_column(conn, "mlflow_model_version", "model_kind", "TEXT")
        ensure_table_column(conn, "mlflow_model_version", "training_mode", "TEXT")
        ensure_table_column(conn, "mlflow_model_version", "base_model", "TEXT")
        ensure_table_column(conn, "mlflow_model_version", "artifact_uri", "TEXT")
        ensure_table_column(conn, "mlflow_model_version", "bundle_path", "TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mlflow_training_artifact_source_run_id
            ON mlflow_training_artifact(source_run_id)
            WHERE source_run_id IS NOT NULL
            """
        )

        migrate_training_tracker_lora_terminology(conn)
        seed_training_tracker_default(conn)
        conn.commit()


@app.on_event("startup")
def initialize_feedback_database() -> None:
    init_feedback_db()


def fetch_training_tracker_payload() -> Dict[str, Any]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        phases = conn.execute(
            """
            SELECT id, title, sort_order, created_at, updated_at
            FROM training_tracker_phase
            ORDER BY sort_order ASC, created_at ASC
            """
        ).fetchall()
        groups = conn.execute(
            """
            SELECT id, phase_id, title, sort_order, created_at, updated_at
            FROM training_tracker_group
            ORDER BY phase_id ASC, sort_order ASC, created_at ASC
            """
        ).fetchall()
        tasks = conn.execute(
            """
            SELECT id, phase_id, group_id, label, param, sort_order, checked, created_at, updated_at
            FROM training_tracker_task
            ORDER BY phase_id ASC, COALESCE(group_id, ''), sort_order ASC, created_at ASC
            """
        ).fetchall()
        results = conn.execute(
            """
            SELECT id, scenario_name, phase_id, macro_f1, f1_toxic, precision_toxic, recall_toxic,
                   val_loss, best_threshold_macro_f1, best_threshold_f1_toxic, notes, created_at
            FROM training_tracker_result
            ORDER BY created_at DESC
            """
        ).fetchall()

    groups_by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for row in groups:
        groups_by_phase.setdefault(row["phase_id"], []).append(
            {
                "id": row["id"],
                "title": row["title"],
                "sort_order": row["sort_order"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    grouped_tasks: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
    for row in tasks:
        key = (row["phase_id"], row["group_id"])
        grouped_tasks.setdefault(key, []).append(
            {
                "id": row["id"],
                "label": row["label"],
                "param": row["param"],
                "sort_order": row["sort_order"],
                "checked": bool(row["checked"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    phase_items: List[Dict[str, Any]] = []
    for phase in phases:
        phase_id = phase["id"]
        phase_groups = groups_by_phase.get(phase_id, [])
        groups_payload: List[Dict[str, Any]] = []
        for group in phase_groups:
            task_items = grouped_tasks.get((phase_id, group["id"]), [])
            groups_payload.append({
                **group,
                "tasks": task_items,
            })

        direct_tasks = grouped_tasks.get((phase_id, None), [])
        phase_items.append(
            {
                "id": phase_id,
                "title": phase["title"],
                "sort_order": phase["sort_order"],
                "created_at": phase["created_at"],
                "updated_at": phase["updated_at"],
                "groups": groups_payload,
                "tasks": direct_tasks,
            }
        )

    result_items = []
    for row in results:
        result_items.append(
            {
                "id": row["id"],
                "scenario_name": row["scenario_name"],
                "macro_f1": row["macro_f1"],
                "f1_toxic": row["f1_toxic"],
                "precision_toxic": row["precision_toxic"],
                "recall_toxic": row["recall_toxic"],
                "val_loss": row["val_loss"],
                "best_threshold_macro_f1": row["best_threshold_macro_f1"],
                "best_threshold_f1_toxic": row["best_threshold_f1_toxic"],
                "notes": row["notes"] or "",
                "created_at": row["created_at"],
            }
        )

    return {"phases": phase_items, "results": result_items}


def create_training_phase(title: str) -> Dict[str, Any]:
    init_feedback_db()
    phase_id = uuid.uuid4().hex
    clean_title = title.strip()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        current_max = conn.execute("SELECT COALESCE(MAX(sort_order), -1) FROM training_tracker_phase").fetchone()
        next_order = int(current_max[0]) + 1 if current_max else 0
        conn.execute(
            """
            INSERT INTO training_tracker_phase (id, title, sort_order, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (phase_id, clean_title, next_order),
        )
        conn.commit()
    return {"id": phase_id, "title": clean_title, "sort_order": next_order}


def update_training_phase_title(phase_id: str, title: str) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE training_tracker_phase
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title.strip(), phase_id),
        )
        conn.commit()
        return cursor.rowcount or 0


def reorder_training_phases(phase_ids: List[str]) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        existing = [row[0] for row in conn.execute("SELECT id FROM training_tracker_phase").fetchall()]
        existing_set = set(existing)
        if set(phase_ids) != existing_set:
            raise HTTPException(status_code=400, detail="phase_ids must include all existing phases")
        for idx, phase_id in enumerate(phase_ids):
            conn.execute(
                "UPDATE training_tracker_phase SET sort_order = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (idx, phase_id),
            )
        conn.commit()
    return len(phase_ids)


def delete_training_phase(phase_id: str) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("DELETE FROM training_tracker_task WHERE phase_id = ?", (phase_id,))
        conn.execute("DELETE FROM training_tracker_group WHERE phase_id = ?", (phase_id,))
        cursor = conn.execute("DELETE FROM training_tracker_phase WHERE id = ?", (phase_id,))
        conn.commit()
        return cursor.rowcount or 0


def create_training_group(phase_id: str, title: str) -> Dict[str, Any]:
    init_feedback_db()
    group_id = uuid.uuid4().hex
    clean_title = title.strip()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        exists = conn.execute("SELECT COUNT(1) FROM training_tracker_phase WHERE id = ?", (phase_id,)).fetchone()
        if not exists or int(exists[0]) == 0:
            raise HTTPException(status_code=404, detail="Phase not found")
        current_max = conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) FROM training_tracker_group WHERE phase_id = ?",
            (phase_id,),
        ).fetchone()
        next_order = int(current_max[0]) + 1 if current_max else 0
        conn.execute(
            """
            INSERT INTO training_tracker_group (id, phase_id, title, sort_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (group_id, phase_id, clean_title, next_order),
        )
        conn.commit()
    return {"id": group_id, "phase_id": phase_id, "title": clean_title, "sort_order": next_order}


def update_training_group_title(group_id: str, title: str) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE training_tracker_group
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title.strip(), group_id),
        )
        conn.commit()
        return cursor.rowcount or 0


def reorder_training_groups(phase_id: str, group_ids: List[str]) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        existing = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM training_tracker_group WHERE phase_id = ? ORDER BY sort_order ASC",
                (phase_id,),
            ).fetchall()
        ]
        if set(existing) != set(group_ids):
            raise HTTPException(status_code=400, detail="group_ids must include all groups in phase")
        for idx, group_id in enumerate(group_ids):
            conn.execute(
                "UPDATE training_tracker_group SET sort_order = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (idx, group_id),
            )
        conn.commit()
    return len(group_ids)


def delete_training_group(group_id: str) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM training_tracker_group WHERE id = ?", (group_id,))
        conn.commit()
        return cursor.rowcount or 0


def create_training_task(phase_id: str, group_id: Optional[str], label: str, param: Optional[str]) -> Dict[str, Any]:
    init_feedback_db()
    task_id = uuid.uuid4().hex
    clean_label = label.strip()
    clean_param = (param or "").strip() or None
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        phase_exists = conn.execute("SELECT COUNT(1) FROM training_tracker_phase WHERE id = ?", (phase_id,)).fetchone()
        if not phase_exists or int(phase_exists[0]) == 0:
            raise HTTPException(status_code=404, detail="Phase not found")
        if group_id:
            group_exists = conn.execute(
                "SELECT COUNT(1) FROM training_tracker_group WHERE id = ? AND phase_id = ?",
                (group_id, phase_id),
            ).fetchone()
            if not group_exists or int(group_exists[0]) == 0:
                raise HTTPException(status_code=404, detail="Group not found")
        current_max = conn.execute(
            """
            SELECT COALESCE(MAX(sort_order), -1)
            FROM training_tracker_task
            WHERE phase_id = ? AND (
                (group_id IS NULL AND ? IS NULL) OR group_id = ?
            )
            """,
            (phase_id, group_id, group_id),
        ).fetchone()
        next_order = int(current_max[0]) + 1 if current_max else 0
        conn.execute(
            """
            INSERT INTO training_tracker_task (
                id, phase_id, group_id, label, param, sort_order, checked, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (task_id, phase_id, group_id, clean_label, clean_param, next_order),
        )
        conn.commit()
    return {
        "id": task_id,
        "phase_id": phase_id,
        "group_id": group_id,
        "label": clean_label,
        "param": clean_param,
        "sort_order": next_order,
    }


def update_training_task(task_id: str, label: str, param: Optional[str]) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE training_tracker_task
            SET label = ?, param = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (label.strip(), (param or "").strip() or None, task_id),
        )
        conn.commit()
        return cursor.rowcount or 0


def reorder_training_tasks(phase_id: str, group_id: Optional[str], task_ids: List[str]) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        existing = [
            row[0]
            for row in conn.execute(
                """
                SELECT id FROM training_tracker_task
                WHERE phase_id = ? AND ((group_id IS NULL AND ? IS NULL) OR group_id = ?)
                ORDER BY sort_order ASC
                """,
                (phase_id, group_id, group_id),
            ).fetchall()
        ]
        if set(existing) != set(task_ids):
            raise HTTPException(status_code=400, detail="task_ids must include all tasks in target scope")
        for idx, task_id in enumerate(task_ids):
            conn.execute(
                "UPDATE training_tracker_task SET sort_order = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (idx, task_id),
            )
        conn.commit()
    return len(task_ids)


def set_training_task_checked(task_id: str, checked: bool) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE training_tracker_task
            SET checked = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (1 if checked else 0, task_id),
        )
        conn.commit()
        return cursor.rowcount or 0


def delete_training_task(task_id: str) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM training_tracker_task WHERE id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount or 0


def create_training_result(item: TrainingTrackerCreateResultRequest) -> Dict[str, Any]:
    init_feedback_db()
    result_id = uuid.uuid4().hex
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO training_tracker_result (
                id, scenario_name, phase_id, macro_f1, f1_toxic, precision_toxic, recall_toxic,
                val_loss, best_threshold_macro_f1, best_threshold_f1_toxic, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                result_id,
                item.scenario_name.strip(),
                item.phase_id,
                item.macro_f1,
                item.f1_toxic,
                item.precision_toxic,
                item.recall_toxic,
                item.val_loss,
                item.best_threshold_macro_f1,
                item.best_threshold_f1_toxic,
                (item.notes or "").strip(),
            ),
        )
        conn.commit()

    payload = fetch_training_tracker_payload()
    result = next((r for r in payload["results"] if r["id"] == result_id), None)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to read inserted result")
    return result


def delete_training_result(result_id: str) -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM training_tracker_result WHERE id = ?", (result_id,))
        conn.commit()
        return cursor.rowcount or 0


def insert_feedback_page(items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    init_feedback_db()
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback_page)")}
        insert_columns = ["job_id", "url", "url_hash", "model_id"]
        if "domain_category" in columns:
            insert_columns.append("domain_category")
        if "domain_override" in columns:
            insert_columns.append("domain_override")
        if "html_tag" in columns:
            insert_columns.append("html_tag")
        if "html_tag_override" in columns:
            insert_columns.append("html_tag_override")
        insert_columns += ["seg_threshold_used", "score_overall", "label", "created_at"]

        rows = []
        for item in items:
            row = [item["job_id"], item["url"], item["url_hash"], item["model_id"]]
            if "domain_category" in columns:
                row.append(item["html_tag"])
            if "domain_override" in columns:
                row.append(item.get("html_tag_override"))
            if "html_tag" in columns:
                row.append(item["html_tag"])
            if "html_tag_override" in columns:
                row.append(item.get("html_tag_override"))
            row += [
                item.get("seg_threshold_used"),
                item.get("score_overall"),
                item["label"],
                now,
            ]
            rows.append(tuple(row))

        placeholders = ", ".join(["?"] * len(insert_columns))
        sql = f"INSERT INTO feedback_page ({', '.join(insert_columns)}) VALUES ({placeholders})"
        conn.executemany(sql, rows)
        conn.commit()
    return len(rows)


def insert_feedback_segment(items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    init_feedback_db()
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback_segment)")}
        insert_columns = ["job_id", "url", "url_hash", "model_id"]
        if "domain_category" in columns:
            insert_columns.append("domain_category")
        if "domain_override" in columns:
            insert_columns.append("domain_override")
        if "html_tag" in columns:
            insert_columns.append("html_tag")
        if "html_tag_override" in columns:
            insert_columns.append("html_tag_override")
        insert_columns += ["segment_id", "text", "score", "seg_threshold_used", "label"]
        if "segment_hash" in columns:
            insert_columns.append("segment_hash")
        if "context_segment_hash" in columns:
            insert_columns.append("context_segment_hash")
        insert_columns.append("created_at")

        dedupe_candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
        passthrough_items: List[Dict[str, Any]] = []
        for item in items:
            effective_hash = (item.get("context_segment_hash") or item.get("segment_hash") or "").strip()
            effective_tag = (item.get("html_tag_override") or item.get("html_tag") or "").strip().lower()
            if effective_hash:
                dedupe_candidates[(effective_hash, effective_tag)] = item
            else:
                passthrough_items.append(item)

        deduped_items = list(dedupe_candidates.values()) + passthrough_items

        rows = []
        for item in deduped_items:
            row = [item["job_id"], item["url"], item["url_hash"], item["model_id"]]
            if "domain_category" in columns:
                row.append(item["html_tag"])
            if "domain_override" in columns:
                row.append(item.get("html_tag_override"))
            if "html_tag" in columns:
                row.append(item["html_tag"])
            if "html_tag_override" in columns:
                row.append(item.get("html_tag_override"))
            row += [
                item["segment_id"],
                item["text"],
                item.get("score"),
                item.get("seg_threshold_used"),
                item["label"],
            ]
            if "segment_hash" in columns:
                row.append(item.get("segment_hash"))
            if "context_segment_hash" in columns:
                row.append(item.get("context_segment_hash"))
            row.append(now)
            rows.append(tuple(row))

        if "segment_hash" in columns:
            for (effective_hash, effective_tag) in dedupe_candidates.keys():
                if "context_segment_hash" in columns:
                    conn.execute(
                        """
                        DELETE FROM feedback_segment
                        WHERE COALESCE(context_segment_hash, segment_hash) = ?
                          AND LOWER(COALESCE(html_tag_override, html_tag, '')) = ?
                        """,
                        (effective_hash, effective_tag),
                    )
                else:
                    conn.execute(
                        """
                        DELETE FROM feedback_segment
                        WHERE segment_hash = ?
                          AND LOWER(COALESCE(html_tag_override, html_tag, '')) = ?
                        """,
                        (effective_hash, effective_tag),
                    )

        placeholders = ", ".join(["?"] * len(insert_columns))
        sql = f"INSERT INTO feedback_segment ({', '.join(insert_columns)}) VALUES ({placeholders})"
        conn.executemany(sql, rows)
        conn.commit()
    return len(rows)


def load_threshold_overrides(model_id: str) -> Dict[str, float]:
    return {}


def delete_threshold_overrides(model_id: str, categories: List[str]) -> int:
    return 0


def save_threshold_overrides(model_id: str, values: Dict[str, float]) -> None:
    return None


def get_effective_thresholds(model_id: str) -> Dict[str, float]:
    return {**CATEGORY_THRESHOLDS}


def normalize_segment_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def build_segment_hash(text: str, html_tag: str) -> str:
    base = f"{normalize_segment_text(text)}|{(html_tag or '').strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def load_learned_segments(model_id: Optional[str] = None) -> Dict[Tuple[str, str], Dict[str, float]]:
    init_feedback_db()
    learned: Dict[Tuple[str, str], Dict[str, float]] = {}
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        query = """
            SELECT id, segment_hash, context_segment_hash, html_tag_override, html_tag, label
            FROM feedback_segment
            ORDER BY id DESC
        """
        rows = conn.execute(query).fetchall()

    # Deduplicate by semantic unit to prevent repeated re-scans from inflating support.
    # Keep only the latest label per (effective_hash, effective_tag).
    latest_by_unit: Dict[Tuple[str, str], str] = {}
    for _id, segment_hash, context_segment_hash, html_tag_override, html_tag, label in rows:
        normalized = safe_label(label)
        if normalized not in {"toxic", "clean"}:
            continue

        tag = (html_tag_override or html_tag or "").strip().lower()
        effective_hash = (context_segment_hash or segment_hash or "").strip()
        if not effective_hash:
            continue

        unit_key = (effective_hash, tag)
        if unit_key not in latest_by_unit:
            latest_by_unit[unit_key] = normalized

    for (effective_hash, tag), normalized in latest_by_unit.items():
        keys: List[Tuple[str, str]] = [
            (effective_hash, tag),
            (effective_hash, ""),
        ]

        for key in keys:
            stats = learned.setdefault(key, {"toxic_count": 0.0, "clean_count": 0.0, "support": 0.0, "agreement": 0.0})
            if normalized == "toxic":
                stats["toxic_count"] += 1.0
            else:
                stats["clean_count"] += 1.0

    for stats in learned.values():
        support = stats["toxic_count"] + stats["clean_count"]
        stats["support"] = support
        stats["agreement"] = (max(stats["toxic_count"], stats["clean_count"]) / support) if support else 0.0

    return learned


def safe_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"toxic", "clean", "unsure"}:
        return normalized
    return None


def safe_label_int(value: Optional[str]) -> Optional[int]:
    normalized = safe_label(value)
    if normalized == "toxic":
        return 1
    if normalized == "clean":
        return 0
    return None


def compute_f1(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)






def normalize_dataset_version(value: Optional[str]) -> str:
    normalized = (value or DEFAULT_DATASET_VERSION).strip().lower()
    resolved = DATASET_VERSION_ALIASES.get(normalized)
    if resolved:
        return resolved
    raise HTTPException(
        status_code=400,
        detail={
            "message": "Unsupported dataset_version",
            "value": value,
            "allowed": sorted(DATASET_VERSION_ALIASES.keys()),
        },
    )


def resolve_dataset_dir(dataset_version: str) -> Path:
    dataset_dir = DATASET_VERSION_DIRS.get(dataset_version)
    if dataset_dir is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported canonical dataset_version",
                "value": dataset_version,
                "allowed": sorted(DATASET_VERSION_DIRS.keys()),
            },
        )
    if not dataset_dir.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Dataset directory does not exist",
                "dataset_version": dataset_version,
                "path": to_relative(str(dataset_dir)),
            },
        )
    return dataset_dir


def iter_dataset_files(dataset_version: str) -> List[Tuple[Path, str, bool]]:
    dataset_dir = resolve_dataset_dir(dataset_version)
    files: List[Tuple[Path, str, bool]] = []
    for path in sorted(dataset_dir.glob("*.jsonl")):
        name = path.name.lower()
        if "train" in name:
            split = "train"
        elif "validation" in name or "valid" in name:
            split = "validation"
        elif "test" in name:
            split = "test"
        else:
            split = "unknown"
        is_augmented = "augmented" in name
        files.append((path, split, is_augmented))
    return files


def normalize_source(value: Optional[str]) -> str:
    if not value:
        return "victsd"
    cleaned = value.strip().lower()
    if cleaned in {"vihsd", "vihsd_v1", "vihsd_v2"}:
        return "vihsd"
    if cleaned == "victsd":
        return "victsd"
    return cleaned


def iter_dataset_rows(dataset_version: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path, split, is_augmented in iter_dataset_files(dataset_version):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text")
                    label = normalize_int(obj.get("label"))
                    constructiveness = normalize_int(obj.get("constructiveness"))
                    if label is None:
                        label = normalize_int(obj.get("toxicity"))
                    if text is None or label is None:
                        continue
                    if constructiveness not in {0, 1}:
                        constructiveness = None
                    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
                    source = normalize_source(meta.get("source"))
                    if is_augmented:
                        source = f"{source}_augmented"
                    meta_out = {
                        **meta,
                        "source": source,
                        "split": split,
                        "is_augmented": is_augmented,
                    }
                    rows.append({"text": text, "label": label, "constructiveness": constructiveness, "meta": meta_out})
        except Exception:
            continue
    return rows


def iter_feedback_rows() -> List[Dict[str, Any]]:
    init_feedback_db()
    rows: List[Dict[str, Any]] = []
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        results = conn.execute(
            """
            SELECT id, text, label, model_id, html_tag, html_tag_override, score, seg_threshold_used, created_at
            FROM feedback_segment
            ORDER BY id DESC
            """
        ).fetchall()

    for feedback_id, text, label, model_id, html_tag, html_tag_override, score, seg_threshold_used, created_at in results:
        label_int = safe_label_int(label)
        if label_int is None:
            continue
        meta = {
            "source": "new_collected",
            "split": "feedback",
            "is_augmented": False,
            "feedback_id": feedback_id,
            "model_id": model_id,
            "html_tag": html_tag,
            "html_tag_override": html_tag_override,
            "score": normalize_score(score),
            "seg_threshold_used": normalize_score(seg_threshold_used),
            "created_at": created_at,
        }
        rows.append({"text": text, "label": label_int, "constructiveness": None, "meta": meta})
    return rows


def build_dataset_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, int]] = {}
    total = 0
    for row in rows:
        meta = row.get("meta") or {}
        source = meta.get("source") or "unknown"
        label = row.get("label")
        total += 1
        source_stats = by_source.setdefault(source, {"total": 0, "clean": 0, "toxic": 0})
        source_stats["total"] += 1
        if label == 1:
            source_stats["toxic"] += 1
        elif label == 0:
            source_stats["clean"] += 1
    return {"total": total, "by_source": by_source}


def load_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return default


def build_registry_from_models(
    model_root: Path,
    legacy_registry: Optional[Dict[str, Any]] = None,
    merge_legacy: bool = True,
) -> Dict[str, Any]:
    from registry_builder import build_registry_from_models as build_registry

    return build_registry(
        model_root=model_root,
        base_dir=BASE_DIR,
        legacy_registry=legacy_registry,
        merge_legacy=merge_legacy,
    )


def file_last_updated(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).isoformat()
    except OSError as exc:
        logger.warning("Failed to read mtime for %s: %s", path, exc)
        return None


def utc_timestamp_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def slugify_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def build_artifact_versions(
    *,
    dataset_version: Optional[str],
    model_version: Optional[str],
    policy_version: Optional[str],
) -> Dict[str, str]:
    versions = {
        "dataset_version": (dataset_version or "").strip(),
        "model_version": (model_version or "").strip(),
        "policy_version": (policy_version or "").strip(),
    }
    return versions


def find_missing_required_versions(versions: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    for key in REQUIRED_VERSION_KEYS:
        value = versions.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
    return missing


def filter_dataset_rows(
    rows: List[Dict[str, Any]],
    sources: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
    splits: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    source_set = {s.strip().lower() for s in sources or [] if s}
    label_set = {int(v) for v in labels or [] if isinstance(v, (int, float, str)) and str(v).isdigit()}
    split_set = {s.strip().lower() for s in splits or [] if s}

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta") or {}
        source = str(meta.get("source") or "").lower()
        split = str(meta.get("split") or "").lower()
        label = row.get("label")
        if source_set and source not in source_set:
            continue
        if label_set and label not in label_set:
            continue
        if split_set and split not in split_set:
            continue
        filtered.append(row)
    return filtered


def delete_feedback_rows(ids: List[int]) -> int:
    if not ids:
        return 0
    normalized = [int(v) for v in ids if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())]
    if not normalized:
        return 0
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        placeholders = ", ".join(["?"] * len(normalized))
        cursor = conn.execute(
            f"DELETE FROM feedback_segment WHERE id IN ({placeholders})",
            tuple(normalized),
        )
        conn.commit()
        return cursor.rowcount or 0


def insert_synthetic_batch(
    *,
    batch_id: str,
    domain: str,
    style: str,
    target_label: int,
    requested_count: int,
    generated_count: int,
    generator_model: str,
    rows: List[Dict[str, Any]],
) -> int:
    if not rows:
        return 0

    init_feedback_db()
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO synthetic_generation_batch (
                batch_id, domain, style, target_label, requested_count,
                generated_count, generator_model, prompt_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                domain,
                style,
                target_label,
                requested_count,
                generated_count,
                generator_model,
                SYNTHETIC_PROMPT_VERSION,
                now,
            ),
        )

        payload_rows = []
        for row in rows:
            payload_rows.append(
                (
                    batch_id,
                    row["text"],
                    row["label"],
                    row.get("constructiveness") if row.get("constructiveness") in {0, 1} else None,
                    domain,
                    style,
                    0,
                    row.get("structure_fingerprint"),
                    row.get("text_hash"),
                    json.dumps(row.get("validation_flags") or {}, ensure_ascii=False),
                    json.dumps(row.get("meta") or {}, ensure_ascii=False),
                    now,
                    None,
                )
            )

        conn.executemany(
            """
            INSERT INTO synthetic_dataset_row (
                batch_id, text, label, constructiveness, domain, style, is_accepted,
                structure_fingerprint, text_hash, validation_flags, meta_json,
                created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload_rows,
        )
        conn.commit()

    return len(rows)


def load_synthetic_rows(
    *,
    batch_id: Optional[str] = None,
    domain: Optional[str] = None,
    style: Optional[str] = None,
    label: Optional[int] = None,
    accepted: Optional[bool] = None,
    reviewed: bool = Query(default=False),
) -> List[Dict[str, Any]]:
    init_feedback_db()
    clauses: List[str] = []
    params: List[Any] = []

    if batch_id:
        clauses.append("batch_id = ?")
        params.append(batch_id)
    if domain:
        clauses.append("domain = ?")
        params.append(domain)
    if style:
        clauses.append("style = ?")
        params.append(style)
    if label is not None:
        clauses.append("label = ?")
        params.append(int(label))
    if accepted is not None:
        clauses.append("is_accepted = ?")
        params.append(1 if accepted else 0)
    if reviewed is True:
        clauses.append("reviewed_at IS NOT NULL")
    elif reviewed is False:
        clauses.append("reviewed_at IS NULL")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT id, batch_id, text, label, constructiveness, domain, style, is_accepted,
               validation_flags, meta_json, created_at, reviewed_at,
               review_method, label_confidence, reviewed_by, review_provider, review_model_name
        FROM synthetic_dataset_row
        {where_sql}
        ORDER BY id DESC
    """

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        results = conn.execute(query, tuple(params)).fetchall()

    rows: List[Dict[str, Any]] = []
    for (
        sample_id,
        row_batch_id,
        text,
        row_label,
        row_constructiveness,
        row_domain,
        row_style,
        is_accepted,
        validation_flags,
        meta_json,
        created_at,
        reviewed_at,
        review_method,
        label_confidence,
        reviewed_by,
        review_provider,
        review_model_name,
    ) in results:
        meta: Dict[str, Any] = {}
        if isinstance(meta_json, str) and meta_json.strip():
            try:
                parsed_meta = json.loads(meta_json)
                if isinstance(parsed_meta, dict):
                    meta = parsed_meta
            except Exception:
                meta = {}

        if not meta:
            meta = build_synthetic_meta(
                sample_id=sample_id,
                batch_id=row_batch_id,
                domain=row_domain,
                style=row_style,
                model_name=SYNTHETIC_FALLBACK_MODEL,
                created_at=created_at,
            )

        flags: Dict[str, Any] = {}
        if isinstance(validation_flags, str) and validation_flags.strip():
            try:
                parsed_flags = json.loads(validation_flags)
                if isinstance(parsed_flags, dict):
                    flags = parsed_flags
            except Exception:
                flags = {}

        normalized_constructiveness = normalize_int(row_constructiveness)
        rows.append(
            {
                "id": sample_id,
                "batch_id": row_batch_id,
                "text": text,
                "label": row_label,
                "toxicity": row_label,
                "constructiveness": normalized_constructiveness if normalized_constructiveness in {0, 1} else None,
                "domain": row_domain,
                "style": row_style,
                "is_accepted": bool(is_accepted),
                "meta": meta,
                "validation_flags": flags,
                "created_at": created_at,
                "reviewed_at": reviewed_at,
                "review_method": review_method,
                "label_confidence": label_confidence,
                "reviewed_by": reviewed_by,
                "review_provider": review_provider,
                "review_model_name": review_model_name,
            }
        )

    return rows


def delete_synthetic_rows(ids: List[int]) -> int:
    if not ids:
        return 0
    normalized = [int(v) for v in ids if isinstance(v, (int, float)) or (isinstance(v, str) and str(v).isdigit())]
    if not normalized:
        return 0
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        placeholders = ", ".join(["?"] * len(normalized))
        cursor = conn.execute(
            f"DELETE FROM synthetic_dataset_row WHERE id IN ({placeholders})",
            tuple(normalized),
        )
        conn.commit()
        return cursor.rowcount or 0


def update_synthetic_review(items: List[Dict[str, Any]], reviewed_by: str) -> int:
    if not items:
        return 0
    init_feedback_db()
    now = datetime.utcnow().isoformat()

    normalized: List[Tuple[Any, ...]] = []
    for item in items:
        sample_id = normalize_int(item.get("id"))
        if sample_id is None:
            continue

        reviewed_text = item.get("text")
        cleaned_text = normalize_synthetic_text(str(reviewed_text)) if reviewed_text is not None else None
        if cleaned_text == "":
            cleaned_text = None

        reviewed_label = normalize_int(item.get("label"))
        if reviewed_label not in {0, 1}:
            reviewed_label = None
        has_constructiveness = "constructiveness" in item
        reviewed_constructiveness = normalize_int(item.get("constructiveness"))
        if reviewed_constructiveness not in {0, 1}:
            reviewed_constructiveness = None

        text_hash = build_text_hash(cleaned_text) if cleaned_text is not None else None
        review_method = str(item.get("review_method") or "manual").strip().lower()
        if review_method not in {"manual", "gemini_assisted"}:
            review_method = "manual"
        label_confidence = str(item.get("label_confidence") or "").strip().lower() or None
        if label_confidence not in {None, "low", "medium", "high"}:
            label_confidence = None
        review_provider = str(item.get("review_provider") or "").strip().lower() or None
        review_model_name = normalize_gemini_model_name(item.get("review_model_name"))
        if review_method != "gemini_assisted":
            review_provider = None
            review_model_name = None
        elif review_provider != "gemini" or not review_model_name:
            raise HTTPException(status_code=400, detail=f"Gemini review provenance is required for synthetic row {sample_id}")
        normalized.append(
            (
                1 if bool(item.get("is_accepted")) else 0,
                cleaned_text,
                reviewed_label,
                has_constructiveness,
                reviewed_constructiveness,
                text_hash,
                review_method,
                label_confidence,
                review_provider,
                review_model_name,
                sample_id,
            )
        )

    if not normalized:
        return 0

    changed = 0
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        for (
            is_accepted,
            cleaned_text,
            reviewed_label,
            has_constructiveness,
            reviewed_constructiveness,
            text_hash,
            review_method,
            label_confidence,
            review_provider,
            review_model_name,
            sample_id,
        ) in normalized:
            existing = conn.execute(
                "SELECT text, label, constructiveness, domain, style, meta_json FROM synthetic_dataset_row WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if not existing:
                continue

            old_text, old_label, old_constructiveness, domain, style, old_meta_json = existing
            final_text = cleaned_text if cleaned_text is not None else old_text
            final_label = reviewed_label if reviewed_label is not None else old_label
            final_constructiveness = reviewed_constructiveness if has_constructiveness else old_constructiveness

            meta: Dict[str, Any] = {}
            if isinstance(old_meta_json, str) and old_meta_json.strip():
                try:
                    parsed = json.loads(old_meta_json)
                    if isinstance(parsed, dict):
                        meta = parsed
                except Exception:
                    meta = {}

            if cleaned_text is not None and cleaned_text != old_text:
                meta["edited_by_reviewer"] = True
                meta["edited_at"] = now

            meta["domain"] = domain
            meta["style"] = style
            meta["toxicity"] = final_label
            meta["constructiveness"] = final_constructiveness if normalize_int(final_constructiveness) in {0, 1} else None
            meta["review_method"] = review_method
            meta["reviewed_by"] = reviewed_by
            meta["review_provider"] = review_provider
            meta["review_model_name"] = review_model_name

            conn.execute(
                """
                UPDATE synthetic_dataset_row
                SET is_accepted = ?,
                    text = ?,
                    label = ?,
                    constructiveness = ?,
                    text_hash = ?,
                    meta_json = ?,
                    reviewed_at = ?,
                    review_method = ?,
                    label_confidence = ?,
                    reviewed_by = ?,
                    review_provider = ?,
                    review_model_name = ?
                WHERE id = ?
                """,
                (
                    is_accepted,
                    final_text,
                    final_label,
                    final_constructiveness,
                    text_hash if text_hash is not None else build_text_hash(final_text),
                    json.dumps(meta, ensure_ascii=False),
                    now,
                    review_method,
                    label_confidence,
                    reviewed_by,
                    review_provider,
                    review_model_name,
                    sample_id,
                ),
            )
            changed += 1

        conn.commit()
    return changed


def summarize_synthetic_training_preview_transfer(batch_id: Optional[str] = None) -> Dict[str, Any]:
    init_feedback_db()
    clauses = ["s.reviewed_at IS NOT NULL", "s.is_accepted = 1"]
    params: List[Any] = []
    if batch_id and batch_id.strip():
        clauses.append("s.batch_id = ?")
        params.append(batch_id.strip())
    where_sql = " AND ".join(clauses)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT s.id, s.label, s.constructiveness,
                   CASE WHEN m.id IS NULL THEN 0 ELSE 1 END AS already_transferred
            FROM synthetic_dataset_row AS s
            LEFT JOIN mlflow_comment_item AS m
              ON m.source_type = 'synthetic' AND m.source_row_id = s.id
            WHERE {where_sql}
            ORDER BY s.id ASC
            """,
            tuple(params),
        ).fetchall()

    eligible = [row for row in rows if int(row["already_transferred"] or 0) == 0]
    return {
        "batch_id": batch_id.strip() if batch_id and batch_id.strip() else None,
        "eligible": len(eligible),
        "toxic": sum(1 for row in eligible if normalize_int(row["label"]) == 1),
        "clean": sum(1 for row in eligible if normalize_int(row["label"]) == 0),
        "constructive": sum(1 for row in eligible if normalize_int(row["constructiveness"]) == 1),
        "non_constructive": sum(1 for row in eligible if normalize_int(row["constructiveness"]) == 0),
        "constructiveness_masked": sum(1 for row in eligible if normalize_int(row["constructiveness"]) not in {0, 1}),
        "already_transferred": sum(1 for row in rows if int(row["already_transferred"] or 0) == 1),
        "ids": [int(row["id"]) for row in eligible],
    }


def transfer_synthetic_rows_to_training_preview(ids: List[int], admin_username: str) -> Dict[str, Any]:
    normalized_ids = sorted({int(value) for value in ids})
    if not normalized_ids:
        raise HTTPException(status_code=400, detail="No synthetic row ids provided")

    init_feedback_db()
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ", ".join(["?"] * len(normalized_ids))
    transferred = 0
    toxic = 0
    clean = 0
    skipped = 0

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            f"""
            SELECT s.id, s.batch_id, s.text, s.label, s.constructiveness, s.domain,
                   s.style, s.created_at, s.reviewed_at, s.review_method,
                   s.label_confidence, s.reviewed_by, s.review_provider,
                   s.review_model_name, b.generator_model
            FROM synthetic_dataset_row AS s
            JOIN synthetic_generation_batch AS b ON b.batch_id = s.batch_id
            WHERE s.id IN ({placeholders})
              AND s.reviewed_at IS NOT NULL
              AND s.is_accepted = 1
            ORDER BY s.id ASC
            """,
            tuple(normalized_ids),
        ).fetchall()

        for row in rows:
            source_row_id = int(row["id"])
            existing = conn.execute(
                "SELECT id FROM mlflow_comment_item WHERE source_type = 'synthetic' AND source_row_id = ?",
                (source_row_id,),
            ).fetchone()
            if existing:
                skipped += 1
                continue

            source_batch_id = str(row["batch_id"])
            mlflow_batch_id = f"synthetic_{source_batch_id}"
            options_json = json.dumps(
                {
                    "source": "synthetic_reviewed",
                    "synthetic_batch_id": source_batch_id,
                    "transferred_by": admin_username,
                    "reviewed_by": row["reviewed_by"],
                },
                ensure_ascii=False,
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO mlflow_crawl_batch (
                    batch_id, model_id, status, source_job_id, created_at, completed_at, options_json
                ) VALUES (?, ?, 'completed', NULL, ?, ?, ?)
                """,
                (mlflow_batch_id, str(row["generator_model"]), now, now, options_json),
            )

            text_value = normalize_synthetic_text(str(row["text"] or ""))
            label = normalize_int(row["label"])
            if not text_value or label not in {0, 1}:
                skipped += 1
                continue
            text_hash = hashlib.sha256(text_value.encode("utf-8")).hexdigest()
            source_url = f"synthetic://{source_batch_id}/{source_row_id}"
            url_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
            gemini_assisted = str(row["review_method"] or "manual") == "gemini_assisted"
            training_review_status = "manual_gemini" if gemini_assisted else "manual_approved"
            label_source = "gemini_assist" if gemini_assisted else "synthetic_review"
            label_confidence = str(row["label_confidence"] or "high")
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO mlflow_comment_item (
                    batch_id, job_id, url, url_hash, domain_category, segment_id, text,
                    score, pseudo_label, constructiveness_score, constructiveness_label,
                    constructiveness_confidence, selected_for_training, training_review_status,
                    gate_bucket, verification_status, segment_hash, context_segment_hash,
                    dedupe_key, html_tag, seg_threshold_used, label_source, label_confidence,
                    review_provider, review_model_name, source_type, source_row_id, created_at, reviewed_at
                ) VALUES (
                    ?, NULL, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, 1, ?,
                    'accepted', 'manual_accepted', ?, ?, ?, 'synthetic', NULL,
                    ?, ?, ?, ?, 'synthetic', ?, ?, ?
                )
                """,
                (
                    mlflow_batch_id,
                    source_url,
                    url_hash,
                    str(row["domain"]),
                    f"synthetic-{source_row_id}",
                    text_value,
                    label,
                    normalize_int(row["constructiveness"])
                    if normalize_int(row["constructiveness"]) in {0, 1}
                    else None,
                    "high" if normalize_int(row["constructiveness"]) in {0, 1} else None,
                    training_review_status,
                    text_hash,
                    text_hash,
                    f"synthetic:{source_row_id}",
                    label_source,
                    label_confidence,
                    row["review_provider"] if gemini_assisted else None,
                    row["review_model_name"] if gemini_assisted else None,
                    source_row_id,
                    str(row["created_at"] or now),
                    now,
                ),
            )
            if int(cursor.rowcount or 0) == 1:
                transferred += 1
                toxic += int(label == 1)
                clean += int(label == 0)
            else:
                skipped += 1

        skipped += len(normalized_ids) - len(rows)
        conn.commit()

    result = {
        "transferred": transferred,
        "toxic": toxic,
        "clean": clean,
        "skipped": skipped,
    }
    result["automation_scheduled_for"] = _schedule_automation_for_new_training_rows(
        transferred,
        "synthetic_training_preview_transfer",
    )
    return result


def build_synthetic_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    accepted = sum(1 for row in rows if row.get("is_accepted"))
    rejected = total - accepted

    by_domain: Dict[str, Dict[str, int]] = {}
    by_style: Dict[str, Dict[str, int]] = {}
    by_label: Dict[str, Dict[str, int]] = {}
    by_constructiveness: Dict[str, Dict[str, int]] = {}
    by_combo: Dict[str, Dict[str, int]] = {}

    for row in rows:
        domain = str(row.get("domain") or "unknown")
        style = str(row.get("style") or "unknown")
        label = str(row.get("label") if row.get("label") in {0, 1} else "unknown")
        constructiveness = str(row.get("constructiveness") if row.get("constructiveness") in {0, 1} else "masked")
        bucket_status = "accepted" if row.get("is_accepted") else "rejected"

        for group, key in [(by_domain, domain), (by_style, style), (by_label, label), (by_constructiveness, constructiveness)]:
            stats = group.setdefault(key, {"total": 0, "accepted": 0, "rejected": 0})
            stats["total"] += 1
            stats[bucket_status] += 1

        combo_key = f"{domain}|{style}|{label}|{constructiveness}"
        combo_stats = by_combo.setdefault(combo_key, {"total": 0, "accepted": 0, "rejected": 0})
        combo_stats["total"] += 1
        combo_stats[bucket_status] += 1

    return {
        "total_generated": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": round((accepted / total), 4) if total else 0.0,
        "by_domain": by_domain,
        "by_style": by_style,
        "by_label": by_label,
        "by_constructiveness": by_constructiveness,
        "by_combo": by_combo,
    }


def validate_synthetic_candidate(
    *,
    candidate: Dict[str, Any],
    expected_label: int,
    expected_constructiveness: Optional[int] = None,
    domain: str,
    style: str,
    seen_hashes: set,
    seen_fingerprints: set,
    length_bounds: Optional[Tuple[int, int, int]] = None,
) -> Optional[Dict[str, Any]]:
    text = normalize_synthetic_text(str(candidate.get("text") or ""))

    raw_label = candidate.get("label", candidate.get("toxicity"))
    label = normalize_int(raw_label)
    if label is None and isinstance(raw_label, str):
        lowered = raw_label.strip().lower()
        if lowered in {"0", "1"}:
            label = int(lowered)
        elif lowered in {"toxic", "clean", "unsure"}:
            label = safe_label_int(lowered)

    if not text or label is None:
        return None
    if label != expected_label:
        return None
    raw_constructiveness = candidate.get("constructiveness")
    constructiveness = normalize_int(raw_constructiveness)
    if constructiveness not in {0, 1}:
        constructiveness = None
    if expected_constructiveness is not None and constructiveness != expected_constructiveness:
        return None
    if PLACEHOLDER_PATTERN.search(text):
        return None

    text_hash = build_text_hash(text)
    structure_fingerprint = build_structure_fingerprint(text)
    if text_hash in seen_hashes or structure_fingerprint in seen_fingerprints:
        return None

    seen_hashes.add(text_hash)
    seen_fingerprints.add(structure_fingerprint)

    word_length = synthetic_word_length(text)
    bucket = classify_synthetic_length_bucket(word_length, length_bounds or get_synthetic_length_bounds())

    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    meta_out = {
        **meta,
        "source": "synthetic_llm",
        "split": "synthetic",
        "is_augmented": True,
        "domain": domain,
        "style": style,
        "toxicity": expected_label,
        "constructiveness": constructiveness,
        "word_length": word_length,
        "length_bucket": bucket,
    }

    return {
        "text": text,
        "label": expected_label,
        "toxicity": expected_label,
        "constructiveness": constructiveness,
        "meta": meta_out,
        "structure_fingerprint": structure_fingerprint,
        "text_hash": text_hash,
        "word_length": word_length,
        "length_bucket": bucket,
        "validation_flags": {},
    }


def cleanup_old_jobs(ttl_hours: float = 24.0) -> int:
    processed_dir = PROCESSED_DATA_DIR
    if not processed_dir.exists():
        return 0
    now = time.time()
    ttl_seconds = ttl_hours * 3600.0
    deleted = 0
    for path in processed_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("job_"):
            continue
        try:
            age_seconds = now - path.stat().st_mtime
        except OSError:
            continue
        if age_seconds >= ttl_seconds:
            try:
                shutil.rmtree(path)
                deleted += 1
            except Exception:
                logger.warning("Failed to remove job directory %s", path)
                continue
    return deleted


def resolve_mlflow_batch_id(batch_id: Optional[str], strict: bool = False) -> str:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if batch_id and batch_id.strip():
            row = conn.execute(
                "SELECT batch_id FROM mlflow_crawl_batch WHERE batch_id = ?",
                (batch_id.strip(),),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
            return str(row["batch_id"])

        if strict:
            raise HTTPException(status_code=400, detail="batch_id is required when strict_batch=true")

        latest = conn.execute(
            "SELECT batch_id FROM mlflow_crawl_batch ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not latest:
            raise HTTPException(status_code=404, detail="No mlflow batch found")
        return str(latest["batch_id"])


def resolve_mlflow_batch_id_or_none(batch_id: Optional[str], strict: bool = False) -> Optional[str]:
    try:
        return resolve_mlflow_batch_id(batch_id, strict=strict)
    except HTTPException as exc:
        requested_batch = bool(batch_id and batch_id.strip())
        if (
            exc.status_code == 404
            and not requested_batch
            and not strict
            and exc.detail == "No mlflow batch found"
        ):
            return None
        raise


def build_mlflow_gate_counts(conn: sqlite3.Connection, batch_id: str) -> Dict[str, int]:
    rows = conn.execute(
        """
        SELECT item.gate_bucket, COUNT(1) AS c
        FROM mlflow_comment_item AS item
        WHERE item.batch_id = ?
           OR EXISTS (
                SELECT 1
                FROM mlflow_comment_prediction AS prediction
                WHERE prediction.sample_item_id = item.id
                  AND prediction.batch_id = ?
           )
        GROUP BY item.gate_bucket
        """,
        (batch_id, batch_id),
    ).fetchall()
    counts = {"accepted": 0, "candidate": 0, "discarded": 0}
    for bucket, value in rows:
        key = str(bucket or "").strip().lower()
        if key in counts:
            counts[key] = int(value or 0)
    counts["total"] = counts["accepted"] + counts["candidate"] + counts["discarded"]
    return counts


def attach_mlflow_prediction_history(
    conn: sqlite3.Connection,
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_ids = [int(item["id"]) for item in items if item.get("id") is not None]
    history_by_sample: Dict[int, List[Dict[str, Any]]] = {sample_id: [] for sample_id in sample_ids}
    for group in chunked(sample_ids):
        placeholders = ", ".join("?" for _ in group)
        prediction_rows = conn.execute(
            f"""
            SELECT id, sample_item_id, batch_id, job_id, model_id,
                   raw_toxicity_score, adjusted_toxicity_score, predicted_label,
                   constructiveness_score, constructiveness_label, constructiveness_confidence, seg_threshold_used,
                   record_origin, created_at
            FROM mlflow_comment_prediction
            WHERE sample_item_id IN ({placeholders})
            ORDER BY created_at DESC, id DESC
            """,
            tuple(group),
        ).fetchall()
        for prediction_row in prediction_rows:
            prediction = dict(prediction_row)
            history_by_sample.setdefault(int(prediction["sample_item_id"]), []).append(prediction)

    for item in items:
        sample_id = int(item["id"])
        human_label = (
            normalize_int(item.get("pseudo_label"))
            if str(item.get("verification_status") or "") == "manual_accepted"
            else None
        )
        prediction_history = history_by_sample.get(sample_id, [])
        for prediction in prediction_history:
            predicted_label = normalize_int(prediction.get("predicted_label"))
            prediction["agreement_with_human"] = (
                predicted_label == human_label
                if predicted_label in {0, 1} and human_label in {0, 1}
                else None
            )
        item["human_label"] = human_label
        item["latest_prediction"] = prediction_history[0] if prediction_history else None
        item["previous_predictions"] = prediction_history[1:]
        item["prediction_history"] = prediction_history
        re_evaluation_history = [
            prediction
            for prediction in prediction_history
            if str(prediction.get("record_origin") or "") == "model_re_evaluation"
        ]
        latest_re_evaluation = re_evaluation_history[0] if re_evaluation_history else None
        review_reason = str(item.get("review_reason") or "")
        if review_reason == "model_conflict":
            re_evaluation_status = "conflict"
        elif review_reason == "model_uncertain":
            re_evaluation_status = "uncertain"
        elif review_reason.endswith("_resolved"):
            re_evaluation_status = "human_resolved"
        elif review_reason.endswith("_removed"):
            re_evaluation_status = "human_removed"
        elif latest_re_evaluation is not None:
            latest_label = normalize_int(latest_re_evaluation.get("predicted_label"))
            current_label = normalize_int(item.get("pseudo_label"))
            re_evaluation_status = (
                "agreement" if latest_label in {0, 1} and latest_label == current_label else "recorded"
            )
        else:
            re_evaluation_status = None
        item["latest_re_evaluation"] = latest_re_evaluation
        item["re_evaluation_status"] = re_evaluation_status
        item["requires_human_review"] = review_reason in {"model_conflict", "model_uncertain"}
    return items


def build_mlflow_required_bundle_contents(bundle_profile: str = "clean_victsd_gold") -> List[str]:
    if bundle_profile == "full_bundle":
        return [
            "dataset/accepted_pseudo.jsonl",
            "dataset/candidates_unverified.jsonl",
            "dataset/victsd_gold/train.jsonl",
            "dataset/victsd_gold/validation.jsonl",
            "dataset/victsd_gold/test.jsonl",
            "pseudo/accepted.jsonl",
            "pseudo/manifest.json",
            "scripts/train_phobert.py",
            "manifest.json",
            "config/training_config.yaml",
            "config/gate_policy.json",
        ]

    return [
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "build_report.json",
    ]


def infer_constructiveness_label(score: Optional[float], explicit_label: Optional[int] = None) -> Tuple[Optional[int], str]:
    if explicit_label in {0, 1}:
        return int(explicit_label), "model_label"
    if score is None:
        return None, "missing"
    if score >= 0.70:
        return 1, "high"
    if score <= 0.30:
        return 0, "high"
    return None, "masked"


def chunked(values: List[Any], size: int = 900) -> List[List[Any]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def normalize_mlflow_html_tag(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip().lower()


def get_mlflow_effective_hash(segment_hash: Optional[str], context_segment_hash: Optional[str]) -> str:
    return str(context_segment_hash or segment_hash or "").strip()


def build_mlflow_dedupe_key(
    text: str,
    html_tag: str,
    segment_hash: Optional[str],
    context_segment_hash: Optional[str],
) -> str:
    tag = normalize_mlflow_html_tag(html_tag)
    effective_hash = get_mlflow_effective_hash(segment_hash, context_segment_hash)
    if not effective_hash:
        effective_hash = build_segment_hash(text, tag)
    return f"comment_only_v3:{effective_hash}:{tag}"


def classify_mlflow_gate(
    score: float,
    accept_threshold: float,
    discard_threshold: float,
) -> Dict[str, Any]:
    if score >= accept_threshold:
        return {
            "gate_bucket": "accepted",
            "verification_status": "auto_accepted",
            "pseudo_label": 1,
            "label_source": "auto_gate",
            "label_confidence": "high",
            "selected_for_training": 1,
            "training_review_status": "auto",
        }
    if score <= discard_threshold:
        return {
            "gate_bucket": "accepted",
            "verification_status": "auto_accepted",
            "pseudo_label": 0,
            "label_source": "auto_gate",
            "label_confidence": "high",
            "selected_for_training": 1,
            "training_review_status": "auto",
        }
    return {
        "gate_bucket": "candidate",
        "verification_status": "unverified",
        "pseudo_label": 1 if score >= 0.5 else 0,
        "label_source": "auto_gate",
        "label_confidence": "medium",
        "selected_for_training": 0,
        "training_review_status": "pending",
    }


def build_mlflow_comment_rows(
    response_results: List[Dict[str, Any]],
    batch_id: str,
    job_id: str,
    accept_threshold: float,
    discard_threshold: float,
    created_at: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in response_results:
        if result.get("status") != "ok":
            continue
        url = str(result.get("url") or "")
        url_hash = str(result.get("url_hash") or hash_url(url))
        for seg in (result.get("toxicity") or {}).get("by_segment") or []:
            score = normalize_score(seg.get("score"))
            adjusted_score = normalize_score(seg.get("toxic_prob_adjusted"))
            predicted_label = normalize_int(seg.get("toxic_label"))
            text = str(seg.get("text") or seg.get("text_preview") or "").strip()
            if score is None or not text:
                continue

            seg_threshold_used = normalize_score(seg.get("seg_threshold_used"))
            constructiveness_score = normalize_score(seg.get("constructiveness_score"))
            constructiveness_label_raw = normalize_int(seg.get("constructiveness_label"))
            constructiveness_label, constructiveness_confidence = infer_constructiveness_label(
                constructiveness_score,
                constructiveness_label_raw,
            )
            segment_hash = str(seg.get("segment_hash") or "").strip() or None
            context_segment_hash = str(seg.get("context_segment_hash") or "").strip() or None
            html_tag = normalize_mlflow_html_tag(seg.get("html_tags"))
            gate = classify_mlflow_gate(score, accept_threshold, discard_threshold)

            rows.append(
                {
                    "batch_id": batch_id,
                    "job_id": job_id,
                    "url": url,
                    "url_hash": url_hash,
                    "segment_id": seg.get("segment_id"),
                    "domain_category": seg.get("domain_category") or result.get("domain_category"),
                    "text": text,
                    "score": score,
                    "adjusted_score": adjusted_score,
                    "predicted_label": predicted_label if predicted_label in {0, 1} else None,
                    "pseudo_label": gate["pseudo_label"],
                    "constructiveness_score": constructiveness_score,
                    "constructiveness_label": constructiveness_label,
                    "constructiveness_confidence": constructiveness_confidence,
                    "selected_for_training": gate["selected_for_training"],
                    "training_review_status": gate["training_review_status"],
                    "gate_bucket": gate["gate_bucket"],
                    "verification_status": gate["verification_status"],
                    "segment_hash": segment_hash,
                    "context_segment_hash": context_segment_hash,
                    "_effective_hash": get_mlflow_effective_hash(segment_hash, context_segment_hash),
                    "dedupe_key": build_mlflow_dedupe_key(text, html_tag, segment_hash, context_segment_hash),
                    "html_tag": html_tag,
                    "seg_threshold_used": seg_threshold_used,
                    "label_source": gate["label_source"],
                    "label_confidence": gate["label_confidence"],
                    "created_at": created_at,
                }
            )
    return rows


def run_mlflow_model_re_evaluation(
    rows: List[sqlite3.Row],
    model_id: str,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """Run the existing crawler inference implementation against stored canonical text."""
    model_root = resolve_model_root()
    model_type, model_name, model_path = resolve_model_path(model_root, model_id)
    resolved_model_id = f"{model_type}/{model_name}"
    thresholds_by_domain = get_effective_thresholds(resolved_model_id)

    with tempfile.TemporaryDirectory(prefix="viettoxic_model_reevaluation_") as temp_dir:
        temp_root = Path(temp_dir)
        input_dir = temp_root / "stored_samples"
        output_dir = temp_root / "predictions"
        input_dir.mkdir(parents=True, exist_ok=True)
        grouped_rows: Dict[Tuple[str, str], List[sqlite3.Row]] = {}
        for row in rows:
            group_key = (str(row["url_hash"] or ""), str(row["url"] or ""))
            grouped_rows.setdefault(group_key, []).append(row)
        sample_ids_by_folder: Dict[str, List[int]] = {}

        for group_index, ((_, url), group_rows) in enumerate(grouped_rows.items()):
            folder_name = f"stored_page_{group_index}"
            sample_ids_by_folder[folder_name] = [int(row["id"]) for row in group_rows]
            sample_dir = input_dir / folder_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "url": url or f"mlflow://stored-page/{group_index}",
                        "status": "stored_sample",
                        "method": "model_re_evaluation",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            segment_lines = [
                json.dumps(
                    {
                        "text": str(row["text"] or ""),
                        "segment_hash": row["segment_hash"],
                        "html_tag_effective": row["html_tag"],
                    },
                    ensure_ascii=False,
                )
                for row in group_rows
            ]
            (sample_dir / "segments.jsonl").write_text("\n".join(segment_lines) + "\n", encoding="utf-8")

        inference_result = infer_crawled(
            model_path=str(model_path),
            model_type=model_type,
            data_dir=str(input_dir),
            out_dir=str(output_dir),
            batch_size=8,
            max_length=256,
            page_threshold=0.25,
            seg_threshold=0.4,
            threshold_news=thresholds_by_domain.get("news"),
            threshold_social=thresholds_by_domain.get("social"),
            threshold_forum=thresholds_by_domain.get("forum"),
            threshold_unknown=thresholds_by_domain.get("unknown"),
            quiet=True,
            learned_feedback=load_learned_segments(resolved_model_id),
        )

    predictions: Dict[int, Dict[str, Any]] = {}
    prediction_index_by_folder: Dict[str, int] = {}
    for prediction in inference_result.get("segment_results") or []:
        folder_name = str(prediction.get("url_hash") or "")
        sample_ids = sample_ids_by_folder.get(folder_name) or []
        prediction_index = prediction_index_by_folder.get(folder_name, 0)
        if prediction_index >= len(sample_ids):
            continue
        sample_id = sample_ids[prediction_index]
        prediction_index_by_folder[folder_name] = prediction_index + 1
        constructiveness_score = normalize_score(
            prediction.get("constructiveness_prob", prediction.get("constructiveness_score"))
        )
        constructiveness_label, constructiveness_confidence = infer_constructiveness_label(
            constructiveness_score,
            normalize_int(prediction.get("constructiveness_label")),
        )
        predictions[sample_id] = {
            "raw_toxicity_score": normalize_score(prediction.get("score", prediction.get("toxic_prob"))),
            "adjusted_toxicity_score": normalize_score(prediction.get("toxic_prob_adjusted")),
            "predicted_label": normalize_int(prediction.get("toxic_label", prediction.get("label"))),
            "constructiveness_score": constructiveness_score,
            "constructiveness_label": constructiveness_label,
            "constructiveness_confidence": constructiveness_confidence,
            "seg_threshold_used": normalize_score(prediction.get("seg_threshold_used")),
        }
    return resolved_model_id, predictions


def load_existing_mlflow_samples(
    conn: sqlite3.Connection,
    rows: List[Dict[str, Any]],
) -> Dict[str, sqlite3.Row]:
    existing: Dict[str, sqlite3.Row] = {}
    dedupe_keys = sorted({str(row.get("dedupe_key") or "") for row in rows if row.get("dedupe_key")})
    for group in chunked(dedupe_keys):
        placeholders = ", ".join("?" for _ in group)
        db_rows = conn.execute(
            f"""
            SELECT id, dedupe_key, gate_bucket, verification_status, pseudo_label,
                   selected_for_training, training_review_status
            FROM mlflow_comment_item
            WHERE dedupe_key IN ({placeholders})
            """,
            tuple(group),
        ).fetchall()
        for db_row in db_rows:
            key = str(db_row["dedupe_key"] or "")
            if key:
                existing[key] = db_row

    hashes = sorted({str(row.get("_effective_hash") or "") for row in rows if row.get("_effective_hash")})
    if not hashes:
        return existing

    for group in chunked(hashes):
        placeholders = ", ".join("?" for _ in group)
        db_rows = conn.execute(
            f"""
            SELECT id, dedupe_key, gate_bucket, verification_status, pseudo_label,
                   selected_for_training, training_review_status,
                   COALESCE(NULLIF(context_segment_hash, ''), NULLIF(segment_hash, '')) AS effective_hash,
                   COALESCE(html_tag, '') AS html_tag
            FROM mlflow_comment_item
            WHERE COALESCE(NULLIF(context_segment_hash, ''), NULLIF(segment_hash, '')) IN ({placeholders})
            """,
            tuple(group),
        ).fetchall()
        for db_row in db_rows:
            effective_hash = db_row["effective_hash"]
            if effective_hash:
                legacy_key = f"comment_only_v3:{str(effective_hash).strip()}:{normalize_mlflow_html_tag(db_row['html_tag'])}"
                existing.setdefault(legacy_key, db_row)
    return existing


def insert_mlflow_comment_rows(
    *,
    batch_id: str,
    model_id: str,
    source_job_id: str,
    rows: List[Dict[str, Any]],
    options_json: str,
    created_at: str,
    batch_created: bool = False,
) -> Dict[str, Any]:
    init_feedback_db()
    stats: Dict[str, Any] = {
        "enabled": True,
        "batch_id": batch_id if batch_created else None,
        "candidate_rows": len(rows),
        "inserted": 0,
        "samples_inserted": 0,
        "samples_reused": 0,
        "predictions_inserted": 0,
        "skipped_existing_url": 0,
        "skipped_duplicate_item": 0,
        "counts": {"accepted": 0, "candidate": 0, "discarded": 0, "total": 0},
    }

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        existing_samples = load_existing_mlflow_samples(conn, rows)

        def ensure_batch() -> None:
            nonlocal batch_created
            if batch_created:
                return
            conn.execute(
                """
                INSERT INTO mlflow_crawl_batch (batch_id, model_id, status, source_job_id, created_at, options_json)
                VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (batch_id, model_id, source_job_id, created_at, options_json),
            )
            stats["batch_id"] = batch_id
            batch_created = True

        for row in rows:
            dedupe_key = str(row.get("dedupe_key") or "")
            sample = existing_samples.get(dedupe_key)
            sample_was_existing = sample is not None

            if sample is not None:
                duplicate_prediction = conn.execute(
                    """
                    SELECT 1
                    FROM mlflow_comment_prediction
                    WHERE sample_item_id = ? AND model_id = ?
                    """,
                    (int(sample["id"]), model_id),
                ).fetchone()
                if duplicate_prediction:
                    stats["skipped_duplicate_item"] += 1
                    continue

            ensure_batch()

            if sample is None:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO mlflow_comment_item (
                        batch_id, job_id, url, url_hash, segment_id, domain_category, text, score, pseudo_label,
                        constructiveness_score, constructiveness_label, constructiveness_confidence,
                        selected_for_training, training_review_status,
                        gate_bucket, verification_status, segment_hash, context_segment_hash, dedupe_key,
                        html_tag, seg_threshold_used, label_source, label_confidence, created_at
                    ) VALUES (
                        :batch_id, :job_id, :url, :url_hash, :segment_id, :domain_category, :text, :score, :pseudo_label,
                        :constructiveness_score, :constructiveness_label, :constructiveness_confidence,
                        :selected_for_training, :training_review_status,
                        :gate_bucket, :verification_status, :segment_hash, :context_segment_hash, :dedupe_key,
                        :html_tag, :seg_threshold_used, :label_source, :label_confidence, :created_at
                    )
                    """,
                    row,
                )
                if int(cursor.rowcount or 0) > 0:
                    stats["samples_inserted"] += 1
                    stats["inserted"] += 1
                    sample_id = int(cursor.lastrowid)
                else:
                    sample_row = conn.execute(
                        """
                        SELECT id, dedupe_key, gate_bucket, verification_status, pseudo_label,
                               selected_for_training, training_review_status
                        FROM mlflow_comment_item
                        WHERE dedupe_key = ?
                        """,
                        (dedupe_key,),
                    ).fetchone()
                    if sample_row is None:
                        raise RuntimeError(f"Unable to resolve MLflow sample for dedupe key: {dedupe_key}")
                    sample = sample_row
                    sample_id = int(sample["id"])
                    sample_was_existing = True
                sample = conn.execute(
                    """
                    SELECT id, dedupe_key, gate_bucket, verification_status, pseudo_label,
                           selected_for_training, training_review_status
                    FROM mlflow_comment_item WHERE id = ?
                    """,
                    (sample_id,),
                ).fetchone()
                existing_samples[dedupe_key] = sample
            else:
                sample_id = int(sample["id"])

            prediction_cursor = conn.execute(
                """
                INSERT OR IGNORE INTO mlflow_comment_prediction (
                    sample_item_id, batch_id, job_id, model_id,
                    raw_toxicity_score, adjusted_toxicity_score, predicted_label,
                    constructiveness_score, constructiveness_label, constructiveness_confidence, seg_threshold_used,
                    record_origin, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'inference', ?)
                """,
                (
                    sample_id,
                    batch_id,
                    row.get("job_id") or source_job_id,
                    model_id,
                    row.get("score"),
                    row.get("adjusted_score"),
                    row.get("predicted_label"),
                    row.get("constructiveness_score"),
                    row.get("constructiveness_label"),
                    row.get("constructiveness_confidence"),
                    row.get("seg_threshold_used"),
                    row.get("created_at") or created_at,
                ),
            )
            if int(prediction_cursor.rowcount or 0) == 0:
                stats["skipped_duplicate_item"] += 1
                continue

            stats["predictions_inserted"] += 1
            if sample_was_existing:
                stats["samples_reused"] += 1

            verification_status = str(sample["verification_status"] or "")
            if (
                sample_was_existing
                and verification_status not in {"manual_accepted", "manual_rejected"}
                and str(row.get("gate_bucket") or "") == "candidate"
                and str(sample["gate_bucket"] or "") != "candidate"
            ):
                conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET score = ?, pseudo_label = ?, gate_bucket = 'candidate', verification_status = 'unverified',
                        selected_for_training = 0, training_review_status = 'pending',
                        seg_threshold_used = ?, label_source = ?, label_confidence = ?, reviewed_at = NULL
                    WHERE id = ?
                    """,
                    (
                        row.get("score"),
                        row.get("pseudo_label"),
                        row.get("seg_threshold_used"),
                        row.get("label_source"),
                        row.get("label_confidence"),
                        sample_id,
                    ),
                )
                sample = conn.execute(
                    """
                    SELECT id, dedupe_key, gate_bucket, verification_status, pseudo_label,
                           selected_for_training, training_review_status
                    FROM mlflow_comment_item WHERE id = ?
                    """,
                    (sample_id,),
                ).fetchone()
                existing_samples[dedupe_key] = sample

        if batch_created:
            conn.execute(
                """
                UPDATE mlflow_crawl_batch
                SET status = 'completed', completed_at = ?
                WHERE batch_id = ?
                """,
                (datetime.utcnow().isoformat() + "Z", batch_id),
            )
            stats["counts"] = build_mlflow_gate_counts(conn, batch_id)

        conn.commit()

    stats["automation_scheduled_for"] = _schedule_automation_for_new_training_rows(
        int(stats["inserted"]),
        "mlflow_comment_collection",
    )
    return stats


def balance_training_rows(rows: List[sqlite3.Row], strategy: str) -> Tuple[List[sqlite3.Row], Dict[str, Any]]:
    toxic_rows = [row for row in rows if normalize_int(row["pseudo_label"]) == 1]
    clean_rows = [row for row in rows if normalize_int(row["pseudo_label"]) == 0]
    if strategy != "balanced_50_50" or not toxic_rows or not clean_rows:
        return rows, {
            "strategy": strategy,
            "selected_toxic": len(toxic_rows),
            "selected_clean": len(clean_rows),
            "dropped_toxic": 0,
            "dropped_clean": 0,
            "fallback": "single_class_or_all_strategy" if strategy == "balanced_50_50" else None,
        }

    limit = min(len(toxic_rows), len(clean_rows))
    balanced = toxic_rows[:limit] + clean_rows[:limit]
    balanced.sort(key=lambda row: int(row["id"]))
    return balanced, {
        "strategy": strategy,
        "selected_toxic": limit,
        "selected_clean": limit,
        "dropped_toxic": len(toxic_rows) - limit,
        "dropped_clean": len(clean_rows) - limit,
        "fallback": None,
    }


def select_mlflow_training_rows(
    conn: sqlite3.Connection,
    resolved_batch_id: Optional[str],
    balance_strategy: str,
) -> Tuple[List[sqlite3.Row], List[sqlite3.Row], Dict[str, Any]]:
    accepted_where = (
        "item.gate_bucket = 'accepted' AND item.pseudo_label IN (0, 1) "
        "AND COALESCE(item.selected_for_training, 1) = 1"
    )
    accepted_params: List[Any] = []
    if resolved_batch_id:
        accepted_where += (
            " AND (item.batch_id = ? OR EXISTS (SELECT 1 FROM mlflow_comment_prediction AS scoped_prediction "
            "WHERE scoped_prediction.sample_item_id = item.id AND scoped_prediction.batch_id = ?))"
        )
        accepted_params.extend([resolved_batch_id, resolved_batch_id])

    accepted_rows_all = conn.execute(
        f"""
        SELECT item.id, item.batch_id, item.text, item.pseudo_label, item.constructiveness_score, item.constructiveness_label,
               item.constructiveness_confidence, item.training_review_status, item.score, item.url, item.url_hash,
               item.segment_hash, item.context_segment_hash, item.html_tag, item.review_provider, item.review_model_name,
               item.source_type, item.source_row_id
        FROM mlflow_comment_item AS item
        WHERE {accepted_where}
        ORDER BY item.id ASC
        """,
        tuple(accepted_params),
    ).fetchall()
    accepted_rows, balance_stats = balance_training_rows(accepted_rows_all, balance_strategy)
    return accepted_rows_all, accepted_rows, balance_stats


def build_training_merge_plan(
    accepted_rows: List[sqlite3.Row],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], Dict[int, Dict[str, Any]]]:
    train_rows = load_victsd_gold_split("train")
    validation_rows = load_victsd_gold_split("validation")
    test_rows = load_victsd_gold_split("test")
    existing_train_texts = {
        normalize_training_text(str(item.get("text", "")))
        for item in train_rows
        if isinstance(item.get("text"), str) and normalize_training_text(str(item.get("text", "")))
    }
    base_train_count = len(existing_train_texts)
    added_to_train = 0
    skipped_empty = 0
    skipped_duplicate = 0
    row_statuses: Dict[int, Dict[str, Any]] = {}

    for row in accepted_rows:
        item_id = int(row["id"])
        normalized_text = normalize_training_text(str(row["text"] or ""))
        if not normalized_text:
            skipped_empty += 1
            row_statuses[item_id] = {
                "will_finetune": False,
                "reason_code": "empty_text",
                "reason": "Không finetune: nội dung trống sau chuẩn hóa.",
            }
            continue
        if normalized_text in existing_train_texts:
            skipped_duplicate += 1
            row_statuses[item_id] = {
                "will_finetune": False,
                "reason_code": "duplicate",
                "reason": "Không finetune: trùng nội dung đã có trong tập train.",
            }
            continue
        source_type = str(row["source_type"] or "crawl")
        source_meta: Dict[str, Any] = {
            "source": "SyntheticReviewed" if source_type == "synthetic" else "MLFlowAccepted",
            "mlflow_comment_id": item_id,
            "batch_id": row["batch_id"],
            "score": row["score"],
            "url": row["url"],
            "url_hash": row["url_hash"],
            "segment_hash": row["segment_hash"],
            "constructiveness_score": row["constructiveness_score"],
            "constructiveness_confidence": row["constructiveness_confidence"],
            "review_provider": row["review_provider"],
            "review_model_name": row["review_model_name"],
        }
        if source_type == "synthetic":
            source_meta["synthetic_row_id"] = normalize_int(row["source_row_id"])
        train_rows.append(
            {
                "text": normalized_text,
                "toxicity": int(row["pseudo_label"]),
                "constructiveness": normalize_int(row["constructiveness_label"])
                if normalize_int(row["constructiveness_label"]) in {0, 1}
                else None,
                "meta": source_meta,
            }
        )
        existing_train_texts.add(normalized_text)
        added_to_train += 1
        row_statuses[item_id] = {
            "will_finetune": True,
            "reason_code": "included",
            "reason": "Sẽ finetune: được thêm vào train.jsonl của bundle kế tiếp.",
        }

    merge_stats = {
        "base_train_count": base_train_count,
        "added_to_train": added_to_train,
        "skipped_duplicate": skipped_duplicate,
        "skipped_empty": skipped_empty,
        "final_train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
    }
    return train_rows, validation_rows, test_rows, merge_stats, row_statuses


def build_mlflow_training_plan(
    conn: sqlite3.Connection,
    resolved_batch_id: Optional[str],
    balance_strategy: str,
) -> Dict[str, Any]:
    accepted_rows_all, accepted_rows, balance_stats = select_mlflow_training_rows(
        conn,
        resolved_batch_id,
        balance_strategy,
    )
    _, _, _, merge_stats, row_statuses = build_training_merge_plan(accepted_rows)
    eligible_ids = {int(row["id"]) for row in accepted_rows_all}
    balanced_ids = {int(row["id"]) for row in accepted_rows}
    where_sql = "gate_bucket = 'accepted'"
    params: List[Any] = []
    if resolved_batch_id:
        where_sql += " AND batch_id = ?"
        params.append(resolved_batch_id)
    preview_rows = conn.execute(
        f"SELECT id, pseudo_label, selected_for_training FROM mlflow_comment_item WHERE {where_sql} ORDER BY id ASC",
        tuple(params),
    ).fetchall()

    for row in preview_rows:
        item_id = int(row["id"])
        selected = int(row["selected_for_training"] if row["selected_for_training"] is not None else 1) == 1
        label = normalize_int(row["pseudo_label"])
        if not selected:
            row_statuses[item_id] = {
                "will_finetune": False,
                "reason_code": "not_selected",
                "reason": "Không finetune: mẫu đã bị bỏ chọn khỏi training.",
            }
        elif label not in {0, 1}:
            row_statuses[item_id] = {
                "will_finetune": False,
                "reason_code": "invalid_label",
                "reason": "Không finetune: chưa có nhãn Độc hại/Sạch hợp lệ.",
            }
        elif item_id in eligible_ids and item_id not in balanced_ids:
            row_statuses[item_id] = {
                "will_finetune": False,
                "reason_code": "balance_dropped",
                "reason": "Không finetune: bị loại bởi chính sách cân bằng 50/50.",
            }

    return {
        "scope": "batch" if resolved_batch_id else "all_batches",
        "batch_id": resolved_batch_id,
        "balance_strategy": balance_strategy,
        "summary": {
            "gold_train": merge_stats["base_train_count"],
            "eligible_mlflow": len(accepted_rows_all),
            "after_balance": len(accepted_rows),
            "duplicates_skipped": merge_stats["skipped_duplicate"],
            "empty_skipped": merge_stats["skipped_empty"],
            "mlflow_added": merge_stats["added_to_train"],
            "final_train": merge_stats["final_train_count"],
            "gold_validation": merge_stats["validation_count"],
            "gold_test": merge_stats["test_count"],
        },
        "balance": balance_stats,
        "row_statuses": {str(item_id): value for item_id, value in row_statuses.items()},
    }


def build_pseudo_training_row(row: sqlite3.Row, source: str = "mlflow_pseudo") -> Dict[str, Any]:
    constructiveness_label = normalize_int(row["constructiveness_label"])
    payload: Dict[str, Any] = {
        "text": row["text"],
        "label": int(row["pseudo_label"]),
        "toxicity": int(row["pseudo_label"]),
        "constructiveness": constructiveness_label if constructiveness_label in {0, 1} else None,
        "meta": {
            "source": source,
            "batch_id": row["batch_id"],
            "score": row["score"],
            "url": row["url"],
            "url_hash": row["url_hash"],
            "segment_hash": row["segment_hash"],
            "context_segment_hash": row["context_segment_hash"],
            "html_tag": row["html_tag"],
            "constructiveness_score": row["constructiveness_score"],
            "constructiveness_confidence": row["constructiveness_confidence"],
            "training_review_status": row["training_review_status"],
        },
    }
    return payload


def normalize_training_text(text: str) -> str:
    value = unicodedata.normalize("NFC", (text or "").strip())
    return " ".join(value.split())


def load_victsd_gold_split(split_name: str) -> List[Dict[str, Any]]:
    dataset_dir = resolve_dataset_dir("victsd_gold")
    path = dataset_dir / f"{split_name}.jsonl"
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing victsd_gold split file",
                "split": split_name,
                "path": to_relative(str(path)),
            },
        )

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Invalid JSONL in victsd_gold split",
                        "split": split_name,
                        "error": str(exc),
                    },
                ) from exc
            if not isinstance(obj, dict):
                continue
            text_value = obj.get("text")
            toxicity_value = obj.get("toxicity")
            if not isinstance(text_value, str):
                continue
            if toxicity_value not in (0, 1):
                continue
            rows.append(obj)
    return rows


def do_headers() -> Dict[str, str]:
    token = os.getenv("DO_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="DO_API_TOKEN is not configured")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _do_format_http_error(exc: urllib.error.HTTPError) -> str:
    raw_error = ""
    try:
        raw_error = (exc.read() or b"").decode("utf-8", errors="replace")
    except Exception:
        raw_error = ""

    err_id = ""
    message = ""
    request_id = ""
    details = ""
    if raw_error:
        try:
            payload = json.loads(raw_error)
            if isinstance(payload, dict):
                err_id = str(payload.get("id") or "").strip()
                message = str(payload.get("message") or "").strip()
                request_id = str(payload.get("request_id") or "").strip()
                details_value = payload.get("details")
                if isinstance(details_value, str):
                    details = details_value.strip()
                elif isinstance(details_value, dict):
                    details = json.dumps(details_value, ensure_ascii=False)
                elif isinstance(details_value, list):
                    details = json.dumps(details_value, ensure_ascii=False)
        except json.JSONDecodeError:
            message = raw_error.strip()

    parts = [f"DigitalOcean API error {exc.code}"]
    if err_id:
        parts.append(f"id={err_id}")
    if message:
        parts.append(f"message={message}")
    if details:
        parts.append(f"details={details}")
    if request_id:
        parts.append(f"request_id={request_id}")

    return " | ".join(parts)


def do_call(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    else:
        url = f"{DO_API_BASE}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method.upper(), headers=do_headers(), data=body)
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=_do_format_http_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DigitalOcean API request failed: {exc}") from exc

    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _do_next_page_url(payload: Dict[str, Any]) -> Optional[str]:
    links = payload.get("links") if isinstance(payload.get("links"), dict) else None
    pages = links.get("pages") if isinstance(links, dict) and isinstance(links.get("pages"), dict) else None
    next_link = pages.get("next") if isinstance(pages, dict) else None
    if not isinstance(next_link, str) or not next_link.strip():
        return None
    return next_link.strip()


def _do_is_gpu_size(size_item: Dict[str, Any]) -> bool:
    slug = str(size_item.get("slug") or "").lower()
    description = str(size_item.get("description") or "").lower()
    text = f"{slug} {description}"
    gpu_markers = ["gpu", "h100", "a100", "l40", "v100", "nvidia", "tesla", "rtx"]
    return any(marker in text for marker in gpu_markers)


def _do_list_sizes(region: str) -> List[Dict[str, Any]]:
    sizes: List[Dict[str, Any]] = []
    next_url: Optional[str] = f"/sizes?per_page=200"
    seen: set[str] = set()

    while next_url:
        if next_url in seen:
            break
        seen.add(next_url)

        payload = do_call("GET", next_url)
        page_sizes = payload.get("sizes") if isinstance(payload.get("sizes"), list) else []
        for item in page_sizes:
            if not isinstance(item, dict):
                continue
            regions = item.get("regions") if isinstance(item.get("regions"), list) else []
            if region not in [str(r) for r in regions]:
                continue
            if not bool(item.get("available", False)):
                continue
            sizes.append(item)

        next_link = _do_next_page_url(payload)
        if not next_link:
            next_url = None
        else:
            parsed = urllib.parse.urlparse(next_link)
            if parsed.netloc:
                next_url = next_link
            elif parsed.path:
                next_url = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            else:
                next_url = None

    return sizes


def _do_extract_ssh_key_material(public_key: str) -> str:
    parts = [p for p in str(public_key or "").strip().split() if p]
    if len(parts) < 2:
        return ""
    return parts[1]


def _do_read_local_public_key_material() -> str:
    if not DO_SSH_PRIVATE_KEY_PATH:
        return ""
    private_path = Path(DO_SSH_PRIVATE_KEY_PATH).expanduser()
    public_path = Path(f"{private_path}.pub")
    if not public_path.exists() or not public_path.is_file():
        return ""
    try:
        for line in public_path.read_text(encoding="utf-8").splitlines():
            material = _do_extract_ssh_key_material(line)
            if material:
                return material
    except Exception:
        return ""
    return ""


def _do_list_account_ssh_keys() -> List[Dict[str, Any]]:
    keys: List[Dict[str, Any]] = []
    next_url: Optional[str] = "/account/keys?per_page=200"
    seen: set[str] = set()

    while next_url:
        if next_url in seen:
            break
        seen.add(next_url)

        payload = do_call("GET", next_url)
        page_keys = payload.get("ssh_keys") if isinstance(payload.get("ssh_keys"), list) else []
        for item in page_keys:
            if not isinstance(item, dict):
                continue
            key_id_raw = item.get("id")
            key_id: Optional[int] = None
            if isinstance(key_id_raw, int):
                key_id = key_id_raw
            elif isinstance(key_id_raw, str) and key_id_raw.strip().isdigit():
                key_id = int(key_id_raw.strip())
            key_fp = str(item.get("fingerprint") or "").strip()
            key_name = str(item.get("name") or "").strip()
            key_material = _do_extract_ssh_key_material(str(item.get("public_key") or ""))
            keys.append(
                {
                    "id": key_id,
                    "fingerprint": key_fp,
                    "name": key_name,
                    "material": key_material,
                }
            )

        next_link = _do_next_page_url(payload)
        if not next_link:
            next_url = None
        else:
            parsed = urllib.parse.urlparse(next_link)
            if parsed.netloc:
                next_url = next_link
            elif parsed.path:
                next_url = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            else:
                next_url = None

    return keys


def _do_pick_cpu_size(preferred_slug: Optional[str], region: str) -> Dict[str, Any]:
    sizes = _do_list_sizes(region)
    cpu_sizes = [item for item in sizes if not _do_is_gpu_size(item)]
    if not cpu_sizes:
        raise RuntimeError(f"No CPU droplet size available in region {region}")

    preferred = (preferred_slug or "").strip()
    if preferred:
        for item in cpu_sizes:
            if str(item.get("slug") or "") == preferred:
                return {
                    "slug": preferred,
                    "reason": "user_selected",
                    "vcpus": int(item.get("vcpus") or 0),
                    "memory": int(item.get("memory") or 0),
                    "price_monthly": float(item.get("price_monthly") or 0.0),
                }

    high_tier_candidates = [
        item
        for item in cpu_sizes
        if int(item.get("vcpus") or 0) >= 16 and int(item.get("memory") or 0) >= 32768
    ]
    if high_tier_candidates:
        high_tier_candidates.sort(
            key=lambda item: (
                int(item.get("vcpus") or 0),
                int(item.get("memory") or 0),
                -float(item.get("price_monthly") or 10**9),
            ),
            reverse=True,
        )
        picked = high_tier_candidates[0]
        return {
            "slug": str(picked.get("slug") or ""),
            "reason": "auto_high_tier",
            "vcpus": int(picked.get("vcpus") or 0),
            "memory": int(picked.get("memory") or 0),
            "price_monthly": float(picked.get("price_monthly") or 0.0),
        }

    threshold_candidates = [
        item
        for item in cpu_sizes
        if int(item.get("vcpus") or 0) >= DO_CPU_MIN_VCPUS and int(item.get("memory") or 0) >= DO_CPU_MIN_MEMORY_MB
    ]

    if threshold_candidates:
        threshold_candidates.sort(
            key=lambda item: (
                int(item.get("vcpus") or 0),
                int(item.get("memory") or 0),
                -float(item.get("price_monthly") or 10**9),
            ),
            reverse=True,
        )
        picked = threshold_candidates[0]
        return {
            "slug": str(picked.get("slug") or ""),
            "reason": "auto_threshold",
            "vcpus": int(picked.get("vcpus") or 0),
            "memory": int(picked.get("memory") or 0),
            "price_monthly": float(picked.get("price_monthly") or 0.0),
        }

    cpu_sizes.sort(
        key=lambda item: (
            -int(item.get("vcpus") or 0),
            -int(item.get("memory") or 0),
            float(item.get("price_monthly") or 10**9),
        )
    )
    fallback = cpu_sizes[0]
    return {
        "slug": str(fallback.get("slug") or ""),
        "reason": "auto_best_effort",
        "vcpus": int(fallback.get("vcpus") or 0),
        "memory": int(fallback.get("memory") or 0),
        "price_monthly": float(fallback.get("price_monthly") or 0.0),
    }


def _do_parse_kv_line(line: str, prefix: str) -> Dict[str, str]:
    if not line.startswith(prefix):
        return {}
    payload = line[len(prefix) :].strip()
    if not payload:
        return {}

    result: Dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key:
            continue
        result[key.strip()] = value.strip()
    return result


def _do_extract_runtime_metadata(logs: List[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for line in logs:
        meta = _do_parse_kv_line(line, "[META]")
        if meta:
            output["compute_mode"] = meta.get("compute_mode") or output.get("compute_mode")
            output["droplet_profile"] = meta.get("droplet_size") or output.get("droplet_profile")
            output["model_kind"] = meta.get("model_kind") or output.get("model_kind")
            output["training_mode"] = meta.get("training_mode") or output.get("training_mode")
            output["base_model"] = meta.get("base_model") or output.get("base_model")

        eta = _do_parse_kv_line(line, "[ETA]")
        if eta and eta.get("estimate_minutes"):
            try:
                output["eta_estimate_minutes"] = int(eta["estimate_minutes"])
            except Exception:
                pass

        timing = _do_parse_kv_line(line, "[TIMING]")
        if timing:
            if timing.get("train_started_at"):
                output["train_started_at"] = timing.get("train_started_at")
            if timing.get("train_finished_at"):
                output["train_finished_at"] = timing.get("train_finished_at")
            if timing.get("duration_minutes"):
                try:
                    output["train_duration_minutes"] = float(timing["duration_minutes"])
                except Exception:
                    pass

        telemetry = _do_parse_kv_line(line, "[TELEMETRY]")
        if telemetry:
            if telemetry.get("sampled_at"):
                output["telemetry_last_sample_at"] = telemetry.get("sampled_at")
            if telemetry.get("interval_sec"):
                try:
                    output["telemetry_interval_sec"] = int(telemetry["interval_sec"])
                except Exception:
                    pass
            if telemetry.get("cpu_pct"):
                try:
                    output["cpu_percent"] = float(telemetry["cpu_pct"])
                except Exception:
                    pass
            if telemetry.get("memory_pct"):
                try:
                    output["memory_percent"] = float(telemetry["memory_pct"])
                except Exception:
                    pass

    return output


def _do_resolve_training_mode(request: "MlflowDOTriggerRequest") -> str:
    if request.model_kind == "lr_smoke":
        return "retrain"
    mode = (request.training_mode or "retrain").strip().lower()
    if mode not in {"retrain", "finetune"}:
        return "retrain"
    return mode


def _do_resolve_base_model(request: "MlflowDOTriggerRequest") -> Optional[str]:
    if request.model_kind == "lr_smoke":
        return None
    if _do_resolve_training_mode(request) == "retrain":
        return PHOBERT_V2_BASE_MODEL
    raw = (request.base_model or "").strip()
    if not raw:
        raise RuntimeError("PhoBERT finetune requires an explicitly selected base_model")
    if any(x in raw for x in ("..", "\\")):
        raise RuntimeError("Invalid base_model")
    return raw


def _validate_phobert_checkpoint_dir(model_path: Path, *, label: str) -> None:
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (model_path / name).is_file()]
    if not ((model_path / "model.safetensors").is_file() or (model_path / "pytorch_model.bin").is_file()):
        missing.append("model.safetensors or pytorch_model.bin")
    if not ((model_path / "vocab.txt").is_file() or (model_path / "tokenizer.json").is_file()):
        missing.append("vocab.txt or tokenizer.json")
    if missing:
        raise RuntimeError(f"PhoBERT finetune base model '{label}' is incomplete: missing {', '.join(missing)}")


def _resolve_phobert_finetune_base_model(model_id: str) -> Tuple[str, Path, Dict[str, Any]]:
    try:
        model_type, model_name, model_path = resolve_model_path(resolve_model_root(), model_id)
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve PhoBERT finetune base model '{model_id}': {exc}") from exc
    if model_type != "phobert":
        raise RuntimeError("PhoBERT finetune requires a selected PhoBERT model artifact")
    _validate_phobert_checkpoint_dir(model_path, label=model_id)
    resolved_model_id = f"{model_type}/{model_name}"
    provenance: Dict[str, Any] = {
        "type": "existing_model_artifact",
        "model_id": resolved_model_id,
        "source_path": str(model_path),
    }
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute(
            "SELECT source_run_id, artifact_checksum, artifact_uri, status FROM mlflow_model_version WHERE model_id = ?",
            (resolved_model_id,),
        ).fetchone()
    if row:
        provenance.update({
            "source_run_id": row[0],
            "artifact_checksum": row[1],
            "artifact_uri": row[2],
            "registry_status": row[3],
        })
    return resolved_model_id, model_path, provenance


def _mlflow_stages_for_mode(compute_mode: Optional[str]) -> List[str]:
    mode = (compute_mode or "").strip().lower()
    if mode == "local_m1":
        return LOCAL_M1_STAGES
    return DO_STAGES


def _do_get_run(conn: sqlite3.Connection, run_id: str) -> Optional[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT run_id, batch_id, provider, gpu_profile, status, current_stage, logs_json,
               created_at, updated_at, droplet_id, artifact_uri, artifact_checksum,
               spaces_bucket, spaces_key, error_message, bundle_path, bundle_url,
               bundle_checksum, bundle_manifest_json
        FROM mlflow_do_run
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()


def _do_extract_droplet_ipv4(droplet_id: str) -> Optional[str]:
    details = do_call("GET", f"/droplets/{droplet_id}")
    droplet = details.get("droplet") if isinstance(details, dict) else None
    if not isinstance(droplet, dict):
        return None

    networks = droplet.get("networks") if isinstance(droplet.get("networks"), dict) else {}
    v4 = networks.get("v4") if isinstance(networks.get("v4"), list) else []
    for item in v4:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "public" and item.get("ip_address"):
            return str(item.get("ip_address"))
    return None


def _do_run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        return completed.stdout or ""
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        tail = output[-4000:]
        raise RuntimeError(
            f"Command failed (exit={exc.returncode}) ({' '.join(cmd)}): {tail}"
        ) from exc


def _do_ssh_base(ip_addr: str) -> List[str]:
    if not DO_SSH_PRIVATE_KEY_PATH:
        raise RuntimeError("DO_SSH_PRIVATE_KEY_PATH is required for SSH bootstrap")
    return [
        "ssh",
        "-i",
        DO_SSH_PRIVATE_KEY_PATH,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{DO_SSH_USER}@{ip_addr}",
    ]


def _do_scp_base() -> List[str]:
    if not DO_SSH_PRIVATE_KEY_PATH:
        raise RuntimeError("DO_SSH_PRIVATE_KEY_PATH is required for SCP")
    return [
        "scp",
        "-i",
        DO_SSH_PRIVATE_KEY_PATH,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


def _do_ssh_exec(ip_addr: str, command: str) -> str:
    return _do_run_cmd(_do_ssh_base(ip_addr) + [command])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _do_update_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    logs: Optional[List[Any]] = None,
    gpu_profile: Optional[str] = None,
    droplet_id: Optional[str] = None,
    artifact_uri: Optional[str] = None,
    artifact_checksum: Optional[str] = None,
    spaces_bucket: Optional[str] = None,
    spaces_key: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    fields: List[str] = ["updated_at = ?"]
    values: List[Any] = [datetime.utcnow().isoformat() + "Z"]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if stage is not None:
        fields.append("current_stage = ?")
        values.append(stage)
    if logs is not None:
        fields.append("logs_json = ?")
        values.append(json.dumps(logs, ensure_ascii=False))
    if gpu_profile is not None:
        fields.append("gpu_profile = ?")
        values.append(gpu_profile)
    if droplet_id is not None:
        fields.append("droplet_id = ?")
        values.append(droplet_id)
    if artifact_uri is not None:
        fields.append("artifact_uri = ?")
        values.append(artifact_uri)
    if artifact_checksum is not None:
        fields.append("artifact_checksum = ?")
        values.append(artifact_checksum)
    if spaces_bucket is not None:
        fields.append("spaces_bucket = ?")
        values.append(spaces_bucket)
    if spaces_key is not None:
        fields.append("spaces_key = ?")
        values.append(spaces_key)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)

    values.append(run_id)
    conn.execute(f"UPDATE mlflow_do_run SET {', '.join(fields)} WHERE run_id = ?", tuple(values))


def _do_load_logs(row: sqlite3.Row) -> List[str]:
    events = _do_load_log_events(row)
    return [str(item.get("message") or "") for item in events if str(item.get("message") or "").strip()]


def _do_load_log_events(row: sqlite3.Row) -> List[Dict[str, Any]]:
    raw_logs = row["logs_json"] if row else None
    if not raw_logs:
        return []

    events: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(raw_logs)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    for entry in parsed:
        if isinstance(entry, dict):
            message = str(entry.get("message") or "").strip()
            if not message:
                continue
            events.append(
                {
                    "ts": str(entry.get("ts") or "").strip() or None,
                    "message": message,
                    "stage": str(entry.get("stage") or "").strip() or None,
                    "source": str(entry.get("source") or "").strip() or None,
                }
            )
            continue

        text = str(entry).strip()
        if not text:
            continue
        events.append({"ts": None, "message": text, "stage": None, "source": None})

    return events


def _do_append_log(
    conn: sqlite3.Connection,
    run_id: str,
    message: str,
    *,
    stage: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    row = _do_get_run(conn, run_id)
    events = _do_load_log_events(row) if row else []
    events.append(
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "message": message,
            "stage": stage,
            "source": source,
        }
    )
    logs: List[Any] = events
    _do_update_run(conn, run_id, logs=logs)


def _do_infer_run_mode(job_id: Optional[str], artifact_uri: Optional[str]) -> str:
    jid = str(job_id or "").strip().lower()
    uri = str(artifact_uri or "").strip().lower()
    if jid.startswith("mock_") or uri.startswith("mock://"):
        return "mock"
    if jid or uri:
        return "real"
    return "unknown"


def _do_infer_status_source(status_url: str, run_mode: str) -> str:
    if run_mode == "mock":
        return "mock_webhook"
    if status_url:
        return "status_webhook"
    return "local_db"


def _do_build_stage_timestamps(events: List[Dict[str, Any]], row: sqlite3.Row) -> Dict[str, Optional[str]]:
    stage_ts: Dict[str, Optional[str]] = {stage: None for stage in KAGGLE_STAGES}

    created_at = str(row["created_at"] or "").strip() or None
    updated_at = str(row["updated_at"] or "").strip() or None
    current_stage = str(row["current_stage"] or "").strip()

    if created_at:
        stage_ts[KAGGLE_STAGES[0]] = created_at

    for event in events:
        stage = str(event.get("stage") or "").strip()
        ts = str(event.get("ts") or "").strip()
        if stage and ts and stage in stage_ts and not stage_ts[stage]:
            stage_ts[stage] = ts

    if current_stage in stage_ts and updated_at:
        stage_ts[current_stage] = stage_ts.get(current_stage) or updated_at

    return stage_ts



@app.post("/api/mlflow/ingest", dependencies=[Depends(require_admin)])
def mlflow_ingest(request: MlflowIngestRequest) -> Dict[str, Any]:
    try:
        cleanup_old_jobs(float(os.getenv("JOB_RETENTION_HOURS", "24")))
        options = request.options or MlflowIngestOptions()

        urls = normalize_input_urls(request.urls)
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")

        accept_threshold = float(options.gate_accept_threshold)
        discard_threshold = float(options.gate_discard_threshold)
        if discard_threshold > accept_threshold:
            raise HTTPException(status_code=400, detail="gate_discard_threshold must be <= gate_accept_threshold")

        model_root = resolve_model_root()
        try:
            model_type, model_name, model_path = resolve_model_path(model_root, options.model_name)
            model_id = f"{model_type}/{model_name}"
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (PermissionError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to access model directory: {exc}") from exc

        source_job_id = uuid.uuid4().hex
        out_dir = PROCESSED_DATA_DIR / f"job_{source_job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        batch_id = f"mlf_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat() + "Z"

        init_feedback_db()
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO mlflow_crawl_batch (batch_id, model_id, status, source_job_id, created_at, options_json)
                VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (
                    batch_id,
                    model_id,
                    source_job_id,
                    now,
                    json.dumps(
                        {
                            "source": "admin_ingest",
                            "batch_size": options.batch_size,
                            "max_length": options.max_length,
                            "page_threshold": options.page_threshold,
                            "seg_threshold": options.seg_threshold,
                            "gate_accept_threshold": accept_threshold,
                            "gate_discard_threshold": discard_threshold,
                            "persist_unused": options.persist_unused,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            conn.commit()

        crawl_results = crawl_urls(
            urls,
            out_dir=str(DATA_DIR),
            timeout=options.crawl_timeout_sec,
            max_load_more=options.max_load_more_clicks,
            max_comments_per_url=options.max_comments_per_url,
        )
        ok_hashes = [r["url_hash"] for r in crawl_results if r.get("status") == "ok"]

        crawl_status_counts = {
            "ok": 0,
            "blocked": 0,
            "no_comments": 0,
            "unsupported": 0,
            "error": 0,
            "from_cache": 0,
            "retried": 0,
        }
        crawl_timeout_count = 0
        for crawl in crawl_results:
            crawl_status = str(crawl.get("crawl_status") or "error")
            if crawl_status in crawl_status_counts:
                crawl_status_counts[crawl_status] += 1
            else:
                crawl_status_counts["error"] += 1
            if crawl.get("from_cache"):
                crawl_status_counts["from_cache"] += 1
            attempts = int(crawl.get("attempts") or 1)
            if attempts > 1:
                crawl_status_counts["retried"] += 1
            warnings = [str(w).lower() for w in (crawl.get("warnings") or [])]
            if any("timeout" in w for w in warnings):
                crawl_timeout_count += 1

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=source_job_id,
                urls=urls,
                url_hashes=ok_hashes,
                model_ids=[model_id],
                enable_video=False,
                merged_used=False,
            ),
        )

        thresholds_by_domain = get_effective_thresholds(model_id)
        if ok_hashes:
            infer_crawled(
                model_path=str(model_path),
                model_type=model_type,
                data_dir=str(DATA_DIR),
                out_dir=str(out_dir),
                batch_size=options.batch_size,
                max_length=options.max_length,
                page_threshold=options.page_threshold,
                seg_threshold=options.seg_threshold,
                threshold_news=thresholds_by_domain.get("news"),
                threshold_social=thresholds_by_domain.get("social"),
                threshold_forum=thresholds_by_domain.get("forum"),
                threshold_unknown=thresholds_by_domain.get("unknown"),
                only_url_hashes=ok_hashes,
                quiet=True,
                learned_feedback=load_learned_segments(),
                html_dir=str(DATA_DIR),
            )

        page_by_hash, page_by_url = load_page_results_map(out_dir)
        segment_results = load_segment_results(out_dir)
        seg_by_hash: Dict[str, List[Dict[str, Any]]] = {}
        seg_by_url: Dict[str, List[Dict[str, Any]]] = {}
        for seg in segment_results:
            if seg.get("url_hash"):
                seg_by_hash.setdefault(seg["url_hash"], []).append(seg)
            if seg.get("url"):
                seg_by_url.setdefault(seg["url"], []).append(seg)

        response_results = map_results_to_response(
            crawl_results,
            page_by_hash,
            page_by_url,
            seg_by_hash,
            seg_by_url,
        )

        created_at = datetime.utcnow().isoformat() + "Z"
        mlflow_options_json = json.dumps(
            {
                "source": "admin_ingest",
                "batch_size": options.batch_size,
                "max_length": options.max_length,
                "page_threshold": options.page_threshold,
                "seg_threshold": options.seg_threshold,
                "gate_accept_threshold": accept_threshold,
                "gate_discard_threshold": discard_threshold,
                "persist_unused": options.persist_unused,
            },
            ensure_ascii=False,
        )
        mlflow_collection = insert_mlflow_comment_rows(
            batch_id=batch_id,
            model_id=model_id,
            source_job_id=source_job_id,
            rows=build_mlflow_comment_rows(
                response_results,
                batch_id,
                source_job_id,
                accept_threshold,
                discard_threshold,
                created_at,
            ),
            options_json=mlflow_options_json,
            created_at=now,
            batch_created=True,
        )
        counts = mlflow_collection["counts"]

        return {
            "batch_id": batch_id,
            "source_job_id": source_job_id,
            "model_name": model_id,
            "status": "completed",
            "gate_thresholds": {
                "accept": accept_threshold,
                "discard": discard_threshold,
            },
            "counts": counts,
            "dedupe": {
                "candidate_rows": mlflow_collection["candidate_rows"],
                "inserted": mlflow_collection["inserted"],
                "samples_inserted": mlflow_collection["samples_inserted"],
                "samples_reused": mlflow_collection["samples_reused"],
                "predictions_inserted": mlflow_collection["predictions_inserted"],
                "skipped_existing_url": mlflow_collection["skipped_existing_url"],
                "skipped_duplicate_item": mlflow_collection["skipped_duplicate_item"],
            },
            "crawl_summary": {
                "status_counts": crawl_status_counts,
                "timeout_count": crawl_timeout_count,
                "total_urls": len(crawl_results),
            },
            "created_at": now,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("MLFlow ingest failed")
        raise HTTPException(status_code=500, detail=f"MLFlow ingest failed: {exc}")


@app.get("/api/mlflow/overview", dependencies=[Depends(require_admin)])
def mlflow_overview(
    batch_id: Optional[str] = None,
    strict_batch: bool = Query(default=False),
) -> Dict[str, Any]:
    resolved_batch_id = resolve_mlflow_batch_id_or_none(batch_id, strict=strict_batch)
    if not resolved_batch_id:
        return {
            "active_batch_id": "",
            "model_name": None,
            "status": "empty",
            "source_job_id": None,
            "last_run_at": None,
            "pipeline_counts": {
                "crawled": 0,
                "inferred": 0,
                "accepted": 0,
                "candidate": 0,
                "discarded": 0,
            },
            "has_data": False,
        }

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        batch_row = conn.execute(
            """
            SELECT batch_id, model_id, status, source_job_id, created_at, completed_at
            FROM mlflow_crawl_batch
            WHERE batch_id = ?
            """,
            (resolved_batch_id,),
        ).fetchone()
        if not batch_row:
            raise HTTPException(status_code=404, detail=f"Batch not found: {resolved_batch_id}")

        counts = build_mlflow_gate_counts(conn, resolved_batch_id)

    return {
        "active_batch_id": resolved_batch_id,
        "model_name": batch_row["model_id"],
        "status": batch_row["status"],
        "source_job_id": batch_row["source_job_id"],
        "last_run_at": batch_row["completed_at"] or batch_row["created_at"],
        "pipeline_counts": {
            "crawled": counts["total"],
            "inferred": counts["total"],
            "accepted": counts["accepted"],
            "candidate": counts["candidate"],
            "discarded": counts["discarded"],
        },
        "has_data": True,
    }


@app.get("/api/mlflow/batches", dependencies=[Depends(require_admin)])
def mlflow_batches(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT batch_id, model_id, status, source_job_id, created_at, completed_at
            FROM mlflow_crawl_batch
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            batch_id = str(row["batch_id"])
            counts = build_mlflow_gate_counts(conn, batch_id)
            items.append(
                {
                    "batch_id": batch_id,
                    "model_id": row["model_id"],
                    "status": row["status"],
                    "source_job_id": row["source_job_id"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                    "counts": counts,
                }
            )

    return {"items": items, "total": len(items)}


@app.get("/api/mlflow/crawl-history", dependencies=[Depends(require_admin)])
def mlflow_crawl_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    offset = (page - 1) * page_size
    init_feedback_db()

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(
            """
            WITH batch_samples AS (
                SELECT batch_id, id AS sample_item_id, created_at AS seen_at
                FROM mlflow_comment_item
                UNION ALL
                SELECT batch_id, sample_item_id, created_at AS seen_at
                FROM mlflow_comment_prediction
            ), resolved AS (
                SELECT batch_id, sample_item_id, MAX(seen_at) AS seen_at
                FROM batch_samples
                GROUP BY batch_id, sample_item_id
            )
            SELECT COUNT(1)
            FROM (
                SELECT resolved.batch_id, item.url_hash
                FROM resolved
                JOIN mlflow_comment_item AS item ON item.id = resolved.sample_item_id
                GROUP BY resolved.batch_id, item.url_hash
            )
            """
        ).fetchone()
        total = int(total_row[0] if total_row else 0)

        comment_item_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(mlflow_comment_item)").fetchall()
            if len(row) >= 2
        }
        domain_category_expr = (
            "MAX(item.domain_category) AS domain_category"
            if "domain_category" in comment_item_columns
            else "NULL AS domain_category"
        )

        rows = conn.execute(
            f"""
            WITH batch_samples AS (
                SELECT batch_id, id AS sample_item_id, created_at AS seen_at
                FROM mlflow_comment_item
                UNION ALL
                SELECT batch_id, sample_item_id, created_at AS seen_at
                FROM mlflow_comment_prediction
            ), resolved AS (
                SELECT batch_id, sample_item_id, MAX(seen_at) AS seen_at
                FROM batch_samples
                GROUP BY batch_id, sample_item_id
            )
            SELECT
                resolved.batch_id,
                item.url,
                item.url_hash,
                {domain_category_expr},
                COUNT(1) AS segment_count,
                SUM(CASE WHEN item.gate_bucket = 'accepted' THEN 1 ELSE 0 END) AS accepted_count,
                SUM(CASE WHEN item.gate_bucket = 'candidate' THEN 1 ELSE 0 END) AS candidate_count,
                SUM(CASE WHEN item.gate_bucket = 'discarded' THEN 1 ELSE 0 END) AS discarded_count,
                MAX(resolved.seen_at) AS last_seen_at
            FROM resolved
            JOIN mlflow_comment_item AS item ON item.id = resolved.sample_item_id
            GROUP BY resolved.batch_id, item.url_hash, item.url
            ORDER BY last_seen_at DESC
            LIMIT ? OFFSET ?
            """,
            (page_size, offset),
        ).fetchall()
        items = [dict(row) for row in rows]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/api/mlflow/clear-batch", dependencies=[Depends(require_admin)])
def mlflow_clear_batch(request: MlflowClearBatchRequest) -> Dict[str, Any]:
    batch_id = resolve_mlflow_batch_id(request.batch_id, strict=True)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        deleted_do_run = int(
            conn.execute("DELETE FROM mlflow_do_run WHERE batch_id = ?", (batch_id,)).rowcount or 0
        )
        deleted_prediction = int(
            conn.execute("DELETE FROM mlflow_comment_prediction WHERE batch_id = ?", (batch_id,)).rowcount or 0
        )

        owned_samples = conn.execute(
            """
            SELECT id, verification_status, selected_for_training, is_locked
            FROM mlflow_comment_item
            WHERE batch_id = ?
            ORDER BY id ASC
            """,
            (batch_id,),
        ).fetchall()
        deleted_comment_item = 0
        preserved_comment_item = 0
        reassigned_comment_item = 0
        for sample in owned_samples:
            sample_id = int(sample["id"])
            replacement_prediction = conn.execute(
                """
                SELECT batch_id
                FROM mlflow_comment_prediction
                WHERE sample_item_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (sample_id,),
            ).fetchone()
            if replacement_prediction is not None:
                conn.execute(
                    "UPDATE mlflow_comment_item SET batch_id = ? WHERE id = ?",
                    (str(replacement_prediction["batch_id"]), sample_id),
                )
                reassigned_comment_item += 1
                continue

            requires_preservation = (
                str(sample["verification_status"] or "") in {"manual_accepted", "manual_rejected"}
                or int(sample["selected_for_training"] or 0) == 1
                or int(sample["is_locked"] or 0) == 1
            )
            if requires_preservation:
                preserved_comment_item += 1
                continue
            deleted_comment_item += int(
                conn.execute("DELETE FROM mlflow_comment_item WHERE id = ?", (sample_id,)).rowcount or 0
            )

        if preserved_comment_item:
            deleted_crawl_batch = 0
        else:
            deleted_crawl_batch = int(
                conn.execute("DELETE FROM mlflow_crawl_batch WHERE batch_id = ?", (batch_id,)).rowcount or 0
            )

        conn.commit()

    return {
        "scope": "batch",
        "batch_id": batch_id,
        "deleted_rows": {
            "mlflow_do_run": deleted_do_run,
            "mlflow_comment_prediction": deleted_prediction,
            "mlflow_comment_item": deleted_comment_item,
            "mlflow_comment_item_preserved": preserved_comment_item,
            "mlflow_comment_item_reassigned": reassigned_comment_item,
            "mlflow_crawl_batch": deleted_crawl_batch,
            "mlflow_training_artifact": 0,
        },
    }


@app.post("/api/mlflow/clear-all", dependencies=[Depends(require_admin)])
def mlflow_clear_all(request: MlflowClearAllRequest) -> Dict[str, Any]:
    confirm_token = (request.confirm_token or "").strip()
    if confirm_token != MLFLOW_CLEAR_ALL_CONFIRM_TOKEN:
        raise HTTPException(status_code=400, detail="Invalid confirm_token")

    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        deleted_do_run = int(conn.execute("DELETE FROM mlflow_do_run").rowcount or 0)
        deleted_training_artifact = int(conn.execute("DELETE FROM mlflow_training_artifact").rowcount or 0)
        deleted_prediction = int(conn.execute("DELETE FROM mlflow_comment_prediction").rowcount or 0)
        deleted_comment_item = int(conn.execute("DELETE FROM mlflow_comment_item").rowcount or 0)
        deleted_crawl_batch = int(conn.execute("DELETE FROM mlflow_crawl_batch").rowcount or 0)

        conn.commit()

    return {
        "scope": "all",
        "deleted_rows": {
            "mlflow_do_run": deleted_do_run,
            "mlflow_training_artifact": deleted_training_artifact,
            "mlflow_comment_prediction": deleted_prediction,
            "mlflow_comment_item": deleted_comment_item,
            "mlflow_crawl_batch": deleted_crawl_batch,
        },
    }


@app.get("/api/mlflow/review-history", dependencies=[Depends(require_admin)])
def mlflow_review_history(
    batch_id: Optional[str] = None,
    decision: str = Query(default="all"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    strict_batch: bool = Query(default=False),
    scope: Literal["batch", "all_batches"] = Query(default="batch"),
) -> Dict[str, Any]:
    resolved_batch_id: Optional[str] = None
    offset = (page - 1) * page_size

    decision_normalized = (decision or "all").strip().lower()
    where_parts = ["item.verification_status != 'unverified'"]
    params: List[Any] = []

    if scope == "batch":
        resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
        where_parts.append(
            "(item.batch_id = ? OR EXISTS (SELECT 1 FROM mlflow_comment_prediction AS prediction "
            "WHERE prediction.sample_item_id = item.id AND prediction.batch_id = ?))"
        )
        params.extend([resolved_batch_id, resolved_batch_id])

    if decision_normalized == "accepted":
        where_parts.append("item.gate_bucket = 'accepted'")
    elif decision_normalized == "rejected":
        where_parts.append("item.verification_status = 'manual_rejected'")
    elif decision_normalized == "discarded":
        where_parts.append("item.gate_bucket = 'discarded'")
    elif decision_normalized != "all":
        raise HTTPException(status_code=400, detail=f"Unsupported decision filter: {decision}")

    where_sql = " AND ".join(where_parts)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(
            f"SELECT COUNT(1) FROM mlflow_comment_item AS item WHERE {where_sql}",
            tuple(params),
        ).fetchone()
        total = int(total_row[0] if total_row else 0)

        rows = conn.execute(
            f"""
            SELECT item.id, item.batch_id, item.url, item.url_hash, item.segment_id, item.domain_category, item.text, item.score,
                   item.pseudo_label, item.constructiveness_score, item.constructiveness_label, item.constructiveness_confidence,
                   item.selected_for_training, item.training_review_status, item.is_locked, item.gate_bucket, item.verification_status,
                   item.segment_hash, item.context_segment_hash, item.html_tag,
                   item.seg_threshold_used, item.label_source, item.label_confidence, item.review_provider, item.review_model_name,
                   item.review_reason,
                   item.created_at, item.reviewed_at
            FROM mlflow_comment_item AS item
            WHERE {where_sql}
            ORDER BY COALESCE(item.reviewed_at, item.created_at) DESC, item.id DESC
            LIMIT ? OFFSET ?
            """,
            tuple([*params, page_size, offset]),
        ).fetchall()
        items = attach_mlflow_prediction_history(conn, [dict(row) for row in rows])

    return {
        "scope": scope,
        "batch_id": resolved_batch_id,
        "decision": decision_normalized,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/mlflow/candidates", dependencies=[Depends(require_admin)])
def mlflow_candidates(
    batch_id: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    strict_batch: bool = Query(default=False),
    scope: Literal["batch", "all_batches"] = Query(default="batch"),
) -> Dict[str, Any]:
    resolved_batch_id: Optional[str] = None
    offset = (page - 1) * page_size
    where_parts = [
        "item.gate_bucket = 'candidate'",
        "item.verification_status = 'unverified'",
    ]
    params: List[Any] = []

    if scope == "batch":
        resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
        where_parts.insert(
            0,
            "(item.batch_id = ? OR EXISTS (SELECT 1 FROM mlflow_comment_prediction AS prediction "
            "WHERE prediction.sample_item_id = item.id AND prediction.batch_id = ?))",
        )
        params.extend([resolved_batch_id, resolved_batch_id])

    where_sql = " AND ".join(where_parts)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(
            f"""
            SELECT COUNT(1)
            FROM mlflow_comment_item AS item
            WHERE {where_sql}
            """,
            tuple(params),
        ).fetchone()
        total = int(total_row[0] if total_row else 0)

        rows = conn.execute(
            f"""
            SELECT item.id, item.batch_id, item.url, item.url_hash, item.segment_id, item.domain_category, item.text, item.score,
                   item.pseudo_label, item.constructiveness_score, item.constructiveness_label, item.constructiveness_confidence,
                   item.selected_for_training, item.training_review_status, item.is_locked, item.gate_bucket, item.verification_status,
                   item.segment_hash, item.context_segment_hash, item.html_tag,
                   item.seg_threshold_used, item.label_source, item.label_confidence, item.review_provider, item.review_model_name,
                   item.review_reason,
                   item.created_at, item.reviewed_at
            FROM mlflow_comment_item AS item
            WHERE {where_sql}
            ORDER BY item.id DESC
            LIMIT ? OFFSET ?
            """,
            tuple([*params, page_size, offset]),
        ).fetchall()
        items = attach_mlflow_prediction_history(conn, [dict(row) for row in rows])

    return {
        "scope": scope,
        "batch_id": resolved_batch_id,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/api/mlflow/candidates/gemini-review", dependencies=[Depends(require_admin)])
def mlflow_candidates_gemini_review(request: MlflowTrainingPreviewGeminiReviewRequest) -> Dict[str, Any]:
    init_feedback_db()
    ids = list(dict.fromkeys(request.ids))
    validate_gemini_review_item_limit(ids)
    placeholders = ", ".join(["?"] * len(ids))
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, batch_id, url, domain_category, text, score, pseudo_label,
                   constructiveness_score, constructiveness_label, gate_bucket,
                   selected_for_training, training_review_status, is_locked
            FROM mlflow_comment_item
            WHERE id IN ({placeholders})
              AND gate_bucket = 'candidate'
              AND verification_status = 'unverified'
            ORDER BY id ASC
            """,
            tuple(ids),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No manual verify rows found for provided ids")
    suggestions = run_mlflow_gemini_review(rows)
    failed_ids = sorted({int(row["id"]) for row in rows} - {int(item["id"]) for item in suggestions})
    if not suggestions:
        raise HTTPException(status_code=502, detail="Gemini could not produce valid review suggestions after retrying")
    return {**build_gemini_review_response(suggestions, len(ids)), "failed_ids": failed_ids}


@app.post("/api/mlflow/re-evaluate", dependencies=[Depends(require_admin)])
def mlflow_model_re_evaluate(request: MlflowModelReEvaluationRequest) -> Dict[str, Any]:
    init_feedback_db()
    requested_model_id = request.model_id.strip()
    try:
        model_type, model_name, _ = resolve_model_path(resolve_model_root(), requested_model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (PermissionError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to access model directory: {exc}") from exc
    model_id = f"{model_type}/{model_name}"

    sample_ids = list(dict.fromkeys(int(sample_id) for sample_id in request.sample_ids))
    if request.selection == "selected" and not sample_ids:
        raise HTTPException(status_code=400, detail="sample_ids is required for selected re-evaluation")

    resolved_batch_id: Optional[str] = None
    if request.training_scope == "batch":
        if not request.batch_id:
            raise HTTPException(status_code=400, detail="batch_id is required when training_scope=batch")
        resolved_batch_id = resolve_mlflow_batch_id(request.batch_id, strict=True)

    where_parts: List[str] = []
    params: List[Any] = []
    if request.selection == "all_auto_eligible":
        where_parts.extend(
            [
                "item.verification_status = 'auto_accepted'",
                "item.gate_bucket = 'accepted'",
                "COALESCE(item.selected_for_training, 1) = 1",
                "item.pseudo_label IN (0, 1)",
            ]
        )
    else:
        placeholders = ", ".join("?" for _ in sample_ids)
        where_parts.append(f"item.id IN ({placeholders})")
        params.extend(sample_ids)
    if resolved_batch_id:
        where_parts.append(
            "(item.batch_id = ? OR EXISTS (SELECT 1 FROM mlflow_comment_prediction AS scoped_prediction "
            "WHERE scoped_prediction.sample_item_id = item.id AND scoped_prediction.batch_id = ?))"
        )
        params.extend([resolved_batch_id, resolved_batch_id])

    where_sql = " AND ".join(where_parts)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT item.id, item.batch_id, item.url, item.url_hash, item.text, item.pseudo_label,
                   item.selected_for_training, item.gate_bucket, item.verification_status,
                   item.segment_hash, item.html_tag, item.review_reason
            FROM mlflow_comment_item AS item
            WHERE {where_sql}
            ORDER BY item.id ASC
            """,
            tuple(params),
        ).fetchall()
        existing_model_ids = (
            {
                int(existing[0])
                for existing in conn.execute(
                    f"""
                    SELECT prediction.sample_item_id
                    FROM mlflow_comment_prediction AS prediction
                    WHERE prediction.model_id = ?
                      AND prediction.sample_item_id IN ({', '.join('?' for _ in rows)})
                    """,
                    tuple([model_id, *[int(row["id"]) for row in rows]]),
                ).fetchall()
            }
            if rows
            else set()
        )

    requested_count = len(sample_ids) if request.selection == "selected" else len(rows)
    result_by_id: Dict[int, Dict[str, Any]] = {}
    if request.selection == "selected":
        found_ids = {int(row["id"]) for row in rows}
        for missing_id in sample_ids:
            if missing_id not in found_ids:
                result_by_id[missing_id] = {
                    "sample_id": missing_id,
                    "status": "skipped",
                    "message": "Sample not found in the requested training scope",
                }

    inference_rows: List[sqlite3.Row] = []
    for row in rows:
        sample_id = int(row["id"])
        if str(row["verification_status"] or "") == "manual_rejected":
            result_by_id[sample_id] = {
                "sample_id": sample_id,
                "status": "skipped",
                "message": "Removed samples are not re-evaluated automatically",
            }
        elif sample_id in existing_model_ids:
            result_by_id[sample_id] = {
                "sample_id": sample_id,
                "status": "skipped",
                "message": "Already evaluated with this model version",
            }
        else:
            inference_rows.append(row)

    if inference_rows:
        try:
            _, predictions = run_mlflow_model_re_evaluation(inference_rows, model_id)
        except Exception as exc:
            logger.exception("MLflow model re-evaluation failed for model %s", model_id)
            for row in inference_rows:
                sample_id = int(row["id"])
                result_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "message": str(exc),
                }
            predictions = {}

        now = datetime.utcnow().isoformat() + "Z"
        operation_id = f"model_reevaluation_{uuid.uuid4().hex[:12]}"
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            for queued_row in inference_rows:
                sample_id = int(queued_row["id"])
                prediction = predictions.get(sample_id)
                if prediction is None:
                    if sample_id not in result_by_id:
                        result_by_id[sample_id] = {
                            "sample_id": sample_id,
                            "status": "failed",
                            "message": "Selected model did not return a prediction for this sample",
                        }
                    continue

                current = conn.execute(
                    """
                    SELECT id, batch_id, pseudo_label, selected_for_training, gate_bucket,
                           verification_status, review_reason
                    FROM mlflow_comment_item WHERE id = ?
                    """,
                    (sample_id,),
                ).fetchone()
                if current is None:
                    result_by_id[sample_id] = {
                        "sample_id": sample_id,
                        "status": "failed",
                        "message": "Sample disappeared before prediction persistence",
                    }
                    continue

                prediction_cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO mlflow_comment_prediction (
                        sample_item_id, batch_id, job_id, model_id,
                        raw_toxicity_score, adjusted_toxicity_score, predicted_label,
                        constructiveness_score, constructiveness_label, constructiveness_confidence,
                        seg_threshold_used, record_origin, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'model_re_evaluation', ?)
                    """,
                    (
                        sample_id,
                        str(current["batch_id"]),
                        operation_id,
                        model_id,
                        prediction.get("raw_toxicity_score"),
                        prediction.get("adjusted_toxicity_score"),
                        prediction.get("predicted_label"),
                        prediction.get("constructiveness_score"),
                        prediction.get("constructiveness_label"),
                        prediction.get("constructiveness_confidence"),
                        prediction.get("seg_threshold_used"),
                        now,
                    ),
                )
                if int(prediction_cursor.rowcount or 0) == 0:
                    result_by_id[sample_id] = {
                        "sample_id": sample_id,
                        "status": "skipped",
                        "message": "Already evaluated with this model version",
                    }
                    continue

                current_label = normalize_int(current["pseudo_label"])
                predicted_label = normalize_int(prediction.get("predicted_label"))
                raw_score = normalize_score(prediction.get("raw_toxicity_score"))
                verification_status = str(current["verification_status"] or "")
                is_auto_eligible = (
                    verification_status == "auto_accepted"
                    and str(current["gate_bucket"] or "") == "accepted"
                    and int(current["selected_for_training"] or 0) == 1
                    and current_label in {0, 1}
                )

                if raw_score is None:
                    re_evaluation_status = "uncertain"
                else:
                    re_evaluation_gate = classify_mlflow_gate(
                        raw_score,
                        MLFLOW_ACCEPT_THRESHOLD,
                        MLFLOW_DISCARD_THRESHOLD,
                    )
                    re_evaluation_status = (
                        "uncertain"
                        if re_evaluation_gate["gate_bucket"] == "candidate" or predicted_label not in {0, 1}
                        else "agreement"
                        if predicted_label == current_label
                        else "conflict"
                    )

                workflow_changed = False
                if is_auto_eligible and re_evaluation_status in {"conflict", "uncertain"}:
                    reason = "model_conflict" if re_evaluation_status == "conflict" else "model_uncertain"
                    conn.execute(
                        """
                        UPDATE mlflow_comment_item
                        SET gate_bucket = 'candidate', verification_status = 'unverified',
                            selected_for_training = 0, training_review_status = 'pending',
                            review_reason = ?, reviewed_at = NULL
                        WHERE id = ?
                        """,
                        (reason, sample_id),
                    )
                    workflow_changed = True

                if verification_status == "manual_accepted":
                    status = "human_agreement" if predicted_label == current_label else "human_disagreement"
                    message = "Human label remains authoritative; training eligibility was unchanged"
                elif is_auto_eligible:
                    status = re_evaluation_status
                    message = (
                        "Model agreement; automatic training eligibility was preserved"
                        if status == "agreement"
                        else "Model conflict; sample was suspended for Manual Verify"
                        if status == "conflict"
                        else "Model result was uncertain; sample was suspended for Manual Verify"
                    )
                else:
                    status = "pending_manual_review" if verification_status == "unverified" else "recorded"
                    message = "Prediction recorded; existing workflow state was unchanged"

                result_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "status": status,
                    "message": message,
                    "workflow_changed": workflow_changed,
                    "current_label": current_label,
                    "predicted_label": predicted_label,
                    "raw_toxicity_score": raw_score,
                    "adjusted_toxicity_score": prediction.get("adjusted_toxicity_score"),
                    "seg_threshold_used": prediction.get("seg_threshold_used"),
                }
            conn.commit()

    results = [result_by_id[key] for key in sorted(result_by_id)]
    summary = {
        "requested": requested_count,
        "evaluated": sum(1 for item in results if item["status"] not in {"skipped", "failed"}),
        "agreement": sum(1 for item in results if item["status"] in {"agreement", "human_agreement"}),
        "conflict": sum(1 for item in results if item["status"] == "conflict"),
        "uncertain": sum(1 for item in results if item["status"] == "uncertain"),
        "needs_review": sum(1 for item in results if item["status"] in {"conflict", "uncertain"}),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
    }
    return {
        "status": "ok" if summary["failed"] == 0 else "partial",
        "model_id": model_id,
        "selection": request.selection,
        "training_scope": request.training_scope,
        "batch_id": resolved_batch_id,
        "summary": summary,
        "results": results,
    }


@app.post("/api/mlflow/candidates/review", dependencies=[Depends(require_admin)])
def mlflow_candidates_review(request: MlflowCandidateReviewRequest) -> Dict[str, Any]:
    init_feedback_db()
    ids = [item.id for item in request.updates]
    action_by_id = {item.id: item.action for item in request.updates}
    decision_by_id = {item.id: item.decision for item in request.updates}
    lock_state_by_id = {item.id: item.lock_state for item in request.updates}
    pseudo_label_by_id = {
        item.id: (item.pseudo_label if item.pseudo_label in {0, 1} else None)
        for item in request.updates
    }
    constructiveness_label_by_id = {
        item.id: (item.constructiveness_label if item.constructiveness_label in {0, 1} else None)
        for item in request.updates
    }
    clear_constructiveness_by_id = {item.id: item.clear_constructiveness for item in request.updates}
    label_source_by_id = {item.id: item.label_source for item in request.updates}
    label_confidence_by_id = {item.id: item.label_confidence for item in request.updates}
    reviewed_by_gemini_by_id = {item.id: item.reviewed_by_gemini for item in request.updates}
    review_provider_by_id = {item.id: item.review_provider for item in request.updates}
    review_model_by_id = {item.id: normalize_gemini_model_name(item.review_model_name) for item in request.updates}

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, batch_id, pseudo_label, constructiveness_label, is_locked, review_reason FROM mlflow_comment_item WHERE id IN ({', '.join(['?'] * len(ids))})",
            tuple(ids),
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No candidate rows found for provided ids")

        now = datetime.utcnow().isoformat() + "Z"
        affected_batch_ids = {str(row["batch_id"]) for row in rows}
        updated = 0
        locked_updated = 0
        skipped_locked_ids: List[int] = []
        for row in rows:
            item_id = int(row["id"])
            manual_label = pseudo_label_by_id.get(item_id)
            current_label = normalize_int(row["pseudo_label"])
            final_label = manual_label if manual_label in {0, 1} else current_label
            requested_constructiveness = constructiveness_label_by_id.get(item_id)
            if clear_constructiveness_by_id.get(item_id):
                final_constructiveness = None
            elif requested_constructiveness in {0, 1}:
                final_constructiveness = requested_constructiveness
            else:
                final_constructiveness = normalize_int(row["constructiveness_label"])
            reviewed_by_gemini = bool(reviewed_by_gemini_by_id.get(item_id))
            review_status = "manual_gemini" if reviewed_by_gemini else "manual_approved"
            label_source = label_source_by_id.get(item_id) or ("gemini_assist" if reviewed_by_gemini else "manual_override")
            label_confidence = label_confidence_by_id.get(item_id) or "high"
            review_provider = str(review_provider_by_id.get(item_id) or "").strip().lower() or None
            review_model_name = review_model_by_id.get(item_id)
            if reviewed_by_gemini and (review_provider != "gemini" or not review_model_name):
                raise HTTPException(status_code=400, detail=f"Gemini review provenance is required for item {item_id}")
            if not reviewed_by_gemini:
                review_provider = None
                review_model_name = None
            effective_locked = bool(int(row["is_locked"] or 0))
            current_review_reason = str(row["review_reason"] or "")
            resolved_review_reason = (
                f"{current_review_reason}_resolved"
                if current_review_reason in {"model_conflict", "model_uncertain"}
                else current_review_reason or None
            )
            removed_review_reason = (
                f"{current_review_reason}_removed"
                if current_review_reason in {"model_conflict", "model_uncertain"}
                else current_review_reason or None
            )

            requested_lock_state = lock_state_by_id.get(item_id)
            if requested_lock_state is not None:
                next_lock_state = 1 if requested_lock_state else 0
                if next_lock_state != int(row["is_locked"] or 0):
                    cursor_lock = conn.execute(
                        "UPDATE mlflow_comment_item SET is_locked = ?, reviewed_at = ? WHERE id = ?",
                        (next_lock_state, now, item_id),
                    )
                    changed = int(cursor_lock.rowcount or 0)
                    updated += changed
                    locked_updated += changed
                effective_locked = bool(next_lock_state)

            action = action_by_id.get(item_id)
            decision = decision_by_id.get(item_id)
            if action is None:
                if decision == "accept":
                    if final_label not in {0, 1}:
                        raise HTTPException(status_code=400, detail=f"Include action requires label for item {item_id}")
                    action = "include_toxic" if final_label == 1 else "include_clean"
                elif decision == "reject":
                    action = "drop"

            if action is None:
                continue

            if action not in {"include_toxic", "include_clean", "drop"}:
                raise HTTPException(status_code=400, detail=f"Unsupported review action for item {item_id}")

            if effective_locked and action == "drop":
                skipped_locked_ids.append(item_id)
                continue

            if action == "include_toxic":
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, pseudo_label = ?, constructiveness_label = ?,
                        selected_for_training = ?, training_review_status = ?, label_source = ?, label_confidence = ?,
                        review_provider = ?, review_model_name = ?, review_reason = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_accepted", "accepted", 1, final_constructiveness, 1, review_status, label_source, label_confidence, review_provider, review_model_name, resolved_review_reason, now, item_id),
                )
            elif action == "include_clean":
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, pseudo_label = ?, constructiveness_label = ?,
                        selected_for_training = ?, training_review_status = ?, label_source = ?, label_confidence = ?,
                        review_provider = ?, review_model_name = ?, review_reason = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_accepted", "accepted", 0, final_constructiveness, 1, review_status, label_source, label_confidence, review_provider, review_model_name, resolved_review_reason, now, item_id),
                )
            elif final_label in {0, 1}:
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, pseudo_label = ?, selected_for_training = ?, training_review_status = ?,
                        label_source = ?, label_confidence = ?, review_reason = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_rejected", "discarded", final_label, 0, "manual_removed", "manual_override", "high", removed_review_reason, now, item_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, selected_for_training = ?, training_review_status = ?,
                        label_source = ?, label_confidence = ?, review_reason = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_rejected", "discarded", 0, "manual_removed", "manual_rejected_unlabeled", "low", removed_review_reason, now, item_id),
                )
            updated += int(cursor.rowcount or 0)

        counts_by_batch: Dict[str, Dict[str, int]] = {}
        for b_id in affected_batch_ids:
            counts_by_batch[b_id] = build_mlflow_gate_counts(conn, b_id)

        conn.commit()

    primary_batch = sorted(affected_batch_ids)[0]
    return {
        "updated": updated,
        "locked_updated": locked_updated,
        "skipped_locked": len(skipped_locked_ids),
        "skipped_locked_ids": skipped_locked_ids,
        "batch_id": primary_batch,
        "counts": counts_by_batch.get(primary_batch, {"accepted": 0, "candidate": 0, "discarded": 0, "total": 0}),
        "counts_by_batch": counts_by_batch,
    }


@app.get("/api/mlflow/threshold-status", dependencies=[Depends(require_admin)])
def mlflow_threshold_status(
    batch_id: Optional[str] = None,
    strict_batch: bool = Query(default=False),
) -> Dict[str, Any]:
    target_min_rows = get_mlflow_bundle_min_rows()
    resolved_batch_id = resolve_mlflow_batch_id_or_none(batch_id, strict=strict_batch)
    if not resolved_batch_id:
        return {
            "batch_id": "",
            "scope": "all_batches",
            "accepted_count": 0,
            "accepted_count_current_batch": 0,
            "target_max_test_stage": target_min_rows,
            "remaining_to_target": target_min_rows,
            "is_ready": False,
            "has_data": False,
        }

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        accepted_total_row = conn.execute(
            """
            SELECT COUNT(1)
            FROM mlflow_comment_item
            WHERE gate_bucket = 'accepted' AND pseudo_label IN (0, 1)
            """
        ).fetchone()
        accepted_count = int(accepted_total_row[0] if accepted_total_row else 0)

        accepted_batch_row = conn.execute(
            """
            SELECT COUNT(1)
            FROM mlflow_comment_item
            WHERE batch_id = ? AND gate_bucket = 'accepted' AND pseudo_label IN (0, 1)
            """,
            (resolved_batch_id,),
        ).fetchone()
        accepted_count_current_batch = int(accepted_batch_row[0] if accepted_batch_row else 0)

    remaining = max(target_min_rows - accepted_count, 0)
    return {
        "batch_id": resolved_batch_id,
        "scope": "all_batches",
        "accepted_count": accepted_count,
        "accepted_count_current_batch": accepted_count_current_batch,
        "target_max_test_stage": target_min_rows,
        "remaining_to_target": remaining,
        "is_ready": accepted_count >= target_min_rows,
        "has_data": True,
    }


@app.get("/api/mlflow/training-preview", dependencies=[Depends(require_admin)])
def mlflow_training_preview(
    batch_id: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=300),
    strict_batch: bool = Query(default=False),
    scope: Literal["batch", "all_batches"] = Query(default="all_batches"),
    label_filter: Literal["all", "toxic", "clean"] = Query(default="all"),
    constructiveness_filter: Literal["all", "included", "masked"] = Query(default="all"),
) -> Dict[str, Any]:
    init_feedback_db()
    resolved_batch_id: Optional[str] = None
    visibility_clause = (
        "((item.gate_bucket = 'accepted' AND COALESCE(item.selected_for_training, 1) = 1) "
        "OR (item.gate_bucket = 'candidate' AND item.verification_status = 'unverified' "
        "AND item.review_reason IN ('model_conflict', 'model_uncertain')))"
    )
    where_parts = [visibility_clause]
    params: List[Any] = []
    scope_parts: List[str] = []
    scope_params: List[Any] = []
    if scope == "batch":
        resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
        scope_clause = (
            "(item.batch_id = ? OR EXISTS (SELECT 1 FROM mlflow_comment_prediction AS scoped_prediction "
            "WHERE scoped_prediction.sample_item_id = item.id AND scoped_prediction.batch_id = ?))"
        )
        where_parts.insert(0, scope_clause)
        params.extend([resolved_batch_id, resolved_batch_id])
        scope_parts.append(scope_clause)
        scope_params.extend([resolved_batch_id, resolved_batch_id])
    if label_filter == "toxic":
        where_parts.append("item.pseudo_label = 1")
    elif label_filter == "clean":
        where_parts.append("item.pseudo_label = 0")
    if constructiveness_filter == "included":
        where_parts.append("item.constructiveness_label IN (0, 1)")
    elif constructiveness_filter == "masked":
        where_parts.append("item.constructiveness_label IS NULL")

    where_sql = " AND ".join(where_parts)
    scope_sql = " AND ".join(scope_parts) if scope_parts else "1 = 1"
    offset = (page - 1) * page_size

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total = int(
            conn.execute(
                f"SELECT COUNT(1) FROM mlflow_comment_item AS item WHERE {where_sql}",
                tuple(params),
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT item.id, item.batch_id, item.url, item.url_hash, item.segment_id, item.domain_category, item.text, item.score,
                   item.pseudo_label, item.constructiveness_score, item.constructiveness_label, item.constructiveness_confidence,
                   item.selected_for_training, item.training_review_status, item.is_locked, item.gate_bucket, item.verification_status,
                   item.segment_hash, item.context_segment_hash, item.html_tag, item.seg_threshold_used,
                   item.label_source, item.label_confidence, item.review_provider, item.review_model_name,
                   item.review_reason, item.source_type, item.source_row_id, item.created_at, item.reviewed_at
            FROM mlflow_comment_item AS item
            WHERE {where_sql}
            ORDER BY item.gate_bucket ASC, item.selected_for_training DESC, item.id DESC
            LIMIT ? OFFSET ?
            """,
            tuple([*params, page_size, offset]),
        ).fetchall()
        all_rows = conn.execute(
            f"""
            SELECT item.id, item.pseudo_label, item.constructiveness_label, item.selected_for_training,
                   item.gate_bucket, item.verification_status, item.training_review_status,
                   item.is_locked, item.review_reason
            FROM mlflow_comment_item AS item
            WHERE {where_sql}
            """,
            tuple(params),
        ).fetchall()
        removed_count = int(
            conn.execute(
                f"""
                SELECT COUNT(1) FROM mlflow_comment_item AS item
                WHERE {scope_sql}
                  AND (item.gate_bucket = 'discarded' OR item.training_review_status = 'manual_removed')
                """,
                tuple(scope_params),
            ).fetchone()[0]
        )
        items = attach_mlflow_prediction_history(conn, [dict(row) for row in rows])

    selected_rows = [
        row for row in all_rows
        if int(row["selected_for_training"] or 0) == 1
        and str(row["gate_bucket"] or "") == "accepted"
        and normalize_int(row["pseudo_label"]) in {0, 1}
    ]
    toxic_selected = sum(1 for row in selected_rows if normalize_int(row["pseudo_label"]) == 1)
    clean_selected = sum(1 for row in selected_rows if normalize_int(row["pseudo_label"]) == 0)
    constructiveness_included = sum(1 for row in selected_rows if normalize_int(row["constructiveness_label"]) in {0, 1})
    constructive_selected = sum(1 for row in selected_rows if normalize_int(row["constructiveness_label"]) == 1)
    non_constructive_selected = sum(1 for row in selected_rows if normalize_int(row["constructiveness_label"]) == 0)
    return {
        "scope": scope,
        "batch_id": resolved_batch_id,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "counts": {
            "selected": len(selected_rows),
            "selected_toxic": toxic_selected,
            "selected_clean": clean_selected,
            "candidate": sum(1 for row in all_rows if str(row["gate_bucket"] or "") == "candidate"),
            "auto_eligible": sum(
                1
                for row in selected_rows
                if str(row["verification_status"] or "") == "auto_accepted"
            ),
            "requires_human_review": sum(
                1
                for row in all_rows
                if str(row["review_reason"] or "") in {"model_conflict", "model_uncertain"}
            ),
            "model_conflicts": sum(
                1 for row in all_rows if str(row["review_reason"] or "") == "model_conflict"
            ),
            "model_uncertain": sum(
                1 for row in all_rows if str(row["review_reason"] or "") == "model_uncertain"
            ),
            "removed": removed_count,
        },
        "balance": {
            "strategy": "balanced_50_50",
            "balanced_count": 2 * min(toxic_selected, clean_selected) if toxic_selected and clean_selected else len(selected_rows),
            "toxic_available": toxic_selected,
            "clean_available": clean_selected,
        },
        "constructiveness": {
            "included": constructiveness_included,
            "masked": max(0, len(selected_rows) - constructiveness_included),
            "constructive": constructive_selected,
            "non_constructive": non_constructive_selected,
        },
    }


@app.get("/api/mlflow/training-plan", dependencies=[Depends(require_admin)])
def mlflow_training_plan(
    batch_id: Optional[str] = None,
    strict_batch: bool = Query(default=False),
    scope: Literal["batch", "all_batches"] = Query(default="all_batches"),
    balance_strategy: Literal["balanced_50_50", "all"] = Query(default="balanced_50_50"),
) -> Dict[str, Any]:
    init_feedback_db()
    resolved_batch_id: Optional[str] = None
    if scope == "batch":
        resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return build_mlflow_training_plan(conn, resolved_batch_id, balance_strategy)


@app.post("/api/mlflow/training-preview/gemini-review", dependencies=[Depends(require_admin)])
def mlflow_training_preview_gemini_review(request: MlflowTrainingPreviewGeminiReviewRequest) -> Dict[str, Any]:
    init_feedback_db()
    ids = []
    seen: set[int] = set()
    for item_id in request.ids:
        if item_id not in seen:
            ids.append(item_id)
            seen.add(item_id)
    validate_gemini_review_item_limit(ids)
    placeholders = ", ".join(["?"] * len(ids))

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, batch_id, url, domain_category, text, score, pseudo_label,
                   constructiveness_score, constructiveness_label, gate_bucket,
                   selected_for_training, training_review_status, is_locked
            FROM mlflow_comment_item
            WHERE id IN ({placeholders})
              AND gate_bucket = 'accepted'
            ORDER BY id ASC
            """,
            tuple(ids),
        ).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No training preview rows found for provided ids")

    suggestions = run_mlflow_gemini_review(rows)
    failed_ids = sorted({int(row["id"]) for row in rows} - {int(item["id"]) for item in suggestions})
    if not suggestions:
        raise HTTPException(status_code=502, detail="Gemini could not produce valid review suggestions after retrying")
    return {**build_gemini_review_response(suggestions, len(ids)), "failed_ids": failed_ids}


@app.post("/api/mlflow/training-preview/review", dependencies=[Depends(require_admin)])
def mlflow_training_preview_review(request: MlflowTrainingPreviewReviewRequest) -> Dict[str, Any]:
    init_feedback_db()
    ids = [item.id for item in request.updates]
    now = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            f"SELECT id, is_locked, training_review_status, review_reason FROM mlflow_comment_item WHERE id IN ({', '.join(['?'] * len(ids))})",
            tuple(ids),
        ).fetchall()
        if not existing:
            raise HTTPException(status_code=404, detail="No training preview rows found for provided ids")
        existing_by_id = {int(row["id"]): row for row in existing}

        updated = 0
        skipped_locked_ids: List[int] = []
        for item in request.updates:
            existing_row = existing_by_id.get(item.id)
            current_locked = bool(int(existing_row["is_locked"] or 0)) if existing_row is not None else False
            current_review_status = str(existing_row["training_review_status"] or "") if existing_row is not None else ""
            current_review_reason = str(existing_row["review_reason"] or "") if existing_row is not None else ""
            target_locked = item.lock_state if item.lock_state is not None else current_locked
            if item.selected_for_training is False and target_locked:
                skipped_locked_ids.append(item.id)
                continue

            fields = ["reviewed_at = ?"]
            values: List[Any] = [now]
            if item.lock_state is not None:
                fields.append("is_locked = ?")
                values.append(1 if item.lock_state else 0)
            next_review_status: Optional[str] = None
            if item.selected_for_training is not None:
                fields.append("selected_for_training = ?")
                values.append(1 if item.selected_for_training else 0)
                next_review_status = "manual_approved" if item.selected_for_training else "manual_removed"
            if item.pseudo_label in {0, 1}:
                fields.extend([
                    "pseudo_label = ?",
                    "gate_bucket = ?",
                    "verification_status = ?",
                    "label_source = ?",
                    "label_confidence = ?",
                    "review_provider = ?",
                    "review_model_name = ?",
                    "review_reason = ?",
                ])
                source = (item.label_source or "manual_override").strip() or "manual_override"
                confidence = (item.label_confidence or "high").strip() or "high"
                review_provider = str(item.review_provider or "").strip().lower() or None
                review_model_name = normalize_gemini_model_name(item.review_model_name)
                if item.reviewed_by_gemini and (review_provider != "gemini" or not review_model_name):
                    raise HTTPException(status_code=400, detail=f"Gemini review provenance is required for item {item.id}")
                if not item.reviewed_by_gemini:
                    review_provider = None
                    review_model_name = None
                values.extend([
                    int(item.pseudo_label),
                    "accepted",
                    "manual_accepted",
                    source[:64],
                    confidence[:32],
                    review_provider,
                    review_model_name,
                    f"{current_review_reason}_resolved"
                    if current_review_reason in {"model_conflict", "model_uncertain"}
                    else current_review_reason or None,
                ])
                if not item.reviewed_by_gemini:
                    next_review_status = "manual_approved"
            if item.reviewed_by_gemini and item.pseudo_label in {0, 1}:
                next_review_status = "auto_gemini" if current_review_status in {"auto", "auto_gemini"} else "manual_gemini"
            if next_review_status is not None:
                fields.append("training_review_status = ?")
                values.append(next_review_status)
            if item.clear_constructiveness:
                fields.extend(["constructiveness_label = ?", "constructiveness_confidence = ?"])
                values.extend([None, "gemini_masked" if item.reviewed_by_gemini else "masked"])
            elif item.constructiveness_label in {0, 1}:
                fields.extend(["constructiveness_label = ?", "constructiveness_confidence = ?"])
                values.extend([int(item.constructiveness_label), "gemini" if item.reviewed_by_gemini else "manual"])
            values.append(item.id)
            cursor = conn.execute(
                f"UPDATE mlflow_comment_item SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )
            updated += int(cursor.rowcount or 0)
        conn.commit()
    return {"status": "ok", "updated": updated, "skipped_locked": len(skipped_locked_ids), "skipped_locked_ids": skipped_locked_ids}


@app.post("/api/mlflow/manual/export-bundle", dependencies=[Depends(require_admin)])
def mlflow_manual_export_bundle(request: MlflowManualExportBundleRequest) -> Dict[str, Any]:
    init_feedback_db()
    resolved_training_mode = "retrain" if request.model_kind == "lr_smoke" else request.training_mode
    scope = request.scope
    requested_batch_id = (request.batch_id or "").strip()
    resolved_batch_id: Optional[str] = None

    if scope == "batch":
        if not requested_batch_id:
            raise HTTPException(status_code=400, detail="batch_id is required when scope='batch'")
        resolved_batch_id = resolve_mlflow_batch_id(requested_batch_id, strict=True)

    resolved_dataset_version = normalize_dataset_version(request.dataset_version)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        if resolved_batch_id:
            batch_row = conn.execute(
                "SELECT batch_id, model_id, created_at FROM mlflow_crawl_batch WHERE batch_id = ?",
                (resolved_batch_id,),
            ).fetchone()
        else:
            batch_row = conn.execute(
                "SELECT batch_id, model_id, created_at FROM mlflow_crawl_batch ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        accepted_rows_all, accepted_rows, balance_stats = select_mlflow_training_rows(
            conn,
            resolved_batch_id,
            request.balance_strategy,
        )

        candidate_where = "gate_bucket = 'candidate' AND verification_status = 'unverified'"
        candidate_params: List[Any] = []
        if resolved_batch_id:
            candidate_where += " AND batch_id = ?"
            candidate_params.append(resolved_batch_id)

        candidate_rows = conn.execute(
            f"""
            SELECT id, batch_id, text, pseudo_label, constructiveness_score, constructiveness_label,
                   constructiveness_confidence, training_review_status, score, url, url_hash,
                   segment_hash, context_segment_hash, html_tag
            FROM mlflow_comment_item
            WHERE {candidate_where}
            ORDER BY id ASC
            """,
            tuple(candidate_params),
        ).fetchall()

        unused_rows: List[sqlite3.Row] = []
        if request.include_unused:
            unused_where = "gate_bucket = 'discarded'"
            unused_params: List[Any] = []
            if resolved_batch_id:
                unused_where += " AND batch_id = ?"
                unused_params.append(resolved_batch_id)
            if request.unused_scope == "auto_discarded":
                unused_where += " AND verification_status = 'auto_discarded'"
            elif request.unused_scope == "manual_rejected":
                unused_where += " AND verification_status = 'manual_rejected'"

            unused_rows = conn.execute(
                f"""
                SELECT id, batch_id, text, pseudo_label, constructiveness_score, constructiveness_label,
                       constructiveness_confidence, training_review_status, score, url, url_hash,
                       segment_hash, context_segment_hash, html_tag, verification_status
                FROM mlflow_comment_item
                WHERE {unused_where}
                ORDER BY id ASC
                """,
                tuple(unused_params),
            ).fetchall()

    model_version = request.model_version or str(batch_row["model_id"] if batch_row else DEFAULT_MODEL_VERSION)
    policy_version = request.policy_version or DEFAULT_POLICY_VERSION
    versions = build_artifact_versions(
        dataset_version=resolved_dataset_version,
        model_version=model_version,
        policy_version=policy_version,
    )

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp_compact()
    scope_token = "batch" if resolved_batch_id else "all_batches"
    batch_token = slugify_token(resolved_batch_id or "global")
    bundle_profile = request.bundle_profile
    required_zip_contents = build_mlflow_required_bundle_contents(bundle_profile)
    local_base_model_dir: Optional[Path] = None
    base_model_provenance: Optional[Dict[str, Any]] = None
    if bundle_profile == "full_bundle" and request.include_base_model and request.base_model:
        try:
            resolved_base_model_id, model_path, base_model_provenance = _resolve_phobert_finetune_base_model(request.base_model)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        local_base_model_dir = model_path
        required_zip_contents.append("base_model/config.json")
        required_zip_contents.append("base_model/provenance.json")

    lineage_token = re.sub(r"[^a-zA-Z0-9_-]+", "_", (request.lineage_run_id or "").strip()).strip("_")
    lineage_suffix = f"_{lineage_token}" if lineage_token else ""
    if bundle_profile == "full_bundle":
        out_path = out_dir / f"mlflow_bundle_{scope_token}_{batch_token}_{timestamp}{lineage_suffix}.zip"
    else:
        out_path = out_dir / f"victsd_gold_merged_{scope_token}_{batch_token}_{timestamp}{lineage_suffix}.zip"

    accepted_jsonl = "\n".join(
        json.dumps(build_pseudo_training_row(row, "mlflow_pseudo"), ensure_ascii=False)
        for row in accepted_rows
    )
    candidate_jsonl = "\n".join(
        json.dumps(
            build_pseudo_training_row(row, "mlflow_candidate")
            if normalize_int(row["pseudo_label"]) in {0, 1}
            else {"text": row["text"], "label": 0, "toxicity": 0, "constructiveness": None, "meta": {"source": "mlflow_candidate"}},
            ensure_ascii=False,
        )
        for row in candidate_rows
    )
    unused_jsonl = "\n".join(
        json.dumps(
            {
                **(
                    build_pseudo_training_row(row, "mlflow_unused")
                    if normalize_int(row["pseudo_label"]) in {0, 1}
                    else {"text": row["text"], "label": 0, "toxicity": 0, "constructiveness": None, "meta": {"source": "mlflow_unused"}}
                ),
                "verification_status": row["verification_status"],
            },
            ensure_ascii=False,
        )
        for row in unused_rows
    )

    train_rows, validation_rows, test_rows, merge_stats, merge_row_statuses = build_training_merge_plan(accepted_rows)
    included_mlflow_ids = sorted(
        item_id for item_id, status in merge_row_statuses.items() if bool(status.get("will_finetune"))
    )
    included_mlflow_ids_sha256 = hashlib.sha256(
        json.dumps(included_mlflow_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    included_id_set = set(included_mlflow_ids)
    feedback_snapshot_rows = [
        {
            "id": int(row["id"]),
            "text": str(row["text"] or ""),
            "toxicity": normalize_int(row["pseudo_label"]),
            "constructiveness": normalize_int(row["constructiveness_label"]),
        }
        for row in accepted_rows
        if int(row["id"]) in included_id_set
    ]
    feedback_snapshot_rows.sort(key=lambda item: item["id"])
    feedback_snapshot_sha256 = hashlib.sha256(
        json.dumps(feedback_snapshot_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    training_config_yaml = (
        f"model: {'tfidf_lr' if request.model_kind == 'lr_smoke' else 'phobert'}\n"
        f"model_kind: {request.model_kind}\n"
        f"training_mode: {resolved_training_mode}\n"
        f"balance_strategy: {request.balance_strategy}\n"
        f"base_model: {request.base_model or ''}\n"
        f"model_version: {versions['model_version']}\n"
        f"dataset_version: {versions['dataset_version']}\n"
        f"policy_version: {versions['policy_version']}\n"
        "batch_size: 16\n"
        "epochs: 3\n"
        "learning_rate: 2e-5\n"
    )

    gate_policy_json = {
        "accept_threshold": MLFLOW_ACCEPT_THRESHOLD,
        "discard_threshold": MLFLOW_DISCARD_THRESHOLD,
        "target_max_test_stage": get_mlflow_bundle_min_rows(),
    }

    manifest_json = {
        "artifact_type": "mlflow_training_bundle",
        "bundle_profile": bundle_profile,
        "lineage_run_id": request.lineage_run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "scope": scope_token,
        "batch_id": resolved_batch_id,
        "record_count": len(accepted_rows),
        "record_count_before_balance": len(accepted_rows_all),
        "candidate_count": len(candidate_rows),
        "unused_count": len(unused_rows),
        "include_unused": request.include_unused,
        "unused_scope": request.unused_scope,
        "model_kind": request.model_kind,
        "training_mode": resolved_training_mode,
        "balance_strategy": request.balance_strategy,
        "balance_stats": balance_stats,
        "base_model": resolved_base_model_id if local_base_model_dir is not None else request.base_model,
        "base_model_bundled": local_base_model_dir is not None,
        "base_model_provenance": base_model_provenance,
        "merge_stats": merge_stats,
        "included_mlflow_ids": included_mlflow_ids,
        "included_mlflow_ids_sha256": included_mlflow_ids_sha256,
        "feedback_snapshot_sha256": feedback_snapshot_sha256,
        **versions,
        "required_zip_contents": required_zip_contents,
    }
    pseudo_manifest_json = {
        "artifact_type": "mlflow_pseudo_labels",
        "batch_id": resolved_batch_id or "all_batches",
        "seed_model": request.base_model or model_version,
        "created_at": manifest_json["created_at"],
        "n_accepted": len(accepted_rows),
        "n_accepted_toxic": sum(1 for row in accepted_rows if normalize_int(row["pseudo_label"]) == 1),
        "n_accepted_clean": sum(1 for row in accepted_rows if normalize_int(row["pseudo_label"]) == 0),
        "constructiveness_included": sum(1 for row in accepted_rows if normalize_int(row["constructiveness_label"]) in {0, 1}),
        "balance_stats": balance_stats,
    }

    train_jsonl = "\n".join(json.dumps(item, ensure_ascii=False) for item in train_rows)
    validation_jsonl = "\n".join(json.dumps(item, ensure_ascii=False) for item in validation_rows)
    test_jsonl = "\n".join(json.dumps(item, ensure_ascii=False) for item in test_rows)

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if bundle_profile == "full_bundle":
            zf.writestr("dataset/accepted_pseudo.jsonl", accepted_jsonl + ("\n" if accepted_jsonl else ""))
            zf.writestr("dataset/candidates_unverified.jsonl", candidate_jsonl + ("\n" if candidate_jsonl else ""))
            if request.include_unused:
                zf.writestr("dataset/unused_discarded.jsonl", unused_jsonl + ("\n" if unused_jsonl else ""))
            zf.writestr("dataset/victsd_gold/train.jsonl", train_jsonl + ("\n" if train_jsonl else ""))
            zf.writestr("dataset/victsd_gold/validation.jsonl", validation_jsonl + ("\n" if validation_jsonl else ""))
            zf.writestr("dataset/victsd_gold/test.jsonl", test_jsonl + ("\n" if test_jsonl else ""))
            zf.writestr("pseudo/accepted.jsonl", accepted_jsonl + ("\n" if accepted_jsonl else ""))
            zf.writestr("pseudo/manifest.json", json.dumps(pseudo_manifest_json, ensure_ascii=False, indent=2))
            train_script_path = Path(__file__).resolve().parents[1] / "scripts" / "06_train_phobert_lora_macro_f1_finetune.py"
            if train_script_path.exists():
                zf.write(train_script_path, "scripts/train_phobert.py")
            if local_base_model_dir is not None:
                for path in local_base_model_dir.rglob("*"):
                    if path.is_file():
                        zf.write(path, f"base_model/{path.relative_to(local_base_model_dir).as_posix()}")
                zf.writestr("base_model/provenance.json", json.dumps(base_model_provenance, ensure_ascii=False, indent=2))
            zf.writestr("manifest.json", json.dumps(manifest_json, ensure_ascii=False, indent=2))
            zf.writestr("config/training_config.yaml", training_config_yaml)
            zf.writestr("config/gate_policy.json", json.dumps(gate_policy_json, ensure_ascii=False, indent=2))
        else:
            zf.writestr("train.jsonl", train_jsonl + ("\n" if train_jsonl else ""))
            zf.writestr("validation.jsonl", validation_jsonl + ("\n" if validation_jsonl else ""))
            zf.writestr("test.jsonl", test_jsonl + ("\n" if test_jsonl else ""))
            zf.writestr(
                "build_report.json",
                json.dumps(
                    {
                        "profile": "clean_victsd_gold",
                        "generated_at": datetime.utcnow().isoformat() + "Z",
                        "scope": scope_token,
                        "batch_id": resolved_batch_id,
                        "lineage_run_id": request.lineage_run_id,
                        "artifact_versions": versions,
                        "merge_stats": manifest_json["merge_stats"],
                        "included_mlflow_ids": included_mlflow_ids,
                        "included_mlflow_ids_sha256": included_mlflow_ids_sha256,
                        "feedback_snapshot_sha256": feedback_snapshot_sha256,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    relative_bundle_path = encode_artifact_ref(out_path)
    download_path = f"/api/mlflow/manual/export-bundle/download?bundle_path={urllib.parse.quote(relative_bundle_path)}"

    return {
        "bundle_path": relative_bundle_path,
        "download_url": download_path,
        "bundle_profile": bundle_profile,
        "model_kind": request.model_kind,
        "training_mode": resolved_training_mode,
        "balance_strategy": request.balance_strategy,
        "scope": scope_token,
        "batch_id": resolved_batch_id,
        "count": len(accepted_rows),
        "count_before_balance": len(accepted_rows_all),
        "candidate_count": len(candidate_rows),
        "unused_count": len(unused_rows),
        "include_unused": request.include_unused,
        "unused_scope": request.unused_scope,
        "required_zip_contents": required_zip_contents,
        "artifact_versions": versions,
        "balance_stats": balance_stats,
        "constructiveness": {
            "included": pseudo_manifest_json["constructiveness_included"],
            "masked": max(0, len(accepted_rows) - int(pseudo_manifest_json["constructiveness_included"])),
        },
        "base_model_bundled": local_base_model_dir is not None,
        "base_model_provenance": base_model_provenance,
        "merge_stats": manifest_json["merge_stats"],
        "included_mlflow_ids": included_mlflow_ids,
        "included_mlflow_ids_sha256": included_mlflow_ids_sha256,
        "feedback_snapshot_sha256": feedback_snapshot_sha256,
    }


@app.get("/api/mlflow/manual/export-bundle/download", dependencies=[Depends(require_admin)])
def mlflow_manual_export_bundle_download(bundle_path: str = Query(..., min_length=1)) -> FileResponse:
    candidate = (bundle_path or "").strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="bundle_path is required")

    try:
        resolved = resolve_artifact_ref(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid bundle_path") from exc
    processed_dir = PROCESSED_DATA_DIR.resolve()

    try:
        resolved.relative_to(processed_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid bundle_path") from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Bundle file not found: {candidate}")

    return FileResponse(path=str(resolved), filename=resolved.name, media_type="application/zip")


@app.post("/api/mlflow/manual/import-artifact", dependencies=[Depends(require_admin)])
def mlflow_manual_import_artifact(request: MlflowManualImportArtifactRequest) -> Dict[str, Any]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO mlflow_training_artifact (run_name, artifact_path, notes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                request.run_name.strip(),
                encode_artifact_ref(request.artifact_path.strip()),
                request.notes.strip() if request.notes else None,
                datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
        artifact_id = int(cursor.lastrowid)
    return {
        "import_id": artifact_id,
        "status": "recorded",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


KAGGLE_STAGES = [
    "prepare_bundle",
    "submit_kaggle_job",
    "train",
    "import_artifact",
    "complete",
]


KAGGLE_ARTIFACT_ROOT = get_kaggle_runtime_dir().resolve()
MODEL_REGISTRY_ARTIFACT_ROOT = get_model_registry_dir().resolve()


def _kaggle_artifact_download_url(run_id: str, artifact_uri: Optional[str]) -> Optional[str]:
    if not artifact_uri or str(artifact_uri).lower().startswith("mock://"):
        return None
    return f"/api/mlflow/kaggle/artifact/download?run_id={urllib.parse.quote(run_id)}"


def _resolve_kaggle_artifact_path(artifact_uri: Optional[str]) -> Path:
    raw = str(artifact_uri or "").strip()
    if not raw:
        raise HTTPException(status_code=404, detail="Kaggle artifact is not available")
    if raw.lower().startswith("mock://"):
        raise HTTPException(status_code=400, detail="Mock Kaggle artifacts are not downloadable")
    if raw.startswith("http://") or raw.startswith("https://"):
        raise HTTPException(status_code=400, detail="Remote artifact downloads are not supported yet")

    if raw.startswith(("data://", "runtime://", "model://")):
        try:
            candidate = resolve_artifact_ref(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid portable artifact reference") from exc
    elif raw.lower().startswith("file://"):
        parsed = urllib.parse.urlparse(raw)
        if parsed.netloc and re.fullmatch(r"[A-Za-z]:", parsed.netloc):
            candidate = Path(urllib.request.url2pathname(f"{parsed.netloc}{parsed.path}"))
        elif parsed.netloc and parsed.netloc not in {"", "localhost"}:
            raise HTTPException(status_code=400, detail="Only local file:// artifacts are supported")
        else:
            candidate = Path(urllib.request.url2pathname(parsed.path))
    else:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate

    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(KAGGLE_ARTIFACT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Kaggle artifact path") from exc
    if "output" not in resolved.relative_to(KAGGLE_ARTIFACT_ROOT).parts:
        raise HTTPException(status_code=400, detail="Kaggle artifact must be under a job output directory")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Kaggle artifact file not found: {resolved.name}")
    return resolved


def _registry_model_id(model_family: str, run_id: str) -> str:
    return f"{model_family}/{_sanitize_import_model_name(run_id)}"


def _resolve_registry_artifact_path(artifact_uri: Optional[str]) -> Path:
    raw = str(artifact_uri or "").strip()
    if not raw:
        raise HTTPException(status_code=404, detail="Registry artifact is not available")
    if raw.startswith(("data://", "runtime://", "model://")):
        try:
            candidate = resolve_artifact_ref(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid portable registry artifact reference") from exc
    elif raw.lower().startswith("file://"):
        parsed = urllib.parse.urlparse(raw)
        candidate = Path(urllib.request.url2pathname(f"{parsed.netloc}{parsed.path}")) if parsed.netloc else Path(urllib.request.url2pathname(parsed.path))
    else:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(MODEL_REGISTRY_ARTIFACT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid registry artifact path") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Registry artifact file not found: {resolved.name}")
    return resolved


def _copy_registry_artifact(source: Path, model_family: str, run_id: str, checksum: str) -> Path:
    target_dir = (MODEL_REGISTRY_ARTIFACT_ROOT / model_family / _sanitize_import_model_name(run_id)).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "artifact.zip"
    if target.is_file() and hmac.compare_digest(_sha256_file(target), checksum):
        return target
    staging = target_dir / f".artifact-{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(source, staging)
        if not hmac.compare_digest(_sha256_file(staging), checksum):
            raise HTTPException(status_code=409, detail="Registry artifact checksum changed while copying")
        staging.replace(target)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)
    return target


def _coerce_numeric_metric_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for key, raw in value.items():
        score = normalize_score(raw)
        if score is not None:
            out[str(key)] = score
    return out


def _extract_kaggle_metrics_from_zip(artifact_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if artifact_path.suffix.lower() != ".zip":
        return None, None
    try:
        with zipfile.ZipFile(artifact_path, "r") as zf:
            metric_members = [
                name
                for name in zf.namelist()
                if not name.endswith("/") and Path(name).name.lower() == "metrics.json"
            ]
            if not metric_members:
                return None, None

            def rank_member(name: str) -> Tuple[int, int, str]:
                lowered = name.lower()
                if lowered == "metrics.json":
                    return (0, len(name), name)
                if lowered.endswith("/results/metrics.json"):
                    return (1, len(name), name)
                return (2, len(name), name)

            member = sorted(metric_members, key=rank_member)[0]
            with zf.open(member) as f:
                payload = json.loads(f.read().decode("utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else None, member
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse Kaggle artifact metrics from %s: %s", artifact_path, exc)
        return None, None


def _normalize_kaggle_metrics(raw_metrics: Optional[Dict[str, Any]], source_member: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return None

    test_metrics = raw_metrics.get("test") if isinstance(raw_metrics.get("test"), dict) else None
    validation_metrics = raw_metrics.get("validation") if isinstance(raw_metrics.get("validation"), dict) else None
    final_metrics = raw_metrics.get("final_test_rich") if isinstance(raw_metrics.get("final_test_rich"), dict) else None
    argmax_metrics = raw_metrics.get("test_argmax_basic") if isinstance(raw_metrics.get("test_argmax_basic"), dict) else None
    primary = test_metrics or final_metrics or argmax_metrics or raw_metrics

    normalized = {
        "f1_toxic": normalize_score(primary.get("f1_toxic") or primary.get("toxic_f1")),
        "macro_f1": normalize_score(primary.get("macro_f1") or primary.get("f1")),
        "accuracy": normalize_score(primary.get("accuracy")),
        "precision": normalize_score(primary.get("precision") or primary.get("precision_toxic")),
        "recall": normalize_score(primary.get("recall") or primary.get("recall_toxic")),
        "source_member": source_member,
        "run_name": raw_metrics.get("run_name") if isinstance(raw_metrics.get("run_name"), str) else None,
        "mode": raw_metrics.get("mode") if isinstance(raw_metrics.get("mode"), str) else None,
        "sizes": _coerce_numeric_metric_map(raw_metrics.get("sizes") if isinstance(raw_metrics.get("sizes"), dict) else None),
        "dataset_evidence": raw_metrics.get("dataset_evidence") if isinstance(raw_metrics.get("dataset_evidence"), dict) else None,
        "confusion_matrix": raw_metrics.get("confusion_matrix") if isinstance(raw_metrics.get("confusion_matrix"), dict) else None,
        "splits": {
            "validation": _coerce_numeric_metric_map(validation_metrics),
            "test": _coerce_numeric_metric_map(test_metrics),
        },
    }
    return normalized


def _load_kaggle_artifact_metrics(artifact_uri: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        artifact_path = _resolve_kaggle_artifact_path(artifact_uri)
    except HTTPException:
        return None
    raw_metrics, source_member = _extract_kaggle_metrics_from_zip(artifact_path)
    return _normalize_kaggle_metrics(raw_metrics, source_member)


def _record_kaggle_training_artifact(
    run_id: str,
    artifact_uri: Optional[str],
    metrics: Optional[Dict[str, Any]],
    notes: Optional[str] = None,
) -> None:
    if not artifact_uri or str(artifact_uri).lower().startswith("mock://"):
        return
    init_feedback_db()
    now = datetime.utcnow().isoformat() + "Z"
    run_name = str((metrics or {}).get("run_name") or run_id).strip() or run_id
    metrics_json = json.dumps(metrics, ensure_ascii=False) if metrics else None
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        existing = conn.execute(
            "SELECT id FROM mlflow_training_artifact WHERE source_run_id = ?",
            (run_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE mlflow_training_artifact
                SET run_name = ?, artifact_path = ?, notes = ?, metrics_json = ?, created_at = ?
                WHERE source_run_id = ?
                """,
                (run_name, artifact_uri, notes, metrics_json, now, run_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO mlflow_training_artifact (run_name, artifact_path, notes, created_at, source_run_id, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_name, artifact_uri, notes, now, run_id, metrics_json),
            )
        conn.commit()


def _load_previous_completed_kaggle_run(
    current_row: sqlite3.Row,
    current_model_kind: str,
) -> Optional[Dict[str, Any]]:
    current_created_at = str(current_row["created_at"] or "").strip()
    if not current_created_at:
        return None

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        candidate_ids = conn.execute(
            """
            SELECT run_id
            FROM mlflow_do_run
            WHERE status = 'completed'
              AND artifact_uri IS NOT NULL
              AND artifact_uri <> ''
              AND created_at < ?
              AND run_id <> ?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (current_created_at, current_row["run_id"]),
        ).fetchall()

        for candidate_id in candidate_ids:
            candidate = _do_get_run(conn, str(candidate_id["run_id"]))
            if not candidate:
                continue
            candidate_logs = _do_load_logs(candidate)
            candidate_meta = _do_extract_runtime_metadata(candidate_logs)
            candidate_model_kind = str(candidate_meta.get("model_kind") or "phobert").strip().lower()
            if candidate_model_kind != current_model_kind:
                continue
            artifact_uri = str(candidate["artifact_uri"] or "").strip()
            metrics = _load_kaggle_artifact_metrics(artifact_uri)
            if not metrics:
                continue
            return {
                "run_id": candidate["run_id"],
                "created_at": candidate["created_at"],
                "updated_at": candidate["updated_at"],
                "artifact_checksum": candidate["artifact_checksum"],
                "model_kind": candidate_model_kind,
                "training_mode": candidate_meta.get("training_mode") or "retrain",
                "metrics": metrics,
            }
    return None


def _kaggle_http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method=method.upper(),
        headers={"Content-Type": "application/json"},
        data=body,
    )
    try:
        timeout_sec = get_int_setting("KAGGLE_WEBHOOK_TIMEOUT_SEC", 180, min_value=10)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = (resp.read() or b"").decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = (exc.read() or b"").decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Kaggle webhook HTTP {exc.code} ({url}): {raw[:500]}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise HTTPException(status_code=502, detail=f"Kaggle webhook timeout ({url}): {exc}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Kaggle webhook unreachable ({url}): {exc.reason}") from exc

    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _kaggle_webhook_health_url(webhook_url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse((webhook_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    if not parsed.path.rstrip("/").startswith("/kaggle"):
        return None
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def _kaggle_webhook_reachability(webhook_url: str) -> tuple[bool, Optional[str]]:
    health_url = _kaggle_webhook_health_url(webhook_url)
    if not health_url:
        return True, None

    req = urllib.request.Request(url=health_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            if 200 <= resp.status < 500:
                return True, None
            return False, f"Webhook health returned HTTP {resp.status} ({health_url})"
    except urllib.error.HTTPError as exc:
        if 200 <= exc.code < 500:
            return True, None
        return False, f"Webhook health returned HTTP {exc.code} ({health_url})"
    except (TimeoutError, socket.timeout) as exc:
        return False, f"Webhook health timeout ({health_url}): {exc}"
    except urllib.error.URLError as exc:
        return False, f"Webhook receiver unreachable ({health_url}): {exc.reason}"


def _kaggle_public_bundle_reachability(public_base_url: str) -> tuple[bool, Optional[str]]:
    health_url = f"{(public_base_url or '').strip().rstrip('/')}/health"
    if not health_url.startswith(("http://", "https://")):
        return False, "KAGGLE_BUNDLE_PUBLIC_BASE_URL must use http:// or https://"
    req = urllib.request.Request(
        url=health_url,
        method="GET",
        headers={"ngrok-skip-browser-warning": "true"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if 200 <= resp.status < 300:
                return True, None
            return False, f"Public bundle tunnel health returned HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"Public bundle tunnel health returned HTTP {exc.code}"
    except (TimeoutError, socket.timeout) as exc:
        return False, f"Public bundle tunnel health timeout: {exc}"
    except urllib.error.URLError as exc:
        return False, f"Public bundle tunnel unreachable: {exc.reason}"


def _automation_mode(model_family: str) -> str:
    key = "MLFLOW_AUTOMATION_TFIDF_LR_MODE" if model_family == "tfidf_lr" else "MLFLOW_AUTOMATION_PHOBERT_MODE"
    mode = (get_setting(key, "disabled") or "disabled").strip().lower()
    return mode if mode in {"disabled", "train_only", "full_auto"} else "disabled"


def _automation_policy(model_family: str) -> Dict[str, Any]:
    return {
        "enabled": get_bool_setting("MLFLOW_AUTOMATION_ENABLED", False),
        "mode": _automation_mode(model_family),
        "min_new_rows": get_int_setting("MLFLOW_AUTOMATION_MIN_NEW_ROWS", 50, min_value=1),
        "cooldown_minutes": get_int_setting("MLFLOW_AUTOMATION_COOLDOWN_MINUTES", 1440, min_value=0),
        "dry_run": get_bool_setting("MLFLOW_AUTOMATION_DRY_RUN", True),
        "poll_seconds": get_int_setting("MLFLOW_AUTOMATION_POLL_SECONDS", 30, min_value=10),
        "max_poll_minutes": get_int_setting("MLFLOW_AUTOMATION_MAX_POLL_MINUTES", 180, min_value=1),
    }


def _automation_eligible_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(1)
        FROM mlflow_comment_item
        WHERE gate_bucket = 'accepted'
          AND pseudo_label IN (0, 1)
          AND COALESCE(selected_for_training, 1) = 1
        """
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _automation_record_event(
    model_family: str,
    action: str,
    status: str,
    *,
    source_run_id: Optional[str] = None,
    eligible_count: Optional[int] = None,
    detail: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    params = (
        model_family,
        action,
        source_run_id,
        status,
        eligible_count,
        detail,
        datetime.now(timezone.utc).isoformat(),
    )
    query = """
        INSERT INTO mlflow_automation_event (
            model_family, action, source_run_id, status, eligible_count, detail, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    if conn is not None:
        conn.execute(query, params)
        return
    with sqlite3.connect(FEEDBACK_DB_PATH) as event_conn:
        event_conn.execute(query, params)
        event_conn.commit()


def _automation_parse_timestamp(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _repair_dry_run_checkpoint_if_safe(conn: sqlite3.Connection, model_family: str) -> bool:
    """Restore the old dry-run-only baseline once; never undo a real automation run."""
    state = conn.execute("SELECT last_run_id, active_run_id FROM mlflow_automation_state WHERE model_family = ?", (model_family,)).fetchone()
    if not state or state[1] or not state[0]:
        return False
    last_run_id = str(state[0])
    dry_event = conn.execute(
        "SELECT 1 FROM mlflow_automation_event WHERE model_family = ? AND action = 'train_started' AND source_run_id = ? AND status = 'dry_run'",
        (model_family, last_run_id),
    ).fetchone()
    real_event = conn.execute(
        "SELECT 1 FROM mlflow_automation_event WHERE model_family = ? AND action = 'train_started' AND status NOT IN ('dry_run', 'failed') LIMIT 1",
        (model_family,),
    ).fetchone()
    if not dry_event or real_event:
        return False
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE mlflow_automation_state SET last_triggered_eligible_count = 0, last_triggered_at = NULL, last_run_id = NULL, updated_at = ? WHERE model_family = ?",
        (now, model_family),
    )
    _automation_record_event(model_family, "dry_run_checkpoint_restored", "repaired", source_run_id=last_run_id, detail="Dry-run-only checkpoint restored; no real automation run exists.", conn=conn)
    return True


def _automation_state_snapshot(model_family: str) -> Dict[str, Any]:
    init_feedback_db()
    policy = _automation_policy(model_family)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR IGNORE INTO mlflow_automation_state (model_family, updated_at)
            VALUES (?, ?)
            """,
            (model_family, now),
        )
        state = conn.execute(
            "SELECT * FROM mlflow_automation_state WHERE model_family = ?", (model_family,)
        ).fetchone()
        if _repair_dry_run_checkpoint_if_safe(conn, model_family):
            state = conn.execute("SELECT * FROM mlflow_automation_state WHERE model_family = ?", (model_family,)).fetchone()
        eligible_count = _automation_eligible_count(conn)
        conn.commit()

    state_payload = dict(state) if state else {}
    new_rows = max(0, eligible_count - int(state_payload.get("last_triggered_eligible_count") or 0))
    reason: Optional[str] = None
    if not policy["enabled"]:
        reason = "global_disabled"
    elif policy["mode"] == "disabled":
        reason = "family_disabled"
    elif state_payload.get("active_run_id"):
        reason = "run_active"
    elif new_rows < policy["min_new_rows"]:
        reason = "minimum_new_rows_not_reached"
    else:
        last_triggered_at = _automation_parse_timestamp(state_payload.get("last_triggered_at"))
        if last_triggered_at and policy["cooldown_minutes"] > 0:
            elapsed_minutes = (datetime.now(timezone.utc) - last_triggered_at).total_seconds() / 60
            if elapsed_minutes < policy["cooldown_minutes"]:
                reason = "cooldown_active"
    return {
        "model_family": model_family,
        "policy": policy,
        "state": state_payload,
        "eligible_count": eligible_count,
        "new_eligible_rows": new_rows,
        "ready": reason is None,
        "blocked_reason": reason,
    }


def _claim_automation_cycle(model_family: str, cause: str) -> Dict[str, Any]:
    """Atomically reserve one family cycle before an external trigger is issued."""
    init_feedback_db()
    policy = _automation_policy(model_family)
    now = datetime.now(timezone.utc)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR IGNORE INTO mlflow_automation_state (model_family, updated_at) VALUES (?, ?)",
            (model_family, now.isoformat()),
        )
        state = conn.execute(
            "SELECT * FROM mlflow_automation_state WHERE model_family = ?", (model_family,)
        ).fetchone()
        if _repair_dry_run_checkpoint_if_safe(conn, model_family):
            state = conn.execute("SELECT * FROM mlflow_automation_state WHERE model_family = ?", (model_family,)).fetchone()
        eligible_count = _automation_eligible_count(conn)
        state_payload = dict(state) if state else {}
        new_rows = max(0, eligible_count - int(state_payload.get("last_triggered_eligible_count") or 0))
        active_run_id = str(state_payload.get("active_run_id") or "").strip()

        blocked_reason: Optional[str] = None
        if not policy["enabled"]:
            blocked_reason = "global_disabled"
        elif policy["mode"] == "disabled":
            blocked_reason = "family_disabled"
        elif active_run_id:
            if active_run_id.startswith("claim:"):
                claimed_at = _automation_parse_timestamp(state_payload.get("updated_at"))
                if claimed_at and (now - claimed_at).total_seconds() > 300:
                    conn.execute(
                        "UPDATE mlflow_automation_state SET active_run_id = NULL, updated_at = ? WHERE model_family = ?",
                        (now.isoformat(), model_family),
                    )
                else:
                    blocked_reason = "run_active"
            else:
                run = conn.execute("SELECT status FROM mlflow_do_run WHERE run_id = ?", (active_run_id,)).fetchone()
                if run and str(run[0] or "").lower() in {"queued", "running"}:
                    blocked_reason = "run_active"
                else:
                    conn.execute(
                        "UPDATE mlflow_automation_state SET active_run_id = NULL, updated_at = ? WHERE model_family = ?",
                        (now.isoformat(), model_family),
                    )
        if not blocked_reason and new_rows < policy["min_new_rows"]:
            blocked_reason = "minimum_new_rows_not_reached"
        last_triggered_at = _automation_parse_timestamp(state_payload.get("last_triggered_at"))
        if not blocked_reason and last_triggered_at and policy["cooldown_minutes"] > 0:
            elapsed_minutes = (now - last_triggered_at).total_seconds() / 60
            if elapsed_minutes < policy["cooldown_minutes"]:
                blocked_reason = "cooldown_active"
        if blocked_reason:
            conn.commit()
            return {
                "started": False,
                "model_family": model_family,
                "eligible_count": eligible_count,
                "new_eligible_rows": new_rows,
                "blocked_reason": blocked_reason,
                "policy": policy,
            }

        claim_id = f"claim:{uuid.uuid4().hex}"
        conn.execute(
            "UPDATE mlflow_automation_state SET active_run_id = ?, updated_at = ? WHERE model_family = ?",
            (claim_id, now.isoformat(), model_family),
        )
        _automation_record_event(
            model_family,
            "cycle_claimed",
            "scheduled",
            eligible_count=eligible_count,
            detail=f"cause={cause}; mode={policy['mode']}; dry_run={policy['dry_run']}; new_eligible_rows={new_rows}; minimum={policy['min_new_rows']}",
            conn=conn,
        )
        conn.commit()
    return {
        "started": True,
        "claim_id": claim_id,
        "model_family": model_family,
        "eligible_count": eligible_count,
        "new_eligible_rows": new_rows,
        "policy": policy,
    }


def _automation_request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/mlflow/automation/cycle",
            "raw_path": b"/api/mlflow/automation/cycle",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 0),
            "server": ("automation.local", 80),
        }
    )


def _start_automation_watcher(run_id: str, model_family: str, policy: Dict[str, Any]) -> None:
    with AUTOMATION_WATCHER_LOCK:
        if run_id in AUTOMATION_WATCH_RUN_IDS:
            return
        AUTOMATION_WATCH_RUN_IDS.add(run_id)

    def watch() -> None:
        deadline = time.monotonic() + int(policy["max_poll_minutes"]) * 60
        try:
            while time.monotonic() < deadline:
                try:
                    payload = mlflow_kaggle_status(run_id)
                    if str(payload.get("status") or "").lower() in {"completed", "failed", "dry_run"}:
                        return
                except Exception as exc:
                    _automation_record_event(
                        model_family,
                        "status_watch",
                        "warning",
                        source_run_id=run_id,
                        detail=str(exc),
                    )
                time.sleep(int(policy["poll_seconds"]))
            _automation_record_event(
                model_family,
                "status_watch",
                "timed_out",
                source_run_id=run_id,
                detail="Watcher timed out; refresh Kaggle status to resume terminal handling.",
            )
        finally:
            with AUTOMATION_WATCHER_LOCK:
                AUTOMATION_WATCH_RUN_IDS.discard(run_id)

    threading.Thread(target=watch, name=f"mlflow-automation-{run_id}", daemon=True).start()


def _run_automation_cycle(model_family: str, cause: str) -> Dict[str, Any]:
    claim = _claim_automation_cycle(model_family, cause)
    if not claim["started"]:
        return claim
    claim_id = str(claim["claim_id"])
    policy = claim["policy"]
    try:
        if not policy["dry_run"] and not (get_setting("KAGGLE_BUNDLE_PUBLIC_BASE_URL", "") or "").strip():
            raise HTTPException(
                status_code=400,
                detail="Automation requires KAGGLE_BUNDLE_PUBLIC_BASE_URL when dry-run is disabled",
            )
        model_kind = "lr_smoke" if model_family == "tfidf_lr" else "phobert"
        result = mlflow_kaggle_trigger(
            MlflowDOTriggerRequest(
                model_kind=model_kind,
                training_mode="retrain",
                training_scope="light_only",
                balance_strategy="balanced_50_50",
                bundle_scope="all_batches",
                dry_run=bool(policy["dry_run"]),
            ),
            _automation_request(),
        )
        run_id = str(result["run_id"])
        terminal = str(result.get("status") or "").lower() in {"dry_run", "failed"}
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            if policy["dry_run"]:
                conn.execute("UPDATE mlflow_automation_state SET active_run_id = NULL, updated_at = ? WHERE model_family = ? AND active_run_id = ?", (datetime.now(timezone.utc).isoformat(), model_family, claim_id))
            else:
                conn.execute(
                    """
                    UPDATE mlflow_automation_state
                    SET last_triggered_eligible_count = ?, last_triggered_at = ?, last_run_id = ?,
                        active_run_id = ?, updated_at = ?
                    WHERE model_family = ? AND active_run_id = ?
                    """,
                    (claim["eligible_count"], datetime.now(timezone.utc).isoformat(), run_id, None if terminal else run_id, datetime.now(timezone.utc).isoformat(), model_family, claim_id),
                )
            _automation_record_event(
                model_family,
                "train_started",
                "dry_run" if terminal else "running",
                source_run_id=run_id,
                eligible_count=claim["eligible_count"],
                detail=f"cause={cause}; mode={policy['mode']}; new_eligible_rows={claim['new_eligible_rows']}; minimum={policy['min_new_rows']}; bundle={result.get('bundle_path') or '-'}",
                conn=conn,
            )
            conn.commit()
        if not terminal:
            _start_automation_watcher(run_id, model_family, policy)
        return {**claim, "run_id": run_id, "status": result.get("status"), "dry_run": bool(policy["dry_run"])}
    except Exception as exc:
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            conn.execute(
                "UPDATE mlflow_automation_state SET active_run_id = NULL, updated_at = ? WHERE model_family = ? AND active_run_id = ?",
                (datetime.now(timezone.utc).isoformat(), model_family, claim_id),
            )
            _automation_record_event(
                model_family,
                "train_start",
                "failed",
                eligible_count=claim["eligible_count"],
                detail=str(exc),
                conn=conn,
            )
            conn.commit()
        raise


def _schedule_automation_for_new_training_rows(new_rows: int, cause: str) -> List[str]:
    if new_rows <= 0 or not get_bool_setting("MLFLOW_AUTOMATION_ENABLED", False):
        return []
    scheduled: List[str] = []
    for model_family in ("tfidf_lr", "phobert"):
        if _automation_mode(model_family) == "disabled":
            continue
        threading.Thread(
            target=_run_automation_cycle,
            args=(model_family, f"{cause}; new_rows={new_rows}"),
            name=f"mlflow-automation-trigger-{model_family}",
            daemon=True,
        ).start()
        scheduled.append(model_family)
    return scheduled


@app.get("/api/mlflow/kaggle/preflight", dependencies=[Depends(require_admin)])
def mlflow_kaggle_preflight() -> Dict[str, Any]:
    checked_at = datetime.utcnow().isoformat() + "Z"
    webhook_mode = (get_setting("KAGGLE_WEBHOOK_MODE", "mock") or "mock").strip().lower()
    if webhook_mode not in {"mock", "real"}:
        webhook_mode = "mock"
    webhook_url = (get_setting("KAGGLE_WEBHOOK_URL", "") or "").strip()
    required = {
        "KAGGLE_NOTEBOOK_URL": bool((get_setting("KAGGLE_NOTEBOOK_URL", "") or "").strip()),
        "KAGGLE_WEBHOOK_URL": bool(webhook_url),
    }
    if webhook_mode == "real":
        required["KAGGLE_USERNAME"] = bool((get_setting("KAGGLE_USERNAME", "") or "").strip())
        required["KAGGLE_KEY"] = bool((get_setting("KAGGLE_KEY", "") or "").strip())
    missing = [key for key, ok in required.items() if not ok]
    warnings: List[str] = []
    webhook_reachable = True
    if webhook_url:
        webhook_reachable, webhook_reachability_error = _kaggle_webhook_reachability(webhook_url)
        required["KAGGLE_WEBHOOK_REACHABLE"] = webhook_reachable
        if not webhook_reachable and webhook_reachability_error:
            warnings.append(webhook_reachability_error)
    public_bundle_url = (get_setting("KAGGLE_BUNDLE_PUBLIC_BASE_URL", "") or "").strip()
    public_bundle_reachable = True
    if webhook_mode == "real" and public_bundle_url:
        public_bundle_reachable, public_bundle_error = _kaggle_public_bundle_reachability(public_bundle_url)
        required["KAGGLE_BUNDLE_PUBLIC_REACHABLE"] = public_bundle_reachable
        if not public_bundle_reachable and public_bundle_error:
            warnings.append(public_bundle_error)
    if webhook_mode == "mock":
        warnings.append("KAGGLE_WEBHOOK_MODE=mock: trigger will run in simulation mode and skip credential validation.")
    if not (get_setting("KAGGLE_STATUS_WEBHOOK_URL", "") or "").strip():
        warnings.append("KAGGLE_STATUS_WEBHOOK_URL chưa cấu hình: status chỉ hiển thị từ DB local, không poll cloud realtime.")

    return {
        "ready": len(missing) == 0 and webhook_reachable and public_bundle_reachable,
        "missing": missing,
        "warnings": warnings,
        "checks": required,
        "config": {"provider": "kaggle", "stages": KAGGLE_STAGES, "webhook_mode": webhook_mode},
        "checked_at": checked_at,
    }


@app.get("/api/mlflow/kaggle/bundle")
def mlflow_kaggle_bundle_download(
    run_id: str = Query(..., min_length=1),
    token: str = Query(..., min_length=16),
) -> FileResponse:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute(
            "SELECT bundle_path, bundle_token_hash FROM mlflow_do_run WHERE run_id = ?",
            (run_id.strip(),),
        ).fetchone()
    if not row or not row[0] or not row[1]:
        raise HTTPException(status_code=404, detail="Run bundle not found")
    provided_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(provided_hash, str(row[1])):
        raise HTTPException(status_code=403, detail="Invalid run bundle token")
    resolved = (BASE_DIR / str(row[0])).resolve()
    processed_dir = (BASE_DIR / "data" / "processed").resolve()
    try:
        resolved.relative_to(processed_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run bundle path") from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Run bundle file not found")
    return FileResponse(path=str(resolved), filename=resolved.name, media_type="application/zip")


@app.post("/api/mlflow/kaggle/trigger", dependencies=[Depends(require_admin)])
def mlflow_kaggle_trigger(request: MlflowDOTriggerRequest, http_request: Request) -> Dict[str, Any]:
    init_feedback_db()
    run_id = f"kaggle_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat() + "Z"
    webhook_mode = (get_setting("KAGGLE_WEBHOOK_MODE", "mock") or "mock").strip().lower()
    if webhook_mode not in {"mock", "real"}:
        webhook_mode = "mock"

    training_mode = _do_resolve_training_mode(request)
    try:
        base_model = _do_resolve_base_model(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.training_scope != "light_only":
        raise HTTPException(status_code=400, detail="Kaggle automation only supports light_only scope")
    if training_mode not in {"finetune", "retrain"}:
        raise HTTPException(status_code=400, detail="Kaggle automation only supports finetune/retrain")
    if webhook_mode == "real":
        kaggle_username = (get_setting("KAGGLE_USERNAME", "") or "").strip()
        kaggle_key = (get_setting("KAGGLE_KEY", "") or "").strip()
        if not kaggle_username or not kaggle_key:
            raise HTTPException(
                status_code=400,
                detail="KAGGLE_WEBHOOK_MODE=real requires KAGGLE_USERNAME and KAGGLE_KEY",
            )
    webhook_url = (get_setting("KAGGLE_WEBHOOK_URL", "") or "").strip()
    if not request.dry_run and not webhook_url:
        raise HTTPException(status_code=400, detail="KAGGLE_WEBHOOK_URL is not configured")
    configured_public_base_url = (get_setting("KAGGLE_BUNDLE_PUBLIC_BASE_URL", "") or "").strip()
    if webhook_mode == "real" and not request.dry_run and configured_public_base_url:
        public_bundle_reachable, public_bundle_error = _kaggle_public_bundle_reachability(configured_public_base_url)
        if not public_bundle_reachable:
            raise HTTPException(
                status_code=503,
                detail=public_bundle_error or "Public bundle tunnel is not reachable",
            )

    is_phobert_finetune = request.model_kind == "phobert" and training_mode == "finetune"
    bundle_profile = "full_bundle" if is_phobert_finetune else "clean_victsd_gold"

    bundle_result = mlflow_manual_export_bundle(
        MlflowManualExportBundleRequest(
            batch_id=request.batch_id,
            scope=request.bundle_scope,
            bundle_profile=bundle_profile,
            model_kind=request.model_kind,
            training_mode=training_mode,
            balance_strategy=request.balance_strategy,
            include_base_model=is_phobert_finetune,
            base_model=base_model,
            include_unused=False,
            lineage_run_id=run_id,
        )
    )
    bundle_path = str(bundle_result["bundle_path"])
    resolved_bundle_path = resolve_artifact_ref(bundle_path)
    bundle_checksum = _sha256_file(resolved_bundle_path)
    bundle_token = uuid.uuid4().hex + uuid.uuid4().hex
    bundle_token_hash = hashlib.sha256(bundle_token.encode("utf-8")).hexdigest()
    public_base_url = configured_public_base_url.rstrip("/")
    if not public_base_url and webhook_url:
        parsed_webhook = urllib.parse.urlparse(webhook_url)
        webhook_host = (parsed_webhook.hostname or "").strip().lower()
        if parsed_webhook.scheme in {"http", "https"} and parsed_webhook.netloc and webhook_host not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            public_base_url = f"{parsed_webhook.scheme}://{parsed_webhook.netloc}"
    if not public_base_url:
        public_base_url = str(http_request.base_url).rstrip("/")
    bundle_url = (
        f"{public_base_url}/api/mlflow/kaggle/bundle"
        f"?run_id={urllib.parse.quote(run_id)}&token={urllib.parse.quote(bundle_token)}"
    )
    bundle_manifest = {
        "run_id": run_id,
        "bundle_path": bundle_path,
        "bundle_checksum": bundle_checksum,
        "bundle_profile": bundle_result.get("bundle_profile"),
        "scope": bundle_result.get("scope"),
        "batch_id": bundle_result.get("batch_id"),
        "training_mode": training_mode,
        "base_model": base_model,
        "base_model_bundled": bool(bundle_result.get("base_model_bundled")),
        "base_model_provenance": bundle_result.get("base_model_provenance"),
        "balance_strategy": request.balance_strategy,
        "count_before_balance": bundle_result.get("count_before_balance"),
        "count_after_balance": bundle_result.get("count"),
        "merge_stats": bundle_result.get("merge_stats"),
        "included_mlflow_ids": bundle_result.get("included_mlflow_ids") or [],
        "included_mlflow_ids_sha256": bundle_result.get("included_mlflow_ids_sha256"),
        "feedback_snapshot_sha256": bundle_result.get("feedback_snapshot_sha256"),
        "created_at": now,
    }

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO mlflow_do_run (
                run_id, batch_id, provider, gpu_profile, status, current_stage, logs_json,
                created_at, updated_at, droplet_id, artifact_uri, artifact_checksum, spaces_bucket, spaces_key, error_message,
                bundle_path, bundle_url, bundle_checksum, bundle_token_hash, bundle_manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                request.batch_id,
                "kaggle",
                "kaggle-gpu",
                "queued",
                KAGGLE_STAGES[0],
                json.dumps(
                    [
                        {
                            "ts": now,
                            "message": "Kaggle run queued.",
                            "stage": KAGGLE_STAGES[0],
                            "source": "backend",
                        },
                        {
                            "ts": now,
                            "message": f"[META] model_kind={request.model_kind} training_mode={training_mode} base_model={base_model or 'default'}",
                            "stage": KAGGLE_STAGES[0],
                            "source": "backend",
                        },
                        {
                            "ts": now,
                            "message": f"Bundle snapshot created. path={bundle_path} sha256={bundle_checksum}",
                            "stage": KAGGLE_STAGES[0],
                            "source": "backend",
                        },
                    ],
                    ensure_ascii=False,
                ),
                now,
                now,
                None,
                None,
                None,
                None,
                None,
                None,
                bundle_path,
                bundle_url,
                bundle_checksum,
                bundle_token_hash,
                json.dumps(bundle_manifest, ensure_ascii=False),
            ),
        )
        conn.commit()

    if request.dry_run:
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            _do_append_log(
                conn,
                run_id,
                "Dry-run mode: no Kaggle cloud trigger executed.",
                stage=KAGGLE_STAGES[-1],
                source="backend",
            )
            _do_update_run(conn, run_id, status="dry_run", stage=KAGGLE_STAGES[-1])
            conn.commit()
        return {
            "run_id": run_id,
            "status": "dry_run",
            "stages": KAGGLE_STAGES,
            "dry_run": True,
            "provider": "kaggle",
            "model_kind": request.model_kind,
            "training_mode": training_mode,
            "base_model": base_model,
            "bundle_path": bundle_path,
            "bundle_checksum": bundle_checksum,
            "training_plan": bundle_manifest,
        }

    payload = {
        "run_id": run_id,
        "batch_id": request.batch_id,
        "model_kind": request.model_kind,
        "training_mode": training_mode,
        "base_model": base_model,
        "requested_at": now,
        "notebook_url": (get_setting("KAGGLE_NOTEBOOK_URL", "") or "").strip() or None,
        "bundle_url": bundle_url,
        "bundle_checksum": bundle_checksum,
    }
    try:
        remote = _kaggle_http_json("POST", webhook_url, payload)
    except HTTPException as exc:
        detail = str(exc.detail)
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            _do_append_log(conn, run_id, detail, stage=KAGGLE_STAGES[-1], source="backend")
            _do_update_run(conn, run_id, status="failed", stage=KAGGLE_STAGES[-1], error_message=detail)
            conn.commit()
        raise
    remote_status = str(remote.get("status") or "").strip().lower()
    remote_accepted_raw = remote.get("accepted")
    remote_accepted = bool(remote_accepted_raw) if remote_accepted_raw is not None else True
    remote_error_message = (
        str(remote.get("error_message") or remote.get("message") or remote.get("detail") or "").strip()
    )
    cloud_job_id = str(remote.get("job_id") or remote.get("run_id") or "").strip() or run_id
    if not remote_accepted or remote_status in {"failed", "error", "rejected"}:
        detail = remote_error_message or f"Kaggle webhook rejected run (status={remote_status or 'unknown'})"
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            _do_append_log(conn, run_id, detail, stage=KAGGLE_STAGES[-1], source="kaggle_webhook")
            _do_update_run(conn, run_id, status="failed", stage=KAGGLE_STAGES[-1], error_message=detail)
            conn.commit()
        raise HTTPException(status_code=502, detail=detail)
    if webhook_mode == "real" and cloud_job_id.lower().startswith("mock_"):
        detail = (
            "KAGGLE_WEBHOOK_MODE=real but webhook returned a mock job_id. "
            "Check receiver KAGGLE_WEBHOOK_MODE and restart webhook service."
        )
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            _do_append_log(conn, run_id, detail, stage=KAGGLE_STAGES[-1], source="backend")
            _do_update_run(conn, run_id, status="failed", stage=KAGGLE_STAGES[-1], error_message=detail)
            conn.commit()
        raise HTTPException(status_code=502, detail=detail)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        _do_update_run(conn, run_id, status="running", stage=KAGGLE_STAGES[1], droplet_id=cloud_job_id)
        _do_append_log(
            conn,
            run_id,
            f"Kaggle cloud trigger accepted. job_id={cloud_job_id}",
            stage=KAGGLE_STAGES[1],
            source="mock_webhook" if cloud_job_id.lower().startswith("mock_") else "kaggle_webhook",
        )
        conn.commit()

    return {
        "run_id": run_id,
        "status": "running",
        "stages": KAGGLE_STAGES,
        "dry_run": False,
        "provider": "kaggle",
        "model_kind": request.model_kind,
        "training_mode": training_mode,
        "base_model": base_model,
        "job_id": cloud_job_id,
        "bundle_path": bundle_path,
        "bundle_checksum": bundle_checksum,
        "training_plan": bundle_manifest,
    }


@app.get("/api/mlflow/kaggle/status", dependencies=[Depends(require_admin)])
def mlflow_kaggle_status(run_id: str = Query(..., min_length=1)) -> Dict[str, Any]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = _do_get_run(conn, run_id.strip())
    if not row:
        raise HTTPException(status_code=404, detail=f"Kaggle run not found: {run_id}")

    cloud_job_id = str(row["droplet_id"] or "").strip()
    status_url = (get_setting("KAGGLE_STATUS_WEBHOOK_URL", "") or "").strip()

    if status_url and cloud_job_id and row["status"] in {"queued", "running"}:
        try:
            prev_status = str(row["status"] or "").strip().lower() or "queued"
            prev_stage = str(row["current_stage"] or "").strip() or KAGGLE_STAGES[0]
            prev_artifact_uri = str(row["artifact_uri"] or "").strip()
            remote = _kaggle_http_json("GET", f"{status_url}?job_id={urllib.parse.quote(cloud_job_id)}")
            remote_status = str(remote.get("status") or "").strip().lower()
            remote_stage = str(remote.get("current_stage") or remote.get("stage") or "").strip()
            artifact_uri = str(remote.get("artifact_uri") or "").strip() or None
            artifact_checksum = str(remote.get("artifact_checksum") or "").strip() or None
            error_message = str(remote.get("error_message") or "").strip() or None

            stage = remote_stage if remote_stage in KAGGLE_STAGES else KAGGLE_STAGES[2]
            if remote_status in {"completed", "succeeded", "success"}:
                remote_status = "completed"
                stage = KAGGLE_STAGES[-1]
            elif remote_status in {"failed", "error", "cancelled"}:
                remote_status = "failed"
                stage = KAGGLE_STAGES[-1]
            elif remote_status in {"running", "queued"}:
                stage = remote_stage if remote_stage in KAGGLE_STAGES else KAGGLE_STAGES[2]
            else:
                remote_status = "running"

            with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
                _do_update_run(
                    conn,
                    run_id,
                    status=remote_status,
                    stage=stage,
                    artifact_uri=artifact_uri,
                    artifact_checksum=artifact_checksum,
                    error_message=error_message,
                )
                status_changed = remote_status != prev_status or stage != prev_stage
                artifact_became_available = bool(artifact_uri) and artifact_uri != prev_artifact_uri
                if status_changed or artifact_became_available:
                    source_tag = "mock_webhook" if cloud_job_id.lower().startswith("mock_") else "status_webhook"
                    message = (
                        f"Status polled: status={remote_status} stage={stage}"
                        + (f" artifact_uri={artifact_uri}" if artifact_became_available else "")
                    )
                    _do_append_log(conn, run_id, message, stage=stage, source=source_tag)
                conn.commit()
        except Exception:
            pass

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = _do_get_run(conn, run_id.strip())

    logs = _do_load_logs(row)
    log_events = _do_load_log_events(row)
    runtime_meta = _do_extract_runtime_metadata(logs)
    current_model_kind = str(runtime_meta.get("model_kind") or "phobert").strip().lower()
    artifact_uri_value = str(row["artifact_uri"] or "").strip() or None
    run_mode = _do_infer_run_mode(str(row["droplet_id"] or "").strip() or None, artifact_uri_value)
    artifact_kind = "none"
    if artifact_uri_value:
        artifact_kind = "mock" if artifact_uri_value.lower().startswith("mock://") else "real"
    status_source = _do_infer_status_source(status_url, run_mode)
    stage_timestamps = _do_build_stage_timestamps(log_events, row)
    artifact_download_url = _kaggle_artifact_download_url(row["run_id"], artifact_uri_value) if artifact_kind == "real" else None
    kaggle_metrics = _load_kaggle_artifact_metrics(artifact_uri_value) if artifact_kind == "real" else None
    previous_run = (
        _load_previous_completed_kaggle_run(row, current_model_kind)
        if row["status"] == "completed" and artifact_kind == "real" and kaggle_metrics
        else None
    )
    mlflow_ingestion: Optional[Dict[str, Any]] = None
    if cloud_job_id:
        existing_ingestion = get_kaggle_ingestion_record(
            FEEDBACK_DB_PATH,
            source_job_id=cloud_job_id,
            source_run_id=str(row["run_id"]),
        )
        if existing_ingestion:
            mlflow_ingestion = {
                "ingestion_status": existing_ingestion.get("ingestion_status"),
                "canonical_mlflow_run_id": existing_ingestion.get("canonical_mlflow_run_id"),
                "evidence_sha256": existing_ingestion.get("evidence_sha256"),
                "tracking_status": existing_ingestion.get("tracking_status"),
                "artifact_status": existing_ingestion.get("artifact_status"),
                "retriable": bool(existing_ingestion.get("retriable")),
                "detail": existing_ingestion.get("error_message"),
            }
    if row["status"] == "completed" and artifact_kind == "real":
        _record_kaggle_training_artifact(
            row["run_id"],
            artifact_uri_value,
            kaggle_metrics,
            notes="Kaggle retrain artifact",
        )
        candidate = _load_kaggle_candidate(str(row["run_id"]))
        try:
            mlflow_ingestion = _ingest_kaggle_mlflow_evidence(row, candidate)
        except Exception as exc:
            logger.exception("Kaggle MLflow evidence ingestion failed for %s", row["run_id"])
            mlflow_ingestion = {
                "ingestion_status": "failed",
                "retriable": True,
                "detail": f"Unexpected ingestion failure: {type(exc).__name__}",
            }
        try:
            _register_kaggle_candidate(str(row["run_id"]))
        except Exception:
            logger.exception("Model Registry candidate registration failed for %s", row["run_id"])
    if row["status"] in {"completed", "failed", "dry_run"}:
        try:
            _automation_handle_terminal_run(str(row["run_id"]))
        except Exception:
            logger.exception("Automation terminal handling failed for %s", row["run_id"])
    gemini_evaluation = _load_gemini_evaluation(str(row["run_id"])) if row["status"] == "completed" else None

    return {
        "run_id": row["run_id"],
        "batch_id": row["batch_id"],
        "provider": row["provider"] or "kaggle",
        "gpu_profile": row["gpu_profile"],
        "compute_mode": "kaggle",
        "model_kind": current_model_kind,
        "training_mode": "retrain" if current_model_kind == "lr_smoke" else runtime_meta.get("training_mode") or "retrain",
        "base_model": None if current_model_kind == "lr_smoke" else runtime_meta.get("base_model"),
        "status": row["status"],
        "current_stage": row["current_stage"],
        "logs": logs,
        "log_events": log_events,
        "stages": KAGGLE_STAGES,
        "artifact_uri": row["artifact_uri"],
        "artifact_kind": artifact_kind,
        "artifact_download_url": artifact_download_url,
        "artifact_checksum": row["artifact_checksum"],
        "bundle_path": row["bundle_path"],
        "bundle_url": row["bundle_url"],
        "bundle_checksum": row["bundle_checksum"],
        "training_plan": json.loads(row["bundle_manifest_json"]) if row["bundle_manifest_json"] else None,
        "metrics": kaggle_metrics,
        "previous_run": previous_run,
        "gemini_evaluation": gemini_evaluation,
        "mlflow_ingestion": mlflow_ingestion,
        "error_message": row["error_message"],
        "run_mode": run_mode,
        "status_source": status_source,
        "trigger_source": "automation" if _automation_run_was_started(str(row["run_id"])) else "manual",
        "stage_timestamps": stage_timestamps,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "job_id": (row["droplet_id"] or None),
    }


@app.get("/api/mlflow/kaggle/artifact/download", dependencies=[Depends(require_admin)])
def mlflow_kaggle_artifact_download(run_id: str = Query(..., min_length=1)) -> FileResponse:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = _do_get_run(conn, run_id.strip())
    if not row:
        raise HTTPException(status_code=404, detail=f"Kaggle run not found: {run_id}")
    if str(row["status"] or "").strip().lower() != "completed":
        raise HTTPException(status_code=400, detail="Kaggle artifact is only downloadable after completion")

    resolved = _resolve_kaggle_artifact_path(str(row["artifact_uri"] or "").strip())
    return FileResponse(path=str(resolved), filename=resolved.name, media_type="application/zip")


def _registry_row_payload(row: sqlite3.Row) -> Dict[str, Any]:
    artifact_available = False
    try:
        _resolve_registry_artifact_path(row["artifact_uri"])
        artifact_available = True
    except HTTPException:
        pass
    try:
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
    except json.JSONDecodeError:
        metrics = {}
    return {
        "model_id": row["model_id"],
        "source_run_id": row["source_run_id"],
        "model_family": row["model_family"],
        "model_kind": row["model_kind"],
        "training_mode": row["training_mode"],
        "base_model": row["base_model"],
        "status": row["status"],
        "artifact_available": artifact_available,
        "artifact_checksum": row["artifact_checksum"],
        "metrics": _normalize_saved_model_metrics(metrics),
        "created_at": row["created_at"],
        "promoted_at": row["promoted_at"],
    }


@app.get("/api/mlflow/registry", dependencies=[Depends(require_admin)])
def mlflow_model_registry(include_deleted: bool = Query(False)) -> Dict[str, Any]:
    _backfill_model_registry_candidates()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM mlflow_model_version"
        if not include_deleted:
            query += " WHERE status <> 'deleted'"
        rows = conn.execute(query + " ORDER BY created_at DESC").fetchall()
        active_rows = conn.execute("SELECT model_family, active_model_id FROM mlflow_production_slot").fetchall()
    active = {str(row["model_family"]): str(row["active_model_id"]) for row in active_rows}
    items = []
    for row in rows:
        item = _registry_row_payload(row)
        item["is_current_production"] = active.get(str(row["model_family"])) == str(row["model_id"])
        items.append(item)
    return {"items": items, "backfilled": True}


@app.get("/api/mlflow/registry/detail", dependencies=[Depends(require_admin)])
def mlflow_model_registry_detail(model_id: str = Query(..., min_length=3)) -> Dict[str, Any]:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM mlflow_model_version WHERE model_id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Registry model not found")
        events = conn.execute(
            "SELECT action, source_run_id, from_model_id, to_model_id, status, detail, created_at FROM mlflow_promotion_event WHERE to_model_id = ? OR from_model_id = ? ORDER BY id DESC",
            (model_id, model_id),
        ).fetchall()
    payload = _registry_row_payload(row)
    payload.update({
        "artifact_uri": row["artifact_uri"],
        "artifact_path": row["artifact_path"],
        "bundle_path": row["bundle_path"],
        "bundle_checksum": row["bundle_checksum"],
        "test_fingerprint": row["test_fingerprint"],
        "promotion_history": [dict(event) for event in events],
    })
    return payload


@app.get("/api/mlflow/registry/download", dependencies=[Depends(require_admin)])
def mlflow_model_registry_download(model_id: str = Query(..., min_length=3)) -> FileResponse:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute("SELECT status, artifact_uri FROM mlflow_model_version WHERE model_id = ?", (model_id,)).fetchone()
    if not row or str(row[0]) == "deleted":
        raise HTTPException(status_code=404, detail="Registry model is unavailable")
    artifact = _resolve_registry_artifact_path(row[1])
    return FileResponse(path=str(artifact), filename=f"{_sanitize_import_model_name(model_id)}.zip", media_type="application/zip")


@app.post("/api/mlflow/registry/archive", dependencies=[Depends(require_admin)])
def mlflow_model_registry_archive(request: MlflowRegistryLifecycleRequest) -> Dict[str, Any]:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute("SELECT status FROM mlflow_model_version WHERE model_id = ?", (request.model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Registry model not found")
        if str(row[0]) == "production":
            raise HTTPException(status_code=409, detail="Production model cannot be archived")
        if str(row[0]) == "deleted":
            raise HTTPException(status_code=409, detail="Deleted registry model cannot be archived")
        conn.execute("UPDATE mlflow_model_version SET status = 'archived' WHERE model_id = ?", (request.model_id,))
        conn.commit()
    return {"status": "archived", "model_id": request.model_id}


@app.post("/api/mlflow/registry/delete", dependencies=[Depends(require_admin)])
def mlflow_model_registry_delete(request: MlflowRegistryLifecycleRequest) -> Dict[str, Any]:
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required to delete a registry model")
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute("SELECT status FROM mlflow_model_version WHERE model_id = ?", (request.model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Registry model not found")
        referenced = conn.execute(
            "SELECT 1 FROM mlflow_production_slot WHERE active_model_id = ? OR previous_model_id = ? LIMIT 1",
            (request.model_id, request.model_id),
        ).fetchone()
        if str(row[0]) == "production" or referenced:
            raise HTTPException(status_code=409, detail="A production-slot model cannot be deleted")
        conn.execute("UPDATE mlflow_model_version SET status = 'deleted' WHERE model_id = ?", (request.model_id,))
        conn.commit()
    return {"status": "deleted", "model_id": request.model_id, "artifact_deleted": False, "training_run_deleted": False}


def _model_family_from_kind(model_kind: Optional[str]) -> str:
    normalized = str(model_kind or "").strip().lower()
    if normalized in {"lr_smoke", "tfidf_lr"}:
        return "tfidf_lr"
    return "phobert"


def _normalize_saved_model_metrics(raw: Any) -> Dict[str, Optional[float]]:
    payload = raw if isinstance(raw, dict) else {}
    test_payload = payload.get("test") if isinstance(payload.get("test"), dict) else payload
    return {
        "f1_toxic": normalize_score(test_payload.get("f1_toxic") or test_payload.get("toxic_f1")),
        "macro_f1": normalize_score(test_payload.get("macro_f1") or test_payload.get("f1")),
        "accuracy": normalize_score(test_payload.get("accuracy")),
        "precision": normalize_score(test_payload.get("precision") or test_payload.get("precision_toxic")),
        "recall": normalize_score(test_payload.get("recall") or test_payload.get("recall_toxic")),
    }


def _semantic_jsonl_fingerprint_bytes(raw: bytes) -> Tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for raw_line in raw.decode("utf-8", errors="strict").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError("JSONL row must be an object")
        normalized = {
            "text": str(row.get("text") or ""),
            "toxicity": normalize_int(row.get("toxicity") if row.get("toxicity") is not None else row.get("label")),
            "constructiveness": normalize_int(row.get("constructiveness")),
        }
        digest.update(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def _local_gold_test_fingerprint() -> Tuple[Optional[str], Optional[int]]:
    test_path = DATASET_VERSION_DIRS.get("victsd_gold", PROCESSED_DATA_DIR / "victsd_gold") / "test.jsonl"
    try:
        return _semantic_jsonl_fingerprint_bytes(test_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None, None


def _bundle_test_fingerprint(bundle_path: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    raw_path = str(bundle_path or "").strip()
    if not raw_path:
        return None, None
    try:
        resolved = resolve_artifact_ref(raw_path)
        resolved.relative_to(PROCESSED_DATA_DIR.resolve())
        with zipfile.ZipFile(resolved, "r") as zf:
            members = [name for name in zf.namelist() if Path(name).name == "test.jsonl"]
            if not members:
                return None, None
            return _semantic_jsonl_fingerprint_bytes(zf.read(sorted(members, key=len)[0]))
    except (OSError, ValueError, zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError):
        return None, None


def _artifact_contract(artifact_path: Path, model_family: str) -> Tuple[bool, str]:
    try:
        with zipfile.ZipFile(artifact_path, "r") as zf:
            _validate_model_import_zip(zf)
            files = {Path(name).name.lower() for name in zf.namelist() if not name.endswith("/")}
            if model_family == "tfidf_lr":
                has_model = bool(files & {"model_lr.joblib", "model_lr.pkl"})
                has_vectorizer = bool(files & {"vectorizer.joblib", "vectorizer.pkl"})
                if has_model and has_vectorizer:
                    return True, "TF-IDF/LR serving files found"
                return False, "Artifact must contain model_lr and vectorizer files"

            members = [Path(name) for name in zf.namelist() if not name.endswith("/")]
            config_parents = {member.parent for member in members if member.name.lower() == "config.json"}
            weight_parents = {
                member.parent
                for member in members
                if member.name.lower() in {"model.safetensors", "pytorch_model.bin"}
            }
            if config_parents & weight_parents:
                return True, "PhoBERT config and weights found"
            return False, "Artifact must contain PhoBERT config and weights in the same directory"
    except (OSError, zipfile.BadZipFile, HTTPException) as exc:
        return False, f"Invalid artifact ZIP: {exc}"


def _load_kaggle_candidate(run_id: str) -> Optional[Dict[str, Any]]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = _do_get_run(conn, run_id)
    if not row:
        return None
    logs = _do_load_logs(row)
    runtime_meta = _do_extract_runtime_metadata(logs)
    model_family = _model_family_from_kind(runtime_meta.get("model_kind"))
    artifact_uri = str(row["artifact_uri"] or "").strip()
    metrics = _load_kaggle_artifact_metrics(artifact_uri) if artifact_uri else None
    artifact_path: Optional[Path] = None
    artifact_actual_checksum: Optional[str] = None
    contract_ok = False
    contract_detail = "Artifact is unavailable"
    try:
        artifact_path = _resolve_kaggle_artifact_path(artifact_uri)
        artifact_actual_checksum = _sha256_file(artifact_path)
        contract_ok, contract_detail = _artifact_contract(artifact_path, model_family)
    except HTTPException as exc:
        contract_detail = str(exc.detail)
    test_fingerprint, test_size = _bundle_test_fingerprint(row["bundle_path"])
    try:
        training_plan = json.loads(row["bundle_manifest_json"]) if row["bundle_manifest_json"] else {}
    except json.JSONDecodeError:
        training_plan = {}
    return {
        "row": row,
        "run_id": str(row["run_id"]),
        "model_family": model_family,
        "model_kind": runtime_meta.get("model_kind") or "phobert",
        "training_mode": "retrain" if model_family == "tfidf_lr" else runtime_meta.get("training_mode") or "retrain",
        "metrics": metrics,
        "artifact_uri": artifact_uri,
        "artifact_path": artifact_path,
        "artifact_expected_checksum": str(row["artifact_checksum"] or "").strip().lower() or None,
        "artifact_actual_checksum": artifact_actual_checksum,
        "contract_ok": contract_ok,
        "contract_detail": contract_detail,
        "bundle_checksum": str(row["bundle_checksum"] or "").strip() or None,
        "included_mlflow_ids_sha256": training_plan.get("included_mlflow_ids_sha256") if isinstance(training_plan, dict) else None,
        "feedback_snapshot_sha256": training_plan.get("feedback_snapshot_sha256") if isinstance(training_plan, dict) else None,
        "test_fingerprint": test_fingerprint,
        "test_size": test_size,
    }


def _kaggle_candidate_artifact_is_verified(candidate: Optional[Dict[str, Any]]) -> bool:
    if not candidate:
        return False
    expected = str(candidate.get("artifact_expected_checksum") or "").lower()
    actual = str(candidate.get("artifact_actual_checksum") or "").lower()
    return bool(
        isinstance(candidate.get("artifact_path"), Path)
        and expected
        and actual
        and hmac.compare_digest(expected, actual)
        and candidate.get("contract_ok")
    )


def _ingest_kaggle_mlflow_evidence(
    row: sqlite3.Row,
    candidate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    source_job_id = str(row["droplet_id"] or "").strip()
    source_run_id = str(row["run_id"] or "").strip()
    if not source_job_id:
        return {
            "ingestion_status": "not_available",
            "retriable": False,
            "detail": "Kaggle source job identity is unavailable",
        }
    if not _kaggle_candidate_artifact_is_verified(candidate):
        return {
            "ingestion_status": "not_eligible",
            "retriable": False,
            "detail": "Training artifact must pass its existing checksum and serving contract first",
        }
    artifact_path = candidate.get("artifact_path")
    assert isinstance(artifact_path, Path)
    try:
        evidence = validate_kaggle_evidence(
            artifact_path,
            expected_source_job_id=source_job_id,
            expected_source_run_id=source_run_id,
        )
        result = ingest_kaggle_evidence(evidence, db_path=FEEDBACK_DB_PATH)
        return {
            "ingestion_status": result.get("ingestion_status") or result.get("action"),
            "canonical_mlflow_run_id": result.get("canonical_mlflow_run_id"),
            "evidence_sha256": evidence.evidence_sha256,
            "tracking_status": evidence.manifest["status"]["tracking_status"],
            "artifact_status": evidence.manifest["status"]["artifact_status"],
            "retriable": bool(result.get("retriable")),
            "detail": result.get("error_message"),
        }
    except KaggleEvidenceNotFound:
        return {
            "ingestion_status": "not_available",
            "retriable": False,
            "detail": "Portable MLflow evidence is absent (pre-Phase 2B.2B artifact)",
        }
    except KaggleEvidenceConflictError as exc:
        return {
            "ingestion_status": "conflict",
            "retriable": False,
            "detail": str(exc),
        }
    except KaggleEvidenceValidationError as exc:
        return {
            "ingestion_status": "invalid",
            "retriable": False,
            "detail": str(exc),
        }
    except KaggleEvidenceIngestionUnavailable as exc:
        record = get_kaggle_ingestion_record(
            FEEDBACK_DB_PATH,
            source_job_id=source_job_id,
            source_run_id=source_run_id,
        )
        return {
            "ingestion_status": "failed",
            "canonical_mlflow_run_id": None,
            "evidence_sha256": (record or {}).get("evidence_sha256"),
            "tracking_status": (record or {}).get("tracking_status"),
            "artifact_status": (record or {}).get("artifact_status"),
            "retriable": True,
            "detail": str(exc),
        }


def _register_kaggle_candidate(run_id: str) -> Optional[Dict[str, Any]]:
    """Durably register one completed, verified Kaggle artifact without deploying it."""
    candidate = _load_kaggle_candidate(run_id)
    if not candidate:
        return None
    row = candidate["row"]
    expected = str(candidate.get("artifact_expected_checksum") or "").lower()
    actual = str(candidate.get("artifact_actual_checksum") or "").lower()
    artifact_path = candidate.get("artifact_path")
    if (
        str(row["status"] or "").lower() != "completed"
        or not isinstance(artifact_path, Path)
        or not expected
        or not actual
        or not hmac.compare_digest(expected, actual)
        or not candidate.get("contract_ok")
    ):
        return None
    model_family = str(candidate["model_family"])
    stable_artifact = _copy_registry_artifact(artifact_path, model_family, run_id, actual)
    runtime_meta = _do_extract_runtime_metadata(_do_load_logs(row))
    model_id = _registry_model_id(model_family, run_id)
    metrics_json = json.dumps(candidate.get("metrics") or {}, ensure_ascii=False)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        existing = conn.execute(
            "SELECT status, artifact_path FROM mlflow_model_version WHERE source_run_id = ?",
            (run_id,),
        ).fetchone()
        existing_status = str(existing[0]) if existing else "candidate"
        lifecycle = existing_status if existing_status in {"production", "archived", "deleted"} else "candidate"
        stored_path = str(existing[1]) if existing and existing_status == "production" else str(stable_artifact)
        conn.execute(
            """
            INSERT INTO mlflow_model_version (
                model_family, model_id, source_run_id, artifact_path, artifact_checksum,
                bundle_checksum, test_fingerprint, metrics_json, status, created_at,
                model_kind, training_mode, base_model, artifact_uri, bundle_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_run_id) DO UPDATE SET
                artifact_checksum = excluded.artifact_checksum,
                bundle_checksum = excluded.bundle_checksum,
                test_fingerprint = excluded.test_fingerprint,
                metrics_json = excluded.metrics_json,
                model_kind = excluded.model_kind,
                training_mode = excluded.training_mode,
                base_model = excluded.base_model,
                artifact_uri = excluded.artifact_uri,
                bundle_path = excluded.bundle_path
            """,
            (
                model_family, model_id, run_id, stored_path, actual,
                candidate.get("bundle_checksum"), candidate.get("test_fingerprint"), metrics_json, lifecycle, now,
                candidate.get("model_kind"), candidate.get("training_mode"), runtime_meta.get("base_model"),
                encode_artifact_ref(stable_artifact), encode_artifact_ref(str(row["bundle_path"] or "")),
            ),
        )
        conn.commit()
    return {"model_id": model_id, "artifact_path": str(stable_artifact), "status": lifecycle}


def _backfill_model_registry_candidates() -> int:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT run_id FROM mlflow_do_run WHERE status = 'completed' AND artifact_uri IS NOT NULL AND artifact_uri <> ''"
        ).fetchall()
    registered = 0
    for row in rows:
        try:
            if _register_kaggle_candidate(str(row[0])):
                registered += 1
        except (HTTPException, OSError):
            continue
    return registered


def _production_snapshot(model_family: str) -> Dict[str, Any]:
    model_id = get_family_default_model_id(resolve_model_root(), model_family)
    if not model_id:
        return {"model": None, "model_family": model_family, "metrics": {}, "test_fingerprint": None, "test_size": None}
    try:
        resolved_type, resolved_name, model_path = resolve_model_path(resolve_model_root(), model_id)
    except (FileNotFoundError, OSError, ValueError):
        return {"model": None, "model_family": model_family, "metrics": {}, "test_fingerprint": None, "test_size": None}
    metrics_raw = _load_model_json(model_path, "metrics.json")
    production_manifest = _load_model_json(model_path, "production_manifest.json")
    training_manifest = _load_model_json(model_path, "training_manifest.json")
    run_config = _load_model_json(model_path, "run_config.json")
    test_fingerprint = str(production_manifest.get("test_fingerprint") or "").strip() or None
    test_size = normalize_int(production_manifest.get("test_size"))
    if not test_fingerprint:
        test_fingerprint, local_test_size = _local_gold_test_fingerprint()
        test_size = test_size if test_size is not None else local_test_size
    run_id = production_manifest.get("source_run_id") or training_manifest.get("run_id") or run_config.get("run_id")
    previous_model_id: Optional[str] = None
    try:
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            slot = conn.execute(
                "SELECT previous_model_id FROM mlflow_production_slot WHERE model_family = ?",
                (model_family,),
            ).fetchone()
            previous_model_id = str(slot[0]).strip() if slot and slot[0] else None
    except sqlite3.Error:
        previous_model_id = None
    return {
        "model": f"{resolved_type}/{resolved_name}",
        "model_family": model_family,
        "run_id": run_id,
        "metrics": _normalize_saved_model_metrics(metrics_raw),
        "artifact_checksum": production_manifest.get("artifact_checksum"),
        "test_fingerprint": test_fingerprint,
        "test_size": test_size,
        "previous_model": previous_model_id,
        "rollback_available": bool(previous_model_id),
        "created_at": production_manifest.get("promoted_at") or training_manifest.get("trained_at") or run_config.get("created_at"),
    }


def _build_family_comparison(run_id: str) -> Dict[str, Any]:
    candidate = _load_kaggle_candidate(run_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"Kaggle run not found: {run_id}")
    row = candidate["row"]
    candidate_metrics = candidate.get("metrics") or {}
    model_family = str(candidate["model_family"])
    current = _production_snapshot(model_family)
    current_metrics = current.get("metrics") or {}

    deltas: Dict[str, Optional[float]] = {}
    for metric in ("accuracy", "macro_f1", "f1_toxic", "precision", "recall"):
        candidate_value = normalize_score(candidate_metrics.get(metric))
        current_value = normalize_score(current_metrics.get(metric))
        deltas[metric] = candidate_value - current_value if candidate_value is not None and current_value is not None else None

    expected_checksum = candidate.get("artifact_expected_checksum")
    actual_checksum = candidate.get("artifact_actual_checksum")
    artifact_verified = bool(expected_checksum and actual_checksum and hmac.compare_digest(expected_checksum, actual_checksum))
    test_verified = bool(
        candidate.get("test_fingerprint")
        and current.get("test_fingerprint")
        and hmac.compare_digest(str(candidate["test_fingerprint"]), str(current["test_fingerprint"]))
    )
    gate_checks = [
        {"name": "run completed with real artifact", "delta": None, "passed": str(row["status"] or "").lower() == "completed" and candidate.get("artifact_path") is not None},
        {"name": "artifact SHA-256 verified", "delta": None, "passed": artifact_verified},
        {"name": f"{model_family} serving contract", "delta": None, "passed": bool(candidate.get("contract_ok")), "detail": candidate.get("contract_detail")},
        {"name": "same semantic test set", "delta": None, "passed": test_verified},
        {"name": "f1_toxic delta >= 0", "delta": deltas["f1_toxic"], "passed": bool(deltas["f1_toxic"] is not None and deltas["f1_toxic"] >= 0.0)},
        {"name": "macro delta >= -0.01", "delta": deltas["macro_f1"], "passed": bool(deltas["macro_f1"] is not None and deltas["macro_f1"] >= -0.01)},
        {"name": "candidate f1_toxic >= 0.45", "delta": normalize_score(candidate_metrics.get("f1_toxic")), "passed": bool(normalize_score(candidate_metrics.get("f1_toxic")) is not None and float(candidate_metrics["f1_toxic"]) >= 0.45)},
    ]
    promotion_enabled = bool(gate_checks and all(bool(check["passed"]) for check in gate_checks))
    return {
        "model_family": model_family,
        "current": current,
        "candidate": {
            "model": f"{model_family}/{_sanitize_import_model_name(run_id)}",
            "model_family": model_family,
            "artifact_path": candidate.get("artifact_uri"),
            "artifact_checksum": expected_checksum,
            "artifact_actual_checksum": actual_checksum,
            "artifact_verified": artifact_verified,
            "artifact_contract": candidate.get("contract_detail"),
            "metrics": candidate_metrics,
            "source_run_id": run_id,
            "raw_metrics": candidate_metrics,
            "created_at": row["created_at"],
            "bundle_checksum": candidate.get("bundle_checksum"),
            "included_mlflow_ids_sha256": candidate.get("included_mlflow_ids_sha256"),
            "feedback_snapshot_sha256": candidate.get("feedback_snapshot_sha256"),
            "test_fingerprint": candidate.get("test_fingerprint"),
            "test_size": candidate.get("test_size"),
        },
        "deltas": deltas,
        "test_comparability_verified": test_verified,
        "gate_checks": gate_checks,
        "promotion_enabled": promotion_enabled,
        "promotion_mode": "family_production_slot",
    }


def _latest_candidate_run_id() -> Optional[str]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT source_run_id
            FROM mlflow_training_artifact
            WHERE source_run_id IS NOT NULL AND source_run_id <> ''
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    return str(row[0]) if row and row[0] else None


@app.get("/api/mlflow/compare/latest", dependencies=[Depends(require_admin)])
def mlflow_compare_latest(run_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    resolved_run_id = str(run_id or _latest_candidate_run_id() or "").strip()
    if not resolved_run_id:
        return {
            "current": {},
            "candidate": {},
            "gate_checks": [],
            "promotion_enabled": False,
            "promotion_mode": "family_production_slot",
            "message": "No completed Kaggle candidate is available yet.",
        }
    return _build_family_comparison(resolved_run_id)


def _load_gemini_evaluation(run_id: str) -> Optional[Dict[str, Any]]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT run_id, model_family, previous_run_id, evaluation_json, model_name, created_at
            FROM mlflow_gemini_evaluation WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if not row:
        return None
    try:
        evaluation = json.loads(str(row["evaluation_json"] or "{}"))
    except json.JSONDecodeError:
        evaluation = {}
    return {
        "run_id": row["run_id"],
        "model_family": row["model_family"],
        "previous_run_id": row["previous_run_id"],
        "evaluation": evaluation if isinstance(evaluation, dict) else {},
        "model": row["model_name"],
        "created_at": row["created_at"],
    }


def _parse_gemini_evaluation(raw: str) -> Dict[str, Any]:
    cleaned = str(raw or "").strip()
    parsed: Any = None
    candidates = [cleaned]
    if cleaned.startswith("```") and cleaned.endswith("```"):
        candidates.append(cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            start = candidate.find("{")
            while start >= 0:
                try:
                    parsed, _ = decoder.raw_decode(candidate[start:])
                except json.JSONDecodeError:
                    start = candidate.find("{", start + 1)
                    continue
                if isinstance(parsed, dict):
                    break
                start = candidate.find("{", start + 1)
        if isinstance(parsed, dict):
            break
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Gemini Evaluate did not return a valid JSON object")

    verdict = str(parsed.get("verdict") or "review").strip().lower()
    if verdict not in {"promote", "review", "hold"}:
        verdict = "review"

    def text_field(name: str, limit: int = 900) -> str:
        return str(parsed.get(name) or "").strip()[:limit]

    def text_list(name: str) -> List[str]:
        raw_items = parsed.get(name)
        if not isinstance(raw_items, list):
            return []
        return [str(item).strip()[:220] for item in raw_items if str(item).strip()][:5]

    return {
        "summary": text_field("summary"),
        "verdict": verdict,
        "recommendation": text_field("recommendation"),
        "strengths": text_list("strengths"),
        "risks": text_list("risks"),
        "metric_observations": text_list("metric_observations"),
    }


def _build_gemini_evaluate_prompt(
    instruction: str,
    comparison: Dict[str, Any],
    previous_run: Optional[Dict[str, Any]],
) -> str:
    evidence = {
        "model_family": comparison.get("model_family"),
        "production": comparison.get("current"),
        "candidate": comparison.get("candidate"),
        "deltas_vs_production": comparison.get("deltas"),
        "same_semantic_test_set": comparison.get("test_comparability_verified"),
        "production_gate": comparison.get("gate_checks"),
        "previous_completed_same_model_kind": previous_run,
    }
    return (
        f"{instruction}\n\n"
        "Dưới đây là bằng chứng có cấu trúc của một candidate train mới. "
        "Chỉ dựa vào các số liệu này; nếu thiếu bằng chứng, nêu rõ giới hạn. "
        "Không tự promote model và không xem nhận định này là thay thế production gate.\n"
        "Chỉ trả về một JSON object hợp lệ, không markdown; bắt đầu ngay bằng { và kết thúc bằng }, với schema:\n"
        '{"summary": string, "verdict": "promote"|"review"|"hold", '
        '"recommendation": string, "strengths": [string], "risks": [string], '
        '"metric_observations": [string]}.\n'
        "Quy ước verdict: promote chỉ là đề xuất khi mọi gate pass; review khi cần admin xem thêm; "
        "hold khi rủi ro/gate không đạt. Hãy trả lời bằng tiếng Việt.\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)}"
    )


@app.post("/api/mlflow/kaggle/evaluate", dependencies=[Depends(require_admin)])
def mlflow_kaggle_gemini_evaluate(request: MlflowGeminiEvaluateRequest) -> Dict[str, Any]:
    run_id = request.run_id.strip()
    candidate = _load_kaggle_candidate(run_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"Kaggle run not found: {run_id}")
    row = candidate["row"]
    if str(row["status"] or "").lower() != "completed" or candidate.get("artifact_path") is None:
        raise HTTPException(status_code=409, detail="Gemini Evaluate requires a completed run with a real artifact")

    if not request.force:
        cached = _load_gemini_evaluation(run_id)
        if cached:
            return {"status": "cached", **cached}

    runtime_meta = _do_extract_runtime_metadata(_do_load_logs(row))
    model_kind = str(runtime_meta.get("model_kind") or "phobert").strip().lower()
    comparison = _build_family_comparison(run_id)
    previous_run = _load_previous_completed_kaggle_run(row, model_kind)
    instruction = (
        get_setting(
            "GEMINI_EVALUATE_INSTRUCTION",
            "Bạn là trợ lý đánh giá thí nghiệm MLOps tiếng Việt.",
        )
        or ""
    ).strip()
    evaluation_prompt = _build_gemini_evaluate_prompt(instruction, comparison, previous_run)
    raw_evaluation: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    for attempt in range(1, 3):
        raw_evaluation = call_gemini_with_model(evaluation_prompt)
        try:
            evaluation = _parse_gemini_evaluation(raw_evaluation)
            break
        except HTTPException:
            if attempt < 2:
                logger.warning("Gemini Evaluate returned invalid JSON; retrying once for run %s", run_id)
    if evaluation is None or raw_evaluation is None:
        raise HTTPException(status_code=502, detail="Gemini Evaluate did not return a valid JSON object after retry")
    model_name = getattr(raw_evaluation, "model", None) or get_setting("GEMINI_MODEL", "gemini-1.5-flash-latest")
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO mlflow_gemini_evaluation (
                run_id, model_family, previous_run_id, prompt_instruction, evaluation_json, model_name, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                model_family = excluded.model_family,
                previous_run_id = excluded.previous_run_id,
                prompt_instruction = excluded.prompt_instruction,
                evaluation_json = excluded.evaluation_json,
                model_name = excluded.model_name,
                created_at = excluded.created_at
            """,
            (
                run_id,
                candidate["model_family"],
                previous_run.get("run_id") if previous_run else None,
                instruction,
                json.dumps(evaluation, ensure_ascii=False),
                model_name,
                now,
            ),
        )
        conn.commit()
    return {
        "status": "evaluated",
        "run_id": run_id,
        "model_family": candidate["model_family"],
        "previous_run_id": previous_run.get("run_id") if previous_run else None,
        "evaluation": evaluation,
        "model": model_name,
        "created_at": now,
    }


def _find_extracted_model_dir(extract_root: Path, model_family: str) -> Path:
    directories = [extract_root] + [path for path in extract_root.rglob("*") if path.is_dir()]
    if model_family == "phobert":
        for directory in directories:
            try:
                validate_model_artifacts("phobert", directory)
                return directory
            except (FileNotFoundError, OSError, ValueError):
                continue
        raise HTTPException(status_code=400, detail="PhoBERT artifact does not contain a serving-compatible checkpoint")
    return extract_root


def _smoke_validate_staged_model(model_family: str, model_dir: Path) -> None:
    if model_family == "tfidf_lr":
        try:
            import joblib

            vectorizer = joblib.load(model_dir / "vectorizer.pkl")
            model = joblib.load(model_dir / "model_lr.pkl")
            features = vectorizer.transform(["kiểm tra artifact production"])
            prediction = model.predict(features)
            if len(prediction) != 1:
                raise ValueError("unexpected prediction shape")
            for filename in ("model_constructiveness_lr.joblib", "model_lr_constructiveness.pkl"):
                constructiveness_path = model_dir / filename
                if not constructiveness_path.is_file():
                    continue
                constructiveness_model = joblib.load(constructiveness_path)
                if not hasattr(constructiveness_model, "predict_proba"):
                    raise ValueError("constructiveness classifier lacks predict_proba")
                constructiveness_probs = constructiveness_model.predict_proba(features)
                if len(constructiveness_probs) != 1:
                    raise ValueError("unexpected constructiveness prediction shape")
                break
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"TF-IDF/LR serving smoke check failed: {exc}") from exc
        return

    config = _load_model_json(model_dir, "config.json")
    if not config or not isinstance(config.get("model_type"), str):
        raise HTTPException(status_code=400, detail="PhoBERT config.json is invalid or missing model_type")
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as weights:
                if not list(weights.keys()):
                    raise ValueError("checkpoint has no tensors")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PhoBERT safetensors smoke check failed: {exc}") from exc
        return
    pytorch_path = model_dir / "pytorch_model.bin"
    if pytorch_path.stat().st_size <= 0:
        raise HTTPException(status_code=400, detail="PhoBERT pytorch_model.bin is empty")


def _install_candidate_artifact(candidate: Dict[str, Any], comparison: Dict[str, Any]) -> Tuple[str, Path]:
    model_family = str(candidate["model_family"])
    run_id = str(candidate["run_id"])
    artifact_path = candidate.get("artifact_path")
    if not isinstance(artifact_path, Path):
        raise HTTPException(status_code=400, detail="Candidate artifact path is unavailable")
    version_name = _sanitize_import_model_name(run_id)
    family_root = (resolve_model_root() / model_family).resolve()
    family_root.mkdir(parents=True, exist_ok=True)
    target_dir = (family_root / version_name).resolve()
    try:
        target_dir.relative_to(family_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid target model path") from exc

    if target_dir.exists():
        manifest = _load_model_json(target_dir, "production_manifest.json")
        if manifest.get("artifact_checksum") == candidate.get("artifact_actual_checksum"):
            validate_model_artifacts(model_family, target_dir)
            return f"{model_family}/{version_name}", target_dir
        raise HTTPException(status_code=409, detail=f"Model version already exists: {model_family}/{version_name}")

    staging_dir = family_root / f".staging-{version_name}-{uuid.uuid4().hex[:8]}"
    extract_dir = family_root / f".extract-{version_name}-{uuid.uuid4().hex[:8]}"
    try:
        staging_dir.mkdir(parents=False, exist_ok=False)
        extract_dir.mkdir(parents=False, exist_ok=False)
        with zipfile.ZipFile(artifact_path, "r") as zf:
            _validate_model_import_zip(zf)
            zf.extractall(extract_dir)

        if model_family == "tfidf_lr":
            file_aliases = {
                "model_lr.pkl": ("model_lr.pkl", "model_lr.joblib"),
                "vectorizer.pkl": ("vectorizer.pkl", "vectorizer.joblib"),
            }
            for target_name, source_names in file_aliases.items():
                source = next((path for source_name in source_names for path in extract_dir.rglob(source_name)), None)
                if source is None:
                    raise HTTPException(status_code=400, detail=f"Artifact missing {target_name}")
                shutil.copy2(source, staging_dir / target_name)
            constructiveness_source = next(
                (
                    path
                    for filename in ("model_constructiveness_lr.joblib", "model_lr_constructiveness.pkl")
                    for path in extract_dir.rglob(filename)
                ),
                None,
            )
            if constructiveness_source:
                shutil.copy2(constructiveness_source, staging_dir / "model_constructiveness_lr.joblib")
            for optional_name in ("training_evidence.json", "run_summary.json"):
                optional_source = next(iter(extract_dir.rglob(optional_name)), None)
                if optional_source:
                    shutil.copy2(optional_source, staging_dir / optional_name)
        else:
            model_source = _find_extracted_model_dir(extract_dir, model_family)
            for source in model_source.iterdir():
                destination = staging_dir / source.name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)

        metrics = comparison.get("candidate", {}).get("metrics") or {}
        (staging_dir / "metrics.json").write_text(
            json.dumps(_normalize_saved_model_metrics(metrics), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        production_manifest = {
            "model_family": model_family,
            "model_id": f"{model_family}/{version_name}",
            "source_run_id": run_id,
            "artifact_checksum": candidate.get("artifact_actual_checksum"),
            "bundle_checksum": candidate.get("bundle_checksum"),
            "included_mlflow_ids_sha256": candidate.get("included_mlflow_ids_sha256"),
            "feedback_snapshot_sha256": candidate.get("feedback_snapshot_sha256"),
            "test_fingerprint": candidate.get("test_fingerprint"),
            "test_size": candidate.get("test_size"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging_dir / "production_manifest.json").write_text(
            json.dumps(production_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_model_artifacts(model_family, staging_dir)
        _smoke_validate_staged_model(model_family, staging_dir)
        staging_dir.replace(target_dir)
    except HTTPException:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to install candidate artifact: {exc}") from exc
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
    return f"{model_family}/{version_name}", target_dir


@app.post("/api/mlflow/promote", dependencies=[Depends(require_admin)])
def mlflow_promote(request: MlflowPromoteRequest) -> Dict[str, Any]:
    run_id = str(request.run_id or request.candidate_model or "").strip()
    if not run_id:
        raise HTTPException(status_code=422, detail="run_id is required")
    comparison = _build_family_comparison(run_id)
    if not comparison.get("promotion_enabled"):
        failed = [str(check.get("name")) for check in comparison.get("gate_checks", []) if not check.get("passed")]
        raise HTTPException(status_code=409, detail=f"Promotion gate failed: {', '.join(failed)}")
    candidate = _load_kaggle_candidate(run_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"Kaggle run not found: {run_id}")
    actual_checksum = str(candidate.get("artifact_actual_checksum") or "")
    if request.artifact_checksum and not hmac.compare_digest(request.artifact_checksum.lower(), actual_checksum.lower()):
        raise HTTPException(status_code=409, detail="Artifact checksum changed since comparison")

    model_family = str(candidate["model_family"])
    current = comparison.get("current") or {}
    current_model_id = str(current.get("model") or "") or None
    if request.expected_current_version and request.expected_current_version != current_model_id:
        raise HTTPException(status_code=409, detail="Production model changed since comparison; refresh before promoting")

    model_id, installed_path = _install_candidate_artifact(candidate, comparison)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        slot = conn.execute(
            "SELECT active_model_id, active_run_id FROM mlflow_production_slot WHERE model_family = ?",
            (model_family,),
        ).fetchone()
        active_model_id = str(slot[0]) if slot and slot[0] else current_model_id
        active_run_id = str(slot[1]) if slot and slot[1] else current.get("run_id")
        if request.expected_current_version and active_model_id != request.expected_current_version:
            conn.rollback()
            raise HTTPException(status_code=409, detail="Production model changed during promotion; refresh and retry")
        conn.execute(
            "UPDATE mlflow_model_version SET status = 'archived' WHERE model_family = ? AND status = 'production'",
            (model_family,),
        )
        conn.execute(
            """
            INSERT INTO mlflow_model_version (
                model_family, model_id, source_run_id, artifact_path, artifact_checksum,
                bundle_checksum, test_fingerprint, metrics_json, status, created_at, promoted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'production', ?, ?)
            ON CONFLICT(source_run_id) DO UPDATE SET
                model_id = excluded.model_id,
                artifact_path = excluded.artifact_path,
                artifact_checksum = excluded.artifact_checksum,
                bundle_checksum = excluded.bundle_checksum,
                test_fingerprint = excluded.test_fingerprint,
                metrics_json = excluded.metrics_json,
                status = 'production',
                promoted_at = excluded.promoted_at
            """,
            (
                model_family,
                model_id,
                run_id,
                str(installed_path),
                actual_checksum,
                candidate.get("bundle_checksum"),
                candidate.get("test_fingerprint"),
                json.dumps(comparison.get("candidate", {}).get("metrics") or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO mlflow_production_slot (
                model_family, active_model_id, active_run_id, artifact_checksum,
                previous_model_id, previous_run_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_family) DO UPDATE SET
                active_model_id = excluded.active_model_id,
                active_run_id = excluded.active_run_id,
                artifact_checksum = excluded.artifact_checksum,
                previous_model_id = mlflow_production_slot.active_model_id,
                previous_run_id = mlflow_production_slot.active_run_id,
                updated_at = excluded.updated_at
            """,
            (model_family, model_id, run_id, actual_checksum, active_model_id, active_run_id, now),
        )
        conn.execute(
            """
            INSERT INTO mlflow_promotion_event (
                model_family, action, source_run_id, from_model_id, to_model_id,
                artifact_checksum, status, detail, created_at
            ) VALUES (?, 'promote', ?, ?, ?, ?, 'completed', ?, ?)
            """,
            (model_family, run_id, active_model_id, model_id, actual_checksum, "Artifact installed and production pointer updated", now),
        )
        conn.commit()
    return {
        "status": "promoted",
        "model_family": model_family,
        "candidate_model": model_id,
        "previous_model": current_model_id,
        "artifact_checksum": actual_checksum,
        "serving_reload": "next_request",
        "message": f"Promoted {model_id} to the {model_family} production slot.",
    }


def _automation_run_was_started(run_id: str) -> bool:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM mlflow_automation_event
            WHERE source_run_id = ? AND action = 'train_started'
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    return bool(row)


def _automation_finalize_run_state(model_family: str, run_id: str) -> None:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            UPDATE mlflow_automation_state
            SET active_run_id = NULL, updated_at = ?
            WHERE model_family = ? AND active_run_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), model_family, run_id),
        )
        conn.commit()


def _automation_handle_terminal_run(run_id: str) -> None:
    """Advance an automation-created run after its terminal Kaggle status is known."""
    if not _automation_run_was_started(run_id):
        return
    candidate = _load_kaggle_candidate(run_id)
    if not candidate:
        return
    row = candidate["row"]
    model_family = str(candidate["model_family"])
    run_status = str(row["status"] or "").lower()
    policy = _automation_policy(model_family)
    if run_status != "completed":
        _automation_record_event(
            model_family,
            "train_terminal",
            run_status or "unknown",
            source_run_id=run_id,
            detail="Automation run reached a non-completed terminal status.",
        )
        _automation_finalize_run_state(model_family, run_id)
        return

    if not candidate.get("artifact_path"):
        return

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        existing = conn.execute(
            """
            SELECT 1
            FROM mlflow_automation_event
            WHERE source_run_id = ? AND action IN ('candidate_ready', 'auto_promote')
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if existing:
        _automation_finalize_run_state(model_family, run_id)
        return

    if not policy["enabled"] or policy["mode"] != "full_auto":
        _automation_record_event(
            model_family,
            "candidate_ready",
            "awaiting_approval",
            source_run_id=run_id,
            detail=f"Automation mode is {policy['mode']}.",
        )
        _automation_finalize_run_state(model_family, run_id)
        return

    try:
        comparison = _build_family_comparison(run_id)
        failed = [str(check.get("name")) for check in comparison.get("gate_checks", []) if not check.get("passed")]
        if not comparison.get("promotion_enabled"):
            _automation_record_event(
                model_family,
                "auto_promote",
                "rejected",
                source_run_id=run_id,
                detail=f"Promotion gate failed: {', '.join(failed)}",
            )
            return
        result = mlflow_promote(
            MlflowPromoteRequest(
                run_id=run_id,
                artifact_checksum=str(candidate.get("artifact_actual_checksum") or "") or None,
                expected_current_version=str((comparison.get("current") or {}).get("model") or "") or None,
            )
        )
        _automation_record_event(
            model_family,
            "auto_promote",
            "promoted",
            source_run_id=run_id,
            detail=str(result.get("message") or "Production slot updated."),
        )
    except Exception as exc:
        _automation_record_event(
            model_family,
            "auto_promote",
            "failed",
            source_run_id=run_id,
            detail=str(exc),
        )
    finally:
        _automation_finalize_run_state(model_family, run_id)


@app.get("/api/mlflow/automation/status", dependencies=[Depends(require_admin)])
def mlflow_automation_status() -> Dict[str, Any]:
    init_feedback_db()
    families = ("tfidf_lr", "phobert")
    snapshots = [_automation_state_snapshot(family) for family in families]
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        events = conn.execute(
            """
            SELECT model_family, action, source_run_id, status, eligible_count, detail, created_at
            FROM mlflow_automation_event
            ORDER BY id DESC
            LIMIT 30
            """
        ).fetchall()
    return {"families": snapshots, "events": [dict(event) for event in events]}


@app.post("/api/mlflow/automation/cycle", dependencies=[Depends(require_admin)])
def mlflow_automation_cycle(request: MlflowAutomationCycleRequest) -> Dict[str, Any]:
    families = (request.model_family,) if request.model_family else ("tfidf_lr", "phobert")
    results: List[Dict[str, Any]] = []
    for model_family in families:
        try:
            results.append(_run_automation_cycle(model_family, "admin_requested_cycle"))
        except HTTPException as exc:
            results.append({"model_family": model_family, "started": False, "error": exc.detail})
        except Exception as exc:
            results.append({"model_family": model_family, "started": False, "error": str(exc)})
    return {"results": results}


@app.post("/api/mlflow/rollback", dependencies=[Depends(require_admin)])
def mlflow_rollback(request: MlflowRollbackRequest) -> Dict[str, Any]:
    init_feedback_db()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        slot = conn.execute(
            """
            SELECT active_model_id, active_run_id, artifact_checksum, previous_model_id, previous_run_id
            FROM mlflow_production_slot
            WHERE model_family = ?
            """,
            (request.model_family,),
        ).fetchone()
        if not slot or not slot[3]:
            conn.rollback()
            raise HTTPException(status_code=409, detail=f"No rollback target for {request.model_family}")
        active_model_id, active_run_id, _, previous_model_id, previous_run_id = slot
        if request.expected_current_version and request.expected_current_version != active_model_id:
            conn.rollback()
            raise HTTPException(status_code=409, detail="Production model changed; refresh before rollback")
        try:
            resolved_type, resolved_name, _ = resolve_model_path(resolve_model_root(), str(previous_model_id))
        except (FileNotFoundError, OSError, ValueError) as exc:
            conn.rollback()
            raise HTTPException(status_code=409, detail=f"Rollback target is unavailable: {exc}") from exc
        if resolved_type != request.model_family:
            conn.rollback()
            raise HTTPException(status_code=409, detail="Rollback target belongs to a different model family")
        resolved_previous_id = f"{resolved_type}/{resolved_name}"
        previous_version = conn.execute(
            "SELECT artifact_checksum FROM mlflow_model_version WHERE model_id = ?",
            (resolved_previous_id,),
        ).fetchone()
        previous_checksum = str(previous_version[0]) if previous_version and previous_version[0] else None
        conn.execute("UPDATE mlflow_model_version SET status = 'archived' WHERE model_id = ?", (active_model_id,))
        conn.execute("UPDATE mlflow_model_version SET status = 'production' WHERE model_id = ?", (resolved_previous_id,))
        conn.execute(
            """
            UPDATE mlflow_production_slot
            SET active_model_id = ?, active_run_id = ?, artifact_checksum = ?,
                previous_model_id = ?, previous_run_id = ?, updated_at = ?
            WHERE model_family = ?
            """,
            (resolved_previous_id, previous_run_id, previous_checksum, active_model_id, active_run_id, now, request.model_family),
        )
        conn.execute(
            """
            INSERT INTO mlflow_promotion_event (
                model_family, action, source_run_id, from_model_id, to_model_id,
                artifact_checksum, status, detail, created_at
            ) VALUES (?, 'rollback', ?, ?, ?, ?, 'completed', ?, ?)
            """,
            (request.model_family, previous_run_id, active_model_id, resolved_previous_id, previous_checksum, "Production pointer rolled back", now),
        )
        conn.commit()
    return {
        "status": "rolled_back",
        "model_family": request.model_family,
        "active_model": resolved_previous_id,
        "previous_model": active_model_id,
        "serving_reload": "next_request",
        "message": f"Rolled back {request.model_family} production to {resolved_previous_id}.",
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    try:
        cleanup_old_jobs(float(os.getenv("JOB_RETENTION_HOURS", "24")))

        options = request.options or AnalyzeOptions()

        urls = normalize_input_urls(request.urls)
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")
        mlflow_accept_threshold = float(options.mlflow_gate_accept_threshold)
        mlflow_discard_threshold = float(options.mlflow_gate_discard_threshold)
        if mlflow_discard_threshold > mlflow_accept_threshold:
            raise HTTPException(status_code=400, detail="mlflow_gate_discard_threshold must be <= mlflow_gate_accept_threshold")

        job_id = uuid.uuid4().hex
        out_dir = PROCESSED_DATA_DIR / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        analysis_started_at = time.perf_counter()

        def log_analysis_stage(stage: str, **fields: Any) -> None:
            detail = " ".join(f"{key}={value}" for key, value in fields.items())
            logger.info(
                "[analysis:%s] %s elapsed_ms=%d%s",
                job_id,
                stage,
                int((time.perf_counter() - analysis_started_at) * 1000),
                f" {detail}" if detail else "",
            )

        model_root = resolve_model_root()
        try:
            if options.model_path:
                requested_model_path = Path(options.model_path).expanduser().resolve()
                model_root_resolved = model_root.resolve()
                try:
                    requested_model_path.relative_to(model_root_resolved)
                except ValueError as exc:
                    raise ValueError(
                        f"Model path must be under {model_root_resolved}: {requested_model_path}"
                    ) from exc
                relative_parts = requested_model_path.relative_to(model_root_resolved).parts
                if len(relative_parts) < 2:
                    raise ValueError(f"Model path must point to a model directory under {model_root_resolved}")
                model_type = relative_parts[0]
                model_name = requested_model_path.name
                _, _, model_path = resolve_model_path(model_root, f"{model_type}/{model_name}")
                model_id = f"{model_type}/{model_name}"
            else:
                model_type, model_name, model_path = resolve_model_path(model_root, options.model_name)
                model_id = f"{model_type}/{model_name}"
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (PermissionError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to access model directory: {exc}") from exc

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=job_id,
                urls=urls,
                url_hashes=[],
                model_ids=[model_id],
                enable_video=False,
                merged_used=False,
            ),
        )

        model_root = resolve_model_root()
        try:
            model_type, model_name, model_path = resolve_model_path(model_root, model_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (PermissionError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to access model directory: {exc}") from exc

        thresholds_by_domain = get_effective_thresholds(model_id)

        logger.info("Job %s: start analyze for %s urls", job_id, len(urls))
        logger.info("Job %s: using model '%s' (%s) from %s", job_id, model_id, model_type, model_path)
        log_analysis_stage("crawl:start", urls=len(urls))

        crawl_results = crawl_urls(
            urls,
            out_dir=str(DATA_DIR),
            timeout=options.crawl_timeout_sec,
            max_load_more=options.max_load_more_clicks,
            max_comments_per_url=options.max_comments_per_url,
        )
        log_analysis_stage(
            "crawl:done",
            ok_urls=sum(1 for result in crawl_results if result.get("status") == "ok"),
            segments=sum(int(result.get("num_segments") or 0) for result in crawl_results),
        )

        for r in crawl_results:
            logger.info(
                "Job %s: crawl result url=%s status=%s method=%s segments_path=%s error=%s",
                job_id,
                r.get("url"),
                r.get("status"),
                r.get("method"),
                r.get("segments_path"),
                r.get("error"),
            )

        ok_hashes = [r["url_hash"] for r in crawl_results if r.get("status") == "ok"]

        infer_data_dir = DATA_DIR
        merged_used = False

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=job_id,
                urls=urls,
                url_hashes=ok_hashes,
                model_ids=[model_id],
                enable_video=False,
                merged_used=merged_used,
            ),
        )

        if ok_hashes:
            logger.info("Job %s: running inference on %s crawled urls", job_id, len(ok_hashes))
            log_analysis_stage("inference:start", model=model_id, urls=len(ok_hashes))
            infer_crawled(
                model_path=str(model_path),
                model_type=model_type,
                data_dir=str(infer_data_dir),
                out_dir=str(out_dir),
                batch_size=options.batch_size,
                max_length=options.max_length,
                page_threshold=options.page_threshold,
                seg_threshold=options.seg_threshold,
                threshold_news=thresholds_by_domain.get("news"),
                threshold_social=thresholds_by_domain.get("social"),
                threshold_forum=thresholds_by_domain.get("forum"),
                threshold_unknown=thresholds_by_domain.get("unknown"),
                only_url_hashes=ok_hashes,
                quiet=True,
                learned_feedback=load_learned_segments(),
                html_dir=str(DATA_DIR),
            )
            log_analysis_stage("inference:done", model=model_id)
        else:
            logger.warning("Job %s: no successful crawls to run inference", job_id)

        segment_results = load_segment_results(out_dir)

        page_by_hash, page_by_url = load_page_results_map(out_dir)

        seg_by_hash: Dict[str, List[Dict[str, Any]]] = {}
        seg_by_url: Dict[str, List[Dict[str, Any]]] = {}
        for seg in segment_results:
            if seg.get("url_hash"):
                seg_by_hash.setdefault(seg["url_hash"], []).append(seg)
            if seg.get("url"):
                seg_by_url.setdefault(seg["url"], []).append(seg)

        response_results = map_results_to_response(
            crawl_results,
            page_by_hash,
            page_by_url,
            seg_by_hash,
            seg_by_url,
        )
        log_analysis_stage("aggregate:done", results=len(response_results))

        mlflow_collection: Dict[str, Any] = {
            "enabled": bool(options.collect_for_mlflow),
            "batch_id": None,
            "candidate_rows": 0,
            "inserted": 0,
            "samples_inserted": 0,
            "samples_reused": 0,
            "predictions_inserted": 0,
            "skipped_existing_url": 0,
            "skipped_duplicate_item": 0,
            "counts": {"accepted": 0, "candidate": 0, "discarded": 0, "total": 0},
        }
        if options.collect_for_mlflow:
            log_analysis_stage("persistence:start")
            collection_created_at = datetime.utcnow().isoformat() + "Z"
            collection_batch_id = f"mlf_auto_{uuid.uuid4().hex[:12]}"
            mlflow_options_json = json.dumps(
                {
                    "source": "user_analyze",
                    "batch_size": options.batch_size,
                    "max_length": options.max_length,
                    "page_threshold": options.page_threshold,
                    "seg_threshold": options.seg_threshold,
                    "gate_accept_threshold": mlflow_accept_threshold,
                    "gate_discard_threshold": mlflow_discard_threshold,
                },
                ensure_ascii=False,
            )
            mlflow_collection = insert_mlflow_comment_rows(
                batch_id=collection_batch_id,
                model_id=model_id,
                source_job_id=job_id,
                rows=build_mlflow_comment_rows(
                    response_results,
                    collection_batch_id,
                    job_id,
                    mlflow_accept_threshold,
                    mlflow_discard_threshold,
                    collection_created_at,
                ),
                options_json=mlflow_options_json,
                created_at=collection_created_at,
                batch_created=False,
            )
            log_analysis_stage("persistence:done", inserted=mlflow_collection.get("inserted", 0))

        logger.info("Job %s: completed", job_id)
        log_analysis_stage("response:done")
        production_manifest = _load_model_json(model_path, "production_manifest.json")
        family_production_model = get_family_default_model_id(model_root, model_type)
        return {
            "job_id": job_id,
            "flow_state": "completed",
            "model_name": model_id,
            "serving_evidence": {
                "model_family": model_type,
                "model_version": model_id,
                "production_slot": model_type if family_production_model == model_id else None,
                "artifact_checksum": production_manifest.get("artifact_checksum"),
                "source_run_id": production_manifest.get("source_run_id"),
            },
            "thresholds": {
                "seg_threshold": options.seg_threshold,
                "page_threshold": options.page_threshold,
            },
            "thresholds_by_domain": thresholds_by_domain,
            "mlflow_collection": mlflow_collection,
            "results": response_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}")


@app.get("/api/models")
def get_models() -> Dict[str, Any]:
    try:
        model_root = resolve_model_root()
        models = list_all_models(model_root)
        model_ids = [m["id"] for m in models]
        production_slots = {
            family: get_family_default_model_id(model_root, family)
            for family in ("tfidf_lr", "phobert")
        }

        def describe_model(model_id: Optional[str]) -> Dict[str, Any]:
            detail: Dict[str, Any] = {
                "model_id": model_id,
                "family": None,
                "version": None,
                "artifact_path": None,
                "artifact_available": False,
                "base_model": None,
            }
            if not model_id:
                return detail
            try:
                model_type, model_name, model_dir = resolve_model_path(model_root, model_id)
            except (FileNotFoundError, OSError, ValueError):
                return detail
            run_config = _load_model_json(model_dir, "run_config.json")
            detail.update({
                "model_id": f"{model_type}/{model_name}",
                "family": model_type,
                "version": run_config.get("model_version") or run_config.get("model_name") or f"{model_type}/{model_name}",
                "artifact_path": to_relative(str(model_dir)),
                "artifact_available": True,
                "base_model": get_phobert_base_model(model_dir) if model_type == "phobert" else None,
            })
            return detail

        model_details: Dict[str, Dict[str, Any]] = {}
        for model in models:
            model_id = str(model["id"])
            model_details[model_id] = describe_model(model_id)

        configured_slots: Dict[str, Dict[str, Any]] = {}
        for family in ("tfidf_lr", "phobert"):
            state = _read_production_slot_state(family)
            configured = describe_model(state["active_model_id"])
            previous = describe_model(state["previous_model_id"])
            configured_slots[family] = {
                "configured": configured,
                "previous": previous,
                "updated_at": state["updated_at"],
                "resolved": describe_model(production_slots[family]),
            }

        default_model_id = get_default_model_id(model_root)
        runtime_default = describe_model(default_model_id)
        phobert_configured = configured_slots["phobert"]["configured"]
        if phobert_configured["model_id"] == runtime_default["model_id"]:
            runtime_default["resolution_source"] = "phobert_production_slot"
            runtime_default["fallback_reason"] = None
        else:
            runtime_default["resolution_source"] = "phobert_fallback"
            runtime_default["fallback_reason"] = (
                "No configured PhoBERT production slot; selected the first compatible PhoBERT fallback."
                if not phobert_configured["model_id"]
                else "Configured PhoBERT production slot was not compatible; selected a compatible PhoBERT fallback."
            )
        return {
            "models": model_ids,
            "default": default_model_id,
            "production_slots": production_slots,
            "model_details": model_details,
            "configured_production_slots": configured_slots,
            "runtime_default": runtime_default,
            "labels": {
                model_id: MODEL_DISPLAY_NAMES.get(model_id, model_id)
                for model_id in model_ids
            },
        }
    except (PermissionError, OSError, NotADirectoryError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {exc}") from exc


@app.post("/api/models/import-zip", dependencies=[Depends(require_admin)])
async def import_model_zip(request: Request) -> Dict[str, Any]:
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model ZIP import requires python-multipart. Install dependency: pip install python-multipart",
        ) from exc

    model_name_raw = form.get("model_name")
    if not isinstance(model_name_raw, str) or not model_name_raw.strip():
        raise HTTPException(status_code=400, detail="model_name is required")

    model_zip = form.get("model_zip")
    if model_zip is None or not hasattr(model_zip, "filename") or not hasattr(model_zip, "read"):
        raise HTTPException(status_code=400, detail="model_zip is required")

    sanitized_name = _sanitize_import_model_name(model_name_raw)
    filename = str(getattr(model_zip, "filename", "") or "").lower()
    if not filename.endswith(".zip"):
        raise HTTPException(status_code=415, detail="model_zip must be a .zip file")

    model_root = resolve_model_root()
    target_dir = model_root / "phobert" / sanitized_name
    if target_dir.exists():
        raise HTTPException(status_code=409, detail=f"Model already exists: phobert/{sanitized_name}")

    with tempfile.TemporaryDirectory(prefix="model_import_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        zip_path = tmp_dir / "uploaded_model.zip"
        extract_root = tmp_dir / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        with zip_path.open("wb") as out:
            while True:
                chunk = await model_zip.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MODEL_IMPORT_MAX_ZIP_BYTES:
                    raise HTTPException(status_code=413, detail="ZIP exceeds size limit")
                out.write(chunk)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _validate_model_import_zip(zf)
                safe_root = extract_root.resolve()
                for info in zf.infolist():
                    member_target = (extract_root / info.filename).resolve()
                    try:
                        member_target.relative_to(safe_root)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail="ZIP contains unsafe path") from exc

                    if info.is_dir():
                        member_target.mkdir(parents=True, exist_ok=True)
                        continue

                    member_target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, "r") as src, member_target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Invalid ZIP file") from exc

        imported_model_dir = _find_imported_model_dir(extract_root)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(imported_model_dir, target_dir)

    try:
        validate_model_artifacts("phobert", target_dir)
    except Exception as exc:
        try:
            shutil.rmtree(target_dir)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Imported model is invalid: {exc}") from exc
    finally:
        try:
            close_result = model_zip.close()
            if hasattr(close_result, "__await__"):
                await close_result
        except Exception:
            pass

    return {
        "status": "imported",
        "model_id": f"phobert/{sanitized_name}",
        "model_name": sanitized_name,
        "model_type": "phobert",
        "model_path": to_relative(str(target_dir)) or str(target_dir),
        "validated": True,
    }


@app.get("/api/training-tracker")
def get_training_tracker() -> Dict[str, Any]:
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/phases")
def api_create_training_phase(request: TrainingTrackerCreatePhaseRequest) -> Dict[str, Any]:
    create_training_phase(request.title)
    return fetch_training_tracker_payload()


@app.patch("/api/training-tracker/phases/{phase_id}")
def api_update_training_phase(phase_id: str, request: TrainingTrackerUpdatePhaseRequest) -> Dict[str, Any]:
    updated = update_training_phase_title(phase_id, request.title)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Phase not found")
    return fetch_training_tracker_payload()


@app.delete("/api/training-tracker/phases/{phase_id}")
def api_delete_training_phase(phase_id: str) -> Dict[str, Any]:
    deleted = delete_training_phase(phase_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Phase not found")
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/phases/reorder")
def api_reorder_training_phases(request: TrainingTrackerReorderPhasesRequest) -> Dict[str, Any]:
    reorder_training_phases(request.phase_ids)
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/groups")
def api_create_training_group(request: TrainingTrackerCreateGroupRequest) -> Dict[str, Any]:
    create_training_group(request.phase_id, request.title)
    return fetch_training_tracker_payload()


@app.patch("/api/training-tracker/groups/{group_id}")
def api_update_training_group(group_id: str, request: TrainingTrackerUpdateGroupRequest) -> Dict[str, Any]:
    updated = update_training_group_title(group_id, request.title)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Group not found")
    return fetch_training_tracker_payload()


@app.delete("/api/training-tracker/groups/{group_id}")
def api_delete_training_group(group_id: str) -> Dict[str, Any]:
    deleted = delete_training_group(group_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Group not found")
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/groups/reorder")
def api_reorder_training_groups(request: TrainingTrackerReorderGroupsRequest) -> Dict[str, Any]:
    reorder_training_groups(request.phase_id, request.group_ids)
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/tasks")
def api_create_training_task(request: TrainingTrackerCreateTaskRequest) -> Dict[str, Any]:
    create_training_task(request.phase_id, request.group_id, request.label, request.param)
    return fetch_training_tracker_payload()


@app.patch("/api/training-tracker/tasks/{task_id}")
def api_update_training_task(task_id: str, request: TrainingTrackerUpdateTaskRequest) -> Dict[str, Any]:
    updated = update_training_task(task_id, request.label, request.param)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/tasks/reorder")
def api_reorder_training_tasks(request: TrainingTrackerReorderTasksRequest) -> Dict[str, Any]:
    reorder_training_tasks(request.phase_id, request.group_id, request.task_ids)
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/tasks/{task_id}/check")
def api_check_training_task(task_id: str, request: TrainingTrackerTaskCheckRequest) -> Dict[str, Any]:
    updated = set_training_task_checked(task_id, request.checked)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return fetch_training_tracker_payload()


@app.delete("/api/training-tracker/tasks/{task_id}")
def api_delete_training_task(task_id: str) -> Dict[str, Any]:
    deleted = delete_training_task(task_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return fetch_training_tracker_payload()


@app.post("/api/training-tracker/results")
def api_create_training_result(request: TrainingTrackerCreateResultRequest) -> Dict[str, Any]:
    create_training_result(request)
    return fetch_training_tracker_payload()


@app.delete("/api/training-tracker/results/{result_id}")
def api_delete_training_result(result_id: str) -> Dict[str, Any]:
    deleted = delete_training_result(result_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Result not found")
    return fetch_training_tracker_payload()


@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for item in request.items:
        normalized = safe_label(item.label)
        if normalized is None:
            raise HTTPException(status_code=400, detail=f"Invalid label: {item.label}")
        items.append(
            {
                "job_id": request.job_id,
                "url": item.url,
                "url_hash": item.url_hash,
                "model_id": request.model_id,
                "html_tag": item.html_tag,
                "html_tag_override": item.html_tag_override,
                "seg_threshold_used": item.seg_threshold_used,
                "score_overall": item.score_overall,
                "label": normalized,
            }
        )

    inserted = insert_feedback_page(items)
    return {"inserted": inserted}


@app.post("/api/analyze_compare")
def analyze_compare(request: AnalyzeCompareRequest) -> Dict[str, Any]:
    try:
        options = request.options

        urls = normalize_input_urls(request.urls)
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")
        model_ids = [m.strip() for m in options.model_names if m and m.strip()]
        if len(model_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 model_names")
        job_id = uuid.uuid4().hex
        out_dir = PROCESSED_DATA_DIR / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_root = resolve_model_root()
        model_infos: List[Dict[str, Any]] = []
        for model_id in model_ids:
            try:
                model_type, model_name, model_path = resolve_model_path(model_root, model_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except (PermissionError, OSError) as exc:
                raise HTTPException(status_code=500, detail=f"Unable to access model directory: {exc}") from exc
            model_infos.append(
                {
                    "model_id": model_id,
                    "model_type": model_type,
                    "model_name": model_name,
                    "model_path": model_path,
                }
            )

        logger.info("Compare job %s: start analyze for %s urls", job_id, len(urls))

        crawl_results = crawl_urls(
            urls,
            out_dir=str(DATA_DIR),
            timeout=options.crawl_timeout_sec,
            max_load_more=options.max_load_more_clicks,
            max_comments_per_url=options.max_comments_per_url,
        )

        ok_hashes = [r["url_hash"] for r in crawl_results if r.get("status") == "ok"]
        infer_data_dir = DATA_DIR
        merged_used = False

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=job_id,
                urls=urls,
                url_hashes=ok_hashes,
                model_ids=model_ids,
                enable_video=False,
                merged_used=merged_used,
            ),
        )

        compare_results: Dict[str, Any] = {}
        for info in model_infos:
            model_id = info["model_id"]
            model_type = info["model_type"]
            model_path = info["model_path"]
            thresholds_by_domain = get_effective_thresholds(model_id)

            model_out_dir = out_dir / "models" / model_id.replace("/", "-")
            model_out_dir.mkdir(parents=True, exist_ok=True)

            if ok_hashes:
                logger.info(
                    "Compare job %s: running inference for model %s on %s urls",
                    job_id,
                    model_id,
                    len(ok_hashes),
                )
                infer_crawled(
                    model_path=str(model_path),
                    model_type=model_type,
                    data_dir=str(infer_data_dir),
                    out_dir=str(model_out_dir),
                    batch_size=options.batch_size,
                    max_length=options.max_length,
                    page_threshold=options.page_threshold,
                    seg_threshold=options.seg_threshold,
                    threshold_news=thresholds_by_domain.get("news"),
                    threshold_social=thresholds_by_domain.get("social"),
                    threshold_forum=thresholds_by_domain.get("forum"),
                    threshold_unknown=thresholds_by_domain.get("unknown"),
                    only_url_hashes=ok_hashes,
                    quiet=True,
                    learned_feedback=load_learned_segments(),
                    html_dir=str(DATA_DIR),
                )
            else:
                logger.warning("Compare job %s: no successful crawls to run inference", job_id)

            page_by_hash, page_by_url = load_page_results_map(model_out_dir)
            segment_results = load_segment_results(model_out_dir)
            seg_by_hash: Dict[str, List[Dict[str, Any]]] = {}
            seg_by_url: Dict[str, List[Dict[str, Any]]] = {}
            for seg in segment_results:
                if seg.get("url_hash"):
                    seg_by_hash.setdefault(seg["url_hash"], []).append(seg)
                if seg.get("url"):
                    seg_by_url.setdefault(seg["url"], []).append(seg)

            response_results = map_results_to_response(
                crawl_results,
                page_by_hash,
                page_by_url,
                seg_by_hash,
                seg_by_url,
            )

            compare_results[model_id] = {
                "model_name": model_id,
                "thresholds": {
                    "seg_threshold": options.seg_threshold,
                    "page_threshold": options.page_threshold,
                },
                "thresholds_by_domain": thresholds_by_domain,
                "results": response_results,
            }

        return {
            "job_id": job_id,
            "flow_state": "completed",
            "models": compare_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analyze compare failed")
        raise HTTPException(status_code=500, detail=f"Analyze compare failed: {exc}")


@app.post("/api/analyze/rerun")
def analyze_rerun(request: AnalyzeRerunRequest) -> Dict[str, Any]:
    try:
        cleanup_old_jobs(float(os.getenv("JOB_RETENTION_HOURS", "24")))
        options = request.options or AnalyzeOptions()
        job_id = request.job_id.strip()
        if not job_id:
            raise HTTPException(status_code=400, detail="Missing job_id")

        source_dir = PROCESSED_DATA_DIR / f"job_{job_id}"
        if not source_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        compare_root = source_dir / "models"
        is_compare = compare_root.exists() and any(compare_root.iterdir())
        if is_compare and not request.model_name:
            raise HTTPException(status_code=400, detail="Missing model_name for compare rerun")

        model_root = resolve_model_root()
        try:
            if request.model_name:
                model_type, model_name, model_path = resolve_model_path(model_root, request.model_name)
                model_id = f"{model_type}/{model_name}"
            else:
                model_id = get_default_model_id(model_root)
                model_type, model_name, model_path = resolve_model_path(model_root, model_id)
                model_id = f"{model_type}/{model_name}"
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (PermissionError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to access model directory: {exc}") from exc

        source_results_dir = source_dir
        if is_compare and request.model_name:
            source_results_dir = compare_root / request.model_name.replace("/", "-")

        source_pages = load_page_results(source_results_dir)
        if not source_pages:
            raise HTTPException(status_code=404, detail="No page results found for source job")

        url_entries: List[Dict[str, Any]] = []
        url_hashes: List[str] = []
        for row in source_pages:
            url = row.get("url")
            url_hash = row.get("url_hash") or (hash_url(url) if url else None)
            if not url or not url_hash:
                continue
            url_entries.append({"url": url, "url_hash": url_hash})
            url_hashes.append(url_hash)

        if not url_entries:
            raise HTTPException(status_code=404, detail="No URLs found for source job")

        infer_data_dir = DATA_DIR
        merged_root = source_dir / "merged_crawl"
        if request.prefer_merged and merged_root.exists():
            for entry in url_entries:
                seg_path = merged_root / entry["url_hash"] / "segments.jsonl"
                if seg_path.exists():
                    infer_data_dir = merged_root
                    break

        filtered_entries: List[Dict[str, Any]] = []
        for entry in url_entries:
            seg_path = infer_data_dir / entry["url_hash"] / "segments.jsonl"
            if seg_path.exists():
                entry["segments_path"] = str(seg_path)
                entry["status"] = "ok"
                entry["output_dir"] = str(infer_data_dir / entry["url_hash"])
            else:
                entry["segments_path"] = str(seg_path)
                entry["status"] = "error"
                entry["error"] = "segments.jsonl not found"
            filtered_entries.append(entry)

        ok_hashes = [e["url_hash"] for e in filtered_entries if e.get("status") == "ok"]
        if not ok_hashes:
            raise HTTPException(status_code=404, detail="No segments.jsonl found for this job")

        thresholds_by_domain = get_effective_thresholds(model_id)
        rerun_job_id = uuid.uuid4().hex
        out_dir = PROCESSED_DATA_DIR / f"job_{rerun_job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=rerun_job_id,
                urls=[e["url"] for e in filtered_entries],
                url_hashes=ok_hashes,
                model_ids=[model_id],
                enable_video=False,
                merged_used=bool(infer_data_dir == merged_root),
            ),
        )

        infer_crawled(
            model_path=str(model_path),
            model_type=model_type,
            data_dir=str(infer_data_dir),
            out_dir=str(out_dir),
            batch_size=options.batch_size,
            max_length=options.max_length,
            page_threshold=options.page_threshold,
            seg_threshold=options.seg_threshold,
            threshold_news=thresholds_by_domain.get("news"),
            threshold_social=thresholds_by_domain.get("social"),
            threshold_forum=thresholds_by_domain.get("forum"),
            threshold_unknown=thresholds_by_domain.get("unknown"),
            only_url_hashes=ok_hashes,
            quiet=True,
            learned_feedback=load_learned_segments(model_id),
            html_dir=str(DATA_DIR),
        )

        page_by_hash, page_by_url = load_page_results_map(out_dir)
        segment_results = load_segment_results(out_dir)
        seg_by_hash: Dict[str, List[Dict[str, Any]]] = {}
        seg_by_url: Dict[str, List[Dict[str, Any]]] = {}
        for seg in segment_results:
            if seg.get("url_hash"):
                seg_by_hash.setdefault(seg["url_hash"], []).append(seg)
            if seg.get("url"):
                seg_by_url.setdefault(seg["url"], []).append(seg)

        response_results = map_results_to_response(
            filtered_entries,
            page_by_hash,
            page_by_url,
            seg_by_hash,
            seg_by_url,
        )

        return {
            "job_id": rerun_job_id,
            "source_job_id": job_id,
            "model_name": model_id,
            "thresholds": {
                "seg_threshold": options.seg_threshold,
                "page_threshold": options.page_threshold,
            },
            "thresholds_by_domain": thresholds_by_domain,
            "results": response_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analyze rerun failed")
        raise HTTPException(status_code=500, detail=f"Analyze rerun failed: {exc}")


@app.post("/api/ask-ai")
def ask_ai(request: AskAIRequest) -> Dict[str, Any]:
    segments = request.segments[:5]
    prompt_lines = [
        "Bạn là chuyên gia an toàn thông tin. Hãy giải thích ngắn gọn mức độ rủi ro nội dung.",
        f"URL: {request.url}",
    ]
    if request.html_tag:
        prompt_lines.append(f"HTML tag: {request.html_tag}")
    if request.overall is not None:
        prompt_lines.append(f"Điểm độc hại tổng thể (0-1): {request.overall:.3f}")
    if request.thresholds:
        prompt_lines.append(f"Ngưỡng đang dùng: {json.dumps(request.thresholds, ensure_ascii=False)}")

    if segments:
        prompt_lines.append("Các đoạn rủi ro cao nhất:")
        for idx, seg in enumerate(segments, start=1):
            text = seg.get("text") or seg.get("text_preview") or ""
            score = seg.get("score")
            prompt_lines.append(f"{idx}. ({score}) {text}")

    if request.question:
        prompt_lines.append(f"Yêu cầu người dùng: {request.question}")

    prompt = "\n".join(prompt_lines)
    answer = call_gemini(prompt)
    return {"answer": answer}


@app.get("/api/gemini/models")
def gemini_models() -> Dict[str, Any]:
    return list_gemini_models()




@app.post("/api/feedback/segment")
def submit_segment_feedback(request: SegmentFeedbackRequest) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for item in request.items:
        normalized = safe_label(item.label)
        if normalized is None:
            raise HTTPException(status_code=400, detail=f"Invalid label: {item.label}")
        items.append(
            {
                "job_id": request.job_id,
                "url": item.url,
                "url_hash": item.url_hash,
                "model_id": item.model_id,
                "html_tag": item.html_tag,
                "html_tag_override": item.html_tag_override,
                "segment_id": item.segment_id,
                "text": item.text,
                "score": item.score,
                "seg_threshold_used": item.seg_threshold_used,
                "label": normalized,
                "segment_hash": build_segment_hash(item.text, item.html_tag_override or item.html_tag),
                "context_segment_hash": item.context_segment_hash,
            }
        )

    inserted = insert_feedback_segment(items)
    return {"inserted": inserted}


@app.post("/api/feedback/segment/delete")
def delete_segment_feedback(request: FeedbackDeleteRequest) -> Dict[str, Any]:
    ids = [int(v) for v in request.ids if isinstance(v, (int, float)) or (isinstance(v, str) and str(v).isdigit())]
    if not ids:
        raise HTTPException(status_code=400, detail="No valid feedback ids provided")
    deleted = delete_feedback_rows(ids)
    return {"deleted": deleted}




@app.get("/api/dataset/preview")
def dataset_preview(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    source: Optional[str] = None,
    label: Optional[int] = Query(default=None, ge=0, le=1),
    split: Optional[str] = None,
    include_stats: bool = False,
    dataset_version: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_dataset_version = normalize_dataset_version(dataset_version)
    include_feedback = (split or "").strip().lower() == "feedback" or (source or "").strip().lower() == "new_collected"
    rows = iter_dataset_rows(resolved_dataset_version) + (iter_feedback_rows() if include_feedback else [])
    sources = [source] if source else None
    labels = [label] if label is not None else None
    splits = [split] if split else None
    filtered = filter_dataset_rows(rows, sources=sources, labels=labels, splits=splits)
    total = len(filtered)
    total_pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]
    payload: Dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "items": items,
        "dataset_version": resolved_dataset_version,
    }
    if include_stats:
        payload["stats"] = build_dataset_stats(filtered)
    return payload


@app.post("/api/dataset/export")
def dataset_export(request: DatasetExportRequest) -> Dict[str, Any]:
    resolved_dataset_version = normalize_dataset_version(request.dataset_version)
    rows = iter_dataset_rows(resolved_dataset_version) + iter_feedback_rows()
    filtered = filter_dataset_rows(
        rows,
        sources=request.source,
        labels=request.label,
        splits=request.split,
    )

    model_version = request.model_version or DEFAULT_MODEL_VERSION
    policy_version = request.policy_version or DEFAULT_POLICY_VERSION

    versions = build_artifact_versions(
        dataset_version=resolved_dataset_version,
        model_version=model_version,
        policy_version=policy_version,
    )
    missing_versions = find_missing_required_versions(versions)
    if missing_versions:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required version metadata",
                "missing": missing_versions,
                "required": list(REQUIRED_VERSION_KEYS),
            },
        )

    out_dir = PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = utc_timestamp_compact()
    short_id = uuid.uuid4().hex[:8]
    dataset_token = slugify_token(versions["dataset_version"])
    out_path = out_dir / f"combined_dataset_{dataset_token}_{timestamp}_{short_id}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = build_dataset_stats(filtered)
    manifest = {
        "artifact_type": "dataset_export",
        "artifact_path": to_relative(str(out_path)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "filters": {
            "source": request.source or [],
            "label": request.label or [],
            "split": request.split or [],
        },
        "record_count": len(filtered),
        **versions,
    }

    manifest_path = out_dir / f"combined_dataset_{dataset_token}_{timestamp}_{short_id}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "path": to_relative(str(out_path)),
        "artifact_path": to_relative(str(out_path)),
        "manifest_path": to_relative(str(manifest_path)),
        "count": len(filtered),
        "stats": stats,
        "artifact_versions": versions,
    }


@app.post("/api/dataset/synthetic/generate", dependencies=[Depends(require_admin)])
def synthetic_generate(request: SyntheticGenerateRequest) -> Dict[str, Any]:
    target_count = int(request.count)
    expected_label = int(request.label)
    expected_constructiveness = request.constructiveness if request.constructiveness in {0, 1} else None
    domain = request.domain
    style = request.style
    model_name = normalize_gemini_model_name(request.model) or get_setting("GEMINI_MODEL", SYNTHETIC_FALLBACK_MODEL)

    existing_rows = load_synthetic_rows(domain=domain, style=style, label=expected_label)
    seen_hashes = {build_text_hash(row.get("text") or "") for row in existing_rows}
    seen_fingerprints = {build_structure_fingerprint(row.get("text") or "") for row in existing_rows}

    accepted_rows: List[Dict[str, Any]] = []
    total_rejected_placeholder = 0
    total_rejected_duplicate = 0
    total_candidates_seen = 0
    total_rejected_invalid = 0
    last_llm_preview = ""

    length_bounds = get_synthetic_length_bounds()
    length_bucket_target = build_length_bucket_targets(target_count)
    length_bucket_generated = {key: 0 for key in SYNTHETIC_LENGTH_BUCKET_ORDER}
    length_bucket_rejected = {key: 0 for key in SYNTHETIC_LENGTH_BUCKET_ORDER}

    for _ in range(SYNTHETIC_MAX_RETRIES):
        remaining = target_count - len(accepted_rows)
        if remaining <= 0:
            break

        remaining_targets = {
            key: max(0, length_bucket_target.get(key, 0) - length_bucket_generated.get(key, 0))
            for key in SYNTHETIC_LENGTH_BUCKET_ORDER
        }
        prompt = build_synthetic_prompt(
            domain=domain,
            style=style,
            label=expected_label,
            count=remaining,
            constructiveness=expected_constructiveness,
            length_guidance=build_length_bucket_guidance(remaining_targets, length_bounds),
        )
        llm_raw = call_gemini_with_model(prompt, model_name)
        candidates = parse_json_array_from_llm(llm_raw)
        if not candidates:
            last_llm_preview = (llm_raw or "")[:500]

        for candidate in candidates:
            total_candidates_seen += 1
            text = normalize_synthetic_text(str(candidate.get("text") or ""))
            if PLACEHOLDER_PATTERN.search(text):
                total_rejected_placeholder += 1
                continue

            text_hash_before = build_text_hash(text)
            fingerprint_before = build_structure_fingerprint(text)
            if text_hash_before in seen_hashes or fingerprint_before in seen_fingerprints:
                total_rejected_duplicate += 1
                continue

            validated = validate_synthetic_candidate(
                candidate=candidate,
                expected_label=expected_label,
                expected_constructiveness=expected_constructiveness,
                domain=domain,
                style=style,
                seen_hashes=seen_hashes,
                seen_fingerprints=seen_fingerprints,
                length_bounds=length_bounds,
            )
            if not validated:
                total_rejected_invalid += 1
                continue

            bucket = str(validated.get("length_bucket") or "")
            if bucket not in length_bucket_generated:
                bucket = classify_synthetic_length_bucket(int(validated.get("word_length") or 0), length_bounds)
            if length_bucket_generated.get(bucket, 0) >= length_bucket_target.get(bucket, 0):
                length_bucket_rejected[bucket] = length_bucket_rejected.get(bucket, 0) + 1
                continue

            length_bucket_generated[bucket] = length_bucket_generated.get(bucket, 0) + 1
            accepted_rows.append(validated)
            if len(accepted_rows) >= target_count:
                break

    if not accepted_rows:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Synthetic generation failed: no valid samples returned",
                "debug": {
                    "model_name": model_name,
                    "retries": SYNTHETIC_MAX_RETRIES,
                    "candidates_seen": total_candidates_seen,
                    "invalid_rejected": total_rejected_invalid,
                    "placeholder_rejected": total_rejected_placeholder,
                    "structure_or_duplicate_rejected": total_rejected_duplicate,
                    "length_bucket_target": length_bucket_target,
                    "length_bucket_generated": length_bucket_generated,
                    "length_bucket_rejected": length_bucket_rejected,
                    "last_llm_preview": last_llm_preview,
                },
            },
        )

    batch_id = uuid.uuid4().hex
    inserted = insert_synthetic_batch(
        batch_id=batch_id,
        domain=domain,
        style=style,
        target_label=expected_label,
        requested_count=target_count,
        generated_count=len(accepted_rows),
        generator_model=model_name,
        rows=accepted_rows,
    )

    saved_rows = load_synthetic_rows(batch_id=batch_id)
    for row in saved_rows:
        row["meta"] = {
            **(row.get("meta") or {}),
            "sample_id": row.get("id"),
            "batch_id": batch_id,
            "domain": domain,
            "style": style,
            "toxicity": row.get("label"),
            "constructiveness": row.get("constructiveness"),
            "generator_model": model_name,
            "prompt_version": SYNTHETIC_PROMPT_VERSION,
        }

    return {
        "batch_id": batch_id,
        "requested_count": target_count,
        "generated_count": inserted,
        "accepted_default": inserted,
        "items": saved_rows,
        "validation_summary": {
            "candidates_seen": total_candidates_seen,
            "invalid_rejected": total_rejected_invalid,
            "placeholder_rejected": total_rejected_placeholder,
            "structure_or_duplicate_rejected": total_rejected_duplicate,
            "length_bucket_target": length_bucket_target,
            "length_bucket_generated": length_bucket_generated,
            "length_bucket_rejected": length_bucket_rejected,
            "length_bounds_words": {
                "very_short_max": length_bounds[0],
                "short_medium_max": length_bounds[1],
                "medium_long_max": length_bounds[2],
            },
        },
    }


@app.get("/api/dataset/synthetic/preview", dependencies=[Depends(require_admin)])
def synthetic_preview(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    batch_id: Optional[str] = None,
    domain: Optional[SyntheticDomain] = None,
    style: Optional[SyntheticStyle] = None,
    label: Optional[int] = Query(default=None, ge=0, le=1),
    accepted: Optional[bool] = None,
    reviewed: Optional[bool] = None,
    include_stats: bool = False,
) -> Dict[str, Any]:
    if reviewed:
        raise HTTPException(status_code=400, detail="Reviewed synthetic rows are hidden from the generation page")
    rows = load_synthetic_rows(
        batch_id=batch_id,
        domain=domain,
        style=style,
        label=label,
        accepted=accepted,
        reviewed=False,
    )
    total = len(rows)
    total_pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    items = rows[start:end]

    payload: Dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "items": items,
    }
    if include_stats:
        payload["stats"] = build_synthetic_stats(rows)
    return payload


@app.post("/api/dataset/synthetic/gemini-review", dependencies=[Depends(require_admin)])
def synthetic_gemini_review(request: MlflowTrainingPreviewGeminiReviewRequest) -> Dict[str, Any]:
    init_feedback_db()
    ids = list(dict.fromkeys(request.ids))
    validate_gemini_review_item_limit(ids)
    placeholders = ", ".join(["?"] * len(ids))
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, text, NULL AS score, label AS pseudo_label,
                   NULL AS constructiveness_score, constructiveness AS constructiveness_label,
                   'synthetic' AS gate_bucket, domain AS domain_category,
                   ('synthetic://' || batch_id || '/' || id) AS url
            FROM synthetic_dataset_row
            WHERE id IN ({placeholders})
            ORDER BY id ASC
            """,
            tuple(ids),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No synthetic rows found for provided ids")
    suggestions = run_mlflow_gemini_review(rows)
    return build_gemini_review_response(suggestions, len(ids))


@app.post("/api/dataset/synthetic/review")
def synthetic_review(
    request: SyntheticReviewRequest,
    admin_username: str = Depends(require_admin),
) -> Dict[str, Any]:
    updates: List[Dict[str, Any]] = []
    for item in request.updates:
        fields_set = getattr(item, "model_fields_set", getattr(item, "__fields_set__", set()))
        payload = {
            "id": item.id,
            "is_accepted": item.is_accepted,
            "text": item.text,
            "label": item.label,
            "review_method": item.review_method,
            "label_confidence": item.label_confidence,
            "review_provider": item.review_provider,
            "review_model_name": item.review_model_name,
        }
        if "constructiveness" in fields_set:
            payload["constructiveness"] = item.constructiveness
        updates.append(payload)
    updated = update_synthetic_review(updates, admin_username)
    return {"updated": updated}


@app.get("/api/dataset/synthetic/training-preview-summary", dependencies=[Depends(require_admin)])
def synthetic_training_preview_summary(batch_id: Optional[str] = None) -> Dict[str, Any]:
    return summarize_synthetic_training_preview_transfer(batch_id=batch_id)


@app.post("/api/dataset/synthetic/transfer-to-training-preview")
def synthetic_transfer_to_training_preview(
    request: SyntheticTrainingPreviewTransferRequest,
    admin_username: str = Depends(require_admin),
) -> Dict[str, Any]:
    return transfer_synthetic_rows_to_training_preview(request.ids, admin_username)


@app.post("/api/dataset/synthetic/delete", dependencies=[Depends(require_admin)])
def synthetic_delete(request: SyntheticDeleteRequest) -> Dict[str, Any]:
    deleted = delete_synthetic_rows(request.ids)
    return {"deleted": deleted}


@app.get("/api/dataset/synthetic/stats", dependencies=[Depends(require_admin)])
def synthetic_stats(
    batch_id: Optional[str] = None,
    domain: Optional[SyntheticDomain] = None,
    style: Optional[SyntheticStyle] = None,
    label: Optional[int] = Query(default=None, ge=0, le=1),
    accepted: Optional[bool] = None,
) -> Dict[str, Any]:
    rows = load_synthetic_rows(
        batch_id=batch_id,
        domain=domain,
        style=style,
        label=label,
        accepted=accepted,
    )
    return build_synthetic_stats(rows)


@app.post("/api/dataset/synthetic/export", dependencies=[Depends(require_admin)])
def synthetic_export(request: SyntheticExportRequest) -> Dict[str, Any]:
    rows = load_synthetic_rows(
        batch_id=request.batch_id,
        domain=request.domain,
        style=request.style,
        label=request.label,
        accepted=True if request.accepted_only else None,
        reviewed=True,
    )

    export_rows = [
        {
            "text": row["text"],
            "label": row["label"],
            "toxicity": row["label"],
            "constructiveness": row.get("constructiveness") if row.get("constructiveness") in {0, 1} else None,
            "meta": row.get("meta") or {},
        }
        for row in rows
    ]

    out_dir = PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in export_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "path": to_relative(str(out_path)),
        "count": len(export_rows),
        "stats": build_synthetic_stats(rows),
    }


@app.get("/api/preprocessing/steps")
def preprocessing_steps() -> Dict[str, Any]:
    steps = [
        {"id": "trim", "label": "Loại bỏ khoảng trắng đầu/cuối", "active": True},
        {"id": "normalize_unicode", "label": "Chuẩn hoá Unicode (NFC)", "active": True},
        {"id": "normalize_whitespace", "label": "Chuẩn hoá khoảng trắng", "active": True},
        {"id": "lowercase", "label": "Chuyển lowercase", "active": False},
        {"id": "remove_emoji", "label": "Xử lý emoji", "active": False},
        {"id": "strip_punctuation", "label": "Loại bỏ dấu câu mạnh", "active": False},
        {"id": "teencode", "label": "Chuẩn hoá teencode", "active": False},
    ]
    return {"steps": steps}


@app.get("/api/experiments/registry")
def experiments_registry(refresh: bool = Query(False)) -> Dict[str, Any]:
    registry = load_json_file(EXPERIMENT_REGISTRY_PATH, {"runs": []})
    last_updated = file_last_updated(EXPERIMENT_REGISTRY_PATH)

    runs = registry.get("runs") if isinstance(registry, dict) else []
    if refresh or not runs:
        registry = build_registry_from_models(
            model_root=MODEL_OPTIONS_DIR,
            legacy_registry=registry if isinstance(registry, dict) else {"runs": []},
            merge_legacy=True,
        )
        runs = registry.get("runs") if isinstance(registry, dict) else []

    last_run = None
    if isinstance(runs, list) and runs:
        last_run = max(
            (run.get("created_at") for run in runs if isinstance(run, dict) and run.get("created_at")),
            default=None,
        )
    return {
        "runs": runs if isinstance(runs, list) else [],
        "last_updated": last_run or last_updated,
    }


@app.get("/api/eval/policy")
def eval_policy() -> Dict[str, Any]:
    policy = load_json_file(EVAL_POLICY_PATH, {})
    return {"policy": policy, "last_updated": file_last_updated(EVAL_POLICY_PATH)}


@app.get("/api/eval/errors")
def eval_errors() -> Dict[str, Any]:
    rows = load_json_file(ERROR_ANALYSIS_PATH, [])
    return {"items": rows if isinstance(rows, list) else [], "last_updated": file_last_updated(ERROR_ANALYSIS_PATH)}


@app.get("/api/eval/hard-cases")
def eval_hard_cases() -> Dict[str, Any]:
    rows = load_json_file(HARD_CASES_PATH, [])
    return {"items": rows if isinstance(rows, list) else [], "last_updated": file_last_updated(HARD_CASES_PATH)}
