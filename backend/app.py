import csv
import hashlib
import json
import logging
import math
import os
import statistics
import re
import shutil
import sqlite3
import time
import uuid
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from domain_classifier import CATEGORY_THRESHOLDS
from backend.crawl_adapter import crawl_urls
from infer_crawled_local import infer_crawled, build_segment_hash, build_context_segment_hash

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "crawled_urls"
MODEL_OPTIONS_DIR = Path(
    os.getenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(BASE_DIR / "models" / "options"))
).expanduser()
FEEDBACK_DIR = BASE_DIR / "data" / "processed" / "feedback"
FEEDBACK_DB_PATH = FEEDBACK_DIR / "feedback.db"
EXPERIMENT_REGISTRY_PATH = BASE_DIR / "experiments" / "registry.json"
EVAL_POLICY_PATH = BASE_DIR / "config" / "eval_policy.json"
ERROR_ANALYSIS_PATH = BASE_DIR / "data" / "processed" / "error_analysis.json"
HARD_CASES_PATH = BASE_DIR / "data" / "processed" / "hard_case_candidates.json"
PROTOCOL_BUILD_REPORT_PATH = BASE_DIR / "data" / "victsd" / "victsd_v1_protocol_build_report.json"
PROTOCOL_METRICS_ROOT = BASE_DIR / "viettoxic_outputs"

DEFAULT_DATASET_VERSION = os.getenv("VIETTOXIC_DATASET_VERSION", "victsd_v1")
DEFAULT_MODEL_VERSION = os.getenv("VIETTOXIC_MODEL_VERSION", "unknown")
DEFAULT_POLICY_VERSION = os.getenv("VIETTOXIC_POLICY_VERSION", "policy-v1")
REQUIRED_VERSION_KEYS = ("dataset_version", "model_version", "policy_version")

MLFLOW_ACCEPT_THRESHOLD = float(os.getenv("MLFLOW_ACCEPT_THRESHOLD", "0.8"))
MLFLOW_DISCARD_THRESHOLD = float(os.getenv("MLFLOW_DISCARD_THRESHOLD", "0.2"))
MLFLOW_THRESHOLD_TARGET_MAX = max(1, int(os.getenv("MLFLOW_THRESHOLD_TARGET_MAX", "10")))
MLFLOW_CLEAR_ALL_CONFIRM_TOKEN = os.getenv("MLFLOW_CLEAR_ALL_CONFIRM_TOKEN", "DELETE_ALL_MLFLOW_DATA")

DATASET_VERSION_ALIASES: Dict[str, str] = {
    "v1": "victsd_v1",
    "victsd_v1": "victsd_v1",
    "latest": "victsd_gold",
    "victsd_gold": "victsd_gold",
}
DATASET_VERSION_DIRS: Dict[str, Path] = {
    "victsd_v1": BASE_DIR / "data" / "victsd",
    "victsd_gold": BASE_DIR / "data" / "processed" / "victsd_gold",
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
                "title": "1.4 Learning rate LoRA",
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
        "title": "Giai đoạn 5 — LoRA config",
        "tasks": [
            {"id": "p5_task_1", "label": "Test r=8", "param": "LORA_R=8"},
            {"id": "p5_task_2", "label": "Test r=16", "param": "LORA_R=16"},
            {"id": "p5_task_3", "label": "Test r=32", "param": "LORA_R=32"},
            {"id": "p5_task_4", "label": "Test lora_alpha=16", "param": "LORA_ALPHA=16"},
            {"id": "p5_task_5", "label": "Test lora_alpha=32", "param": "LORA_ALPHA=32"},
            {"id": "p5_task_6", "label": "Test lora_alpha=64", "param": "LORA_ALPHA=64"},
            {"id": "p5_task_7", "label": "Test lora_dropout=0.05", "param": "LORA_DROPOUT=0.05"},
            {"id": "p5_task_8", "label": "Test lora_dropout=0.1", "param": "LORA_DROPOUT=0.1"},
            {"id": "p5_task_9", "label": "Test target_modules: q,v", "param": "LORA_TARGET_MODULES=q,v"},
            {"id": "p5_task_10", "label": "Test target_modules: q,k,v", "param": "LORA_TARGET_MODULES=q,k,v"},
        ],
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("viet-toxic-backend")


ENV_FILES = [
    BASE_DIR / ".env",
    BASE_DIR / ".env.local",
    BASE_DIR / "backend" / ".env",
    BASE_DIR / "backend" / ".env.local",
]


def load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed; skipping .env/.env.local")
        return

    loaded_any = False
    for env_path in ENV_FILES:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            loaded_any = True

    if loaded_any:
        logger.info("Loaded environment variables from .env files")


load_env_files()


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


app = FastAPI(title="VietToxic Local API")
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
        "message": "VietToxic API is running. Use POST /api/analyze.",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


class AnalyzeOptions(BaseModel):
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=256, ge=16)
    page_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    seg_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    enable_video: bool = False
    selenium_fallback_mode: Literal["auto", "ask"] = "auto"


class AnalyzeRequest(BaseModel):
    urls: List[str] = Field(min_items=1)
    options: Optional[AnalyzeOptions] = None


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
    count: int = Field(default=10, ge=1, le=200)
    model: Optional[str] = None


class SyntheticReviewItem(BaseModel):
    id: int
    is_accepted: bool
    text: Optional[str] = None
    label: Optional[int] = Field(default=None, ge=0, le=1)


class SyntheticReviewRequest(BaseModel):
    updates: List[SyntheticReviewItem] = Field(min_items=1)


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


class MlflowCandidateReviewRequest(BaseModel):
    updates: List[MlflowCandidateReviewItem] = Field(min_items=1)


class MlflowManualExportBundleRequest(BaseModel):
    batch_id: str = Field(min_length=1)
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    policy_version: Optional[str] = None
    include_unused: bool = False
    unused_scope: Literal["all", "auto_discarded", "manual_rejected"] = "all"


class MlflowManualImportArtifactRequest(BaseModel):
    run_name: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    notes: Optional[str] = None


class MlflowDOTriggerRequest(BaseModel):
    batch_id: Optional[str] = None
    provider: str = Field(default="digitalocean", min_length=1)
    gpu_profile: str = Field(default="gpu-placeholder", min_length=1)
    dry_run: bool = True


class MlflowPromoteRequest(BaseModel):
    candidate_model: str = Field(min_length=1)


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
            models.append({
                "id": f"{model_type}/{name}",
                "type": model_type,
                "name": name,
            })
    return models


def get_default_model_id(model_root: Path) -> Optional[str]:
    phobert_models = list_models_by_type(model_root, "phobert")
    if phobert_models:
        if "v2" in phobert_models:
            return "phobert/v2"
        return f"phobert/{phobert_models[0]}"
    all_models = list_all_models(model_root)
    if not all_models:
        return None
    return all_models[0]["id"]


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
    if name not in models:
        raise ValueError(f"Model '{model_id}' not found. Available: {models}")

    model_path = base_dir / name
    if not model_path.is_dir():
        raise ValueError(f"Model '{model_id}' is not a directory under {base_dir}")

    validate_model_artifacts(model_type, model_path)
    return model_type, name, model_path


def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


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
        if page_info:
            overall = normalize_score(page_info.get("avg_toxic_prob"))
            if overall is None:
                overall = normalize_score(page_info.get("toxic_ratio"))

        segment_entries = seg_by_hash.get(url_hash) or seg_by_url.get(url) or []
        by_segment = []
        for idx, seg in enumerate(segment_entries):
            score = normalize_score(seg.get("toxic_prob"))
            text = seg.get("text") or seg.get("text_preview") or ""
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
                    "toxic_prob_adjusted": normalize_score(seg.get("toxic_prob_adjusted")),
                    "ai_learned_mode": seg.get("ai_learned_mode"),
                    "learned_support": seg.get("learned_support"),
                    "learned_agreement": normalize_score(seg.get("learned_agreement")),
                    "seg_threshold_used": normalize_score(seg.get("seg_threshold_used")),
                }
            )

        response_results.append(
            {
                "url": url,
                "url_hash": url_hash,
                "domain_category": page_info.get("domain_category") if page_info else None,
                "status": status,
                "error": error,
                "warnings": crawl.get("warnings") or [],
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


def get_gemini_model_candidates() -> List[str]:
    primary = normalize_gemini_model_name(os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest"))
    fallback_raw = os.getenv("GEMINI_FALLBACK_MODELS", "")
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


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    api_version = os.getenv("GEMINI_API_VERSION", "v1beta")
    try:
        max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
    except ValueError:
        max_tokens = 1024

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
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp else str(exc)
            last_error = detail
            if exc.code == 404 and idx < len(candidates) - 1:
                logger.warning("Gemini model not found: %s", model)
                continue
            if is_gemini_rate_limited(exc.code, detail) and idx < len(candidates) - 1:
                logger.warning("Gemini rate limited on %s, trying fallback", model)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}") from exc
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

        try:
            parsed = json.loads(raw)
            model_candidates = parsed.get("candidates") or []
            if not model_candidates:
                raise ValueError("No candidates returned")
            parts = model_candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No content parts returned")
            return "\n".join([p.get("text", "") for p in parts if p.get("text")])
        except Exception as exc:
            last_error = str(exc)
            raise HTTPException(status_code=502, detail=f"Gemini response parse error: {exc}") from exc

    raise HTTPException(status_code=502, detail=f"Gemini API error: {last_error or 'Unknown error'}")


def list_gemini_models() -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    api_version = os.getenv("GEMINI_API_VERSION", "v1beta")
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
        "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest"),
        "fallback_models": get_gemini_model_candidates()[1:],
        "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
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
        BASE_DIR / "data" / "processed" / "victsd_v1" / "train.jsonl",
        BASE_DIR / "data" / "processed" / "victsd_v1" / "validation.jsonl",
        BASE_DIR / "data" / "processed" / "victsd_v1" / "test.jsonl",
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
        return []

    payload = cleaned[start : end + 1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []

    return extract_items(parsed)


def build_synthetic_prompt(
    domain: str,
    style: str,
    label: int,
    count: int,
    length_guidance: Optional[str] = None,
) -> str:
    toxicity = "toxic" if label == 1 else "clean"
    guidance = f"\n7) {length_guidance}" if length_guidance else ""
    return (
        "Bạn là hệ thống tạo dữ liệu tiếng Việt cho phân loại toxic. "
        "Hãy tạo đúng số lượng mẫu theo yêu cầu và trả về JSON array hợp lệ, không có text ngoài JSON.\n"
        f"Yêu cầu: domain={domain}, style={style}, label={label} ({toxicity}), số mẫu={count}.\n"
        "Mỗi phần tử bắt buộc có schema: {\"text\": string, \"label\": 0|1, \"meta\": object}.\n"
        "Ràng buộc bắt buộc:\n"
        "1) Không lặp cấu trúc câu giữa các mẫu.\n"
        "2) Không dùng placeholder dạng [tên], [trường], <name>, {city}.\n"
        "3) Phải dùng tên/tổ chức cụ thể giả định (vd: Trường THPT Nguyễn Trãi, GS. Nguyễn Văn A).\n"
        "4) Dữ liệu phải tự nhiên, đúng tiếng Việt.\n"
        "5) meta phải chứa source=\"synthetic_llm\", domain, style.\n"
        "6) label trong từng sample phải đúng bằng label yêu cầu."
        f"{guidance}"
    )


def call_gemini_with_model(prompt: str, model_name: Optional[str] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    api_version = os.getenv("GEMINI_API_VERSION", "v1beta")
    try:
        max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
    except ValueError:
        max_tokens = 1024

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
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp else str(exc)
            last_error = detail
            if (exc.code == 404 or is_gemini_rate_limited(exc.code, detail)) and idx < len(candidates) - 1:
                logger.warning("Gemini failed on %s, trying fallback", model)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}") from exc
        except urllib.error.URLError as exc:
            last_error = str(exc)
            if idx < len(candidates) - 1:
                logger.warning("Gemini network error on %s, trying fallback", model)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

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
                return text
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


def init_feedback_db() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
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
                domain TEXT NOT NULL,
                style TEXT NOT NULL,
                is_accepted INTEGER NOT NULL DEFAULT 1,
                structure_fingerprint TEXT,
                text_hash TEXT,
                validation_flags TEXT,
                meta_json TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
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
                gate_bucket TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                segment_hash TEXT,
                context_segment_hash TEXT,
                html_tag TEXT,
                seg_threshold_used REAL,
                label_source TEXT,
                label_confidence TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                FOREIGN KEY(batch_id) REFERENCES mlflow_crawl_batch(batch_id) ON DELETE CASCADE
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
        ensure_table_column(conn, "mlflow_comment_item", "label_source", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "label_confidence", "TEXT")
        ensure_table_column(conn, "mlflow_comment_item", "domain_category", "TEXT")

        seed_training_tracker_default(conn)
        conn.commit()


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
                "path": str(dataset_dir.relative_to(BASE_DIR)),
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
    if cleaned in {"victsd", "victsd_v1"}:
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
                    if label is None:
                        label = normalize_int(obj.get("toxicity"))
                    if text is None or label is None:
                        continue
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
                    rows.append({"text": text, "label": label, "meta": meta_out})
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
        rows.append({"text": text, "label": label_int, "meta": meta})
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


def build_domain_mismatch_note(protocol_id: str, train_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    stats = train_stats if isinstance(train_stats, dict) else {}
    sources = stats.get("sources") if isinstance(stats, dict) else {}
    source_map = {str(k): int(v) for k, v in (sources or {}).items() if isinstance(v, (int, float))}
    total = sum(source_map.values())
    vihsd_total = source_map.get("UIT-ViHSD", 0)
    vihsd_ratio = (vihsd_total / total) if total else 0.0

    if protocol_id == "a":
        summary = "ViCTSD-only training set (closest to clean benchmark anchor)."
    elif protocol_id == "b":
        summary = (
            "Train includes ViHSD OFFENSIVE injections while validation/test stay ViCTSD; "
            "watch for social-style toxic bias against formal news text."
        )
    elif protocol_id == "c":
        summary = (
            "Merged benchmark across ViCTSD+ViHSD with global dedup/split; "
            "still requires caution when deploying to formal news domain."
        )
    else:
        summary = "Domain mismatch risk unknown."

    risk_level = "low"
    if vihsd_ratio >= 0.40:
        risk_level = "high"
    elif vihsd_ratio >= 0.20:
        risk_level = "medium"

    return {
        "risk_level": risk_level,
        "vihsd_train_ratio": vihsd_ratio,
        "train_source_mix": source_map,
        "summary": summary,
    }


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
                batch_id, text, label, domain, style, is_accepted,
                structure_fingerprint, text_hash, validation_flags, meta_json,
                created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    reviewed: Optional[bool] = None,
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
        SELECT id, batch_id, text, label, domain, style, is_accepted,
               validation_flags, meta_json, created_at, reviewed_at
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
        row_domain,
        row_style,
        is_accepted,
        validation_flags,
        meta_json,
        created_at,
        reviewed_at,
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

        rows.append(
            {
                "id": sample_id,
                "batch_id": row_batch_id,
                "text": text,
                "label": row_label,
                "domain": row_domain,
                "style": row_style,
                "is_accepted": bool(is_accepted),
                "meta": meta,
                "validation_flags": flags,
                "created_at": created_at,
                "reviewed_at": reviewed_at,
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


def update_synthetic_review(items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    init_feedback_db()
    now = datetime.utcnow().isoformat()

    normalized: List[Tuple[int, Optional[str], Optional[int], Optional[str], int]] = []
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

        text_hash = build_text_hash(cleaned_text) if cleaned_text is not None else None
        normalized.append(
            (
                1 if bool(item.get("is_accepted")) else 0,
                cleaned_text,
                reviewed_label,
                text_hash,
                sample_id,
            )
        )

    if not normalized:
        return 0

    changed = 0
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        for is_accepted, cleaned_text, reviewed_label, text_hash, sample_id in normalized:
            existing = conn.execute(
                "SELECT text, label, domain, style, meta_json FROM synthetic_dataset_row WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if not existing:
                continue

            old_text, old_label, domain, style, old_meta_json = existing
            final_text = cleaned_text if cleaned_text is not None else old_text
            final_label = reviewed_label if reviewed_label is not None else old_label

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

            conn.execute(
                """
                UPDATE synthetic_dataset_row
                SET is_accepted = ?,
                    text = ?,
                    label = ?,
                    text_hash = ?,
                    meta_json = ?,
                    reviewed_at = ?
                WHERE id = ?
                """,
                (
                    is_accepted,
                    final_text,
                    final_label,
                    text_hash if text_hash is not None else build_text_hash(final_text),
                    json.dumps(meta, ensure_ascii=False),
                    now,
                    sample_id,
                ),
            )
            changed += 1

        conn.commit()
    return changed


def build_synthetic_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    accepted = sum(1 for row in rows if row.get("is_accepted"))
    rejected = total - accepted

    by_domain: Dict[str, Dict[str, int]] = {}
    by_style: Dict[str, Dict[str, int]] = {}
    by_label: Dict[str, Dict[str, int]] = {}
    by_combo: Dict[str, Dict[str, int]] = {}

    for row in rows:
        domain = str(row.get("domain") or "unknown")
        style = str(row.get("style") or "unknown")
        label = str(row.get("label") if row.get("label") in {0, 1} else "unknown")
        bucket_status = "accepted" if row.get("is_accepted") else "rejected"

        for group, key in [(by_domain, domain), (by_style, style), (by_label, label)]:
            stats = group.setdefault(key, {"total": 0, "accepted": 0, "rejected": 0})
            stats["total"] += 1
            stats[bucket_status] += 1

        combo_key = f"{domain}|{style}|{label}"
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
        "by_combo": by_combo,
    }


def validate_synthetic_candidate(
    *,
    candidate: Dict[str, Any],
    expected_label: int,
    domain: str,
    style: str,
    seen_hashes: set,
    seen_fingerprints: set,
    length_bounds: Optional[Tuple[int, int, int]] = None,
) -> Optional[Dict[str, Any]]:
    text = normalize_synthetic_text(str(candidate.get("text") or ""))

    raw_label = candidate.get("label")
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
        "word_length": word_length,
        "length_bucket": bucket,
    }

    return {
        "text": text,
        "label": expected_label,
        "meta": meta_out,
        "structure_fingerprint": structure_fingerprint,
        "text_hash": text_hash,
        "word_length": word_length,
        "length_bucket": bucket,
        "validation_flags": {},
    }


def cleanup_old_jobs(ttl_hours: float = 24.0) -> int:
    processed_dir = BASE_DIR / "data" / "processed"
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


def build_mlflow_gate_counts(conn: sqlite3.Connection, batch_id: str) -> Dict[str, int]:
    rows = conn.execute(
        """
        SELECT gate_bucket, COUNT(1) AS c
        FROM mlflow_comment_item
        WHERE batch_id = ?
        GROUP BY gate_bucket
        """,
        (batch_id,),
    ).fetchall()
    counts = {"accepted": 0, "candidate": 0, "discarded": 0}
    for bucket, value in rows:
        key = str(bucket or "").strip().lower()
        if key in counts:
            counts[key] = int(value or 0)
    counts["total"] = counts["accepted"] + counts["candidate"] + counts["discarded"]
    return counts


def build_mlflow_required_bundle_contents() -> List[str]:
    return [
        "dataset/accepted_pseudo.jsonl",
        "dataset/candidates_unverified.jsonl",
        "manifest.json",
        "config/training_config.yaml",
        "config/gate_policy.json",
    ]


@app.post("/api/mlflow/ingest")
def mlflow_ingest(request: MlflowIngestRequest) -> Dict[str, Any]:
    try:
        cleanup_old_jobs(float(os.getenv("JOB_RETENTION_HOURS", "24")))
        options = request.options or MlflowIngestOptions()

        urls = [u.strip() for u in request.urls if u and u.strip()]
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
        out_dir = BASE_DIR / "data" / "processed" / f"job_{source_job_id}"
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

        crawl_results = crawl_urls(urls, out_dir=str(DATA_DIR))
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

        rows_to_insert: List[Tuple[Any, ...]] = []
        created_at = datetime.utcnow().isoformat() + "Z"
        for result in response_results:
            if result.get("status") != "ok":
                continue
            url = str(result.get("url") or "")
            url_hash = str(result.get("url_hash") or hash_url(url))
            for seg in (result.get("toxicity") or {}).get("by_segment") or []:
                score = normalize_score(seg.get("score"))
                if score is None:
                    continue
                if score <= discard_threshold:
                    if not options.persist_unused:
                        continue
                    gate_bucket = "discarded"
                    verification_status = "auto_discarded"
                elif score >= accept_threshold:
                    gate_bucket = "accepted"
                    verification_status = "auto_accepted"
                else:
                    gate_bucket = "candidate"
                    verification_status = "unverified"

                seg_toxic_label = normalize_int(seg.get("toxic_label"))
                seg_threshold_used = normalize_score(seg.get("seg_threshold_used"))

                label_source = "fallback_0_5"
                label_confidence = "low"
                if seg_toxic_label in {0, 1}:
                    pseudo_label = seg_toxic_label
                    label_source = "auto_infer"
                    label_confidence = "high"
                elif seg_threshold_used is not None:
                    pseudo_label = 1 if score >= seg_threshold_used else 0
                    label_source = "auto_infer"
                    label_confidence = "medium"
                else:
                    pseudo_label = 1 if score >= 0.5 else 0

                rows_to_insert.append(
                    (
                        batch_id,
                        source_job_id,
                        url,
                        url_hash,
                        seg.get("segment_id"),
                        seg.get("domain_category") or result.get("domain_category"),
                        seg.get("text") or seg.get("text_preview") or "",
                        score,
                        pseudo_label,
                        gate_bucket,
                        verification_status,
                        seg.get("segment_hash"),
                        seg.get("context_segment_hash"),
                        ((seg.get("html_tags") or [None])[0] if isinstance(seg.get("html_tags"), list) else None),
                        seg_threshold_used,
                        label_source,
                        label_confidence,
                        created_at,
                    )
                )

        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            if rows_to_insert:
                conn.executemany(
                    """
                    INSERT INTO mlflow_comment_item (
                        batch_id, job_id, url, url_hash, segment_id, domain_category, text, score, pseudo_label,
                        gate_bucket, verification_status, segment_hash, context_segment_hash,
                        html_tag, seg_threshold_used, label_source, label_confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_insert,
                )
            conn.execute(
                """
                UPDATE mlflow_crawl_batch
                SET status = 'completed', completed_at = ?
                WHERE batch_id = ?
                """,
                (datetime.utcnow().isoformat() + "Z", batch_id),
            )
            counts = build_mlflow_gate_counts(conn, batch_id)
            conn.commit()

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


@app.get("/api/mlflow/overview")
def mlflow_overview(
    batch_id: Optional[str] = None,
    strict_batch: bool = Query(default=False),
) -> Dict[str, Any]:
    resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
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
    }


@app.get("/api/mlflow/batches")
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


@app.post("/api/mlflow/clear-batch")
def mlflow_clear_batch(request: MlflowClearBatchRequest) -> Dict[str, Any]:
    batch_id = resolve_mlflow_batch_id(request.batch_id, strict=True)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        deleted_do_run = int(
            conn.execute("DELETE FROM mlflow_do_run WHERE batch_id = ?", (batch_id,)).rowcount or 0
        )
        deleted_comment_item = int(
            conn.execute("DELETE FROM mlflow_comment_item WHERE batch_id = ?", (batch_id,)).rowcount or 0
        )
        deleted_crawl_batch = int(
            conn.execute("DELETE FROM mlflow_crawl_batch WHERE batch_id = ?", (batch_id,)).rowcount or 0
        )

        conn.commit()

    return {
        "scope": "batch",
        "batch_id": batch_id,
        "deleted_rows": {
            "mlflow_do_run": deleted_do_run,
            "mlflow_comment_item": deleted_comment_item,
            "mlflow_crawl_batch": deleted_crawl_batch,
            "mlflow_training_artifact": 0,
        },
    }


@app.post("/api/mlflow/clear-all")
def mlflow_clear_all(request: MlflowClearAllRequest) -> Dict[str, Any]:
    confirm_token = (request.confirm_token or "").strip()
    if confirm_token != MLFLOW_CLEAR_ALL_CONFIRM_TOKEN:
        raise HTTPException(status_code=400, detail="Invalid confirm_token")

    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        deleted_do_run = int(conn.execute("DELETE FROM mlflow_do_run").rowcount or 0)
        deleted_training_artifact = int(conn.execute("DELETE FROM mlflow_training_artifact").rowcount or 0)
        deleted_comment_item = int(conn.execute("DELETE FROM mlflow_comment_item").rowcount or 0)
        deleted_crawl_batch = int(conn.execute("DELETE FROM mlflow_crawl_batch").rowcount or 0)

        conn.commit()

    return {
        "scope": "all",
        "deleted_rows": {
            "mlflow_do_run": deleted_do_run,
            "mlflow_training_artifact": deleted_training_artifact,
            "mlflow_comment_item": deleted_comment_item,
            "mlflow_crawl_batch": deleted_crawl_batch,
        },
    }


@app.get("/api/mlflow/review-history")
def mlflow_review_history(
    batch_id: Optional[str] = None,
    decision: str = Query(default="all"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    strict_batch: bool = Query(default=False),
) -> Dict[str, Any]:
    resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
    offset = (page - 1) * page_size

    decision_normalized = (decision or "all").strip().lower()
    where_parts = ["batch_id = ?", "verification_status != 'unverified'"]
    params: List[Any] = [resolved_batch_id]

    if decision_normalized == "accepted":
        where_parts.append("gate_bucket = 'accepted'")
    elif decision_normalized == "rejected":
        where_parts.append("verification_status = 'manual_rejected'")
    elif decision_normalized == "discarded":
        where_parts.append("gate_bucket = 'discarded'")
    elif decision_normalized != "all":
        raise HTTPException(status_code=400, detail=f"Unsupported decision filter: {decision}")

    where_sql = " AND ".join(where_parts)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(
            f"SELECT COUNT(1) FROM mlflow_comment_item WHERE {where_sql}",
            tuple(params),
        ).fetchone()
        total = int(total_row[0] if total_row else 0)

        rows = conn.execute(
            f"""
            SELECT id, batch_id, url, url_hash, segment_id, domain_category, text, score,
                   pseudo_label, gate_bucket, verification_status,
                   segment_hash, context_segment_hash, html_tag,
                   seg_threshold_used, label_source, label_confidence, created_at, reviewed_at
            FROM mlflow_comment_item
            WHERE {where_sql}
            ORDER BY COALESCE(reviewed_at, created_at) DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            tuple([*params, page_size, offset]),
        ).fetchall()

    items = [dict(row) for row in rows]
    return {
        "batch_id": resolved_batch_id,
        "decision": decision_normalized,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/mlflow/candidates")
def mlflow_candidates(
    batch_id: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    strict_batch: bool = Query(default=False),
) -> Dict[str, Any]:
    resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
    offset = (page - 1) * page_size
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(
            """
            SELECT COUNT(1)
            FROM mlflow_comment_item
            WHERE batch_id = ?
              AND gate_bucket = 'candidate'
              AND verification_status = 'unverified'
            """,
            (resolved_batch_id,),
        ).fetchone()
        total = int(total_row[0] if total_row else 0)

        rows = conn.execute(
            """
            SELECT id, batch_id, url, url_hash, segment_id, domain_category, text, score,
                   pseudo_label, gate_bucket, verification_status,
                   segment_hash, context_segment_hash, html_tag,
                   seg_threshold_used, label_source, label_confidence, created_at, reviewed_at
            FROM mlflow_comment_item
            WHERE batch_id = ?
              AND gate_bucket = 'candidate'
              AND verification_status = 'unverified'
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (resolved_batch_id, page_size, offset),
        ).fetchall()

    items = [dict(row) for row in rows]
    return {
        "batch_id": resolved_batch_id,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/api/mlflow/candidates/review")
def mlflow_candidates_review(request: MlflowCandidateReviewRequest) -> Dict[str, Any]:
    init_feedback_db()
    ids = [item.id for item in request.updates]
    action_by_id = {item.id: item.action for item in request.updates}
    decision_by_id = {item.id: item.decision for item in request.updates}
    pseudo_label_by_id = {
        item.id: (item.pseudo_label if item.pseudo_label in {0, 1} else None)
        for item in request.updates
    }

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, batch_id, pseudo_label FROM mlflow_comment_item WHERE id IN ({', '.join(['?'] * len(ids))})",
            tuple(ids),
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No candidate rows found for provided ids")

        now = datetime.utcnow().isoformat() + "Z"
        affected_batch_ids = {str(row["batch_id"]) for row in rows}
        updated = 0
        for row in rows:
            item_id = int(row["id"])
            manual_label = pseudo_label_by_id.get(item_id)
            current_label = normalize_int(row["pseudo_label"])
            final_label = manual_label if manual_label in {0, 1} else current_label

            action = action_by_id.get(item_id)
            decision = decision_by_id.get(item_id)
            if action is None:
                if decision == "accept":
                    if final_label not in {0, 1}:
                        raise HTTPException(status_code=400, detail=f"Include action requires label for item {item_id}")
                    action = "include_toxic" if final_label == 1 else "include_clean"
                elif decision == "reject":
                    action = "drop"

            if action not in {"include_toxic", "include_clean", "drop"}:
                raise HTTPException(status_code=400, detail=f"Unsupported review action for item {item_id}")

            if action == "include_toxic":
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, pseudo_label = ?, label_source = ?, label_confidence = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_accepted", "accepted", 1, "manual_override", "high", now, item_id),
                )
            elif action == "include_clean":
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, pseudo_label = ?, label_source = ?, label_confidence = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_accepted", "accepted", 0, "manual_override", "high", now, item_id),
                )
            elif final_label in {0, 1}:
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, pseudo_label = ?, label_source = ?, label_confidence = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_rejected", "discarded", final_label, "manual_override", "high", now, item_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE mlflow_comment_item
                    SET verification_status = ?, gate_bucket = ?, label_source = ?, label_confidence = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    ("manual_rejected", "discarded", "manual_rejected_unlabeled", "low", now, item_id),
                )
            updated += int(cursor.rowcount or 0)

        counts_by_batch: Dict[str, Dict[str, int]] = {}
        for b_id in affected_batch_ids:
            counts_by_batch[b_id] = build_mlflow_gate_counts(conn, b_id)

        conn.commit()

    primary_batch = sorted(affected_batch_ids)[0]
    return {
        "updated": updated,
        "batch_id": primary_batch,
        "counts": counts_by_batch.get(primary_batch, {"accepted": 0, "candidate": 0, "discarded": 0, "total": 0}),
        "counts_by_batch": counts_by_batch,
    }


@app.get("/api/mlflow/threshold-status")
def mlflow_threshold_status(
    batch_id: Optional[str] = None,
    strict_batch: bool = Query(default=False),
) -> Dict[str, Any]:
    resolved_batch_id = resolve_mlflow_batch_id(batch_id, strict=strict_batch)
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

    remaining = max(MLFLOW_THRESHOLD_TARGET_MAX - accepted_count, 0)
    return {
        "batch_id": resolved_batch_id,
        "scope": "all_batches",
        "accepted_count": accepted_count,
        "accepted_count_current_batch": accepted_count_current_batch,
        "target_max_test_stage": MLFLOW_THRESHOLD_TARGET_MAX,
        "remaining_to_target": remaining,
        "is_ready": accepted_count >= MLFLOW_THRESHOLD_TARGET_MAX,
    }


@app.post("/api/mlflow/manual/export-bundle")
def mlflow_manual_export_bundle(request: MlflowManualExportBundleRequest) -> Dict[str, Any]:
    init_feedback_db()
    batch_id = request.batch_id.strip()
    resolved_dataset_version = normalize_dataset_version(request.dataset_version)

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        batch_row = conn.execute(
            "SELECT batch_id, model_id, created_at FROM mlflow_crawl_batch WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if not batch_row:
            raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

        accepted_rows = conn.execute(
            """
            SELECT text, pseudo_label, score, url, url_hash, segment_hash, context_segment_hash, html_tag
            FROM mlflow_comment_item
            WHERE batch_id = ? AND gate_bucket = 'accepted' AND pseudo_label IN (0, 1)
            ORDER BY id ASC
            """,
            (batch_id,),
        ).fetchall()
        candidate_rows = conn.execute(
            """
            SELECT text, pseudo_label, score, url, url_hash, segment_hash, context_segment_hash, html_tag
            FROM mlflow_comment_item
            WHERE batch_id = ? AND gate_bucket = 'candidate' AND verification_status = 'unverified'
            ORDER BY id ASC
            """,
            (batch_id,),
        ).fetchall()

        unused_rows: List[sqlite3.Row] = []
        if request.include_unused:
            where_sql = "batch_id = ? AND gate_bucket = 'discarded'"
            params: List[Any] = [batch_id]
            if request.unused_scope == "auto_discarded":
                where_sql += " AND verification_status = 'auto_discarded'"
            elif request.unused_scope == "manual_rejected":
                where_sql += " AND verification_status = 'manual_rejected'"

            unused_rows = conn.execute(
                f"""
                SELECT text, pseudo_label, score, url, url_hash, segment_hash, context_segment_hash, html_tag, verification_status
                FROM mlflow_comment_item
                WHERE {where_sql}
                ORDER BY id ASC
                """,
                tuple(params),
            ).fetchall()

    model_version = request.model_version or str(batch_row["model_id"])
    policy_version = request.policy_version or DEFAULT_POLICY_VERSION
    versions = build_artifact_versions(
        dataset_version=resolved_dataset_version,
        model_version=model_version,
        policy_version=policy_version,
    )

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp_compact()
    out_path = out_dir / f"mlflow_bundle_{batch_id}_{timestamp}.zip"

    accepted_jsonl = "\n".join(
        json.dumps(
            {
                "text": row["text"],
                "label": int(row["pseudo_label"]),
                "meta": {
                    "source": "mlflow_pseudo",
                    "batch_id": batch_id,
                    "score": row["score"],
                    "url": row["url"],
                    "url_hash": row["url_hash"],
                    "segment_hash": row["segment_hash"],
                    "context_segment_hash": row["context_segment_hash"],
                    "html_tag": row["html_tag"],
                },
            },
            ensure_ascii=False,
        )
        for row in accepted_rows
    )
    candidate_jsonl = "\n".join(
        json.dumps(
            {
                "text": row["text"],
                "label": int(row["pseudo_label"] if row["pseudo_label"] is not None else 0),
                "meta": {
                    "source": "mlflow_candidate",
                    "batch_id": batch_id,
                    "score": row["score"],
                    "url": row["url"],
                    "url_hash": row["url_hash"],
                    "segment_hash": row["segment_hash"],
                    "context_segment_hash": row["context_segment_hash"],
                    "html_tag": row["html_tag"],
                },
            },
            ensure_ascii=False,
        )
        for row in candidate_rows
    )
    unused_jsonl = "\n".join(
        json.dumps(
            {
                "text": row["text"],
                "label": int(row["pseudo_label"] if row["pseudo_label"] is not None else 0),
                "meta": {
                    "source": "mlflow_unused",
                    "batch_id": batch_id,
                    "score": row["score"],
                    "url": row["url"],
                    "url_hash": row["url_hash"],
                    "segment_hash": row["segment_hash"],
                    "context_segment_hash": row["context_segment_hash"],
                    "html_tag": row["html_tag"],
                    "verification_status": row["verification_status"],
                },
            },
            ensure_ascii=False,
        )
        for row in unused_rows
    )

    training_config_yaml = (
        "model: phobert\n"
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
        "target_max_test_stage": MLFLOW_THRESHOLD_TARGET_MAX,
    }

    manifest_json = {
        "artifact_type": "mlflow_training_bundle",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "batch_id": batch_id,
        "record_count": len(accepted_rows),
        "candidate_count": len(candidate_rows),
        "unused_count": len(unused_rows),
        "include_unused": request.include_unused,
        "unused_scope": request.unused_scope,
        **versions,
        "required_zip_contents": build_mlflow_required_bundle_contents(),
    }

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset/accepted_pseudo.jsonl", accepted_jsonl + ("\n" if accepted_jsonl else ""))
        zf.writestr("dataset/candidates_unverified.jsonl", candidate_jsonl + ("\n" if candidate_jsonl else ""))
        if request.include_unused:
            zf.writestr("dataset/unused_discarded.jsonl", unused_jsonl + ("\n" if unused_jsonl else ""))
        zf.writestr("manifest.json", json.dumps(manifest_json, ensure_ascii=False, indent=2))
        zf.writestr("config/training_config.yaml", training_config_yaml)
        zf.writestr("config/gate_policy.json", json.dumps(gate_policy_json, ensure_ascii=False, indent=2))

    return {
        "bundle_path": str(out_path.relative_to(BASE_DIR)),
        "count": len(accepted_rows),
        "candidate_count": len(candidate_rows),
        "unused_count": len(unused_rows),
        "include_unused": request.include_unused,
        "unused_scope": request.unused_scope,
        "required_zip_contents": build_mlflow_required_bundle_contents(),
        "artifact_versions": versions,
    }


@app.post("/api/mlflow/manual/import-artifact")
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
                request.artifact_path.strip(),
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


@app.post("/api/mlflow/do/trigger")
def mlflow_do_trigger(request: MlflowDOTriggerRequest) -> Dict[str, Any]:
    init_feedback_db()
    run_id = f"do_{uuid.uuid4().hex[:12]}"
    stages = [
        "trigger_vm_gpu",
        "upload_data_and_train_files",
        "train",
        "save_artifact",
        "download_or_upload_destination",
        "destroy_vm",
    ]
    logs = [
        "Placeholder flow only. No infrastructure action executed.",
        "Configure DigitalOcean credentials before enabling real automation.",
    ]
    now = datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO mlflow_do_run (
                run_id, batch_id, provider, gpu_profile, status, current_stage, logs_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                request.batch_id,
                request.provider,
                request.gpu_profile,
                "placeholder",
                stages[0],
                json.dumps(logs, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()

    return {
        "run_id": run_id,
        "status": "placeholder",
        "stages": stages,
        "dry_run": request.dry_run,
    }


@app.get("/api/mlflow/do/status")
def mlflow_do_status(run_id: str = Query(..., min_length=1)) -> Dict[str, Any]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT run_id, batch_id, provider, gpu_profile, status, current_stage, logs_json, created_at, updated_at
            FROM mlflow_do_run
            WHERE run_id = ?
            """,
            (run_id.strip(),),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"DO run not found: {run_id}")

    logs = []
    raw_logs = row["logs_json"]
    if raw_logs:
        try:
            parsed = json.loads(raw_logs)
            if isinstance(parsed, list):
                logs = parsed
        except Exception:
            logs = []

    return {
        "run_id": row["run_id"],
        "batch_id": row["batch_id"],
        "provider": row["provider"],
        "gpu_profile": row["gpu_profile"],
        "status": row["status"],
        "current_stage": row["current_stage"],
        "logs": logs,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.get("/api/mlflow/compare/latest")
def mlflow_compare_latest() -> Dict[str, Any]:
    init_feedback_db()
    registry = load_json_file(EXPERIMENT_REGISTRY_PATH, {"runs": []})
    runs = registry.get("runs") if isinstance(registry, dict) else []
    current_run = runs[0] if isinstance(runs, list) and runs else {}

    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        artifact_row = conn.execute(
            """
            SELECT id, run_name, artifact_path, notes, created_at
            FROM mlflow_training_artifact
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        tracker_row = conn.execute(
            """
            SELECT scenario_name, macro_f1, f1_toxic, val_loss, created_at
            FROM training_tracker_result
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    current_metrics = {
        "f1_toxic": normalize_score((current_run or {}).get("metrics", {}).get("f1_toxic") if isinstance(current_run, dict) else None),
        "macro_f1": normalize_score((current_run or {}).get("metrics", {}).get("f1") if isinstance(current_run, dict) else None),
        "val_loss": None,
    }
    candidate_metrics = {
        "f1_toxic": normalize_score(tracker_row["f1_toxic"]) if tracker_row else None,
        "macro_f1": normalize_score(tracker_row["macro_f1"]) if tracker_row else None,
        "val_loss": normalize_score(tracker_row["val_loss"]) if tracker_row else None,
    }

    f1_delta = None
    macro_delta = None
    if current_metrics["f1_toxic"] is not None and candidate_metrics["f1_toxic"] is not None:
        f1_delta = float(candidate_metrics["f1_toxic"]) - float(current_metrics["f1_toxic"])
    if current_metrics["macro_f1"] is not None and candidate_metrics["macro_f1"] is not None:
        macro_delta = float(candidate_metrics["macro_f1"]) - float(current_metrics["macro_f1"])

    gate_checks = [
        {
            "name": "f1_toxic delta >= 0",
            "delta": f1_delta,
            "passed": bool(f1_delta is not None and f1_delta >= 0.0),
        },
        {
            "name": "macro delta >= -0.01",
            "delta": macro_delta,
            "passed": bool(macro_delta is not None and macro_delta >= -0.01),
        },
        {
            "name": "candidate f1_toxic >= 0.45",
            "delta": candidate_metrics["f1_toxic"],
            "passed": bool(candidate_metrics["f1_toxic"] is not None and candidate_metrics["f1_toxic"] >= 0.45),
        },
    ]

    current_model = (current_run or {}).get("model_name") if isinstance(current_run, dict) else None
    candidate_model = artifact_row["run_name"] if artifact_row else None

    return {
        "current": {
            "model": current_model,
            "metrics": current_metrics,
            "created_at": (current_run or {}).get("created_at") if isinstance(current_run, dict) else None,
        },
        "candidate": {
            "model": candidate_model,
            "artifact_path": artifact_row["artifact_path"] if artifact_row else None,
            "notes": artifact_row["notes"] if artifact_row else None,
            "metrics": candidate_metrics,
            "created_at": artifact_row["created_at"] if artifact_row else None,
        },
        "gate_checks": gate_checks,
        "promotion_enabled": False,
        "promotion_mode": "placeholder",
    }


@app.post("/api/mlflow/promote")
def mlflow_promote(request: MlflowPromoteRequest) -> Dict[str, Any]:
    return {
        "status": "placeholder",
        "candidate_model": request.candidate_model,
        "message": "Promotion workflow is not enabled yet.",
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    try:
        cleanup_old_jobs(float(os.getenv("JOB_RETENTION_HOURS", "24")))

        options = request.options or AnalyzeOptions()

        urls = [u.strip() for u in request.urls if u and u.strip()]
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")
        job_id = uuid.uuid4().hex
        out_dir = BASE_DIR / "data" / "processed" / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

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

        crawl_results = crawl_urls(urls, out_dir=str(DATA_DIR))

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

        logger.info("Job %s: completed", job_id)
        return {
            "job_id": job_id,
            "flow_state": "completed",
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
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}")


@app.get("/api/models")
def get_models() -> Dict[str, Any]:
    try:
        model_root = resolve_model_root()
        models = list_all_models(model_root)
        return {
            "models": [m["id"] for m in models],
            "default": get_default_model_id(model_root),
        }
    except (PermissionError, OSError, NotADirectoryError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {exc}") from exc


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

        urls = [u.strip() for u in request.urls if u and u.strip()]
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")
        model_ids = [m.strip() for m in options.model_names if m and m.strip()]
        if len(model_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 model_names")
        job_id = uuid.uuid4().hex
        out_dir = BASE_DIR / "data" / "processed" / f"job_{job_id}"
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

        crawl_results = crawl_urls(urls, out_dir=str(DATA_DIR))

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

        source_dir = BASE_DIR / "data" / "processed" / f"job_{job_id}"
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
        out_dir = BASE_DIR / "data" / "processed" / f"job_{rerun_job_id}"
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

    out_dir = BASE_DIR / "data" / "processed"
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
        "artifact_path": str(out_path.relative_to(BASE_DIR)),
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
        "path": str(out_path.relative_to(BASE_DIR)),
        "artifact_path": str(out_path.relative_to(BASE_DIR)),
        "manifest_path": str(manifest_path.relative_to(BASE_DIR)),
        "count": len(filtered),
        "stats": stats,
        "artifact_versions": versions,
    }


@app.post("/api/dataset/synthetic/generate")
def synthetic_generate(request: SyntheticGenerateRequest) -> Dict[str, Any]:
    target_count = int(request.count)
    expected_label = int(request.label)
    domain = request.domain
    style = request.style
    model_name = normalize_gemini_model_name(request.model) or os.getenv("GEMINI_MODEL", SYNTHETIC_FALLBACK_MODEL)

    existing_rows = load_synthetic_rows(domain=domain, style=style, label=expected_label)
    seen_hashes = {build_text_hash(row.get("text") or "") for row in existing_rows}
    seen_fingerprints = {build_structure_fingerprint(row.get("text") or "") for row in existing_rows}

    accepted_rows: List[Dict[str, Any]] = []
    total_rejected_placeholder = 0
    total_rejected_duplicate = 0
    total_candidates_seen = 0
    total_rejected_invalid = 0

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
            length_guidance=build_length_bucket_guidance(remaining_targets, length_bounds),
        )
        llm_raw = call_gemini_with_model(prompt, model_name)
        candidates = parse_json_array_from_llm(llm_raw)

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


@app.get("/api/dataset/synthetic/preview")
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
    rows = load_synthetic_rows(
        batch_id=batch_id,
        domain=domain,
        style=style,
        label=label,
        accepted=accepted,
        reviewed=reviewed,
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


@app.post("/api/dataset/synthetic/review")
def synthetic_review(request: SyntheticReviewRequest) -> Dict[str, Any]:
    updates = [
        {
            "id": item.id,
            "is_accepted": item.is_accepted,
            "text": item.text,
            "label": item.label,
        }
        for item in request.updates
    ]
    updated = update_synthetic_review(updates)
    return {"updated": updated}


@app.post("/api/dataset/synthetic/delete")
def synthetic_delete(request: SyntheticDeleteRequest) -> Dict[str, Any]:
    deleted = delete_synthetic_rows(request.ids)
    return {"deleted": deleted}


@app.get("/api/dataset/synthetic/stats")
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


@app.post("/api/dataset/synthetic/export")
def synthetic_export(request: SyntheticExportRequest) -> Dict[str, Any]:
    rows = load_synthetic_rows(
        batch_id=request.batch_id,
        domain=request.domain,
        style=request.style,
        label=request.label,
        accepted=True if request.accepted_only else None,
        reviewed=True,
    )

    export_rows = [{"text": row["text"], "label": row["label"], "meta": row.get("meta") or {}} for row in rows]

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in export_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "path": str(out_path.relative_to(BASE_DIR)),
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


def _parse_optional_seed(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("-"):
            digits = raw[1:]
            if digits.isdigit():
                return int(raw)
        elif raw.isdigit():
            return int(raw)
    return None


def _collect_protocol_seed_runs(protocol_id: str) -> List[Dict[str, Any]]:
    prefix = f"protocol_{protocol_id}"
    runs: List[Dict[str, Any]] = []

    if not PROTOCOL_METRICS_ROOT.exists():
        return runs

    for candidate in sorted(PROTOCOL_METRICS_ROOT.iterdir(), key=lambda p: p.name):
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue

        metrics_path = candidate / "results" / "metrics.json"
        metrics_json = load_json_file(metrics_path, {})
        if not isinstance(metrics_json, dict):
            continue

        final_metrics = metrics_json.get("final_test_rich")
        if not isinstance(final_metrics, dict):
            continue

        run_config = load_json_file(candidate / "models" / "best" / "run_config.json", {})
        seed = None
        if isinstance(run_config, dict):
            seed = _parse_optional_seed((run_config.get("config") or {}).get("SEED"))
        if seed is None:
            seed = _parse_optional_seed((metrics_json.get("hyperparameters") or {}).get("SEED"))

        runs.append(
            {
                "run_key": candidate.name,
                "run_id": metrics_json.get("run_id") if isinstance(metrics_json.get("run_id"), str) else candidate.name,
                "seed": seed,
                "macro_f1": final_metrics.get("macro_f1"),
                "f1_toxic": final_metrics.get("f1_toxic"),
                "accuracy": final_metrics.get("accuracy"),
                "ece": final_metrics.get("ece"),
                "brier": final_metrics.get("brier"),
                "metrics_last_updated": file_last_updated(metrics_path),
            }
        )

    runs.sort(
        key=lambda r: (
            -1 if isinstance(r.get("f1_toxic"), (int, float)) else 1,
            -(r.get("f1_toxic") or -1),
            -(r.get("macro_f1") or -1),
            r.get("seed") is None,
            r.get("seed") if isinstance(r.get("seed"), int) else 0,
            r.get("run_key") or "",
        )
    )
    return runs


@app.get("/api/protocols/summary")
def protocol_summary() -> Dict[str, Any]:
    report = load_json_file(PROTOCOL_BUILD_REPORT_PATH, {})
    report_protocols = report.get("protocols") if isinstance(report, dict) else {}

    dataset_version = (
        (report.get("config") or {}).get("dataset_prefix") if isinstance(report, dict) else None
    ) or DEFAULT_DATASET_VERSION
    versions = build_artifact_versions(
        dataset_version=dataset_version,
        model_version=DEFAULT_MODEL_VERSION,
        policy_version=DEFAULT_POLICY_VERSION,
    )
    missing_versions = find_missing_required_versions(versions)

    warnings: List[str] = []
    if missing_versions:
        warnings.append(f"Missing required version metadata: {missing_versions}")

    protocols: List[Dict[str, Any]] = []

    for protocol_id in ["a", "b", "c"]:
        report_entry = report_protocols.get(protocol_id, {}) if isinstance(report_protocols, dict) else {}
        stats = report_entry.get("stats", {}) if isinstance(report_entry, dict) else {}
        overlap = report_entry.get("overlap_exact", {}) if isinstance(report_entry, dict) else {}

        metrics_path = (
            PROTOCOL_METRICS_ROOT
            / f"protocol_{protocol_id}"
            / "results"
            / "metrics.json"
        )
        metrics_json = load_json_file(metrics_path, {})
        final_metrics = metrics_json.get("final_test_rich", {}) if isinstance(metrics_json, dict) else {}

        available = bool(final_metrics)
        if not available:
            warnings.append(f"Missing or invalid metrics for protocol_{protocol_id}: {metrics_path}")

        seed_runs = _collect_protocol_seed_runs(protocol_id)
        macro_values = [
            float(run["macro_f1"]) for run in seed_runs if isinstance(run.get("macro_f1"), (int, float))
        ]
        f1_toxic_values = [
            float(run["f1_toxic"]) for run in seed_runs if isinstance(run.get("f1_toxic"), (int, float))
        ]

        train_stats = stats.get("train") if isinstance(stats, dict) else None
        source_mix_by_split = {
            "train": (train_stats.get("sources") if isinstance(train_stats, dict) else {}) or {},
            "validation": ((stats.get("validation") or {}).get("sources") if isinstance(stats, dict) else {}) or {},
            "test": ((stats.get("test") or {}).get("sources") if isinstance(stats, dict) else {}) or {},
        }

        overlap_exact = overlap if isinstance(overlap, dict) else {}
        train_validation_overlap = int(overlap_exact.get("train_validation") or 0)
        train_test_overlap = int(overlap_exact.get("train_test") or 0)
        validation_test_overlap = int(overlap_exact.get("validation_test") or 0)

        leakage_evidence = {
            "train_validation": train_validation_overlap,
            "train_test": train_test_overlap,
            "validation_test": validation_test_overlap,
            "has_train_test_leakage": train_test_overlap > 0,
            "has_any_overlap": (train_validation_overlap + train_test_overlap + validation_test_overlap) > 0,
        }

        domain_mismatch = build_domain_mismatch_note(protocol_id, train_stats if isinstance(train_stats, dict) else None)

        protocols.append(
            {
                "id": protocol_id,
                "name": f"Protocol {protocol_id.upper()}",
                "available": available,
                "metrics": {
                    "macro_f1": final_metrics.get("macro_f1"),
                    "f1_toxic": final_metrics.get("f1_toxic"),
                    "accuracy": final_metrics.get("accuracy"),
                    "ece": final_metrics.get("ece"),
                    "brier": final_metrics.get("brier"),
                    "threshold": final_metrics.get("threshold"),
                    "support_clean": final_metrics.get("support_clean"),
                    "support_toxic": final_metrics.get("support_toxic"),
                },
                "stats": {
                    "train": train_stats,
                    "validation": stats.get("validation") if isinstance(stats, dict) else None,
                    "test": stats.get("test") if isinstance(stats, dict) else None,
                },
                "source_mix_by_split": source_mix_by_split,
                "overlap_exact": overlap_exact,
                "leakage_evidence": leakage_evidence,
                "domain_mismatch": domain_mismatch,
                "artifact_versions": versions,
                "metrics_last_updated": file_last_updated(metrics_path),
                "seed_runs": seed_runs,
                "seed_summary": {
                    "n_runs": len(seed_runs),
                    "n_with_seed": sum(1 for run in seed_runs if isinstance(run.get("seed"), int)),
                    "macro_f1_mean": statistics.fmean(macro_values) if macro_values else None,
                    "macro_f1_std": statistics.stdev(macro_values) if len(macro_values) >= 2 else None,
                    "f1_toxic_mean": statistics.fmean(f1_toxic_values) if f1_toxic_values else None,
                    "f1_toxic_std": statistics.stdev(f1_toxic_values) if len(f1_toxic_values) >= 2 else None,
                },
            }
        )

    best_protocol = None
    scored = [
        p for p in protocols
        if p.get("available") and isinstance((p.get("metrics") or {}).get("f1_toxic"), (int, float))
    ]
    if scored:
        best_protocol = max(
            scored,
            key=lambda p: (
                (p.get("metrics") or {}).get("f1_toxic", float("-inf")),
                (p.get("metrics") or {}).get("macro_f1", float("-inf")),
            ),
        ).get("id")

    if isinstance(report, dict):
        report_warnings = report.get("warnings")
        if isinstance(report_warnings, list):
            warnings.extend(str(w) for w in report_warnings)

    return {
        "dataset_version": dataset_version,
        "model_version": versions["model_version"],
        "policy_version": versions["policy_version"],
        "artifact_versions": versions,
        "missing_required_versions": missing_versions,
        "build_report_last_updated": file_last_updated(PROTOCOL_BUILD_REPORT_PATH),
        "protocols": protocols,
        "winner": best_protocol,
        "warnings": warnings,
        "source_note": (
            "Source: data/victsd/*_protocol_build_report.json + "
            "viettoxic_outputs/protocol_{a,b,c}/results/metrics.json (final_test_rich)"
        ),
    }
