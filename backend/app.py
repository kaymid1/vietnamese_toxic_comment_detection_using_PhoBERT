import csv
import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from domain_classifier import CATEGORY_THRESHOLDS
from setup_and_crawl import crawl_urls
from infer_crawled_local import infer_crawled

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


app = FastAPI(title="VietToxic Local API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.ngrok-free\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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


class AnalyzeRequest(BaseModel):
    urls: List[str] = Field(min_items=1)
    options: Optional[AnalyzeOptions] = None


class FeedbackPageItem(BaseModel):
    url: str
    url_hash: str
    domain_category: str
    domain_override: Optional[str] = None
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
    domain_category: str
    domain_override: Optional[str] = None
    segment_id: str
    text: str
    score: Optional[float] = None
    seg_threshold_used: Optional[float] = None
    label: str


class SegmentFeedbackRequest(BaseModel):
    job_id: str
    items: List[SegmentFeedbackItem] = Field(min_items=1)


class ThresholdPreviewRequest(BaseModel):
    model_id: str
    min_samples: int = Field(default=10, ge=1)


class ThresholdApplyRequest(BaseModel):
    model_id: str
    suggested_thresholds: Dict[str, float]
    ema_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    min_samples_apply: int = Field(default=10, ge=1)
    max_delta: float = Field(default=0.03, ge=0.0, le=1.0)


class ThresholdCurrentRequest(BaseModel):
    model_id: str


class ThresholdResetRequest(BaseModel):
    model_id: str
    categories: List[str] = Field(min_items=1)


class DatasetExportRequest(BaseModel):
    source: Optional[List[str]] = None
    label: Optional[List[int]] = None
    split: Optional[List[str]] = None


class FeedbackDeleteRequest(BaseModel):
    ids: List[int] = Field(min_items=1)


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
    domain_category: Optional[str] = None
    overall: Optional[float] = None
    thresholds: Optional[Dict[str, float]] = None
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    question: Optional[str] = None


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
                    "domain_category": seg.get("domain_category"),
                    "seg_threshold_used": normalize_score(seg.get("seg_threshold_used")),
                }
            )

        response_results.append(
            {
                "url": url,
                "url_hash": url_hash,
                "status": status,
                "error": error,
                "warnings": crawl.get("warnings") or [],
                "crawl_output_dir": to_relative(crawl.get("output_dir")),
                "segments_path": to_relative(segments_path),
                "videos": load_video_results(url_hash),
                "domain_category": page_info.get("domain_category") if page_info else None,
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
    if status_code == 429:
        return True
    lowered = detail.lower()
    if "resource_exhausted" in lowered or "rate limit" in lowered or "quota" in lowered:
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


def load_video_results(url_hash: str) -> List[Dict[str, Any]]:
    video_path = DATA_DIR / url_hash / "video_data.jsonl"
    if not video_path.exists():
        return []
    results: List[Dict[str, Any]] = []
    with video_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def build_merged_segments(
    url_hash: str,
    data_dir: Path,
    merged_root: Path,
) -> bool:
    src_folder = data_dir / url_hash
    seg_path = src_folder / "segments.jsonl"
    meta_path = src_folder / "meta.json"
    video_path = src_folder / "video_data.jsonl"
    if not (seg_path.exists() and meta_path.exists()):
        return False

    merged_folder = merged_root / url_hash
    merged_folder.mkdir(parents=True, exist_ok=True)

    try:
        meta_text = meta_path.read_text(encoding="utf-8")
        (merged_folder / "meta.json").write_text(meta_text, encoding="utf-8")
    except Exception:
        return False

    segments: List[str] = []
    try:
        with seg_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if text:
                        segments.append(text)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return False

    if video_path.exists():
        try:
            with video_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    transcript = obj.get("transcript") or []
                    for seg in transcript:
                        text = seg.get("text") if isinstance(seg, dict) else None
                        if text:
                            segments.append(text)
        except Exception:
            pass

    if not segments:
        return False

    try:
        with (merged_folder / "segments.jsonl").open("w", encoding="utf-8") as f:
            for text in segments:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    except Exception:
        return False

    return True


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


def ensure_table_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_feedback_db() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_page (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                url TEXT NOT NULL,
                url_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                domain_category TEXT NOT NULL,
                domain_override TEXT,
                seg_threshold_used REAL,
                score_overall REAL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threshold_overrides (
                model_id TEXT NOT NULL,
                domain_category TEXT NOT NULL,
                threshold REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (model_id, domain_category)
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
                domain_category TEXT NOT NULL,
                domain_override TEXT,
                segment_id TEXT NOT NULL,
                text TEXT NOT NULL,
                score REAL,
                seg_threshold_used REAL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        ensure_table_column(conn, "feedback_page", "domain_override", "TEXT")
        ensure_table_column(conn, "feedback_segment", "domain_override", "TEXT")


def insert_feedback_page(items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    init_feedback_db()
    now = datetime.utcnow().isoformat()
    rows = [
        (
            item["job_id"],
            item["url"],
            item["url_hash"],
            item["model_id"],
            item["domain_category"],
            item.get("domain_override"),
            item.get("seg_threshold_used"),
            item.get("score_overall"),
            item["label"],
            now,
        )
        for item in items
    ]
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO feedback_page (
                job_id, url, url_hash, model_id, domain_category, domain_override,
                seg_threshold_used, score_overall, label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def insert_feedback_segment(items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    init_feedback_db()
    now = datetime.utcnow().isoformat()
    rows = [
        (
            item["job_id"],
            item["url"],
            item["url_hash"],
            item["model_id"],
            item["domain_category"],
            item.get("domain_override"),
            item["segment_id"],
            item["text"],
            item.get("score"),
            item.get("seg_threshold_used"),
            item["label"],
            now,
        )
        for item in items
    ]
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO feedback_segment (
                job_id, url, url_hash, model_id, domain_category, domain_override, segment_id,
                text, score, seg_threshold_used, label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_threshold_overrides(model_id: str) -> Dict[str, float]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT domain_category, threshold
            FROM threshold_overrides
            WHERE model_id = ?
            """,
            (model_id,),
        ).fetchall()
    return {row[0]: float(row[1]) for row in rows}


def delete_threshold_overrides(model_id: str, categories: List[str]) -> int:
    if not categories:
        return 0
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        placeholders = ", ".join(["?"] * len(categories))
        cursor = conn.execute(
            f"DELETE FROM threshold_overrides WHERE model_id = ? AND domain_category IN ({placeholders})",
            (model_id, *categories),
        )
        conn.commit()
        return cursor.rowcount or 0


def save_threshold_overrides(model_id: str, values: Dict[str, float]) -> None:
    if not values:
        return
    init_feedback_db()
    now = datetime.utcnow().isoformat()
    rows = [(model_id, cat, float(thr), now) for cat, thr in values.items()]
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO threshold_overrides (model_id, domain_category, threshold, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_id, domain_category)
            DO UPDATE SET threshold = excluded.threshold, updated_at = excluded.updated_at
            """,
            rows,
        )
        conn.commit()


def get_effective_thresholds(model_id: str) -> Dict[str, float]:
    overrides = load_threshold_overrides(model_id)
    return {**CATEGORY_THRESHOLDS, **overrides}


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


def resolve_effective_category(domain_category: Optional[str], domain_override: Optional[str]) -> Optional[str]:
    if domain_override and domain_override in CATEGORY_THRESHOLDS:
        return domain_override
    if domain_category and domain_category in CATEGORY_THRESHOLDS:
        return domain_category
    return None


def collect_threshold_feedback(model_id: str) -> Dict[str, List[Tuple[float, int]]]:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT domain_category, domain_override, score_overall, label
            FROM feedback_page
            WHERE model_id = ?
            """,
            (model_id,),
        ).fetchall()

    grouped: Dict[str, List[Tuple[float, int]]] = {}
    for domain_category, domain_override, score_overall, label in rows:
        normalized = safe_label(label)
        if normalized not in {"toxic", "clean"} or score_overall is None:
            continue
        effective = resolve_effective_category(domain_category, domain_override)
        if not effective:
            continue
        grouped.setdefault(effective, []).append(
            (float(score_overall), 1 if normalized == "toxic" else 0)
        )
    return grouped


def preview_thresholds(model_id: str, min_samples: int = 10) -> Dict[str, Any]:
    grouped = collect_threshold_feedback(model_id)

    suggestions: Dict[str, float] = {}
    stats: Dict[str, Any] = {}

    for category, pairs in grouped.items():
        if len(pairs) < min_samples:
            stats[category] = {
                "count": len(pairs),
                "status": "insufficient_samples",
            }
            continue

        scores = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        thresholds = sorted(set(scores))
        best = {"f1": -1.0, "thr": None, "precision": 0.0, "recall": 0.0}

        for thr in thresholds:
            tp = fp = fn = 0
            for score, label in pairs:
                pred = 1 if score >= thr else 0
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = compute_f1(precision, recall)
            if f1 > best["f1"]:
                best = {"f1": f1, "thr": thr, "precision": precision, "recall": recall}

        if best["thr"] is not None:
            suggestions[category] = float(best["thr"])
            stats[category] = {
                "count": len(pairs),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
            }

    return {"suggested_thresholds": suggestions, "stats": stats}


def apply_thresholds(
    model_id: str,
    suggested: Dict[str, float],
    ema_weight: float = 0.8,
    min_samples_apply: int = 10,
    max_delta: float = 0.03,
) -> Dict[str, float]:
    effective = get_effective_thresholds(model_id)
    grouped = collect_threshold_feedback(model_id)
    updates: Dict[str, float] = {}
    for category, suggested_value in suggested.items():
        if category not in CATEGORY_THRESHOLDS:
            continue
        if len(grouped.get(category, [])) < min_samples_apply:
            continue
        current = effective.get(category, CATEGORY_THRESHOLDS[category])
        ema_value = (ema_weight * current) + ((1 - ema_weight) * float(suggested_value))
        if max_delta > 0:
            lower = current - max_delta
            upper = current + max_delta
            ema_value = min(max(ema_value, lower), upper)
        updates[category] = round(ema_value, 4)
    if updates:
        save_threshold_overrides(model_id, updates)
    return get_effective_thresholds(model_id)


def iter_dataset_files() -> List[Tuple[Path, str, bool]]:
    dataset_dir = BASE_DIR / "data" / "victsd"
    if not dataset_dir.exists():
        return []
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


def iter_dataset_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path, split, is_augmented in iter_dataset_files():
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
            SELECT id, text, label, model_id, domain_category, domain_override, score, seg_threshold_used, created_at
            FROM feedback_segment
            ORDER BY id DESC
            """
        ).fetchall()

    for feedback_id, text, label, model_id, domain_category, domain_override, score, seg_threshold_used, created_at in results:
        label_int = safe_label_int(label)
        if label_int is None:
            continue
        meta = {
            "source": "new_collected",
            "split": "feedback",
            "is_augmented": False,
            "feedback_id": feedback_id,
            "model_id": model_id,
            "domain_category": domain_category,
            "domain_override": domain_override,
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

        thresholds_by_domain = get_effective_thresholds(model_id)

        logger.info("Job %s: start analyze for %s urls", job_id, len(urls))
        logger.info("Job %s: using model '%s' (%s) from %s", job_id, model_id, model_type, model_path)
        logger.info("Job %s: crawling", job_id)
        crawl_results = crawl_urls(urls, out_dir=str(DATA_DIR), enable_video=options.enable_video)
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
        if options.enable_video and ok_hashes:
            merged_root = out_dir / "merged_crawl"
            merged_root.mkdir(parents=True, exist_ok=True)
            merged_ok: List[str] = []
            for h in ok_hashes:
                if build_merged_segments(h, DATA_DIR, merged_root):
                    merged_ok.append(h)
            if merged_ok:
                logger.info("Job %s: using merged segments (text+video) for %s urls", job_id, len(merged_ok))
                infer_data_dir = merged_root
                ok_hashes = merged_ok
                merged_used = True

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=job_id,
                urls=urls,
                url_hashes=ok_hashes,
                model_ids=[model_id],
                enable_video=options.enable_video,
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
            )
        else:
            logger.warning("Job %s: no successful crawls to run inference", job_id)

        page_results = load_page_results(out_dir)
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
                "domain_category": item.domain_category,
                "domain_override": item.domain_override,
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

        job_id = uuid.uuid4().hex
        out_dir = BASE_DIR / "data" / "processed" / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_root = resolve_model_root()
        model_ids = [m.strip() for m in options.model_names if m and m.strip()]
        if len(model_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 model_names")

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
        logger.info("Compare job %s: crawling", job_id)
        crawl_results = crawl_urls(urls, out_dir=str(DATA_DIR), enable_video=options.enable_video)

        ok_hashes = [r["url_hash"] for r in crawl_results if r.get("status") == "ok"]
        infer_data_dir = DATA_DIR
        merged_used = False
        if options.enable_video and ok_hashes:
            merged_root = out_dir / "merged_crawl"
            merged_root.mkdir(parents=True, exist_ok=True)
            merged_ok: List[str] = []
            for h in ok_hashes:
                if build_merged_segments(h, DATA_DIR, merged_root):
                    merged_ok.append(h)
            if merged_ok:
                infer_data_dir = merged_root
                ok_hashes = merged_ok
                merged_used = True

        save_job_meta(
            out_dir,
            build_job_meta(
                job_id=job_id,
                urls=urls,
                url_hashes=ok_hashes,
                model_ids=model_ids,
                enable_video=options.enable_video,
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
                enable_video=options.enable_video,
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
    if request.domain_category:
        prompt_lines.append(f"Domain category: {request.domain_category}")
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


@app.post("/api/thresholds/preview")
def thresholds_preview(request: ThresholdPreviewRequest) -> Dict[str, Any]:
    result = preview_thresholds(request.model_id, min_samples=request.min_samples)
    return result


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
                "domain_category": item.domain_category,
                "domain_override": item.domain_override,
                "segment_id": item.segment_id,
                "text": item.text,
                "score": item.score,
                "seg_threshold_used": item.seg_threshold_used,
                "label": normalized,
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


@app.post("/api/thresholds/apply")
def thresholds_apply(request: ThresholdApplyRequest) -> Dict[str, Any]:
    updated = apply_thresholds(
        request.model_id,
        request.suggested_thresholds,
        ema_weight=request.ema_weight,
        min_samples_apply=request.min_samples_apply,
        max_delta=request.max_delta,
    )
    return {"thresholds_by_domain": updated}


@app.get("/api/dataset/preview")
def dataset_preview(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    source: Optional[str] = None,
    label: Optional[int] = Query(default=None, ge=0, le=1),
    split: Optional[str] = None,
    include_stats: bool = False,
) -> Dict[str, Any]:
    rows = iter_dataset_rows() + iter_feedback_rows()
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
    }
    if include_stats:
        payload["stats"] = build_dataset_stats(filtered)
    return payload


@app.post("/api/dataset/export")
def dataset_export(request: DatasetExportRequest) -> Dict[str, Any]:
    rows = iter_dataset_rows() + iter_feedback_rows()
    filtered = filter_dataset_rows(
        rows,
        sources=request.source,
        labels=request.label,
        splits=request.split,
    )
    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = build_dataset_stats(filtered)
    return {"path": str(out_path.relative_to(BASE_DIR)), "count": len(filtered), "stats": stats}


@app.post("/api/thresholds/current")
def thresholds_current(request: ThresholdCurrentRequest) -> Dict[str, Any]:
    current = get_effective_thresholds(request.model_id)
    overrides = load_threshold_overrides(request.model_id)
    return {"thresholds_by_domain": current, "overrides": overrides}


@app.post("/api/thresholds/reset")
def thresholds_reset(request: ThresholdResetRequest) -> Dict[str, Any]:
    delete_threshold_overrides(request.model_id, request.categories)
    current = get_effective_thresholds(request.model_id)
    overrides = load_threshold_overrides(request.model_id)
    return {"thresholds_by_domain": current, "overrides": overrides}


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
