import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from domain_classifier import CATEGORY_THRESHOLDS
from setup_and_crawl import crawl_urls
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
                "status": status,
                "error": error,
                "warnings": crawl.get("warnings") or [],
                "crawl_output_dir": to_relative(crawl.get("output_dir")),
                "segments_path": to_relative(segments_path),
                "videos": load_video_results(url_hash),
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
                learned_feedback=load_learned_segments(),
                html_dir=str(DATA_DIR),
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
) -> Dict[str, Any]:
    include_feedback = (split or "").strip().lower() == "feedback" or (source or "").strip().lower() == "new_collected"
    rows = iter_dataset_rows() + (iter_feedback_rows() if include_feedback else [])
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
