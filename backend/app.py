import csv
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from setup_and_crawl import crawl_urls
from infer_crawled_local import infer_crawled

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "crawled_urls"
MODEL_OPTIONS_DIR = Path(
    os.getenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(BASE_DIR / "models" / "options"))
).expanduser()

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


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    try:
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
                only_url_hashes=ok_hashes,
                quiet=True,
            )
        else:
            logger.warning("Job %s: no successful crawls to run inference", job_id)

        page_results = load_page_results(out_dir)
        segment_results = load_segment_results(out_dir)

        page_by_hash = {r.get("url_hash"): r for r in page_results if r.get("url_hash")}
        page_by_url = {r.get("url"): r for r in page_results if r.get("url")}

        seg_by_hash: Dict[str, List[Dict[str, Any]]] = {}
        seg_by_url: Dict[str, List[Dict[str, Any]]] = {}
        for seg in segment_results:
            if seg.get("url_hash"):
                seg_by_hash.setdefault(seg["url_hash"], []).append(seg)
            if seg.get("url"):
                seg_by_url.setdefault(seg["url"], []).append(seg)

        response_results: List[Dict[str, Any]] = []
        for crawl in crawl_results:
            url = crawl.get("url")
            url_hash = crawl.get("url_hash") or hash_url(url)
            status = crawl.get("status", "error")
            error = crawl.get("error")

            segments_path = crawl.get("segments_path")
            if status == "ok" and (not segments_path or not Path(segments_path).exists()):
                logger.warning(
                    "Job %s: segments missing for url=%s path=%s",
                    job_id,
                    url,
                    segments_path,
                )
                status = "error"
                error = "segments.jsonl not found after crawl"

            page_info = None
            if status == "ok":
                page_info = page_by_hash.get(url_hash) or page_by_url.get(url)
                if not page_info:
                    logger.warning("Job %s: no inference result for url=%s hash=%s", job_id, url, url_hash)
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
                    }
                )

            response_results.append(
                {
                    "url": url,
                    "status": status,
                    "error": error,
                    "crawl_output_dir": to_relative(crawl.get("output_dir")),
                    "segments_path": to_relative(segments_path),
                    "videos": load_video_results(url_hash),
                    "toxicity": {
                        "overall": overall,
                        "by_segment": by_segment,
                    },
                }
            )

        logger.info("Job %s: completed", job_id)
        return {
            "job_id": job_id,
            "model_name": model_id,
            "thresholds": {
                "seg_threshold": options.seg_threshold,
                "page_threshold": options.page_threshold,
            },
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
