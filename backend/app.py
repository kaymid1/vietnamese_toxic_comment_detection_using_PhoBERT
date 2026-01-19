import csv
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from setup_and_crawl import crawl_urls
from infer_crawled_local import infer_crawled

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw" / "crawled_urls"
DEFAULT_MODEL_PATH = BASE_DIR / "models_2" / "phobert" / "new"

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
    model_path: Optional[str] = None


class AnalyzeRequest(BaseModel):
    urls: List[str] = Field(min_items=1)
    options: Optional[AnalyzeOptions] = None


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

        model_path = Path(options.model_path).expanduser() if options.model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Model path not found: {model_path}",
            )

        logger.info("Job %s: start analyze for %s urls", job_id, len(urls))
        logger.info("Job %s: crawling", job_id)
        crawl_results = crawl_urls(urls, out_dir=str(DATA_DIR))

        ok_hashes = [r["url_hash"] for r in crawl_results if r.get("status") == "ok"]

        if ok_hashes:
            logger.info("Job %s: running inference on %s crawled urls", job_id, len(ok_hashes))
            infer_crawled(
                model_path=str(model_path),
                data_dir=str(DATA_DIR),
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
                status = "error"
                error = "segments.jsonl not found after crawl"

            page_info = None
            if status == "ok":
                page_info = page_by_hash.get(url_hash) or page_by_url.get(url)
                if not page_info:
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
                    }
                )

            response_results.append(
                {
                    "url": url,
                    "status": status,
                    "error": error,
                    "crawl_output_dir": to_relative(crawl.get("output_dir")),
                    "segments_path": to_relative(segments_path),
                    "toxicity": {
                        "overall": overall,
                        "by_segment": by_segment,
                    },
                }
            )

        logger.info("Job %s: completed", job_id)
        return {"job_id": job_id, "results": response_results}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}")
