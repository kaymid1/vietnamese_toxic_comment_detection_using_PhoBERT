#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
setup_and_crawl.py
- macOS (Apple Silicon/M1) friendly
- Creates venv, installs deps, downloads VnCoreNLP assets, then crawls a curated URL list
Usage:
  python3 setup_and_crawl.py --setup
  source .venv/bin/activate
  python setup_and_crawl.py --crawl
Or all-in-one (will re-exec inside venv):
  python3 setup_and_crawl.py --setup --crawl
"""

import os
import sys
import json
import time
import re
import hashlib
import unicodedata
import subprocess
import random
import glob
import argparse
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
from typing import List, Optional

def ensure_runtime_deps():
    """
    Ensure all required packages are installed into the CURRENT active venv.
    """
    import importlib.util

    required = {
        "trafilatura": "trafilatura>=1.7.0",
        "vncorenlp": "vncorenlp>=1.0.3",
        "tqdm": "tqdm>=4.66.0",
        "pandas": "pandas>=2.0.0",
        "selenium": "selenium>=4.15.0",
        "undetected_chromedriver": "undetected-chromedriver>=3.5.5",
        "requests": "requests>=2.31.0",
        "certifi": "certifi>=2023.11.17",
        "youtube_transcript_api": "youtube-transcript-api>=0.6.2",
        "yt_dlp": "yt-dlp>=2024.12.0",
        "bs4": "beautifulsoup4>=4.12.0",
        "faster_whisper": "faster-whisper>=1.1.0",
    }

    missing = []
    for mod, pkg in required.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)

    if missing:
        print("[INFO] Installing missing packages into current venv:")
        for m in missing:
            print("  -", m)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )


# =========================
# Configuration
# =========================

DEFAULT_REQUIREMENTS = [
    "trafilatura>=1.7.0",
    "vncorenlp>=1.0.3",
    "tqdm>=4.66.0",
    "pandas>=2.0.0",
    "selenium>=4.15.0",
    "undetected-chromedriver>=3.5.5",
    "requests>=2.31.0",
    "certifi>=2023.11.17",
    "youtube-transcript-api>=0.6.2",
    "yt-dlp>=2024.12.0",
    "beautifulsoup4>=4.12.0",
    "faster-whisper>=1.1.0",
]

VNC_JAR_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar"
VNC_VOCAB_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab"
VNC_RDR_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"

VNC_BASE_DIR = "vncorenlp"
VNC_JAR_PATH = os.path.join(VNC_BASE_DIR, "VnCoreNLP-1.1.1.jar")
VNC_VOCAB_PATH = os.path.join(VNC_BASE_DIR, "models", "wordsegmenter", "vi-vocab")
VNC_RDR_PATH = os.path.join(VNC_BASE_DIR, "models", "wordsegmenter", "wordsegmenter.rdr")

DATA_DIR = os.path.join("data", "raw", "crawled_urls")
EXP_DIR = "experiments"
LOG_FILE = os.path.join(EXP_DIR, "crawling_log.md")

URL_LIST = [
    "https://nhandan.vn/video-ha-noi-chu-dong-binh-on-thi-truong-phuc-vu-tet-nguyen-dan-post940462.html",
]

# =========================
# Helpers: shell / python
# =========================

def run(cmd: List[str], check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env)

def is_macos() -> bool:
    return sys.platform == "darwin"

def in_venv() -> bool:
    return (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

# =========================
# Setup: Java, venv, pip deps
# =========================

def check_java() -> bool:
    try:
        cp = subprocess.run(["java", "-version"], capture_output=True, text=True)
        # java -version prints to stderr usually
        out = (cp.stdout or "") + (cp.stderr or "")
        ok = cp.returncode == 0
        if ok:
            print("[OK] Java detected:")
            print(out.strip().splitlines()[0])
        else:
            print("[WARN] Java not available.")
        return ok
    except FileNotFoundError:
        print("[WARN] Java not found in PATH.")
        return False

# =========================
# Download VnCoreNLP assets
# =========================

def download_file(url: str, out_path: str):
    import requests
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def ensure_vncorenlp_assets():
    if not os.path.exists(VNC_JAR_PATH):
        print("[DL] VnCoreNLP jar...")
        download_file(VNC_JAR_URL, VNC_JAR_PATH)
    if not os.path.exists(VNC_VOCAB_PATH):
        print("[DL] VnCoreNLP vocab...")
        download_file(VNC_VOCAB_URL, VNC_VOCAB_PATH)
    if not os.path.exists(VNC_RDR_PATH):
        print("[DL] VnCoreNLP rdr...")
        download_file(VNC_RDR_URL, VNC_RDR_PATH)
    print("[OK] VnCoreNLP assets ready:", VNC_JAR_PATH)

# =========================
# Text processing (same idea as yours)
# =========================

def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.strip().split())

class Segmenter:
    def __init__(self, jar_path: str):
        self.vncorenlp = None
        try:
            from vncorenlp import VnCoreNLP
            self.vncorenlp = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx2g') #wseg only for word segmentation
            print("[OK] VnCoreNLP loaded")
        except Exception as e:
            print(f"[WARN] VnCoreNLP load fail -> fallback regex. Error: {e}")
            self.vncorenlp = None

    def segment_text(self, text: str) -> List[str]:
        if self.vncorenlp:
            try:
                sentences = self.vncorenlp.tokenize(text)
                return [" ".join(sentence) for sentence in sentences]
            except Exception:
                pass
        # Fallback regex
        return [s.strip() for s in re.split(r'[.!?]\s*|\n+\s*|\r+\s*', text) if len(s.strip()) > 10]

# =========================
# Selenium driver (local)
# =========================

def make_driver(timeout: int = 90, headless: bool = True):
    import undetected_chromedriver as uc
    from selenium.webdriver.chrome.options import Options

    def _detect_chrome_major_macos() -> Optional[int]:
        chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        try:
            cp = subprocess.run([chrome_bin, "--version"], capture_output=True, text=True, check=False)
            s = (cp.stdout or "") + (cp.stderr or "")
            m = re.search(r"(\d+)\.", s)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,2000")

    major = _detect_chrome_major_macos()
    print(f"[DEBUG] Detected Chrome major (macOS): {major}")
    if major:
        print(f"[DEBUG] Launching uc.Chrome(version_main={major}, headless={headless})")
        driver = uc.Chrome(options=options, version_main=major, use_subprocess=True)
        driver.set_page_load_timeout(timeout)
        return driver

    try:
        print(f"[DEBUG] Launching uc.Chrome(default), headless={headless}")
        driver = uc.Chrome(options=options, use_subprocess=True)
        driver.set_page_load_timeout(timeout)
        return driver
    except Exception as e:
        msg = str(e)
        m = re.search(r"Current browser version is (\d+)\.", msg)
        if m:
            cur_major = int(m.group(1))
            print(f"[DEBUG] Retry uc.Chrome(version_main={cur_major}) after mismatch")
            driver = uc.Chrome(options=options, version_main=cur_major, use_subprocess=True)
            driver.set_page_load_timeout(timeout)
            return driver
        raise

# =========================
# Hybrid crawl
# =========================

def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def _extract_youtube_id_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if "youtu.be" in host:
        vid = path.strip("/").split("/")[0]
        return vid or None

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        if path.startswith("/embed/"):
            return path.split("/embed/")[-1].split("/")[0] or None
        if path.startswith("/watch"):
            qs = parse_qs(parsed.query or "")
            v = qs.get("v", [None])[0]
            return v or None

    return None

def _is_ad_url(url: str) -> bool:
    bad = [
        "doubleclick",
        "googlesyndication",
        "adservice",
        "adsystem",
        "imasdk",
        "vast",
        "prebid",
        "criteo",
        "taboola",
        "outbrain",
    ]
    low = (url or "").lower()
    return any(b in low for b in bad)

def _detect_videos_from_html(html: str, page_url: str) -> dict:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    youtube_ids = set()

    for iframe in soup.find_all("iframe"):
        src = iframe.get("src") or iframe.get("data-src") or ""
        if not src:
            continue
        abs_src = urljoin(page_url, src)
        if _is_ad_url(abs_src):
            continue
        vid = _extract_youtube_id_from_url(abs_src)
        if vid:
            youtube_ids.add(vid)

    for tag in soup.find_all(["a", "link"]):
        href = tag.get("href") or ""
        if not href:
            continue
        abs_href = urljoin(page_url, href)
        if _is_ad_url(abs_href):
            continue
        vid = _extract_youtube_id_from_url(abs_href)
        if vid:
            youtube_ids.add(vid)

    raw_ids = set()
    for m in re.findall(r"(?:youtube(?:-nocookie)?\.com/(?:embed/|watch\?v=)|youtu\.be/)([A-Za-z0-9_-]{6,})", html or ""):
        raw_ids.add(m)

    youtube_ids.update(raw_ids)

    return {
        "youtube_ids": sorted(youtube_ids),
    }

def _fetch_oembed_metadata(video_url: str) -> Optional[dict]:
    import requests

    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _fetch_transcript_with_retry(video_id: str, max_retries: int = 4) -> Optional[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        TooManyRequests,
        CouldNotRetrieveTranscript,
    )

    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(["vi", "en"])
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(["vi", "en"])
                except Exception:
                    transcript = transcript_list.find_transcript(["vi", "en"])
            segments = transcript.fetch()
            return {
                "segments": segments,
                "language": getattr(transcript, "language_code", None),
                "is_generated": getattr(transcript, "is_generated", None),
            }
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            return None
        except (TooManyRequests, CouldNotRetrieveTranscript):
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
            continue
        except Exception:
            return None
    return None

def _parse_vtt_to_segments(vtt_text: str) -> List[dict]:
    segments: List[dict] = []
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")

    def to_seconds(ts: str) -> float:
        h, m, s = ts.split(":")
        sec, ms = s.split(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0

    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = time_re.match(line)
        if match:
            start_s = to_seconds(match.group(1))
            end_s = to_seconds(match.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                segments.append({
                    "text": text,
                    "start": start_s,
                    "duration": max(0.0, end_s - start_s),
                })
        i += 1
    return segments

def _format_upload_date(ud: Optional[str]) -> Optional[str]:
    if not ud:
        return None
    if len(ud) == 8 and ud.isdigit():
        return f"{ud[0:4]}-{ud[4:6]}-{ud[6:8]}"
    return ud

def _run_yt_dlp_video(video_url: str, videos_dir: str, keep_artifacts: bool = False) -> Optional[dict]:
    if keep_artifacts:
        os.makedirs(videos_dir, exist_ok=True)
        work_dir = videos_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="yt_dlp_")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--write-auto-sub",
        "--sub-lang", "vi,en,vi.*?,en.*?",
        "--output", os.path.join(work_dir, "%(id)s.%(ext)s"),
        video_url,
    ]
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        if not keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)
        return None

    info_files = glob.glob(os.path.join(work_dir, "*.info.json"))
    info = None
    if info_files:
        try:
            with open(info_files[0], "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = None

    vtt_files = glob.glob(os.path.join(work_dir, "*.vtt"))
    transcript_segments = []
    if vtt_files:
        try:
            with open(vtt_files[0], "r", encoding="utf-8") as f:
                transcript_segments = _parse_vtt_to_segments(f.read())
        except Exception:
            transcript_segments = []

    if not keep_artifacts:
        shutil.rmtree(work_dir, ignore_errors=True)
        info_files = []
        vtt_files = []

    return {
        "info": info,
        "transcript_segments": transcript_segments,
        "vtt_files": vtt_files,
        "info_files": info_files,
    }

def _run_yt_dlp_page(page_url: str, videos_dir: str, keep_artifacts: bool = False) -> List[dict]:
    if keep_artifacts:
        os.makedirs(videos_dir, exist_ok=True)
    cmd = ["yt-dlp", "--dump-json", "--skip-download", page_url]
    try:
        cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return []

    if cp.returncode != 0:
        return []

    records = []
    lines = [l for l in (cp.stdout or "").splitlines() if l.strip()]
    for idx, line in enumerate(lines):
        try:
            info = json.loads(line)
        except Exception:
            continue
        if keep_artifacts:
            info_path = os.path.join(videos_dir, f"page_fallback.{idx}.info.json")
            try:
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        records.append(info)
    return records

def _select_best_format_url(info: dict) -> Optional[str]:
    if not info:
        return None
    for key in ["requested_formats", "formats"]:
        formats = info.get(key)
        if not formats:
            continue
        candidates = []
        for fmt in formats:
            furl = fmt.get("url")
            if not furl:
                continue
            ext = (fmt.get("ext") or "").lower()
            proto = (fmt.get("protocol") or "").lower()
            candidates.append((ext, proto, furl))
        for ext, proto, furl in candidates:
            if ext in {"m3u8", "mp4"} or "m3u8" in proto:
                return furl
        if candidates:
            return candidates[0][2]
    return info.get("url")

def _build_video_record(
    *,
    video_id: str,
    page_url: str,
    transcript_data: Optional[dict],
    oembed: Optional[dict],
    ytdlp: Optional[dict],
    video_url: str,
    artifacts_kept: bool = False,
) -> dict:
    title = None
    channel = None
    upload_date = None
    duration = None
    view_count = None
    language = None
    has_auto_generated = None
    transcript = []
    metadata = {}
    transcript_source = "none"

    if oembed:
        metadata["oembed"] = oembed
        title = oembed.get("title") or title
        channel = oembed.get("author_name") or channel

    if transcript_data:
        transcript = [
            {
                "text": seg.get("text"),
                "start": float(seg.get("start", 0.0)),
                "duration": float(seg.get("duration", 0.0)),
            }
            for seg in transcript_data.get("segments", [])
        ] or []
        language = transcript_data.get("language")
        is_generated = transcript_data.get("is_generated")
        if is_generated is not None:
            has_auto_generated = bool(is_generated)
        if transcript:
            transcript_source = "youtube_transcript_api"

    if ytdlp and ytdlp.get("info"):
        info = ytdlp["info"]
        metadata["ytdlp"] = info
        title = info.get("title") or title
        channel = info.get("channel") or info.get("uploader") or channel
        upload_date = _format_upload_date(info.get("upload_date"))
        duration = info.get("duration") or duration
        view_count = info.get("view_count") or view_count
        if info.get("language"):
            language = info.get("language")
        if info.get("automatic_captions") is not None:
            has_auto_generated = True

    if ytdlp and ytdlp.get("transcript_segments"):
        transcript = ytdlp.get("transcript_segments") or transcript
        if transcript:
            transcript_source = "yt_dlp_caption"

    return {
        "video_id": video_id,
        "platform": "youtube",
        "video_url": video_url,
        "page_url": page_url,
        "title": title,
        "channel": channel,
        "upload_date": upload_date,
        "duration": duration,
        "view_count": view_count,
        "transcript": transcript,
        "language": language,
        "has_auto_generated": has_auto_generated,
        "metadata": metadata if metadata else {},
        "transcript_source": transcript_source,
        "artifacts_kept": bool(artifacts_kept),
    }

def _run_asr_on_video_url(
    video_url: str,
    language: str = "vi",
    asr_max_seconds: int = 600,
    keep_artifacts: bool = False,
    artifacts_dir: Optional[str] = None,
    known_duration: Optional[float] = None,
) -> dict:
    result: dict = {
        "segments": [],
        "full_segments": [],
        "model": None,
        "device": None,
        "compute_type": None,
        "reason": None,
    }
    if not shutil.which("yt-dlp"):
        print("[WARN] yt-dlp not found; skip ASR")
        result["reason"] = "missing_yt_dlp"
        return result
    if not shutil.which("ffmpeg"):
        print("[WARN] ffmpeg not found; skip ASR")
        result["reason"] = "missing_ffmpeg"
        return result

    try:
        from faster_whisper import WhisperModel
    except Exception:
        print("[WARN] faster-whisper not available; skip ASR")
        result["reason"] = "missing_whisper"
        return result

    if known_duration and asr_max_seconds and known_duration > asr_max_seconds:
        result["reason"] = "too_long"
        return result

    temp_dir_ctx = tempfile.TemporaryDirectory(prefix="asr_tmp_")
    work_dir = temp_dir_ctx.name
    print(f"[DEBUG] ASR temp dir created: {work_dir}")

    media_path = os.path.join(work_dir, "%(id)s.%(ext)s")
    try:
        subprocess.run(
            ["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "-o", media_path, video_url],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        print(f"[WARN] ASR yt-dlp download failed: {e}")
        result["reason"] = "download_failed"
        if temp_dir_ctx:
            temp_dir_ctx.cleanup()
        return result

    media_files = glob.glob(os.path.join(work_dir, "*.*"))
    media_files = [p for p in media_files if not p.endswith(".json") and not p.endswith(".vtt")]
    if not media_files:
        print("[WARN] ASR yt-dlp produced no media")
        result["reason"] = "download_failed"
        if temp_dir_ctx:
            temp_dir_ctx.cleanup()
        return result
    media_file = media_files[0]

    wav_path = os.path.join(work_dir, "audio_16k.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", media_file, "-ac", "1", "-ar", "16000", wav_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        print(f"[WARN] ASR ffmpeg failed: {e}")
        result["reason"] = "ffmpeg_failed"
        if temp_dir_ctx:
            temp_dir_ctx.cleanup()
        return result

    device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_name = os.getenv("ASR_MODEL", "small")
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        language_arg = None if language == "auto" else language
        segments_gen, info = model.transcribe(wav_path, language=language_arg, vad_filter=True, beam_size=5)
    except Exception as e:
        print(f"[WARN] ASR whisper failed: {e}")
        result["reason"] = "whisper_failed"
        if temp_dir_ctx:
            temp_dir_ctx.cleanup()
        return result

    segments = []
    full_segments = []
    for seg in segments_gen:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append({"text": text, "start": float(seg.start), "duration": float(seg.end - seg.start)})
        full_segments.append({
            "text": text,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "avg_logprob": round(float(seg.avg_logprob), 4),
        })

    result.update({
        "segments": segments,
        "full_segments": full_segments,
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "reason": None,
    })

    if keep_artifacts:
        dst_dir = artifacts_dir
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(work_dir):
                src = os.path.join(work_dir, fname)
                if os.path.isfile(src):
                    try:
                        shutil.copy2(src, os.path.join(dst_dir, fname))
                    except Exception:
                        pass
            print(f"[DEBUG] ASR artifacts copied to: {dst_dir}")

    temp_dir_ctx.cleanup()
    print(f"[DEBUG] ASR temp dir cleaned: {work_dir}")
    return result

def _fetch_youtube_video_data(video_id: str, page_url: str, videos_dir: str, keep_artifacts: bool = False) -> dict:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed = _fetch_oembed_metadata(video_url)
    transcript_data = _fetch_transcript_with_retry(video_id)

    ytdlp = None
    if transcript_data is None or oembed is None:
        ytdlp = _run_yt_dlp_video(video_url, videos_dir, keep_artifacts=keep_artifacts)

    return _build_video_record(
        video_id=video_id,
        page_url=page_url,
        transcript_data=transcript_data,
        oembed=oembed,
        ytdlp=ytdlp,
        video_url=video_url,
        artifacts_kept=keep_artifacts,
    )

def crawl_and_save(
    url: str,
    segmenter: Segmenter,
    timeout: int = 90,
    out_root: str = DATA_DIR,
    enable_video: bool = True,
    enable_asr: bool = True,
    keep_artifacts: bool = False,
    asr_max_seconds: int = 600,
    asr_language: str = "vi",
) -> dict:
    import trafilatura

    url_hash = hash_url(url)
    save_folder = os.path.join(out_root, url_hash)
    os.makedirs(save_folder, exist_ok=True)

    start_total = time.time()

    # Step 1: Fast - Trafilatura
    print(f"[FAST] Trafilatura -> {url}")
    downloaded = trafilatura.fetch_url(url)
    print(f"[DEBUG] trafilatura.fetch_url bytes: {len(downloaded) if downloaded else 0}")
    if not downloaded:
        try:
            import requests
            headers = {
                "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/144.0.0.0 Safari/537.36"),
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            }
            r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            print(f"[DEBUG] requests fallback status={r.status_code} final_url={r.url}")
            downloaded = r.text if r.ok else None
        except Exception:
            downloaded = None
    html_for_video = downloaded
    text = trafilatura.extract(downloaded, include_comments=True, include_tables=False)
    print(f"[DEBUG] trafilatura.extract text_len: {len(text) if text else 0}")
    method = "trafilatura_fast"

    # Step 2: Fallback Selenium
    if not text or len(text) < 800:
        print(f"[FALLBACK] Text short ({len(text) if text else 0} chars) -> Selenium")
        driver = None
        try:
            driver = make_driver(timeout=timeout, headless=True)
            print("[DEBUG] Selenium driver initialized")
            driver.get(url)
            print("[DEBUG] Selenium page loaded")
            time.sleep(6)
            for _ in range(4):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(4)
            html = driver.page_source
            print(f"[DEBUG] Selenium page_source bytes: {len(html) if html else 0}")
            html_for_video = html
            text = trafilatura.extract(html, include_comments=True, include_tables=False)
            print(f"[DEBUG] Selenium extract text_len: {len(text) if text else 0}")
            method = "selenium_fallback"
        except Exception as e:
            print(f"[ERROR] Selenium failed: {e}")
            text = None
            method = "failed"
        finally:
            try:
                if driver:
                    driver.quit()
            except Exception:
                pass

    # Still fail
    if not text or len(text) < 200:
        duration = round(time.time() - start_total, 2)
        meta = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "method": method,
            "duration_sec": duration,
        }
        with open(os.path.join(save_folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {
            "url": url,
            "url_hash": url_hash,
            "status": "error",
            "error": f"Crawl failed via {method}",
            "output_dir": save_folder,
            "segments_path": None,
            "num_segments": 0,
        }

    # success -> segment + save
    segments = segmenter.segment_text(text)
    cleaned_segments = []
    for s in segments:
        s2 = preprocess_text(s)
        if len(s2) > 10:
            cleaned_segments.append(s2)

    with open(os.path.join(save_folder, "extracted.txt"), "w", encoding="utf-8") as f:
        f.write(text.strip())

    with open(os.path.join(save_folder, "segments.jsonl"), "w", encoding="utf-8") as f:
        for seg in cleaned_segments:
            f.write(json.dumps({"text": seg}, ensure_ascii=False) + "\n")

    if enable_video and html_for_video:
        try:
            videos_dir = os.path.join(save_folder, "videos")
            if not keep_artifacts and os.path.exists(videos_dir):
                shutil.rmtree(videos_dir, ignore_errors=True)
            detected = _detect_videos_from_html(html_for_video, url)
            youtube_ids = detected.get("youtube_ids", [])
            video_records: List[dict] = []

            if youtube_ids:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_map = {
                        executor.submit(_fetch_youtube_video_data, vid, url, videos_dir, keep_artifacts): vid
                        for vid in youtube_ids
                    }
                    for future in as_completed(future_map):
                        vid = future_map[future]
                        try:
                            record = future.result()
                            if record:
                                video_records.append(record)
                        except Exception as e:
                                video_records.append({
                                    "video_id": vid,
                                    "platform": "youtube",
                                    "video_url": f"https://www.youtube.com/watch?v={vid}",
                                    "page_url": url,
                                    "title": None,
                                    "channel": None,
                                    "upload_date": None,
                                    "duration": None,
                                    "view_count": None,
                                    "transcript": [],
                                    "language": None,
                                    "has_auto_generated": None,
                                    "metadata": {},
                                    "error": str(e),
                                    "transcript_source": "none",
                                    "artifacts_kept": bool(keep_artifacts),
                                })

            page_infos = _run_yt_dlp_page(url, videos_dir, keep_artifacts=keep_artifacts)
            for info in page_infos:
                entries = info.get("entries")
                if entries:
                    for entry in entries:
                        if not entry:
                            continue
                        video_url = _select_best_format_url(entry)
                        if video_url and _is_ad_url(video_url):
                            continue
                        transcript = []
                        asr_meta = None
                        asr_reason = None
                        if enable_asr:
                            work_dir = os.path.join(videos_dir, f"asr_{entry.get('id') or 'unknown'}") if keep_artifacts else None
                            asr = _run_asr_on_video_url(
                                video_url,
                                language=asr_language,
                                asr_max_seconds=asr_max_seconds,
                                keep_artifacts=keep_artifacts,
                                artifacts_dir=work_dir,
                                known_duration=entry.get("duration"),
                            )
                            if asr and asr.get("segments"):
                                transcript = asr["segments"]
                                asr_meta = {
                                    "model": asr.get("model"),
                                    "device": asr.get("device"),
                                    "compute_type": asr.get("compute_type"),
                                }
                                if keep_artifacts and work_dir:
                                    try:
                                        with open(os.path.join(work_dir, "segments_full_asr.json"), "w", encoding="utf-8") as f:
                                            json.dump(asr.get("full_segments", []), f, ensure_ascii=False, indent=2)
                                    except Exception:
                                        pass
                            asr_reason = asr.get("reason") if asr else "unknown"
                        video_records.append({
                            "video_id": entry.get("id") or hash_url(entry.get("url") or ""),
                            "platform": "native",
                            "video_url": video_url,
                            "page_url": url,
                            "title": entry.get("title"),
                            "channel": entry.get("uploader") or entry.get("channel"),
                            "upload_date": _format_upload_date(entry.get("upload_date")),
                            "duration": entry.get("duration"),
                            "view_count": entry.get("view_count"),
                            "transcript": transcript,
                            "language": entry.get("language"),
                            "has_auto_generated": None,
                            "metadata": {"ytdlp": entry, "asr": asr_meta} if asr_meta else entry,
                            "transcript_source": "asr_ephemeral" if transcript else "none",
                            "artifacts_kept": bool(keep_artifacts),
                            "reason": asr_reason,
                        })
                else:
                    video_url = _select_best_format_url(info)
                    if video_url and _is_ad_url(video_url):
                        continue
                    transcript = []
                    asr_meta = None
                    asr_reason = None
                    if enable_asr:
                        work_dir = os.path.join(videos_dir, f"asr_{info.get('id') or 'unknown'}") if keep_artifacts else None
                        asr = _run_asr_on_video_url(
                            video_url,
                            language=asr_language,
                            asr_max_seconds=asr_max_seconds,
                            keep_artifacts=keep_artifacts,
                            artifacts_dir=work_dir,
                            known_duration=info.get("duration"),
                        )
                        if asr and asr.get("segments"):
                            transcript = asr["segments"]
                            asr_meta = {
                                "model": asr.get("model"),
                                "device": asr.get("device"),
                                "compute_type": asr.get("compute_type"),
                            }
                            if keep_artifacts and work_dir:
                                try:
                                    with open(os.path.join(work_dir, "segments_full_asr.json"), "w", encoding="utf-8") as f:
                                        json.dump(asr.get("full_segments", []), f, ensure_ascii=False, indent=2)
                                except Exception:
                                    pass
                        asr_reason = asr.get("reason") if asr else "unknown"
                    video_records.append({
                        "video_id": info.get("id") or hash_url(info.get("url") or ""),
                        "platform": "native",
                        "video_url": video_url,
                        "page_url": url,
                        "title": info.get("title"),
                        "channel": info.get("uploader") or info.get("channel"),
                        "upload_date": _format_upload_date(info.get("upload_date")),
                        "duration": info.get("duration"),
                        "view_count": info.get("view_count"),
                        "transcript": transcript,
                        "language": info.get("language"),
                        "has_auto_generated": None,
                        "metadata": {"ytdlp": info, "asr": asr_meta} if asr_meta else info,
                        "transcript_source": "asr_ephemeral" if transcript else "none",
                        "artifacts_kept": bool(keep_artifacts),
                        "reason": asr_reason,
                    })

            if video_records:
                video_data_path = os.path.join(save_folder, "video_data.jsonl")
                with open(video_data_path, "w", encoding="utf-8") as f:
                    for rec in video_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] Video crawling failed (non-fatal): {e}")

    duration = round(time.time() - start_total, 2)
    meta = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "num_segments": len(cleaned_segments),
        "duration_sec": duration,
        "method": method,
        "text_length": len(text),
        "status": "success",
    }
    with open(os.path.join(save_folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "url": url,
        "url_hash": url_hash,
        "status": "ok",
        "error": None,
        "output_dir": save_folder,
        "segments_path": os.path.join(save_folder, "segments.jsonl"),
        "num_segments": len(cleaned_segments),
        "method": method,
        "duration_sec": duration,
    }

def run_crawl(
    enable_video: bool = True,
    enable_asr: bool = True,
    keep_artifacts: bool = False,
    asr_max_seconds: int = 600,
    asr_language: str = "vi",
):
    from tqdm import tqdm

    ensure_dirs()
    ensure_vncorenlp_assets()
    seg = Segmenter(VNC_JAR_PATH)

    log_lines = [
        f"## Hybrid Crawl Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Total URLs: {len(URL_LIST)}",
        ""
    ]

    success_count = 0
    for url in tqdm(URL_LIST):
        result = crawl_and_save(
            url.strip(),
            seg,
            out_root=DATA_DIR,
            enable_video=enable_video,
            enable_asr=enable_asr,
            keep_artifacts=keep_artifacts,
            asr_max_seconds=asr_max_seconds,
            asr_language=asr_language,
        )
        status = result.get("status")
        if status == "ok":
            line = f"SUCCESS | {result.get('method', 'unknown')} | {result.get('num_segments', 0)} segments | {result.get('duration_sec', 0)}s"
            success_count += 1
        else:
            line = f"FAILED | {result.get('error', 'unknown')}"
        log_lines.append(f"- {url} → {line}")
        time.sleep(3)

    log_lines.append(f"\n**Summary**: {success_count}/{len(URL_LIST)} success")
    log_lines.append("")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\nDONE!")
    print(f"→ {success_count}/{len(URL_LIST)} URLs success")
    print(f"Data: {DATA_DIR}")
    print(f"Log : {LOG_FILE}")

# =========================
# CLI
# =========================

def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Setup dependencies and assets")
    parser.add_argument("--crawl", action="store_true", help="Run crawler")
    parser.add_argument("--enable-video", action="store_true", help="Enable video crawling")
    parser.add_argument("--disable-video", action="store_true", help="Disable video crawling")
    parser.add_argument("--enable-asr", action="store_true", help="Enable ASR for native video")
    parser.add_argument("--disable-asr", action="store_true", help="Disable ASR for native video")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep video/ASR artifacts on disk")
    parser.add_argument("--asr-max-seconds", type=int, default=600, help="Skip ASR if duration > cap (seconds)")
    parser.add_argument("--asr-language", default="vi", help='ASR language code or "auto"')
    args = parser.parse_args(argv)
    return (
        args.setup,
        args.crawl,
        args.enable_video,
        args.disable_video,
        args.enable_asr,
        args.disable_asr,
        args.keep_artifacts,
        args.asr_max_seconds,
        args.asr_language,
    )

def ensure_runtime_deps():
    """
    Install missing deps into the CURRENT running python environment (your active venv).
    This avoids accidentally installing into .venv or system python.
    """
    import importlib.util
    missing = []

    required = {
        "trafilatura": "trafilatura>=1.7.0",
        "vncorenlp": "vncorenlp>=1.0.3",
        "tqdm": "tqdm>=4.66.0",
        "pandas": "pandas>=2.0.0",
        "selenium": "selenium>=4.15.0",
        "undetected_chromedriver": "undetected-chromedriver>=3.5.5",
        "requests": "requests>=2.31.0",
        "certifi": "certifi>=2023.11.17",
        "youtube_transcript_api": "youtube-transcript-api>=0.6.2",
        "yt_dlp": "yt-dlp>=2024.12.0",
        "bs4": "beautifulsoup4>=4.12.0",
        "faster_whisper": "faster-whisper>=1.1.0",
    }

    for mod, pkg in required.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)

    if missing:
        print("[INFO] Installing missing packages into current venv:")
        for m in missing:
            print("  -", m)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
def main():
    argv = sys.argv[1:]
    (
        do_setup,
        do_crawl,
        enable_video_flag,
        disable_video,
        enable_asr_flag,
        disable_asr,
        keep_artifacts,
        asr_max_seconds,
        asr_language,
    ) = parse_args(argv)

    # Mặc định: nếu không truyền flag gì thì chạy crawl luôn (đỡ phiền)
    if not do_setup and not do_crawl:
        do_crawl = True

    enable_video = not disable_video
    if enable_video_flag:
        enable_video = True

    enable_asr = not disable_asr
    if enable_asr_flag:
        enable_asr = True

    # Setup: chỉ check Java + đảm bảo deps trong current venv
    if do_setup:
        java_ok = check_java()
        if not java_ok:
            print("\n[NOTE] You need Java for VnCoreNLP word segmentation. Install e.g.:")
            print("  brew install --cask temurin@17\n")

        # luôn đảm bảo deps trong env đang chạy
        ensure_runtime_deps()

    # Crawl: đảm bảo deps trước, rồi chạy
    if do_crawl:
        if not in_venv():
            print("[WARN] You are not running inside a virtualenv.")
            print("It's still OK, but recommended to activate your (venv) first.\n")

        ensure_runtime_deps()
        if enable_video:
            print("[INFO] Video crawling enabled.")
        if enable_asr:
            print("[INFO] Native ASR enabled.")
        run_crawl(
            enable_video=enable_video,
            enable_asr=enable_asr,
            keep_artifacts=keep_artifacts,
            asr_max_seconds=asr_max_seconds,
            asr_language=asr_language,
        )


def crawl_urls(
    urls: List[str],
    out_dir: str = DATA_DIR,
    timeout: int = 90,
    enable_video: bool = True,
    enable_asr: bool = True,
    keep_artifacts: bool = False,
    asr_max_seconds: int = 600,
    asr_language: str = "vi",
) -> List[dict]:
    """
    Crawl a list of URLs and return per-URL crawl info.
    This does NOT auto-install dependencies.
    """
    os.makedirs(out_dir, exist_ok=True)
    ensure_vncorenlp_assets()
    seg = Segmenter(VNC_JAR_PATH)

    results: List[dict] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            info = crawl_and_save(
                url,
                seg,
                timeout=timeout,
                out_root=out_dir,
                enable_video=enable_video,
                enable_asr=enable_asr,
                keep_artifacts=keep_artifacts,
                asr_max_seconds=asr_max_seconds,
                asr_language=asr_language,
            )
        except Exception as e:
            info = {
                "url": url,
                "url_hash": hash_url(url),
                "status": "error",
                "error": f"Unexpected crawl error: {e}",
                "output_dir": os.path.join(out_dir, hash_url(url)),
                "segments_path": None,
                "num_segments": 0,
            }
        results.append(info)
    return results


if __name__ == "__main__":
    main()
