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
    "https://vnexpress.net/tranh-cai-ve-so-danh-hieu-cua-messi-4991489.html",
    "https://vnexpress.net/bao-thai-lan-dung-trach-chu-nha-vi-trong-tai-huy-ban-thang-cua-viet-nam-4995254.html",
    "https://vnexpress.net/mot-nam-khung-hoang-hinh-anh-nghe-si-viet-4996888.html",
    "https://tuoitre.vn/cach-nao-de-cham-dut-viec-chui-boi-xuc-pham-tren-mang-20211027223924572.htm",
    "https://tuoitre.vn/chui-boi-tung-toe-tren-mang-cung-hong-luc-ra-toa-20190907220922579.htm",
    "https://vnexpress.net/nhieu-nguoi-gioi-tranh-cai-thay-vi-tranh-luan-4335581.html",
    "https://voz.vn/t/van-hoa-tranh-luan-cua-voz-hien-nay-qua-toxic.354362/",
    "https://voz.vn/f/oto-xe-may.8/",
    "https://www.webtretho.com/f/tam-su",
    "https://vnexpress.net/thoi-su",
    "https://tuoitre.vn",
    "https://vnexpress.net/bong-da",
    "https://thanhnien.vn/cong-nghe",
    "https://vnexpress.net/giai-tri",
    "https://tuoitre.vn/xa-hoi.htm",
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

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,2000")

    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(timeout)
    return driver

# =========================
# Hybrid crawl
# =========================

def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def crawl_and_save(url: str, segmenter: Segmenter, timeout: int = 90, out_root: str = DATA_DIR) -> dict:
    import trafilatura

    url_hash = hash_url(url)
    save_folder = os.path.join(out_root, url_hash)
    os.makedirs(save_folder, exist_ok=True)

    start_total = time.time()

    # Step 1: Fast - Trafilatura
    print(f"[FAST] Trafilatura -> {url}")
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_comments=True, include_tables=False)
    method = "trafilatura_fast"

    # Step 2: Fallback Selenium
    if not text or len(text) < 800:
        print(f"[FALLBACK] Text short ({len(text) if text else 0} chars) -> Selenium")
        driver = None
        try:
            driver = make_driver(timeout=timeout, headless=True)
            driver.get(url)
            time.sleep(6)
            for _ in range(4):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(4)
            html = driver.page_source
            text = trafilatura.extract(html, include_comments=True, include_tables=False)
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

def run_crawl():
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
        result = crawl_and_save(url.strip(), seg, out_root=DATA_DIR)
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
    do_setup = "--setup" in argv
    do_crawl = "--crawl" in argv
    return do_setup, do_crawl

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
    do_setup, do_crawl = parse_args(argv)

    # Mặc định: nếu không truyền flag gì thì chạy crawl luôn (đỡ phiền)
    if not do_setup and not do_crawl:
        do_crawl = True

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
        run_crawl()


def crawl_urls(urls: List[str], out_dir: str = DATA_DIR, timeout: int = 90) -> List[dict]:
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
            info = crawl_and_save(url, seg, timeout=timeout, out_root=out_dir)
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
