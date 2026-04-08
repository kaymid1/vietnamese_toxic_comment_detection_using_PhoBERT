"""
comment_crawl.py — Comment-only crawl pipeline for news sites and Facebook.

Outputs segments.jsonl in the same schema as setup_and_crawl.py so that
infer_crawled_local.py and backend/app.py can consume results with zero refactor.

Usage (standalone):
    python comment_crawl.py "https://vnexpress.net/some-article-123.html"
    python comment_crawl.py "https://www.facebook.com/permalink.php?id=xxx&story_fbid=yyy"

Usage (as library):
    from comment_crawl import crawl_comments_from_url
    result = crawl_comments_from_url(url, output_base_dir="data/raw/crawled_urls")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import time
import unicodedata
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

CRAWL_RETRY_MAX_ATTEMPTS = max(1, int(os.getenv("COMMENT_CRAWL_RETRY_MAX_ATTEMPTS", "3")))
CRAWL_RETRY_BASE_DELAY = max(0.1, float(os.getenv("COMMENT_CRAWL_RETRY_BASE_DELAY", "2.0")))
CRAWL_RETRY_MAX_DELAY = max(0.1, float(os.getenv("COMMENT_CRAWL_RETRY_MAX_DELAY", "12.0")))
CRAWL_RETRY_JITTER = max(0.0, float(os.getenv("COMMENT_CRAWL_RETRY_JITTER", "1.5")))
CRAWL_CACHE_TTL_HOURS = max(0.0, float(os.getenv("COMMENT_CRAWL_CACHE_TTL_HOURS", "2.0")))
BATCH_INTER_URL_DELAY_MIN = max(
    0.0, float(os.getenv("COMMENT_CRAWL_INTER_URL_DELAY_MIN", "2.5"))
)
BATCH_INTER_URL_DELAY_MAX = max(
    BATCH_INTER_URL_DELAY_MIN,
    float(os.getenv("COMMENT_CRAWL_INTER_URL_DELAY_MAX", "6.0")),
)
COMMENT_CRAWL_SCHEMA_VERSION = "comment_only_v3"


def _proxy_pool() -> list[str]:
    raw = os.getenv("COMMENT_CRAWL_PROXY_LIST", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _pick_proxy() -> str | None:
    pool = _proxy_pool()
    if not pool:
        return None
    return random.choice(pool)


def _pick_user_agent() -> str:
    custom_pool = os.getenv("COMMENT_CRAWL_USER_AGENTS", "").strip()
    if custom_pool:
        pool = [item.strip() for item in custom_pool.split("||") if item.strip()]
        if pool:
            return random.choice(pool)
    return random.choice(_DEFAULT_USER_AGENTS)


def _compute_backoff(attempt: int) -> float:
    delay = CRAWL_RETRY_BASE_DELAY * (2 ** max(0, attempt - 1))
    delay += random.uniform(0.0, CRAWL_RETRY_JITTER)
    return min(delay, CRAWL_RETRY_MAX_DELAY)


def _is_transient_crawl_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_tokens = [
        "timeout",
        "timed out",
        "net::err",
        "connection",
        "disconnected",
        "session deleted",
        "target window already closed",
        "chrome not reachable",
    ]
    return any(token in text for token in transient_tokens)


# ---------------------------------------------------------------------------
# Segment schema helpers (mirrors setup_and_crawl.py contract)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """NFC + collapse whitespace + strip."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_stale_element_error(exc: Exception) -> bool:
    return "stale element reference" in str(exc).lower()


def _clean_comment_text(text: str) -> str:
    txt = _normalize_text(text)
    if not txt:
        return ""

    # Strip common reaction/report/time footer noise appended by comment widgets.
    # Keep patterns end-anchored to avoid deleting real comment content in the middle.
    time_suffix = (
        r"(?:"
        r"\d+\s*(?:phút|giờ|ngày|tuần|tháng|năm)\s+trước"
        r"|\d+h\s+trước"
        r"|\d{1,2}:\d{2}(?:\s+\d{1,2}/\d{1,2}(?:/\d{2,4})?)?"
        r")"
    )
    tail_patterns = [
        rf"(?:\s+(?:thích|like|ngạc nhiên|buồn|haha|yêu thích|phẫn nộ|sad|wow|angry|love|\d+)){{0,30}}\s+trả lời(?:\s+báo vi phạm)?(?:\s+{time_suffix})?\s*$",
        rf"(?:\s+\d+){{0,8}}\s+trả lời(?:\s+báo vi phạm)?(?:\s+{time_suffix})?\s*$",
        rf"\s+báo vi phạm(?:\s+{time_suffix})?\s*$",
    ]
    cleaned = txt
    for pattern in tail_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip(" |·-\t")


def _is_comment_like_text(text: str) -> bool:
    txt = _clean_comment_text(text)
    if not txt:
        return False
    if len(txt) < 2:
        return False
    if re.fullmatch(r"[\W_]+", txt):
        return False

    txt_lower = txt.lower()
    for pattern in _UI_NOISE_PATTERNS:
        if re.match(pattern, txt_lower):
            return False

    if len(txt) <= 4 and " " not in txt:
        return False

    return True


def _looks_like_article_blob(text: str) -> bool:
    txt = _normalize_text(text)
    if not txt:
        return False
    if len(txt) < 320:
        return False
    if "\n" in text and text.count("\n") >= 3:
        return True

    sentence_like = len(re.findall(r"[.!?…]", txt))
    if sentence_like >= 6 and len(txt) >= 420:
        return True
    return False


def _make_segment_hash(text: str, html_tag_effective: str) -> str:
    payload = f"{_normalize_text(text)}|{html_tag_effective}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def build_segments_jsonl(
    comments: list[str],
    url: str,
    html_tag_effective: str = "comment",
) -> list[dict]:
    """Convert a list of comment strings into segments.jsonl rows."""
    uhash = _url_hash(url)
    segments: list[dict] = []
    for idx, raw_text in enumerate(comments):
        text = _clean_comment_text(raw_text)
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "segment_index": idx,
                "url_hash": uhash,
                "html_tag_effective": html_tag_effective,
                "segment_hash": _make_segment_hash(text, html_tag_effective),
            }
        )
    return segments


def save_crawl_artifacts(
    output_dir: str | pathlib.Path,
    segments: list[dict],
    meta: dict,
) -> pathlib.Path:
    """Write segments.jsonl, meta.json, and extracted.txt into *output_dir*."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # segments.jsonl
    seg_path = out / "segments.jsonl"
    with open(seg_path, "w", encoding="utf-8") as f:
        for row in segments:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # extracted.txt — concatenation for backward compat
    ext_path = out / "extracted.txt"
    with open(ext_path, "w", encoding="utf-8") as f:
        for row in segments:
            f.write(row["text"] + "\n")

    # meta.json
    meta_path = out / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out


# ---------------------------------------------------------------------------
# URL type detection
# ---------------------------------------------------------------------------

_FB_PATTERNS = re.compile(
    r"(facebook\.com|fb\.com|fb\.watch|m\.facebook\.com|mbasic\.facebook\.com)",
    re.IGNORECASE,
)
_X_PATTERNS = re.compile(r"(twitter\.com|x\.com|mobile\.twitter\.com)", re.IGNORECASE)
_NEWS_DOMAINS: set[str] = {
    "vnexpress.net",
    "tuoitre.vn",
    "thanhnien.vn",
    "dantri.com.vn",
    "vietnamnet.vn",
    "nld.com.vn",
    "baomoi.com",
    "zingnews.vn",
    "kenh14.vn",
    "soha.vn",
    "vtc.vn",
    "vov.vn",
    "tienphong.vn",
    "laodong.vn",
    "plo.vn",
    "cafef.vn",
    "genk.vn",
    "gamek.vn",
    "afamily.vn",
}


def detect_url_type(url: str) -> str:
    """Return 'facebook' | 'x_twitter' | 'news' | 'unknown'."""
    if _FB_PATTERNS.search(url):
        return "facebook"
    if _X_PATTERNS.search(url):
        return "x_twitter"
    parsed = urlparse(url)
    host = parsed.hostname or ""
    host_lower = host.lower().lstrip("www.")
    if host_lower in _NEWS_DOMAINS:
        return "news"
    # Heuristic: path chứa dấu hiệu bài báo
    if re.search(r"\.(html?|htm)$", parsed.path, re.IGNORECASE):
        return "news"
    return "unknown"


# ---------------------------------------------------------------------------
# Selenium helpers
# ---------------------------------------------------------------------------


_EDGE_SEARCH_PATHS_WINDOWS = [
    os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft\EdgeWebView\Application\msedge.exe"
    ),
    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
    os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

_CHROME_SEARCH_PATHS_WINDOWS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
]


def _find_browser_binary() -> tuple[str | None, str]:
    """
    Locate a Chromium-based browser binary.

    Preference order: Edge → Chrome → system PATH.
    Returns (path, browser_type) where browser_type is 'edge' or 'chrome'.
    """
    import platform
    import shutil

    system = platform.system()
    if system == "Windows":
        for candidate in _EDGE_SEARCH_PATHS_WINDOWS:
            if candidate and os.path.isfile(candidate):
                return candidate, "edge"
        for candidate in _CHROME_SEARCH_PATHS_WINDOWS:
            if candidate and os.path.isfile(candidate):
                return candidate, "chrome"
        edge_shutil = shutil.which("msedge") or shutil.which("microsoft-edge")
        if edge_shutil:
            return edge_shutil, "edge"
        chrome_shutil = shutil.which("chrome") or shutil.which("google-chrome")
        if chrome_shutil:
            return chrome_shutil, "chrome"
    elif system == "Darwin":
        edge_mac = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
        if os.path.isfile(edge_mac):
            return edge_mac, "edge"
        chrome_mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.isfile(chrome_mac):
            return chrome_mac, "chrome"
        for cmd, btype in [
            ("microsoft-edge", "edge"),
            ("google-chrome", "chrome"),
            ("chrome", "chrome"),
        ]:
            found = shutil.which(cmd)
            if found:
                return found, btype
    else:
        for cmd, btype in [
            ("microsoft-edge", "edge"),
            ("microsoft-edge-stable", "edge"),
            ("google-chrome", "chrome"),
            ("google-chrome-stable", "chrome"),
            ("chromium-browser", "chrome"),
            ("chromium", "chrome"),
        ]:
            found = shutil.which(cmd)
            if found:
                return found, btype
    return None, "chrome"


def _common_args(user_agent: str) -> list[str]:
    """Shared browser arguments."""
    return [
        f"--user-agent={user_agent}",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--lang=vi-VN",
        "--window-size=1280,900",
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
    ]


def _apply_stealth_patches(driver) -> None:
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'languages', {get: () => ['vi-VN', 'vi', 'en-US', 'en']});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    window.chrome = window.chrome || { runtime: {} };
                """
            },
        )
    except Exception:
        pass


def _get_driver_edge(headless: bool, browser_bin: str | None, proxy: str | None = None):
    """
    Build a standard selenium Edge WebDriver.

    selenium-manager (bundled since Selenium 4.6) auto-downloads msedgedriver
    matching the installed Edge version — no manual driver management needed.
    """
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService

    options = EdgeOptions()
    options.page_load_strategy = "eager"
    if headless:
        options.add_argument("--headless=new")
    user_agent = _pick_user_agent()
    for arg in _common_args(user_agent=user_agent):
        options.add_argument(arg)
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
    if browser_bin:
        options.binary_location = browser_bin

    service = EdgeService()  # selenium-manager resolves msedgedriver automatically
    driver = webdriver.Edge(service=service, options=options)
    _apply_stealth_patches(driver)
    driver.set_page_load_timeout(45)
    driver.implicitly_wait(5)
    return driver


def _detect_chrome_major_version(browser_bin: str | None) -> int | None:
    if not browser_bin:
        return None
    try:
        completed = subprocess.run(
            [browser_bin, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None

    raw = (completed.stdout or completed.stderr or "").strip()
    match = re.search(r"(\d+)\.\d+\.\d+\.\d+", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _get_driver_chrome_uc(
    headless: bool,
    browser_bin: str | None,
    proxy: str | None = None,
):
    """Build undetected_chromedriver for Chrome."""
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    options.page_load_strategy = "eager"
    if headless:
        options.add_argument("--headless=new")
    user_agent = _pick_user_agent()
    for arg in _common_args(user_agent=user_agent):
        options.add_argument(arg)
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
    if browser_bin:
        options.binary_location = browser_bin

    detected_major = _detect_chrome_major_version(browser_bin)
    if detected_major is not None:
        logger.info("Detected Chrome major version: %s", detected_major)
        try:
            driver = uc.Chrome(options=options, version_main=detected_major)
        except Exception:
            logger.warning(
                "uc.Chrome failed with version_main=%s, retrying without version pin",
                detected_major,
            )
            driver = uc.Chrome(options=options)
    else:
        driver = uc.Chrome(options=options)

    _apply_stealth_patches(driver)
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)
    return driver


def _get_undetected_driver(headless: bool = True, proxy: str | None = None):
    """
    Create a browser driver backed by whichever Chromium browser is available.

    - Edge detected  → selenium.webdriver.Edge (selenium-manager handles msedgedriver)
    - Chrome detected → undetected_chromedriver (keeps anti-bot patching)
    """
    browser_bin, browser_type = _find_browser_binary()
    chosen_proxy = proxy or _pick_proxy()
    logger.info(
        "Using %s binary: %s%s",
        browser_type,
        browser_bin,
        " with proxy" if chosen_proxy else "",
    )

    if browser_type == "edge":
        return _get_driver_edge(headless, browser_bin, proxy=chosen_proxy)
    else:
        return _get_driver_chrome_uc(headless, browser_bin, proxy=chosen_proxy)


def _random_delay(lo: float = 1.5, hi: float = 4.0) -> None:
    base = random.uniform(lo, hi)
    if random.random() < 0.15:
        base += random.uniform(1.5, 5.0)
    time.sleep(min(base, hi + 5.0))


def _safe_click(driver, element) -> bool:
    """Click with JS fallback."""
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        StaleElementReferenceException,
    )

    try:
        element.click()
        return True
    except (ElementClickInterceptedException, StaleElementReferenceException):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# News site comment crawler
# ---------------------------------------------------------------------------

# Mapping: domain suffix → dict of CSS selectors + optional load-more info.
# "container"  : wraps all comments
# "item"       : each individual comment
# "text"       : text node inside an item  (relative selector)
# "load_more"  : button/link to expand more comments (optional)
DOMAIN_SELECTORS: dict[str, dict[str, str]] = {
    "vnexpress.net": {
        "container": "#box_comment_vne, #box_comment, .box-comment, .box_comment_vne, [id*='box_comment'], [class*='box_comment_vne']",
        "item": ".comment_item, .item-comment, .comment-item, [class*='comment_item'], [class*='comment-item'], [class*='item-comment']",
        "text": ".full_content, .content-comment, .comment-content, [class*='content-comment'], [class*='full_content'], p",
        "exclude": ".txt-name",
        "load_more": "a.view_more_coment, a.view_more_comment, a.btn-more-comment, .btn-load-more-comment, [class*='view_more_coment'], [class*='load-more-comment']",
    },
    "tuoitre.vn": {
        "container": "#box-comment, .comment-wrapper",
        "item": ".content-comment, .comment-item",
        "text": "p, .comment-text",
        "load_more": ".btn-more-comment, a.viewmore",
    },
    "thanhnien.vn": {
        "container": "#comment-box, .box-comment",
        "item": ".comment-item, .cmt-item",
        "text": ".comment-content, .cmt-content, p",
        "load_more": ".btn-view-more, .more-comment",
    },
    "dantri.com.vn": {
        "container": ".comment-box, #comment-box, [class*='comment-wrap'], [class*='comment-list'], [id*='comment']",
        "item": ".comment-item, [class*='comment-item'], [class*='comment_item'], [class*='item-comment']",
        "text": ".comment-content, .content-comment, [class*='comment-content'], [class*='content-comment'], p",
        "load_more": ".btn-more, .load-more-comment, [class*='load-more'], [class*='view-more'], [data-role*='load-more']",
    },
    "vietnamnet.vn": {
        "container": "#box_comment, .comment-container",
        "item": ".comment-item",
        "text": ".comment-content, p",
        "load_more": ".btn-more-comment",
    },
}

# Fallback heuristic selectors for unknown news domains
_FALLBACK_CONTAINER_SELECTORS = [
    "[id*='comment']",
    "[class*='comment']",
    "[data-component*='comment']",
    "[data-testid*='comment']",
    "[class*='discussion']",
    "[id*='discussion']",
]

_FALLBACK_ITEM_SELECTORS = [
    "[class*='comment-item']",
    "[class*='comment_item']",
    "[class*='item-comment']",
    "[class*='commentRow']",
]

_FALLBACK_TEXT_SELECTORS = [
    "[class*='content-comment']",
    "[class*='comment-content']",
    "[class*='comment-body']",
    "[class*='comment-text']",
    "[data-sigil*='comment-body']",
]

_FALLBACK_COMMENT_SELECTORS = [
    "[class*='comment-item'] [class*='comment-content']",
    "[class*='comment_item'] [class*='comment-content']",
    "[class*='comment-item'] [class*='content-comment']",
    "[class*='comment_item'] [class*='content-comment']",
    "[id*='comment'] [class*='comment-content']",
    "[id*='comment'] [class*='content-comment']",
    ".comment-body",
    ".comment-text",
    "[data-sigil*='comment-body']",
]

_UI_NOISE_PATTERNS = [
    r"^(thích|like)$",
    r"^(trả lời|reply)$",
    r"^(chia sẻ|share)$",
    r"^(xem thêm|view more).*$",
    r"^(phản hồi|comment)$",
    r"^\d+\s*(phút|giờ|ngày|tuần|tháng|năm)\s*(trước)?$",
    r"^\d+[smhdw]$",
]

_NEWS_BLOCK_TEXT_INDICATORS = [
    "too many requests",
    "rate limit",
    "unusual traffic",
    "access denied",
    "temporarily blocked",
    "security check",
    "captcha",
    "verify you are human",
    "cloudflare",
    "request blocked",
    "429",
]

_NEWS_LOAD_MORE_TEXT_KEYWORDS = [
    "xem thêm",
    "xem them",
    "xem thêm bình luận",
    "xem them binh luan",
    "view more",
    "more comments",
    "load more",
    "tải thêm",
    "tai them",
]


class NewsSiteCommentCrawler:
    """Crawl only the comment section of Vietnamese news websites."""

    def __init__(
        self,
        headless: bool = True,
        max_load_more_clicks: int = 15,
        max_runtime_seconds: int = 90,
    ):
        self.headless = headless
        self.max_load_more_clicks = max_load_more_clicks
        self.max_runtime_seconds = max_runtime_seconds
        self.last_block_reason: str | None = None
        self.last_warnings: list[str] = []
        self.last_attempts: int = 0

    # ---- public API ----

    def crawl_comments(self, url: str) -> list[str]:
        """Return list of comment text strings from *url*."""
        self.last_block_reason = None
        self.last_warnings = []
        self.last_attempts = 0

        attempts = max(1, CRAWL_RETRY_MAX_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            self.last_attempts = attempt
            try:
                comments = self._crawl_comments_once(url)
                if attempt > 1:
                    self.last_warnings.append(
                        f"Recovered after retry attempt {attempt}/{attempts}"
                    )
                return comments
            except RuntimeError as exc:
                if self.last_block_reason:
                    self.last_warnings.append(str(exc))
                    logger.warning(
                        "NewsSiteCrawler blocked on attempt %d/%d for %s: %s",
                        attempt,
                        attempts,
                        url,
                        self.last_block_reason,
                    )
                    return []

                if attempt >= attempts or not _is_transient_crawl_error(exc):
                    self.last_warnings.append(str(exc))
                    raise

                backoff = _compute_backoff(attempt)
                self.last_warnings.append(
                    f"Attempt {attempt}/{attempts} failed transiently: {exc}. Backoff {backoff:.2f}s"
                )
                logger.warning(
                    "NewsSiteCrawler transient failure on attempt %d/%d for %s: %s. Retrying in %.2fs",
                    attempt,
                    attempts,
                    url,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        return []

    def _crawl_comments_once(self, url: str) -> list[str]:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        started_at = time.time()
        deadline = started_at + max(10, int(self.max_runtime_seconds))

        def _raise_if_deadline_passed(phase: str) -> None:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Exceeded crawl runtime budget ({self.max_runtime_seconds}s) during {phase}"
                )

        driver = None
        try:
            driver = _get_undetected_driver(headless=self.headless)
            logger.info("NewsSiteCrawler: loading %s", url)
            load_start = time.time()
            try:
                driver.get(url)
            except TimeoutException:
                logger.warning(
                    "Page load timeout for %s after %.1fs; continuing with current DOM",
                    url,
                    time.time() - load_start,
                )
            logger.info(
                "NewsSiteCrawler: page load step done for %s in %.2fs",
                url,
                time.time() - load_start,
            )
            _raise_if_deadline_passed("page load")
            _random_delay(2.0, 4.0)
            _raise_if_deadline_passed("post-load delay")

            block_reason = self._detect_block_or_rate_limit(driver)
            if block_reason:
                self.last_block_reason = block_reason
                logger.warning("NewsSiteCrawler blocked/challenged: %s", block_reason)
                return []

            selectors = self._pick_selectors(url)
            comments: list[str] = []

            logger.info(
                "Selector profile for %s: domain=%s keys=%s",
                url,
                (urlparse(url).hostname or "").lower(),
                sorted(selectors.keys()) if selectors else [],
            )
            if not selectors:
                logger.warning(
                    "No domain-specific selectors matched for %s; using fallback-only extraction",
                    url,
                )

            container_css = selectors.get("container", "")
            if container_css:
                wait_seconds = int(max(1, min(12, deadline - time.time())))
                try:
                    WebDriverWait(driver, wait_seconds).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, container_css))
                    )
                except TimeoutException:
                    logger.warning(
                        "Comment container not found within timeout for %s", url
                    )
                _raise_if_deadline_passed("container wait")

            self._prime_comment_section(driver, container_css)
            _raise_if_deadline_passed("comment priming")

            self._wait_for_comment_readiness(driver, selectors)
            _raise_if_deadline_passed("comment readiness wait")

            self._click_load_more(driver, selectors)
            _raise_if_deadline_passed("load-more expansion")

            self._wait_for_comment_readiness(driver, selectors)
            _raise_if_deadline_passed("post-expand readiness wait")

            comments = self._extract_comments(driver, selectors)
            logger.info(
                "Domain extraction finished with %d comment candidate(s)",
                len(comments),
            )
            _raise_if_deadline_passed("domain extraction")

            if not comments:
                logger.info(
                    "Domain selectors yielded 0 comments, retrying expansion then fallback heuristics"
                )
                self._click_load_more(driver, selectors, css_first=False, max_clicks=4)
                _raise_if_deadline_passed("fallback load-more expansion")
                driver.implicitly_wait(0)
                try:
                    comments = self._extract_fallback(driver, selectors)
                    logger.info(
                        "Fallback extraction finished with %d comment candidate(s)",
                        len(comments),
                    )
                finally:
                    driver.implicitly_wait(5)
                _raise_if_deadline_passed("fallback extraction")

            if not comments:
                block_reason = self._detect_block_or_rate_limit(driver)
                if block_reason:
                    self.last_block_reason = block_reason
                    logger.warning("NewsSiteCrawler blocked/challenged: %s", block_reason)
                else:
                    logger.warning(
                        "No comments extracted and no anti-bot indicator detected. Likely causes: selector mismatch, delayed rendering, or comments hidden by interaction",
                    )

            logger.info(
                "NewsSiteCrawler final result: comments=%d blocked=%s url=%s",
                len(comments),
                bool(self.last_block_reason),
                url,
            )
            return comments

        except TimeoutError as exc:
            logger.warning("NewsSiteCrawler timeout for %s: %s", url, exc)
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            logger.exception("NewsSiteCrawler failed for %s", url)
            raise RuntimeError(f"NewsSiteCrawler failed: {exc}") from exc
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass

    # ---- internals ----

    def _pick_selectors(self, url: str) -> dict[str, str]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().lstrip("www.")
        # Exact match
        if host in DOMAIN_SELECTORS:
            return DOMAIN_SELECTORS[host]
        # Try stripping one subdomain level
        parts = host.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[1:])
            if parent in DOMAIN_SELECTORS:
                return DOMAIN_SELECTORS[parent]
        return {}

    def _prime_comment_section(self, driver, container_css: str) -> None:
        from selenium.webdriver.common.by import By

        try:
            # Many pages lazy-render comments only after the viewport reaches the section.
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")
            _random_delay(0.8, 1.5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            _random_delay(0.8, 1.5)
            if container_css:
                containers = driver.find_elements(By.CSS_SELECTOR, container_css)
                if containers:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});",
                        containers[0],
                    )
                    _random_delay(0.6, 1.2)
        except Exception:
            # Best-effort only; extraction logic below is still the source of truth.
            pass

    def _comment_signal_count(self, driver, selectors: dict[str, str]) -> int:
        from selenium.webdriver.common.by import By

        item_css = selectors.get("item", "")
        text_css = selectors.get("text", "")
        candidate_selectors = [
            item_css,
            text_css,
            "[class*='comment-item']",
            "[class*='comment_item']",
            "[class*='comment'] p",
        ]
        total = 0
        for sel in candidate_selectors:
            if not sel:
                continue
            try:
                total += len(driver.find_elements(By.CSS_SELECTOR, sel))
            except Exception:
                continue
        return total

    def _find_text_load_more_candidate(self, driver, selectors: dict[str, str]):
        from selenium.webdriver.common.by import By

        container_css = selectors.get("container", "")
        searchable_selectors = ["button", "a", "[role='button']"]

        candidates = []
        if container_css:
            try:
                containers = driver.find_elements(By.CSS_SELECTOR, container_css)
                for container in containers:
                    for selector in searchable_selectors:
                        candidates.extend(container.find_elements(By.CSS_SELECTOR, selector))
            except Exception:
                pass

        if not candidates:
            for selector in searchable_selectors:
                try:
                    candidates.extend(driver.find_elements(By.CSS_SELECTOR, selector))
                except Exception:
                    continue

        for element in candidates:
            try:
                if not element.is_displayed():
                    continue
                text = _normalize_text((element.text or "").lower())
                if not text:
                    continue
                if any(keyword in text for keyword in _NEWS_LOAD_MORE_TEXT_KEYWORDS):
                    return element
            except Exception:
                continue
        return None

    def _click_load_more(
        self,
        driver,
        selectors: dict[str, str],
        *,
        css_first: bool = True,
        max_clicks: int | None = None,
    ) -> None:
        from selenium.webdriver.common.by import By

        load_css = selectors.get("load_more")
        click_budget = max_clicks or self.max_load_more_clicks
        clicked = 0

        for idx in range(1, click_budget + 1):
            try:
                target = None
                source = "none"

                if css_first and load_css:
                    btns = driver.find_elements(By.CSS_SELECTOR, load_css)
                    visible = [b for b in btns if b.is_displayed()]
                    if visible:
                        target = visible[0]
                        source = "selector"

                if target is None:
                    target = self._find_text_load_more_candidate(driver, selectors)
                    if target is not None:
                        source = "text"

                if target is None:
                    logger.info(
                        "Load-more phase stopped: no visible controls after %d click(s)",
                        clicked,
                    )
                    break

                before_signal = self._comment_signal_count(driver, selectors)
                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});",
                        target,
                    )
                except Exception:
                    pass

                if not _safe_click(driver, target):
                    logger.info(
                        "Load-more phase stopped: click failed at iteration %d (source=%s)",
                        idx,
                        source,
                    )
                    break

                clicked += 1
                _random_delay(0.8, 1.8)

                for _ in range(6):
                    if self._comment_signal_count(driver, selectors) > before_signal:
                        break
                    time.sleep(0.35)
            except Exception as exc:
                logger.warning(
                    "Load-more phase exception at iteration %d: %s",
                    idx,
                    exc,
                )
                break

        logger.info("Load-more phase completed with %d click(s)", clicked)

    def _wait_for_comment_readiness(self, driver, selectors: dict[str, str]) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        container_css = selectors.get("container", "")
        item_css = selectors.get("item", "")
        text_css = selectors.get("text", "")
        load_css = selectors.get("load_more", "")

        candidate_selectors = [
            item_css,
            text_css,
            "[class*='comment'] p",
            "[class*='comment-item']",
            "[class*='comment_item']",
        ]

        def has_signal(_: Any) -> bool:
            if container_css:
                try:
                    containers = driver.find_elements(By.CSS_SELECTOR, container_css)
                    if containers:
                        for container in containers:
                            for sel in candidate_selectors:
                                if not sel:
                                    continue
                                if container.find_elements(By.CSS_SELECTOR, sel):
                                    return True
                except Exception:
                    pass

            for sel in candidate_selectors:
                if not sel:
                    continue
                try:
                    if driver.find_elements(By.CSS_SELECTOR, sel):
                        return True
                except Exception:
                    continue
            return False

        logger.info(
            "Readiness wait start: container='%s' item='%s' text='%s' load_more='%s'",
            container_css or "<none>",
            item_css or "<none>",
            text_css or "<none>",
            load_css or "<none>",
        )
        try:
            WebDriverWait(driver, 10).until(has_signal)
            logger.info("Readiness wait success: comment signal detected")
        except Exception:
            sample_counts: list[str] = []
            for sel in candidate_selectors:
                if not sel:
                    continue
                try:
                    count = len(driver.find_elements(By.CSS_SELECTOR, sel))
                except Exception:
                    count = -1
                sample_counts.append(f"{sel}={count}")
            logger.warning(
                "Readiness wait timeout: no comment signal in 10s. Selector counts: %s",
                ", ".join(sample_counts) if sample_counts else "<none>",
            )

    def _extract_text_from_node(self, driver, node, exclude_css: str) -> str:
        try:
            if exclude_css:
                txt = driver.execute_script(
                    """
                    var clone = arguments[0].cloneNode(true);
                    var excl = clone.querySelectorAll(arguments[1]);
                    excl.forEach(function(e){ e.remove(); });
                    return clone.textContent;
                    """,
                    node,
                    exclude_css,
                )
                return _normalize_text(txt or "")
            return _normalize_text((node.text or "").strip())
        except Exception as exc:
            if _is_stale_element_error(exc):
                return ""
            raise

    def _detect_block_or_rate_limit(self, driver) -> str | None:
        try:
            current_url = (driver.current_url or "").lower()
        except Exception:
            current_url = ""

        if any(token in current_url for token in ["captcha", "challenge", "blocked"]):
            return f"Redirected to challenge URL: {current_url}"

        page_text = ""
        try:
            page_text = driver.find_element("tag name", "body").text[:5000]
        except Exception:
            return None

        haystack = page_text.lower()
        for indicator in _NEWS_BLOCK_TEXT_INDICATORS:
            if indicator in haystack:
                return f"Block/rate-limit indicator found: '{indicator}'"
        return None

    def _append_candidate(
        self,
        txt: str,
        seen: set[str],
        results: list[str],
        *,
        strict_context: bool,
    ) -> None:
        normalized = _clean_comment_text(txt)
        if not normalized:
            return
        if not _is_comment_like_text(normalized):
            return
        if strict_context and _looks_like_article_blob(txt):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        results.append(normalized)

    def _extract_from_items(
        self,
        driver,
        items,
        text_css: str,
        exclude_css: str,
        seen: set[str],
        results: list[str],
        *,
        strict_context: bool,
    ) -> None:
        from selenium.webdriver.common.by import By

        for item in items:
            txt = ""
            if text_css:
                try:
                    children = item.find_elements(By.CSS_SELECTOR, text_css)
                except Exception as exc:
                    if _is_stale_element_error(exc):
                        continue
                    children = []
                for child in children:
                    txt = self._extract_text_from_node(driver, child, exclude_css)
                    if txt:
                        break
            if not txt:
                txt = self._extract_text_from_node(driver, item, exclude_css)
            self._append_candidate(
                txt,
                seen,
                results,
                strict_context=strict_context,
            )

    def _extract_comments(self, driver, selectors: dict[str, str]) -> list[str]:
        from selenium.webdriver.common.by import By

        container_css = selectors.get("container", "")
        item_css = selectors.get("item", "")
        text_css = selectors.get("text", "")
        exclude_css = selectors.get("exclude", "")

        results: list[str] = []
        seen: set[str] = set()

        has_domain_selectors = bool(container_css or item_css or text_css)
        strict_context_for_backup = not has_domain_selectors

        generic_container_item_css = ", ".join(_FALLBACK_ITEM_SELECTORS)

        if container_css:
            containers = driver.find_elements(By.CSS_SELECTOR, container_css)
            for container in containers:
                items = []
                if item_css:
                    try:
                        items = container.find_elements(By.CSS_SELECTOR, item_css)
                    except Exception:
                        items = []
                if not items and generic_container_item_css:
                    try:
                        items = container.find_elements(
                            By.CSS_SELECTOR, generic_container_item_css
                        )
                    except Exception:
                        items = []
                self._extract_from_items(
                    driver=driver,
                    items=items,
                    text_css=text_css,
                    exclude_css=exclude_css,
                    seen=seen,
                    results=results,
                    strict_context=False,
                )

        # Global pass as backup for layouts where items are not nested under container.
        # If domain selectors are missing, we keep this pass strict to avoid grabbing article body text.
        global_item_css = ", ".join(
            [css for css in [item_css, generic_container_item_css] if css]
        )
        if global_item_css:
            try:
                items = driver.find_elements(By.CSS_SELECTOR, global_item_css)
            except Exception:
                items = []
            self._extract_from_items(
                driver=driver,
                items=items,
                text_css=text_css,
                exclude_css=exclude_css,
                seen=seen,
                results=results,
                strict_context=strict_context_for_backup,
            )

        return results

    def _extract_fallback(self, driver, selectors: dict[str, str]) -> list[str]:
        from selenium.webdriver.common.by import By

        results: list[str] = []
        seen: set[str] = set()

        container_css = selectors.get("container", "")
        containers = []
        container_probe_css = ", ".join(
            [
                css
                for css in [container_css, ", ".join(_FALLBACK_CONTAINER_SELECTORS)]
                if css
            ]
        )

        if container_probe_css:
            try:
                containers = driver.find_elements(By.CSS_SELECTOR, container_probe_css)
            except Exception:
                containers = []

        text_probe_css = ", ".join(_FALLBACK_TEXT_SELECTORS)
        item_probe_css = ", ".join(_FALLBACK_ITEM_SELECTORS)

        # Pass 1: container scoped
        for container in containers:
            for css in [text_probe_css, item_probe_css]:
                if not css:
                    continue
                try:
                    elems = container.find_elements(By.CSS_SELECTOR, css)
                except Exception:
                    elems = []
                for el in elems:
                    try:
                        self._append_candidate(
                            el.text,
                            seen,
                            results,
                            strict_context=True,
                        )
                    except Exception as exc:
                        if _is_stale_element_error(exc):
                            continue
                        raise

        # Pass 2: global fallback selectors
        for sel in _FALLBACK_COMMENT_SELECTORS:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in elems:
                    try:
                        self._append_candidate(
                            el.text,
                            seen,
                            results,
                            strict_context=True,
                        )
                    except Exception as exc:
                        if _is_stale_element_error(exc):
                            continue
                        raise
            except Exception:
                continue

        return results


# ---------------------------------------------------------------------------
# Facebook comment crawler
# ---------------------------------------------------------------------------

# Known indicators that Facebook has blocked or checkpointed us.
_FB_BLOCK_INDICATORS = [
    "checkpoint",
    "login",
    "Bạn không thể sử dụng tính năng này",
    "You can't use this feature",
    "Please log in",
    "Đăng nhập",
    "content isn't available",
    "Nội dung này hiện không",
]


class FacebookCommentCrawler:
    """
    Best-effort Facebook comment crawler using undetected-chromedriver.

    Strategy:
    1. Use mbasic.facebook.com (lightweight HTML, fewer JS detections).
    2. Optionally inject cookies from a real browser session.
    3. Click "Xem thêm bình luận" / "View more comments" links.
    4. Detect blocking and abort gracefully.
    """

    def __init__(
        self,
        headless: bool = True,
        cookie_file: str | None = None,
        max_expand_clicks: int = 10,
        delay_range: tuple[float, float] = (3.0, 6.0),
    ):
        self.headless = headless
        self.cookie_file = cookie_file
        self.max_expand_clicks = max_expand_clicks
        self.delay_range = delay_range

    # ---- public API ----

    def crawl_comments(self, url: str) -> dict[str, Any]:
        """
        Return dict:
            {
                "comments": [...],
                "blocked": False,
                "block_reason": None,
                "warnings": [],
            }
        """
        result: dict[str, Any] = {
            "comments": [],
            "blocked": False,
            "block_reason": None,
            "warnings": [],
        }

        mbasic_url = self._to_mbasic(url)
        driver = _get_undetected_driver(headless=self.headless)
        try:
            # Load cookies if provided
            if self.cookie_file:
                self._inject_cookies(driver, mbasic_url)

            logger.info("FacebookCrawler: loading %s", mbasic_url)
            driver.get(mbasic_url)
            _random_delay(*self.delay_range)

            # Check for blocking
            block_reason = self._detect_block(driver)
            if block_reason:
                result["blocked"] = True
                result["block_reason"] = block_reason
                logger.warning("Facebook BLOCKED: %s", block_reason)
                return result
            logger.info("Facebook block check: no indicator detected")

            # Expand comments
            self._expand_comments(driver)

            # Extract
            comments = self._extract_comments_mbasic(driver)
            result["comments"] = comments
            logger.info("FacebookCrawler: got %d comments from %s", len(comments), url)

            if not comments:
                result["warnings"].append(
                    "No comments found — post may have no comments, "
                    "require login, or be restricted."
                )
            return result

        except Exception as exc:
            logger.exception("FacebookCrawler failed for %s", url)
            result["warnings"].append(f"Crawl exception: {exc}")
            return result
        finally:
            driver.quit()

    # ---- URL rewriting ----

    @staticmethod
    def _to_mbasic(url: str) -> str:
        """Rewrite any facebook URL to mbasic.facebook.com for lightweight HTML."""
        return re.sub(
            r"https?://(www\.|m\.|mobile\.)?facebook\.com",
            "https://mbasic.facebook.com",
            url,
        )

    # ---- cookie injection ----

    def _inject_cookies(self, driver, target_url: str) -> None:
        """
        Load cookies from a JSON file exported from browser.

        Expected format: list of dicts with at least {name, value, domain}.
        You can export cookies using browser extensions like "EditThisCookie" or
        "Cookie-Editor" → export as JSON.
        """
        if not self.cookie_file or not os.path.isfile(self.cookie_file):
            logger.warning("Cookie file not found: %s", self.cookie_file)
            return

        # Must visit the domain first so Selenium allows cookie setting
        driver.get("https://mbasic.facebook.com/")
        _random_delay(1.5, 3.0)

        try:
            with open(self.cookie_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            added = 0
            for c in cookies:
                cookie_dict: dict[str, Any] = {
                    "name": c["name"],
                    "value": c["value"],
                }
                if "domain" in c:
                    cookie_dict["domain"] = c["domain"]
                if "path" in c:
                    cookie_dict["path"] = c["path"]
                if c.get("secure"):
                    cookie_dict["secure"] = True
                try:
                    driver.add_cookie(cookie_dict)
                    added += 1
                except Exception:
                    pass
            logger.info("Injected %d / %d cookies", added, len(cookies))
        except Exception:
            logger.exception("Failed to load cookies from %s", self.cookie_file)

    # ---- block detection ----

    def _detect_block(self, driver) -> str | None:
        """Return reason string if blocked, else None."""
        current_url = driver.current_url.lower()
        if "checkpoint" in current_url or "login" in current_url:
            return f"Redirected to checkpoint/login: {driver.current_url}"

        page_text = ""
        try:
            page_text = driver.find_element("tag name", "body").text[:3000]
        except Exception:
            pass

        for indicator in _FB_BLOCK_INDICATORS:
            if indicator.lower() in page_text.lower():
                return f"Block indicator found: '{indicator}'"
        return None

    # ---- comment expansion ----

    def _expand_comments(self, driver) -> None:
        """Click 'Xem thêm bình luận' / 'View more comments' on mbasic."""
        from selenium.webdriver.common.by import By

        for _ in range(self.max_expand_clicks):
            try:
                # mbasic uses plain <a> links for "more comments"
                links = driver.find_elements(By.CSS_SELECTOR, "a")
                more_link = None
                for link in links:
                    link_text = link.text.strip().lower()
                    if any(
                        kw in link_text
                        for kw in [
                            "xem thêm",
                            "view more",
                            "more comments",
                            "bình luận trước",
                            "previous comments",
                            "xem thêm bình luận",
                        ]
                    ):
                        if link.is_displayed():
                            more_link = link
                            break
                if not more_link:
                    break
                href = more_link.get_attribute("href")
                if href:
                    driver.get(href)
                else:
                    _safe_click(driver, more_link)
                _random_delay(*self.delay_range)

                # Re-check for blocking after each page
                if self._detect_block(driver):
                    logger.warning("Facebook blocked mid-expansion, stopping")
                    break
            except Exception:
                break

    # ---- comment extraction ----

    def _extract_comments_mbasic(self, driver) -> list[str]:
        """
        Extract comment text from mbasic.facebook.com HTML.

        mbasic renders comments as <div> blocks inside the comment section.
        The structure varies, but comments are typically in <div> with
        data-sigil or specific class patterns.
        """
        from selenium.webdriver.common.by import By

        comments: list[str] = []
        seen: set[str] = set()

        # Strategy 1: elements with data-sigil="comment-body"
        try:
            elems = driver.find_elements(
                By.CSS_SELECTOR,
                '[data-sigil="comment-body"], [data-sigil*="comment-body"]',
            )
            for el in elems:
                txt = _clean_comment_text(el.text)
                if txt and txt not in seen:
                    seen.add(txt)
                    comments.append(txt)
        except Exception:
            pass

        # Strategy 2: divs inside comment containers
        if not comments:
            for sel in [
                "div[id*='comment'] div",
                "div.dw, div.do, div.dp",  # mbasic common comment wrapper classes
                "div > div > div > span",  # deeply nested comment text
            ]:
                try:
                    elems = driver.find_elements(By.CSS_SELECTOR, sel)
                    for el in elems:
                        txt = _clean_comment_text(el.text)
                        if txt and len(txt) > 5 and txt not in seen:
                            seen.add(txt)
                            comments.append(txt)
                except Exception:
                    continue

        # Strategy 3: broad text extraction as last resort
        if not comments:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    class_attr = (div.get_attribute("class") or "").lower()
                    id_attr = (div.get_attribute("id") or "").lower()
                    if "comment" in class_attr or "comment" in id_attr:
                        txt = _clean_comment_text(div.text)
                        if txt and len(txt) > 10 and txt not in seen:
                            seen.add(txt)
                            comments.append(txt)
            except Exception:
                pass

        return comments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _load_cached_meta_if_fresh(output_dir: str) -> dict[str, Any] | None:
    if CRAWL_CACHE_TTL_HOURS <= 0:
        return None

    meta_path = os.path.join(output_dir, "meta.json")
    segments_path = os.path.join(output_dir, "segments.jsonl")
    if not (os.path.isfile(meta_path) and os.path.isfile(segments_path)):
        return None

    try:
        age_seconds = time.time() - os.path.getmtime(meta_path)
    except OSError:
        return None

    if age_seconds > CRAWL_CACHE_TTL_HOURS * 3600:
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except Exception:
        return None

    required = {"url", "url_hash", "status", "total_comments"}
    if not isinstance(cached, dict) or not required.issubset(cached.keys()):
        return None

    cache_schema = str(cached.get("crawl_schema") or "")
    if cache_schema != COMMENT_CRAWL_SCHEMA_VERSION:
        logger.info(
            "Cache bypass for %s: crawl_schema=%s expected=%s",
            cached.get("url") or output_dir,
            cache_schema or "<missing>",
            COMMENT_CRAWL_SCHEMA_VERSION,
        )
        return None

    cached["output_dir"] = output_dir
    cached["from_cache"] = True
    return cached


def crawl_comments_from_url(
    url: str,
    output_base_dir: str = "data/raw/crawled_urls",
    headless: bool = True,
    fb_cookie_file: str | None = None,
    max_load_more: int = 15,
    timeout: int = 90,
) -> dict[str, Any]:
    """
    Detect URL type → dispatch to appropriate crawler → save artifacts.

    Returns a dict with:
        url, url_hash, source_type, output_dir, total_comments,
        status, blocked, block_reason, warnings
    """
    uhash = _url_hash(url)
    output_dir = os.path.join(output_base_dir, uhash)
    url_type = detect_url_type(url)

    result_meta: dict[str, Any] = {
        "url": url,
        "url_hash": uhash,
        "source_type": url_type,
        "output_dir": output_dir,
        "total_comments": 0,
        "status": "ok",
        "blocked": False,
        "block_reason": None,
        "warnings": [],
        "from_cache": False,
        "attempts": 1,
        "crawl_schema": COMMENT_CRAWL_SCHEMA_VERSION,
    }

    cached_meta = _load_cached_meta_if_fresh(output_dir)
    if cached_meta is not None:
        cached_meta.setdefault("source_type", url_type)
        cached_meta.setdefault("blocked", False)
        cached_meta.setdefault("block_reason", None)
        cached_meta.setdefault("warnings", [])
        cached_meta.setdefault("attempts", 0)
        logger.info("Cache hit for %s (age < %.2fh), skipping crawl", url, CRAWL_CACHE_TTL_HOURS)
        return cached_meta

    if url_type == "x_twitter":
        result_meta["status"] = "unsupported"
        result_meta["warnings"].append(
            "X/Twitter comment crawling is not implemented yet."
        )
        # Still save empty artifacts so pipeline does not break
        segments = build_segments_jsonl([], url)
        save_crawl_artifacts(output_dir, segments, result_meta)
        return result_meta

    comments: list[str] = []
    html_tag_effective = "comment"

    if url_type == "facebook":
        crawler = FacebookCommentCrawler(
            headless=headless,
            cookie_file=fb_cookie_file,
            max_expand_clicks=max_load_more,
        )
        fb_result = crawler.crawl_comments(url)
        comments = fb_result["comments"]
        result_meta["blocked"] = fb_result["blocked"]
        result_meta["block_reason"] = fb_result["block_reason"]
        result_meta["warnings"].extend(fb_result["warnings"])
        if fb_result["blocked"]:
            result_meta["status"] = "blocked"

    elif url_type in ("news", "unknown"):
        crawler = NewsSiteCommentCrawler(
            headless=headless,
            max_load_more_clicks=max_load_more,
            max_runtime_seconds=timeout,
        )
        comments = crawler.crawl_comments(url)
        result_meta["attempts"] = max(1, int(getattr(crawler, "last_attempts", 1)))
        if crawler.last_warnings:
            result_meta["warnings"].extend(crawler.last_warnings)
        if crawler.last_block_reason:
            result_meta["blocked"] = True
            result_meta["block_reason"] = crawler.last_block_reason
            result_meta["status"] = "blocked"
            result_meta["warnings"].append(
                f"Potential anti-bot/rate-limit detected: {crawler.last_block_reason}"
            )
        elif not comments:
            result_meta["warnings"].append(
                "No comments found after domain + fallback extraction. "
                "The page may truly have no comments, require login, or use an unsupported dynamic structure."
            )
            result_meta["status"] = "no_comments"

    result_meta["total_comments"] = len(comments)

    # Build segments in standard schema
    segments = build_segments_jsonl(comments, url, html_tag_effective)
    save_crawl_artifacts(output_dir, segments, result_meta)

    logger.info(
        "crawl_comments_from_url: %s → %d comments, status=%s",
        url,
        len(comments),
        result_meta["status"],
    )
    return result_meta


def crawl_urls(
    urls: list[str],
    out_dir: str = "data/raw/crawled_urls",
    timeout: int = 90,
    enable_video: bool = False,
    enable_asr: bool = False,
    keep_artifacts: bool = False,
    asr_max_seconds: int = 600,
    asr_language: str = "vi",
    allow_selenium_fallback: bool = True,
    fallback_decisions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Batch crawl adapter compatible with legacy setup_and_crawl.crawl_urls signature."""
    del enable_video, enable_asr, keep_artifacts, asr_max_seconds, asr_language, allow_selenium_fallback, fallback_decisions

    os.makedirs(out_dir, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_urls = len(urls)
    for idx, raw_url in enumerate(urls):
        url = (raw_url or "").strip()
        if not url:
            continue

        start = time.time()
        uhash = _url_hash(url)
        output_dir = os.path.join(out_dir, uhash)
        segments_path = os.path.join(output_dir, "segments.jsonl")

        try:
            crawl_meta = crawl_comments_from_url(
                url=url,
                output_base_dir=out_dir,
                timeout=timeout,
            )
            crawl_status = str(crawl_meta.get("status") or "error")
            if crawl_status in {"ok", "no_comments", "blocked", "unsupported"}:
                mapped_status = "ok" if crawl_status == "ok" else "error"
            else:
                mapped_status = "error"

            warnings = crawl_meta.get("warnings") or []
            error_detail = crawl_meta.get("block_reason") or crawl_status
            if mapped_status != "ok" and crawl_status == "no_comments" and warnings:
                error_detail = f"no_comments: {warnings[-1]}"

            results.append(
                {
                    "url": url,
                    "url_hash": uhash,
                    "status": mapped_status,
                    "crawl_status": crawl_status,
                    "blocked": bool(crawl_meta.get("blocked") or False),
                    "block_reason": crawl_meta.get("block_reason"),
                    "from_cache": bool(crawl_meta.get("from_cache") or False),
                    "attempts": int(crawl_meta.get("attempts") or 1),
                    "error": None if mapped_status == "ok" else error_detail,
                    "output_dir": output_dir,
                    "segments_path": segments_path,
                    "num_segments": int(crawl_meta.get("total_comments") or 0),
                    "method": "comment_crawl",
                    "duration_sec": round(time.time() - start, 3),
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "url": url,
                    "url_hash": uhash,
                    "status": "error",
                    "crawl_status": "error",
                    "blocked": False,
                    "block_reason": None,
                    "from_cache": False,
                    "attempts": 1,
                    "error": f"Unexpected crawl error: {exc}",
                    "output_dir": output_dir,
                    "segments_path": None,
                    "num_segments": 0,
                    "method": "comment_crawl",
                    "duration_sec": round(time.time() - start, 3),
                    "warnings": [],
                }
            )

        if idx < total_urls - 1:
            inter_url_delay = random.uniform(
                BATCH_INTER_URL_DELAY_MIN,
                BATCH_INTER_URL_DELAY_MAX,
            )
            if inter_url_delay > 0:
                logger.info(
                    "Inter-URL throttle: sleeping %.2fs before next URL",
                    inter_url_delay,
                )
                time.sleep(inter_url_delay)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Crawl comments from news sites or Facebook posts."
    )
    parser.add_argument("url", help="URL to crawl comments from")
    parser.add_argument(
        "--output-dir",
        default="data/raw/crawled_urls",
        help="Base output directory (default: data/raw/crawled_urls)",
    )
    parser.add_argument(
        "--fb-cookies",
        default=None,
        help="Path to Facebook cookie JSON file (exported from browser)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode (useful for debugging)",
    )
    parser.add_argument(
        "--max-load-more",
        type=int,
        default=15,
        help="Max times to click load-more / expand button (default: 15)",
    )
    args = parser.parse_args()

    result = crawl_comments_from_url(
        url=args.url,
        output_base_dir=args.output_dir,
        headless=not args.no_headless,
        fb_cookie_file=args.fb_cookies,
        max_load_more=args.max_load_more,
    )

    print("\n" + "=" * 60)
    print(f"URL:            {result['url']}")
    print(f"Type:           {result['source_type']}")
    print(f"Status:         {result['status']}")
    print(f"Comments:       {result['total_comments']}")
    print(f"Output:         {result['output_dir']}")
    if result["blocked"]:
        print(f"BLOCKED:        {result['block_reason']}")
    if result["warnings"]:
        print(f"Warnings:       {result['warnings']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
