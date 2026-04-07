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

# ---------------------------------------------------------------------------
# Segment schema helpers (mirrors setup_and_crawl.py contract)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """NFC + collapse whitespace + strip."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
        text = _normalize_text(raw_text)
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


def _common_args() -> list[str]:
    """Shared browser arguments."""
    return [
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--lang=vi-VN",
        "--window-size=1280,900",
    ]


def _get_driver_edge(headless: bool, browser_bin: str | None):
    """
    Build a standard selenium Edge WebDriver.

    selenium-manager (bundled since Selenium 4.6) auto-downloads msedgedriver
    matching the installed Edge version — no manual driver management needed.
    """
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService

    options = EdgeOptions()
    if headless:
        options.add_argument("--headless=new")
    for arg in _common_args():
        options.add_argument(arg)
    if browser_bin:
        options.binary_location = browser_bin

    service = EdgeService()  # selenium-manager resolves msedgedriver automatically
    driver = webdriver.Edge(service=service, options=options)
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


def _get_driver_chrome_uc(headless: bool, browser_bin: str | None):
    """Build undetected_chromedriver for Chrome."""
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    for arg in _common_args():
        options.add_argument(arg)
    if browser_bin:
        options.binary_location = browser_bin

    detected_major = _detect_chrome_major_version(browser_bin)
    if detected_major is not None:
        logger.info("Detected Chrome major version: %s", detected_major)
        driver = uc.Chrome(options=options, version_main=detected_major)
    else:
        driver = uc.Chrome(options=options)

    driver.set_page_load_timeout(45)
    driver.implicitly_wait(5)
    return driver


def _get_undetected_driver(headless: bool = True):
    """
    Create a browser driver backed by whichever Chromium browser is available.

    - Edge detected  → selenium.webdriver.Edge (selenium-manager handles msedgedriver)
    - Chrome detected → undetected_chromedriver (keeps anti-bot patching)
    """
    browser_bin, browser_type = _find_browser_binary()
    logger.info("Using %s binary: %s", browser_type, browser_bin)

    if browser_type == "edge":
        return _get_driver_edge(headless, browser_bin)
    else:
        return _get_driver_chrome_uc(headless, browser_bin)


def _random_delay(lo: float = 1.5, hi: float = 4.0) -> None:
    time.sleep(random.uniform(lo, hi))


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
        "container": "#box_comment_vne, .box-comment",
        "item": ".comment_item",
        "text": ".full_content",
        "exclude": ".txt-name",
        "load_more": "a.view_more_coment, a.btn-more-comment",
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
        "container": ".comment-box, #comment-box",
        "item": ".comment-item",
        "text": ".comment-content, p",
        "load_more": ".btn-more, .load-more-comment",
    },
    "vietnamnet.vn": {
        "container": "#box_comment, .comment-container",
        "item": ".comment-item",
        "text": ".comment-content, p",
        "load_more": ".btn-more-comment",
    },
}

# Fallback heuristic selectors for unknown news domains
_FALLBACK_COMMENT_SELECTORS = [
    "[class*='comment'] [class*='content']",
    "[class*='comment'] p",
    "[id*='comment'] [class*='content']",
    "[id*='comment'] p",
    "[data-component*='comment'] p",
    ".comment-body",
    ".comment-text",
]


class NewsSiteCommentCrawler:
    """Crawl only the comment section of Vietnamese news websites."""

    def __init__(self, headless: bool = True, max_load_more_clicks: int = 15):
        self.headless = headless
        self.max_load_more_clicks = max_load_more_clicks

    # ---- public API ----

    def crawl_comments(self, url: str) -> list[str]:
        """Return list of comment text strings from *url*."""
        from selenium.common.exceptions import (
            NoSuchElementException,
            TimeoutException,
        )
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        driver = _get_undetected_driver(headless=self.headless)
        try:
            logger.info("NewsSiteCrawler: loading %s", url)
            driver.get(url)
            _random_delay(2.0, 4.0)

            selectors = self._pick_selectors(url)
            comments: list[str] = []

            # Wait for comment container (up to 12s)
            container_css = selectors.get("container", "")
            if container_css:
                try:
                    WebDriverWait(driver, 12).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, container_css))
                    )
                except TimeoutException:
                    logger.warning(
                        "Comment container not found within timeout for %s", url
                    )

            # Expand all comments
            self._click_load_more(driver, selectors)

            # Extract comment texts
            comments = self._extract_comments(driver, selectors)

            if not comments:
                logger.info(
                    "Domain selectors yielded 0 comments, trying fallback heuristics"
                )
                # Avoid implicit-wait amplification: fallback probes many selectors.
                # With implicit_wait=5s, each miss can block, causing long no-comment scans.
                driver.implicitly_wait(0)
                try:
                    comments = self._extract_fallback(driver)
                finally:
                    driver.implicitly_wait(5)

            logger.info("NewsSiteCrawler: got %d comments from %s", len(comments), url)
            return comments

        except Exception:
            logger.exception("NewsSiteCrawler failed for %s", url)
            return []
        finally:
            driver.quit()

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

    def _click_load_more(self, driver, selectors: dict[str, str]) -> None:
        from selenium.webdriver.common.by import By

        load_css = selectors.get("load_more")
        if not load_css:
            return
        for _ in range(self.max_load_more_clicks):
            try:
                btns = driver.find_elements(By.CSS_SELECTOR, load_css)
                visible = [b for b in btns if b.is_displayed()]
                if not visible:
                    break
                if not _safe_click(driver, visible[0]):
                    break
                _random_delay(1.0, 2.5)
            except Exception:
                break

    def _extract_comments(self, driver, selectors: dict[str, str]) -> list[str]:
        from selenium.webdriver.common.by import By

        item_css = selectors.get("item")
        text_css = selectors.get("text")
        exclude_css = selectors.get("exclude")
        if not item_css:
            return []

        items = driver.find_elements(By.CSS_SELECTOR, item_css)
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            txt = ""
            if text_css:
                children = item.find_elements(By.CSS_SELECTOR, text_css)
                for child in children:
                    if exclude_css:
                        # Use JS to clone, strip excluded nodes, get clean text
                        txt = driver.execute_script(
                            """
                            var clone = arguments[0].cloneNode(true);
                            var excl = clone.querySelectorAll(arguments[1]);
                            excl.forEach(function(e){ e.remove(); });
                            return clone.textContent;
                            """,
                            child,
                            exclude_css,
                        )
                    else:
                        txt = child.text
                    txt = (txt or "").strip()
                    if txt:
                        break
            if not txt:
                txt = item.text.strip()
            normalized = _normalize_text(txt)
            if normalized and normalized not in seen:
                seen.add(normalized)
                results.append(normalized)
        return results

    def _extract_fallback(self, driver) -> list[str]:
        from selenium.webdriver.common.by import By

        results: list[str] = []
        seen: set[str] = set()
        for sel in _FALLBACK_COMMENT_SELECTORS:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in elems:
                    txt = _normalize_text(el.text)
                    if txt and len(txt) > 5 and txt not in seen:
                        seen.add(txt)
                        results.append(txt)
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
                txt = _normalize_text(el.text)
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
                        txt = _normalize_text(el.text)
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
                        txt = _normalize_text(div.text)
                        if txt and len(txt) > 10 and txt not in seen:
                            seen.add(txt)
                            comments.append(txt)
            except Exception:
                pass

        return comments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def crawl_comments_from_url(
    url: str,
    output_base_dir: str = "data/raw/crawled_urls",
    headless: bool = True,
    fb_cookie_file: str | None = None,
    max_load_more: int = 15,
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
    }

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
        )
        comments = crawler.crawl_comments(url)
        if not comments:
            result_meta["warnings"].append(
                "No comments found. The page may not have comments, "
                "or the comment section uses an unsupported structure."
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
    del timeout, enable_video, enable_asr, keep_artifacts, asr_max_seconds, asr_language, allow_selenium_fallback, fallback_decisions

    os.makedirs(out_dir, exist_ok=True)

    results: list[dict[str, Any]] = []
    for raw_url in urls:
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
            )
            status = str(crawl_meta.get("status") or "error")
            if status in {"ok", "no_comments", "blocked", "unsupported"}:
                mapped_status = "ok" if status == "ok" else "error"
            else:
                mapped_status = "error"

            results.append(
                {
                    "url": url,
                    "url_hash": uhash,
                    "status": mapped_status,
                    "error": None if mapped_status == "ok" else (crawl_meta.get("block_reason") or status),
                    "output_dir": output_dir,
                    "segments_path": segments_path,
                    "num_segments": int(crawl_meta.get("total_comments") or 0),
                    "method": "comment_crawl",
                    "duration_sec": round(time.time() - start, 3),
                    "warnings": crawl_meta.get("warnings") or [],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "url": url,
                    "url_hash": uhash,
                    "status": "error",
                    "error": f"Unexpected crawl error: {exc}",
                    "output_dir": output_dir,
                    "segments_path": None,
                    "num_segments": 0,
                    "method": "comment_crawl",
                    "duration_sec": round(time.time() - start, 3),
                    "warnings": [],
                }
            )

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
