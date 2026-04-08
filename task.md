Vấn đề hiện tại trong code
Nhìn vào comment_crawl.py, có một số điểm yếu chính:

User-Agent cố định — browser fingerprint dễ bị nhận dạng
Delay quá dự đoán được — _random_delay(1.5, 4.0) vẫn còn pattern
Không có retry/backoff — thất bại là thất bại luôn
Không rotate IP/proxy — cùng IP hit liên tục
Không có request-level throttling — nhiều URL crawl nối tiếp nhau không nghỉ


Giải pháp theo mức độ ưu tiên
1. 🟢 Quick wins — không cần thay đổi architecture
Thêm random User-Agent rotation vào _common_args() và _get_driver_edge/chrome:
pythonfrom fake_useragent import UserAgent

def _common_args() -> list[str]:
    ua = UserAgent()
    return [
        f"--user-agent={ua.random}",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--lang=vi-VN",
        "--window-size=1280,900",
        # Chống phát hiện automation
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
    ]
Tăng entropy cho delay — thêm "human-like" pauses:
pythondef _random_delay(lo: float = 1.5, hi: float = 4.0) -> None:
    base = random.uniform(lo, hi)
    # Đôi khi dừng lâu hơn như người thật đọc trang
    if random.random() < 0.15:
        base += random.uniform(3.0, 8.0)
    time.sleep(base)
Inject JS để ẩn automation fingerprint ngay sau khi tạo driver:
pythondef _patch_driver_stealth(driver) -> None:
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
            Object.defineProperty(navigator, 'languages', {get: () => ['vi-VN','vi','en']});
            window.chrome = { runtime: {} };
        """
    })

2. 🟡 Proxy rotation — giải quyết block theo IP
Thêm proxy support vào _get_driver_edge và _get_driver_chrome_uc:
python# pip install requests[socks]
PROXY_LIST = [
    "socks5://user:pass@proxy1:1080",
    "socks5://user:pass@proxy2:1080",
    # hoặc dùng residential proxies từ Webshare, Oxylabs, BrightData
]

def _get_random_proxy() -> str | None:
    if not PROXY_LIST:
        return None
    return random.choice(PROXY_LIST)

def _get_driver_edge(headless: bool, browser_bin: str | None, proxy: str | None = None):
    from selenium.webdriver.edge.options import Options as EdgeOptions
    options = EdgeOptions()
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
    # ... rest of setup
Tích hợp vào _get_undetected_driver:
pythondef _get_undetected_driver(headless: bool = True, use_proxy: bool = True):
    proxy = _get_random_proxy() if use_proxy else None
    browser_bin, browser_type = _find_browser_binary()
    if browser_type == "edge":
        return _get_driver_edge(headless, browser_bin, proxy)
    else:
        return _get_driver_chrome_uc(headless, browser_bin, proxy)

3. 🟡 Exponential backoff + retry decorator
Bọc các bước crawl chính bằng retry logic:
pythonimport functools

def with_retry(max_attempts: int = 3, base_delay: float = 5.0, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts - 1:
                        raise
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, max_attempts, exc, wait
                    )
                    time.sleep(wait)
        return wrapper
    return decorator

# Áp dụng:
@with_retry(max_attempts=3, base_delay=10.0, exceptions=(RuntimeError,))
def crawl_comments(self, url: str) -> list[str]:
    ...

4. 🟡 Request throttling giữa các URL trong batch
Trong crawl_urls(), thêm nghỉ giữa các URL:
pythonfor i, raw_url in enumerate(urls):
    # ...
    if i > 0:
        # Nghỉ ngẫu nhiên giữa các URL — tránh bị rate limit theo batch
        inter_url_delay = random.uniform(15.0, 45.0)
        logger.info("Waiting %.1fs before next URL...", inter_url_delay)
        time.sleep(inter_url_delay)

5. 🔴 Playwright thay Selenium — cho VNExpress (nếu Selenium bị block)
VNExpress đôi khi detect Selenium rõ hơn. Playwright có stealth tốt hơn:
bashpip install playwright playwright-stealth
playwright install chromium
python# comment_crawl_playwright.py — fallback strategy
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

def crawl_vnexpress_playwright(url: str) -> list[str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="vi-VN",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        )
        page = context.new_page()
        stealth_sync(page)  # inject stealth scripts
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector("#box_comment_vne", timeout=12000)
        # scroll và extract như cũ
        comments = page.eval_on_selector_all(
            ".comment_item .full_content",
            "els => els.map(e => e.textContent.trim())"
        )
        browser.close()
        return [c for c in comments if c]
Tích hợp vào NewsSiteCommentCrawler.crawl_comments() như một fallback sau khi Selenium thất bại.

6. 🔴 Caching / deduplication để giảm số request
Trong crawl_comments_from_url(), skip crawl nếu artifact đã tồn tại và còn mới:
pythondef _is_cache_fresh(output_dir: str, max_age_hours: float = 6.0) -> bool:
    meta_path = os.path.join(output_dir, "meta.json")
    seg_path = os.path.join(output_dir, "segments.jsonl")
    if not os.path.isfile(meta_path) or not os.path.isfile(seg_path):
        return False
    age_seconds = time.time() - os.path.getmtime(meta_path)
    return age_seconds < max_age_hours * 3600

# Trong crawl_comments_from_url():
if _is_cache_fresh(output_dir):
    logger.info("Cache hit for %s, skipping crawl", url)
    # Load và return cached meta
    with open(os.path.join(output_dir, "meta.json")) as f:
        return json.load(f)

Tóm tắt priority
Giải phápEffortImpactLàm ngay?User-Agent rotation + stealth JSThấpCao✅ CóDelay entropyThấpTrung✅ CóRetry + backoffThấpCao✅ CóInter-URL throttlingThấpCao✅ CóArtifact cachingThấpTrung✅ CóProxy rotationTrungRất caoNếu vẫn bị blockPlaywright fallbackTrungCaoNếu vẫn bị block