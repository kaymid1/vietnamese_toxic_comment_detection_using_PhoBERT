# YouTube Comment Collection — Implementation Notes

**Status:** Implemented API-based adapter; supported by mocked/unit tests
**Last updated:** 2026-08-22

---

## Goal

> Given a public YouTube video URL, extract public user comments and convert them into the existing common crawler/segment representation so the existing inference pipeline can process them without modification.

---

## YouTube Data API v3 — Preferred Approach

### Why the official API is preferred

| Concern | API approach | Selenium/DOM approach |
|---------|-------------|----------------------|
| Reliability | High (Google-backed JSON) | Low (DOM changes, anti-bot, dynamic loading) |
| Maintenance | Low (versioned API) | High (CSS selectors break on redesign) |
| Rate limits | Quota-based (predictable) | Unpredictable; IP bans possible |
| Authentication | API key only | Potentially requires login/cookies |
| Speed | Fast HTTP requests | Slow (browser launch + scroll) |
| Data quality | Structured JSON, stable fields | Fragile text scraping |
| Vietnamese Unicode | Correct (JSON encoding) | Browser-dependent |
| Reply threading | Native (`replies` field) | Requires extra scroll/clicks |

The official API satisfies the project requirements for extracting public comments. Selenium/browser crawling should be treated as a last-resort fallback and is **not recommended** for YouTube.

---

## URL Normalisation

YouTube videos are accessible via multiple URL formats. All must be normalised to extract the `videoId` before any API call:

| URL format | Example | Extraction method |
|-----------|---------|------------------|
| Standard watch URL | `https://www.youtube.com/watch?v=dQw4w9WgXcQ` | `urllib.parse.parse_qs(urlparse(url).query)["v"][0]` |
| Short URL | `https://youtu.be/dQw4w9WgXcQ` | `urlparse(url).path.lstrip("/")` |
| YouTube Shorts | `https://www.youtube.com/shorts/dQw4w9WgXcQ` | `urlparse(url).path.split("/shorts/")[1]` |
| Embed URL | `https://www.youtube.com/embed/dQw4w9WgXcQ` | `urlparse(url).path.split("/embed/")[1]` |
| Mobile URL | `https://m.youtube.com/watch?v=dQw4w9WgXcQ` | Same as standard |

The implemented `extract_youtube_video_id(url: str) -> str | None` handles these variants and returns `None` for invalid/non-YouTube URLs. `comment_crawl.py` uses this validation before dispatching to the YouTube crawler.

---

## API Concepts

### Authentication

- Requires a **Google Cloud project** with the **YouTube Data API v3** enabled.
- Authentication uses an **API key** (not OAuth2) since only public data is accessed.
- The key is passed as the `key` query parameter by the lightweight `requests` service wrapper.
- **The API key must never be stored in source control.** Use environment variable `YOUTUBE_DATA_API_KEY`.

### `commentThreads.list` — Top-level comments

Endpoint: `GET https://www.googleapis.com/youtube/v3/commentThreads`

Required parameters:
- `part=snippet,replies` — include comment text and up to 5 inline replies
- `videoId=<videoId>` — the target video
- `maxResults=100` — maximum per page (hard API limit: 100)
- `key=<api_key>`

Optional parameters:
- `order=relevance` (default) or `order=time`
- `pageToken=<nextPageToken>` — for pagination

Response fields of interest:

```json
{
  "items": [
    {
      "id": "<commentThreadId>",
      "snippet": {
        "totalReplyCount": 3,
        "topLevelComment": {
          "id": "<commentId>",
          "snippet": {
            "textDisplay": "Comment text (may contain HTML entities)",
            "textOriginal": "Raw comment text (preferred)",
            "authorDisplayName": "Username",
            "likeCount": 42,
            "publishedAt": "2024-01-15T10:30:00.000Z",
            "updatedAt": "2024-01-15T10:30:00.000Z"
          }
        },
        "replies": {
          "comments": [
            { "id": "...", "snippet": { "textOriginal": "...", ... } }
          ]
        }
      }
    }
  ],
  "nextPageToken": "<token>",
  "pageInfo": { "totalResults": 1500, "resultsPerPage": 100 }
}
```

> Use `textOriginal` (raw text, no HTML) when available. Fall back to `html.unescape(textDisplay)` if `textOriginal` is absent.

### Pagination via `nextPageToken`

```python
page_token = None
while True:
    response = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        pageToken=page_token,
        key=api_key,
    ).execute()
    process_items(response["items"])
    page_token = response.get("nextPageToken")
    if not page_token:
        break
```

### Replies beyond the first 5

`commentThreads.list` with `part=replies` returns up to 5 replies inline. If `totalReplyCount > 5`, fetch all replies with:

```
GET /youtube/v3/comments?part=snippet&parentId=<commentThreadId>&maxResults=100&key=<key>
```

This is a separate quota-consuming call. The current implementation supports optional inline replies returned by `commentThreads.list`; full reply expansion with `comments.list` is not implemented.

### Comments disabled

If comments are disabled, `commentThreads.list` raises an HTTP 403 with error reason `commentsDisabled`. This must be caught and translated to `status="no_comments"` + a `warning` entry — not a hard failure.

### Deleted or unavailable comments

Individual items in the response with missing or empty `textOriginal`/`textDisplay` should be silently skipped.

### API Quota

| Call | Quota cost |
|------|-----------|
| `commentThreads.list` | 1 unit per request |
| `comments.list` (replies) | 1 unit per request |
| Default daily limit | **10,000 units/project** (external API constraint) |

`commentThreads.list` costs 1 unit per request and can return up to 100 comment threads per request.

Therefore 500 top-level threads typically require approximately 5 list requests.

Fetching complete replies can require additional `comments.list` requests, so actual quota cost depends on pagination and reply policy.

Practical considerations:
- Default quota limit of 10,000 units/day is an external Google Cloud project constraint.
- At up to 100 comments per page, retrieving 1,000 top-level comments typically takes ~10 quota units.
- Fetching complete reply threads via `comments.list` beyond the initial 5 inline replies consumes 1 additional quota unit per reply page.
- **Recommendation:** Cap collection per video (e.g. `max_comments_per_url` = 500 top-level + inline replies only) to keep consumption predictable.

A quota increase can be requested through Google Cloud Console if high-volume live collection is needed.

---

## Browser/HTML/Selenium Crawling — Fallback Evaluation

Selenium crawling of YouTube comments is **not recommended** as the primary approach because:

1. **Dynamic loading** — YouTube comments load via XHR after scroll events; initial HTML contains no comments.
2. **Infinite scroll** — the "load more" mechanism requires repeated scroll simulation; complex and fragile.
3. **Anti-bot behaviour** — YouTube actively detects automation; bot detection triggers CAPTCHAs and rate limits.
4. **Login requirements** — Age-restricted or member-only videos require authenticated sessions.
5. **DOM instability** — YouTube's frontend is regularly redesigned; CSS selectors break without warning.
6. **Maintenance cost** — High; requires browser binary management and selector updates on each redesign.
7. **Speed** — Launching a browser per-video is orders of magnitude slower than an API call.

**When Selenium would be the only option:**

- The video is private or requires login and the project has authorised session cookies (unlikely for a public research demo).
- The API quota is permanently exhausted and no alternatives exist.

Even in those cases, the existing `_get_undetected_driver()` infrastructure from `comment_crawl.py` could be reused, but it should be a clearly-documented fallback, not the primary path.

---

## Implemented Integration Design

### Updated URL dispatch

```python
# comment_crawl.py — detect_url_type()
_YOUTUBE_PATTERNS = re.compile(
    r"(youtube\.com/watch|youtube\.com/shorts|youtu\.be|m\.youtube\.com/watch|youtube\.com/embed)",
    re.IGNORECASE,
)

def detect_url_type(url: str) -> str:
    if _FB_PATTERNS.search(url):
        return "facebook"
    if _X_PATTERNS.search(url):
        return "x_twitter"
    if _YOUTUBE_PATTERNS.search(url):
        return "youtube"
    ...
```

### Updated dispatch in `crawl_comments_from_url()`

```python
elif url_type == "youtube":
    from comment_crawl_youtube import YouTubeCommentCrawler
    crawler = YouTubeCommentCrawler(
        api_key=os.getenv("YOUTUBE_DATA_API_KEY", ""),
        max_comments=max_comments_per_url,
    )
    yt_result = crawler.crawl_comments(url)
    comments = yt_result["comments"]
    result_meta["blocked"] = yt_result.get("blocked", False)
    result_meta["block_reason"] = yt_result.get("block_reason")
    result_meta["warnings"].extend(yt_result.get("warnings", []))
    if not comments and not yt_result.get("blocked"):
        result_meta["status"] = "no_comments"
```

### Full adapter flow

```text
YouTube URL
    ↓
detect_url_type() → "youtube"
    ↓
extract_youtube_video_id(url)
    ↓
YouTubeCommentCrawler.crawl_comments(url)
    ├── GET commentThreads.list (paginated, up to max_comments)
    ├── handle: commentsDisabled → status="no_comments"
    ├── handle: quotaExceeded → status="error" + warning
    ├── handle: videoNotFound → status="error" + warning
    └── inline replies (≤ 5 per thread, no extra quota cost)
    ↓
list[str] of cleaned comment texts
    ↓
build_segments_jsonl()          ← shared, unchanged
    ↓
save_crawl_artifacts()          ← shared, unchanged
    ↓
infer_crawled_local.py          ← shared, unchanged
```

### Implemented module: `comment_crawl_youtube.py`

```python
# comment_crawl_youtube.py (simplified shape)

import os, re, html, logging
from urllib.parse import urlparse, parse_qs
from typing import Any

logger = logging.getLogger(__name__)

YOUTUBE_DEFAULT_MAX_RESULTS = 100
YOUTUBE_DEFAULT_MAX_COMMENTS = 500


def extract_youtube_video_id(url: str) -> str | None:
    """Parse videoId from any supported YouTube URL format."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "youtu.be" in host:
        return parsed.path.lstrip("/").split("?")[0] or None
    if "youtube.com" in host or "m.youtube.com" in host:
        if "/shorts/" in parsed.path:
            return parsed.path.split("/shorts/")[1].split("?")[0] or None
        if "/embed/" in parsed.path:
            return parsed.path.split("/embed/")[1].split("?")[0] or None
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    return None


class YouTubeCommentCrawler:
    """
    Crawl public YouTube video comments via the YouTube Data API v3.

    Adapter-pattern: returns list[str] of cleaned comment texts
    for consumption by the shared build_segments_jsonl() pipeline.
    """

    def __init__(
        self,
        api_key: str,
        max_comments: int = YOUTUBE_DEFAULT_MAX_COMMENTS,
    ):
        self.api_key = api_key
        self.max_comments = max(0, int(max_comments))

    def crawl_comments(self, url: str) -> dict[str, Any]:
        """
        Returns:
            {
                "comments": list[str],
                "blocked": bool,
                "block_reason": str | None,
                "warnings": list[str],
            }
        """
        result: dict[str, Any] = {
            "comments": [],
            "blocked": False,
            "block_reason": None,
            "warnings": [],
        }

        video_id = extract_youtube_video_id(url)
        if not video_id:
            result["blocked"] = True
            result["block_reason"] = f"Could not extract videoId from URL: {url}"
            return result

        if not self.api_key:
            result["blocked"] = True
            result["block_reason"] = "YOUTUBE_DATA_API_KEY is not configured"
            return result

        try:
            comments = self._fetch_comments(video_id, result["warnings"])
            result["comments"] = comments
        except Exception as exc:
            result["warnings"].append(f"YouTube API error: {exc}")
            logger.exception("YouTubeCommentCrawler failed for %s", url)

        return result

    def _fetch_comments(self, video_id: str, warnings: list[str]) -> list[str]:
        """Paginated commentThreads.list + inline reply extraction."""
        # Implementation note: use google-api-python-client or plain HTTP requests
        # ...
        pass
```

---

## Schema Mapping

Map YouTube API fields to the existing segment schema:

| YouTube API field | Segment / meta field | Notes |
|-------------------|---------------------|-------|
| `topLevelComment.snippet.textOriginal` | `text` | Preferred; HTML-unescape if absent |
| `topLevelComment.id` | `source_comment_id` (proposed) | Stable, dedup-friendly |
| `topLevelComment.snippet.publishedAt` | `published_at` (proposed) | ISO 8601 |
| `topLevelComment.snippet.likeCount` | `like_count` (proposed) | Optional; aids future ranking |
| `replies.comments[].snippet.textOriginal` | `text` (separate segments) | Reply `reply_to` field maps to parent ID |
| `videoId` | `url_hash` (via MD5 of canonical URL) | Existing mechanism; no change needed |

The existing `segment_hash = sha256(normalized_text + "|" + html_tag_effective)` already provides content-based deduplication. The current YouTube adapter returns comment text into the shared schema; richer source metadata such as `source_comment_id` may be added later if the shared segment schema is intentionally versioned.

---

## Pagination Strategy

```text
page_token = None
collected = 0

while True:
    response = fetch_page(video_id, page_token, max_results=100)

    for thread in response["items"]:
        collect top-level comment
        collect inline replies (≤ 5, no extra quota)
        if max_comments > 0 and collected >= max_comments:
            break

    page_token = response.get("nextPageToken")
    if not page_token or (max_comments > 0 and collected >= max_comments):
        break
```

**Deduplication across pages:** Track seen `source_comment_id` values (not text content) to prevent page-overlap duplicates. The API guarantees non-overlapping pages when using `nextPageToken`, but re-crawl dedup still benefits from ID-based checks.

---

## Reply Handling

**Current behavior:** Collect top-level comments by default. When `include_replies=True`, collect inline replies included in the `commentThreads.list` response.

**Full reply expansion (optional, deferred):**
- Detect threads where `totalReplyCount > len(replies.comments)`.
- Call `comments.list(parentId=<threadId>)` for those threads.
- Each such call costs 1 additional quota unit.

Full reply expansion may be added later if quota and product requirements justify the extra API calls.

---

## Error Handling

| Error condition | Handling |
|----------------|---------|
| `commentsDisabled` (HTTP 403) | `status="no_comments"`, warning logged |
| `videoNotFound` (HTTP 404) | `status="error"`, warning logged |
| `quotaExceeded` (HTTP 403) | `status="error"`, critical warning; stop all requests |
| `forbidden` (private/age-restricted) | `status="blocked"`, warning logged |
| Invalid URL / no `videoId` | `status="error"` before any API call |
| `YOUTUBE_DATA_API_KEY` missing | `status="error"` before any API call |
| Network timeout | Returned as a YouTube API warning by the service wrapper |
| Malformed API response | Skip item, log warning, continue |

---

## Dedup Strategy

1. **Within a crawl session:** Track normalized comment text in a `set` and skip duplicates across pages.
2. **Across crawl sessions:** The existing `segment_hash = sha256(normalized_text + "|" + html_tag_effective)` deduplicates content in the shared pipeline.

---

## Logging

Follow the existing `comment_crawl.py` logging pattern:

```python
logger.info("YouTubeCommentCrawler: fetching comments for videoId=%s", video_id)
logger.info("YouTubeCommentCrawler: page %d → %d comment(s)", page_num, len(page_items))
logger.info("YouTubeCommentCrawler: finished with %d comment(s)", total)
logger.warning("YouTubeCommentCrawler: commentsDisabled for videoId=%s", video_id)
logger.warning("YouTubeCommentCrawler: quotaExceeded — stopping")
logger.exception("YouTubeCommentCrawler: unexpected error for %s", url)
```

All warnings are appended to `result["warnings"]` for inclusion in `meta.json`.

---

## Configuration Requirements

| Variable | Required | Description |
|----------|----------|-------------|
| `YOUTUBE_DATA_API_KEY` | **Yes** | Google Cloud API key with YouTube Data API v3 enabled |
| `YOUTUBE_MAX_COMMENTS_PER_VIDEO` | No | Default cap per video (overridden by `max_comments_per_url`); suggested default 500 |
| `YOUTUBE_INCLUDE_REPLIES` | No | `"true"` to include inline replies returned by `commentThreads.list`; default `"false"` |

### Secret handling

- **Do not store the API key in `.env.local` if committed to version control.**
- Add `YOUTUBE_DATA_API_KEY` to `.gitignore`-protected local env files (already excluded via the existing `.env.local` pattern).
- For production/Docker deployment, inject via environment variable or a secrets manager.
- The existing `backend/.env.local` is already gitignored; this is the correct place for local development.

---

## Implemented Code Changes

### Existing files changed

| File | Change |
|------|--------|
| `comment_crawl.py` | YouTube URL detection, `"youtube"` branch in `detect_url_type()`, and YouTube branch in `crawl_comments_from_url()` |
| `backend/.env.local` (local only) | Add `YOUTUBE_DATA_API_KEY=...` |

### New files

| File | Purpose |
|------|---------|
| `comment_crawl_youtube.py` | `YouTubeCommentCrawler` class + `extract_youtube_video_id()` utility |
| `tests/test_comment_crawl_youtube.py` | Unit tests with mocked API responses |

No changes to:
- `infer_crawled_local.py`
- `backend/app.py` (beyond the env var read, which is already generic)
- `domain_classifier.py`
- `mlflow_*` code
- Frontend code
- `build_segments_jsonl()` / `save_crawl_artifacts()` (shared, unchanged)

---

## Testing

The current tests use mocked API responses; live API collection still depends on a configured key, quota, permissions, comments-enabled videos, and platform availability.

### Test cases

| # | Scenario | Test method |
|---|----------|-------------|
| 1 | Standard `youtube.com/watch?v=ID` URL | `extract_youtube_video_id` returns correct ID |
| 2 | `youtu.be/ID` short URL | `extract_youtube_video_id` returns correct ID |
| 3 | `youtube.com/shorts/ID` URL | `extract_youtube_video_id` returns correct ID |
| 4 | Invalid / non-YouTube URL | `extract_youtube_video_id` returns `None` |
| 5 | Video with comments disabled | API mock returns `commentsDisabled` 403 → `status="no_comments"`, `comments=[]` |
| 6 | Video with 0 comments | API mock returns empty `items` → `comments=[]`, no error |
| 7 | Multi-page comments (500+) | Mock returns 2 pages with `nextPageToken`; verify all comments collected without duplicates |
| 8 | Comments with inline replies | Mock reply items in `replies.comments`; verify replies appear as separate segments |
| 9 | Vietnamese Unicode comments | Verify NFC normalisation preserves diacritics correctly; `textOriginal` → `_normalize_text()` |
| 10 | Duplicate pagination/retry | Same comment ID on two pages; verify dedup via `source_comment_id` set |
| 11 | Deleted/unavailable comment | Mock item with empty `textOriginal`; verify item is silently skipped |
| 12 | API error / rate-limit / quota exceeded | Mock `quotaExceeded` 403; verify `status="error"`, crawl stops, warning logged |
| 13 | Stable schema mapping | Verify `build_segments_jsonl()` output from YouTube comments matches existing segment schema |
| 14 | Existing news crawlers unaffected | `detect_url_type("https://vnexpress.net/...")` still returns `"news"` after YouTube regex added |

### Representative test structure

```python
# tests/test_comment_crawl_youtube.py

import pytest
from unittest.mock import patch, MagicMock
from comment_crawl_youtube import extract_youtube_video_id, YouTubeCommentCrawler


# --- URL parsing ---

def test_standard_url():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_short_url():
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_shorts_url():
    assert extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_invalid_url():
    assert extract_youtube_video_id("https://vnexpress.net/article-123.html") is None

def test_missing_video_id():
    assert extract_youtube_video_id("https://www.youtube.com/") is None


# --- Crawler ---

@patch("comment_crawl_youtube.build_api_client")
def test_comments_disabled(mock_api):
    mock_api.return_value.commentThreads().list().execute.side_effect = HttpError(
        resp=MagicMock(status=403), content=b'{"error":{"errors":[{"reason":"commentsDisabled"}]}}'
    )
    crawler = YouTubeCommentCrawler(api_key="fake_key")
    result = crawler.crawl_comments("https://www.youtube.com/watch?v=DISABLED")
    assert result["comments"] == []
    assert result["blocked"] is False
    assert any("commentsDisabled" in w for w in result["warnings"])


# ... etc.
```

---

## Risks and Limitations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Quota exhaustion** | High | Cap `max_comments_per_url` (default 500). Monitor per-project quota. Consider separate GCP project for research vs. production. |
| **Comments disabled** | Medium | Handle 403 `commentsDisabled` gracefully; return `status="no_comments"`. |
| **Deleted/unavailable comments** | Low | Skip items with empty text; log at DEBUG level. |
| **Reply completeness** | Medium | Default: inline replies only (≤ 5). Full reply expansion is optional and quota-intensive. |
| **No historical comments** | N/A | The API only returns comments currently visible. Deleted comments cannot be recovered. |
| **Vietnamese diacritics** | Low | `textOriginal` is UTF-8 JSON; NFC normalisation preserves all Vietnamese characters. |
| **Private/age-restricted videos** | Medium | Return `status="blocked"` with informative warning. No fallback without login. |
| **API key security** | High | Never commit key. Use `YOUTUBE_DATA_API_KEY` env var. Document in secrets handling guide. |
| **API deprecation/breaking changes** | Low | Tied to YouTube Data API v3, which Google has maintained since 2012. Monitor changelogs. |
| **YouTube Shorts vs. standard videos** | Low | Both use the same `videoId` structure and the same `commentThreads.list` endpoint. |

---

## YouTube Feasibility Assessment

**Classification: Straightforward**

Rationale:
- The YouTube Data API v3 is mature, well-documented, and stable.
- `commentThreads.list` returns structured JSON with all required fields.
- The adapter pattern required (URL dispatch → crawler → `list[str]` → `build_segments_jsonl()`) is already established in `comment_crawl.py`.
- No changes to inference, MLflow, or training pipelines are needed.
- The main operational constraint is the 10,000 unit/day quota, which is manageable with reasonable per-video caps.
- Vietnamese Unicode is handled correctly by the API and the existing `_normalize_text()` function.

The only complexity is the credential management (API key) and quota monitoring — both are standard operational concerns, not architectural ones.

---

## Operational Caveat

The adapter is implemented and unit-tested with mocks, but it is not described here as fully production-certified. Live collection depends on Google Cloud API configuration, quota availability, video permissions, comments-enabled state, and YouTube platform availability.
