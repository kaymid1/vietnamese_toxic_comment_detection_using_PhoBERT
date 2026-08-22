# Crawling — Knowledge Document

**Schema version:** `comment_only_v3`
**Active crawler:** `comment_crawl.py`
**Status:** Production (comment-only flow)

---

## Purpose

The crawler collects **user-generated comments** from public web pages for toxicity and constructiveness inference.

> Article body text, video transcripts, and page metadata are **not** the primary inference unit.
> A URL/page is a grouping context only. Every segment that is inferred is a single comment extracted from that page's comment section.

The pipeline output (`segments.jsonl`) is the shared boundary between the source-specific crawler and the downstream PhoBERT / TF-IDF-LR inference engine. Source adapters, including the YouTube adapter, produce segments in this format and do not require changes to `infer_crawled_local.py` or `backend/app.py`.

---

## Current Architecture

### End-to-end flow

```text
User submits URL(s) via React UI
        ↓
POST /api/analyze  (backend/app.py)
        ↓
backend/crawl_adapter.crawl_urls()
        ↓
comment_crawl.crawl_urls()          ← batch wrapper
        ↓
comment_crawl.crawl_comments_from_url()
        ↓ detect_url_type(url)
        ├── "facebook"  → FacebookCommentCrawler.crawl_comments()
        ├── "news"      → NewsSiteCommentCrawler.crawl_comments()
        ├── "youtube"   → YouTubeCommentCrawler.crawl_comments()
        ├── "unknown"   → NewsSiteCommentCrawler.crawl_comments()
        └── "x_twitter" → unsupported (empty artifacts written)
        ↓
raw comment strings: list[str]
        ↓
_clean_comment_text()  +  _is_comment_like_text()   ← normalization + noise filter
        ↓
deduplication (set-based, exact normalized match)
        ↓
build_segments_jsonl()              ← wraps each comment in segment schema
        ↓
save_crawl_artifacts()              → segments.jsonl, meta.json, extracted.txt
        ↓
infer_crawled_local.py              ← reads segments.jsonl, runs PhoBERT/TF-IDF
        ↓
crawled_predictions.jsonl           ← per-segment toxicity + constructiveness
        ↓
page-level aggregation (backend/app.py)
        ↓
JSON response to frontend
```

### Cache layer

`crawl_comments_from_url()` checks for a fresh `meta.json` + `segments.jsonl` before launching the browser. The cache TTL (default 2 h) is controlled by `COMMENT_CRAWL_CACHE_TTL_HOURS`. Cache is invalidated if the `crawl_schema` version changes.

### Retry and backoff

`NewsSiteCommentCrawler.crawl_comments()` retries up to `CRAWL_RETRY_MAX_ATTEMPTS` (default 3) times on transient errors (timeouts, disconnected sessions). Exponential backoff starts at `CRAWL_RETRY_BASE_DELAY` (default 2 s), caps at `CRAWL_RETRY_MAX_DELAY` (default 12 s), with `CRAWL_RETRY_JITTER` (default 1.5 s) of random jitter.

Inter-URL delay between batched URLs is randomised between `COMMENT_CRAWL_INTER_URL_DELAY_MIN` (default 2.5 s) and `COMMENT_CRAWL_INTER_URL_DELAY_MAX` (default 6 s).

---

## Supported Sources

### Production / Current

| Source | URL type | Crawler class | Strategy |
|--------|----------|---------------|----------|
| VnExpress | `news` | `NewsSiteCommentCrawler` | **Preferred: official JSON comment API** (`usi-saas.vnexpress.net`); Selenium fallback |
| Tuổi Trẻ | `news` | `NewsSiteCommentCrawler` | Selenium + CSS selectors |
| Thanh Niên | `news` | `NewsSiteCommentCrawler` | Selenium + CSS selectors |
| Dân Trí | `news` | `NewsSiteCommentCrawler` | Selenium + CSS selectors |
| VietnamNet | `news` | `NewsSiteCommentCrawler` | Selenium + iframe switch (`iframe[src*='comment.vietnamnet.vn']`) |
| Any `.html`/`.htm` URL | `news` (heuristic) | `NewsSiteCommentCrawler` | Selenium + fallback selectors |
| Unknown domains | `unknown` | `NewsSiteCommentCrawler` | Selenium + generic comment heuristics |
| YouTube videos | `youtube` | `YouTubeCommentCrawler` | API-based `commentThreads.list`; requires `YOUTUBE_DATA_API_KEY`; subject to quota, permissions, comments-disabled state, and platform availability |

### Experimental / Best-effort

| Source | URL type | Crawler class | Limitations |
|--------|----------|---------------|-------------|
| Facebook posts/groups | `facebook` | `FacebookCommentCrawler` | Requires session cookies; frequently blocked by login walls; mbasic.facebook.com strategy |

### Unsupported / Planned

| Source | URL type | Status |
|--------|----------|--------|
| X / Twitter | `x_twitter` | Detected; empty artifacts written; not implemented |

---

## Supported News Domains (hard-coded selector profiles)

```python
# comment_crawl.py — DOMAIN_SELECTORS keys
"vnexpress.net"
"tuoitre.vn"
"thanhnien.vn"
"dantri.com.vn"
"vietnamnet.vn"
```

Additional Vietnamese news sites recognised by `_NEWS_DOMAINS` set (fallback heuristic selectors, no domain-specific profile):

```python
"nld.com.vn", "baomoi.com", "zingnews.vn", "kenh14.vn", "soha.vn",
"vtc.vn", "vov.vn", "tienphong.vn", "laodong.vn", "plo.vn",
"cafef.vn", "genk.vn", "gamek.vn", "afamily.vn"
```

---

## Common Segment Schema

### `segments.jsonl` — one JSON object per line

```json
{
  "text": "...",
  "segment_index": 0,
  "url_hash": "<md5_of_url>",
  "html_tag_effective": "comment",
  "segment_hash": "<sha256(normalized_text + '|' + html_tag_effective)>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Cleaned, NFC-normalised comment text. Always UTF-8. |
| `segment_index` | `int` | 0-based position within this URL's comment list. |
| `url_hash` | `str` | MD5 of the canonical source URL. Groups segments by page. |
| `html_tag_effective` | `str` | Always `"comment"` in the current comment-only flow. |
| `segment_hash` | `str` | SHA-256 of `normalize(text) + "|" + html_tag_effective`. Stable dedup key. |

**Schema version constant:** `COMMENT_CRAWL_SCHEMA_VERSION = "comment_only_v3"` (in `comment_crawl.py`).

### `meta.json` — per-URL crawl metadata

```json
{
  "url": "...",
  "url_hash": "...",
  "source_type": "news",
  "output_dir": "data/raw/crawled_urls/<url_hash>",
  "total_comments": 42,
  "status": "ok",
  "blocked": false,
  "block_reason": null,
  "warnings": [],
  "from_cache": false,
  "attempts": 1,
  "max_comments_per_url": 0,
  "comment_cap_hit": false,
  "crawl_schema": "comment_only_v3"
}
```

Possible `status` values: `ok`, `no_comments`, `blocked`, `unsupported`, `error`.

### Optional source fields

Fields not currently in the shared segment schema but useful for richer sources (see [youtube-comments.md](./youtube-comments.md)):

| Field | Rationale |
|-------|-----------|
| `source_comment_id` | Stable external ID, such as a YouTube comment ID. Enables idempotent re-crawl without content comparison. |
| `author` | Comment author username/channel (optional; may be pseudonymous). |
| `published_at` | ISO 8601 timestamp of comment publication. |
| `like_count` | Raw engagement metric when available from the source API. |
| `reply_to` | Parent comment ID for threaded replies. |
| `parent_source_id` | ID of the parent video/post URL. |

These fields should be added **only when a source provides them reliably**. Downstream inference (`infer_crawled_local.py`) only reads `text`, `segment_hash`, and `html_tag_effective` — additional fields are ignored.

---

## Normalization

All comment text passes through two stages:

### Stage 1 — `_normalize_text(text)` (low-level)

```python
text = unicodedata.normalize("NFC", text)
text = re.sub(r"\s+", " ", text)
return text.strip()
```

- NFC Unicode normalisation preserves Vietnamese diacritics correctly.
- Collapses internal whitespace to a single space.
- No lowercasing. No punctuation removal.

### Stage 2 — `_clean_comment_text(text)` (comment-specific)

Strips common UI noise appended by comment widgets:

- Trailing "Trả lời / Báo vi phạm / 34' trước" reaction-chip suffixes (Vietnamese).
- Repeated reaction tokens ("Thích Thích Vui").
- Uses end-anchored regex patterns to avoid removing real comment content.

### Stage 3 — `_is_comment_like_text(text)` (gate)

Discards strings that are not plausibly comments:

- Empty or single-character strings.
- Strings matching UI noise patterns (timestamps, "Thích", "Trả lời", etc.).
- Short single-token strings (≤ 4 chars, no spaces).
- Strings consisting entirely of non-word characters.

---

## Deduplication

Two levels:

1. **Within a single URL crawl** — `_append_candidate()` uses a `set[str]` of normalised texts to skip exact duplicates during extraction. Applies to all crawlers.
2. **Across crawl sessions** — `segment_hash = sha256(normalized_text + "|" + html_tag_effective)` provides a stable content-based key. `backend/app.py` uses this as a `dedupe_key` when inserting into `mlflow_comment_item`; re-running the same URL is idempotent for existing canonical samples.

**Pagination safety:** `NewsSiteCommentCrawler._merge_comments()` uses a `set` to merge initial and expanded extractions without duplication.

---

## Shared Crawler Rules

These invariants apply to all current and future crawlers:

1. **Preserve Vietnamese Unicode text.** Always use NFC normalisation; never strip diacritics or force ASCII.
2. **Normalize consistently.** All text must pass through `_normalize_text()` before being stored in a segment.
3. **No duplicate comments within a URL.** Use set-based dedup at extraction time.
4. **Stable segment hashes.** `segment_hash` is computed from normalised text + `html_tag_effective`. Do not change the hash formula without incrementing `COMMENT_CRAWL_SCHEMA_VERSION`.
5. **Observable failures.** All block detections, timeouts, and crawl exceptions are logged at WARNING level and recorded in `meta.json["warnings"]`. Empty comment lists are never silently treated as success.
6. **Pagination must not duplicate records.** Retry loops and load-more clicks must track already-seen content.
7. **Source-specific extraction terminates at `build_segments_jsonl()`.** Everything downstream of that function is shared. A source adapter must return `list[str]` of comment texts — or produce segments in the common schema directly.
8. **Anti-bot and login walls abort gracefully.** Crawlers detect block indicators and set `status="blocked"` rather than raising unhandled exceptions.
9. **Cache must be keyed on `crawl_schema` version.** A version bump invalidates stale cache files automatically.

---

## Pipeline Boundary

```
[Source-specific crawler]         [Shared pipeline]
        |                                |
        ↓                                ↓
raw comment list[str]  ──→  build_segments_jsonl()
                                         ↓
                               save_crawl_artifacts()
                                (segments.jsonl)
                                         ↓
                               infer_crawled_local.py
```

**A new source adapter must:**
- Return `list[str]` of cleaned comment texts (or call `build_segments_jsonl()` directly).
- Set `status`, `blocked`, `warnings` fields in `result_meta`.
- Not touch inference, MLflow, or retraining code.

**A new source adapter must not:**
- Write its own custom output schema.
- Modify `build_segments_jsonl()`, `_make_segment_hash()`, or `COMMENT_CRAWL_SCHEMA_VERSION` unless a schema change is intentional and versioned.

---

## Relevant Code Map

| File | Class / Function | Role |
|------|-----------------|------|
| `comment_crawl.py` | `detect_url_type(url)` | URL → source type dispatch (`facebook`, `x_twitter`, `youtube`, `news`, `unknown`) |
| `comment_crawl.py` | `NewsSiteCommentCrawler` | Selenium-based news-site comment crawler |
| `comment_crawl.py` | `NewsSiteCommentCrawler._crawl_vnexpress_comments_via_api()` | VnExpress JSON API (preferred path) |
| `comment_crawl.py` | `NewsSiteCommentCrawler._pick_selectors(url)` | Returns domain-specific CSS selector profile |
| `comment_crawl.py` | `FacebookCommentCrawler` | Selenium-based mbasic.facebook.com crawler |
| `comment_crawl_youtube.py` | `YouTubeCommentCrawler` | API-based YouTube comment crawler using `commentThreads.list` |
| `comment_crawl_youtube.py` | `extract_youtube_video_id()` | Validates and extracts video IDs from supported YouTube URL formats |
| `comment_crawl.py` | `crawl_comments_from_url()` | Single-URL entry point: type detect → crawl → save |
| `comment_crawl.py` | `crawl_urls()` | Batch wrapper; compatible with `setup_and_crawl.crawl_urls()` signature |
| `comment_crawl.py` | `build_segments_jsonl()` | Converts `list[str]` → `list[dict]` segment rows |
| `comment_crawl.py` | `save_crawl_artifacts()` | Writes `segments.jsonl`, `meta.json`, `extracted.txt` |
| `comment_crawl.py` | `_normalize_text()` | NFC + whitespace normalization |
| `comment_crawl.py` | `_clean_comment_text()` | Strips UI noise suffixes |
| `comment_crawl.py` | `_is_comment_like_text()` | Filters non-comment strings |
| `comment_crawl.py` | `_make_segment_hash()` | SHA-256 content hash |
| `comment_crawl.py` | `_url_hash()` | MD5 of URL (directory key) |
| `comment_crawl.py` | `DOMAIN_SELECTORS` | Per-domain CSS selector profiles |
| `comment_crawl.py` | `_NEWS_DOMAINS` | Set of recognised Vietnamese news hostnames |
| `backend/crawl_adapter.py` | `crawl_urls()` | Thin shim: `comment_crawl.crawl_urls → backend` |
| `infer_crawled_local.py` | `load_segments_jsonl()` | Reads `segments.jsonl`; entry point of inference pipeline |
| `domain_classifier.py` | `HybridDomainClassifier` | Domain-aware seg_threshold adjustment (news/social/forum/unknown) |
| `tests/test_comment_crawl_cleaning.py` | — | Unit tests: cleaning, `_is_comment_like_text`, selector profiles |

### Output directories

```
data/raw/crawled_urls/<url_hash>/
    segments.jsonl   ← comment segments (primary inference input)
    meta.json        ← crawl metadata + status
    extracted.txt    ← plain text concatenation (backward compat)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMMENT_CRAWL_RETRY_MAX_ATTEMPTS` | `3` | Max retries on transient crawler errors |
| `COMMENT_CRAWL_RETRY_BASE_DELAY` | `2.0` | Initial backoff delay (seconds) |
| `COMMENT_CRAWL_RETRY_MAX_DELAY` | `12.0` | Maximum backoff delay (seconds) |
| `COMMENT_CRAWL_RETRY_JITTER` | `1.5` | Random jitter added to backoff |
| `COMMENT_CRAWL_CACHE_TTL_HOURS` | `2.0` | Cache TTL; 0 disables caching |
| `COMMENT_CRAWL_INTER_URL_DELAY_MIN` | `2.5` | Minimum inter-URL throttle (seconds) |
| `COMMENT_CRAWL_INTER_URL_DELAY_MAX` | `6.0` | Maximum inter-URL throttle (seconds) |
| `COMMENT_CRAWL_PROXY_LIST` | `""` | Comma-separated proxy URLs (optional) |
| `COMMENT_CRAWL_USER_AGENTS` | `""` | `||`-separated user-agent strings (optional) |
| `YOUTUBE_DATA_API_KEY` | `""` | Google Cloud API key for YouTube Data API v3; required for live YouTube comment collection |
| `YOUTUBE_INCLUDE_REPLIES` | `false` | Include inline replies returned by `commentThreads.list`; full reply expansion is not implemented |

---

## Existing Tests

`tests/test_comment_crawl_cleaning.py` covers:

- `_clean_comment_text()` strips reaction/reply/timestamp suffixes correctly.
- Normal comment content is not modified.
- UI timestamp-only strings are rejected by `_is_comment_like_text()`.
- VietnamNet selector profile targets embedded iframe app.
- VnExpress selector profile remains stable.
- Load-more keywords include ASCII fallback for Vietnamese text.

`tests/test_page_comment_aggregation.py` covers page-level aggregation logic.

`tests/test_comment_crawl_youtube.py` covers YouTube URL detection/video ID extraction, mocked `commentThreads.list` pagination and limits, ordering, Unicode/emoji preservation, deduplication, comments-disabled handling, forbidden/access-blocked handling, optional inline replies, and integration through `comment_crawl.py`.

---

*See [youtube-comments.md](./youtube-comments.md) for the YouTube integration notes and operational caveats.*
