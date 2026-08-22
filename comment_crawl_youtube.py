"""
comment_crawl_youtube.py — YouTube comment crawler adapter.

Collects public comments from YouTube videos via the YouTube Data API v3
and returns them as a list[str] compatible with the existing crawler pipeline.

The crawler is injected with a *service* object so that unit tests can
supply a mock without a live API key or network connection.

Usage (standalone, requires YOUTUBE_DATA_API_KEY env var):
    python comment_crawl_youtube.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

Usage (as library):
    from comment_crawl_youtube import YouTubeCommentCrawler
    crawler = YouTubeCommentCrawler(api_key="...")
    result = crawler.crawl_comments("https://youtu.be/dQw4w9WgXcQ")
    comments = result["comments"]   # list[str] → feed into build_segments_jsonl()

Usage (with injected service for testing):
    from comment_crawl_youtube import YouTubeCommentCrawler
    crawler = YouTubeCommentCrawler(service=fake_service)
    result = crawler.crawl_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

Pipeline boundary:
    YouTubeCommentCrawler.crawl_comments() returns list[str] of raw comment texts.
    Callers (comment_crawl.crawl_comments_from_url) are responsible for passing
    the list through build_segments_jsonl() and save_crawl_artifacts() exactly
    as with existing crawlers. This adapter must not touch downstream inference.

Live integration (deferred):
    When YOUTUBE_DATA_API_KEY is set and no service is injected, the crawler
    uses requests to call the YouTube Data API v3 commentThreads endpoint
    directly.  google-api-python-client is NOT required for this flow.
    If the project later needs OAuth2 or batch requests, adding
    google-api-python-client>=2.100.0 at that time is recommended.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import unicodedata
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOUTUBE_COMMENT_THREADS_URL = (
    "https://www.googleapis.com/youtube/v3/commentThreads"
)

# Hard API limit per page.
_YOUTUBE_MAX_RESULTS_PER_PAGE = 100

# Default cap: collected before pagination stops.
YOUTUBE_DEFAULT_MAX_COMMENTS = 500

# Error reasons returned by the API that indicate a terminal condition.
_COMMENTS_DISABLED_REASONS = frozenset(
    {"commentsDisabled", "commentsDisabledByUser"}
)
_FORBIDDEN_REASONS = frozenset(
    {"forbidden", "accessNotConfigured", "videoForbidden"}
)
_VIDEO_NOT_FOUND_REASONS = frozenset(
    {"videoNotFound", "notFound"}
)
_QUOTA_EXCEEDED_REASONS = frozenset(
    {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}
)

# Regex: valid YouTube video ID characters (base64url without padding).
_VALID_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Recognised YouTube hostnames.
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


def extract_youtube_video_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from any supported URL format.

    Supported formats:
        https://www.youtube.com/watch?v=VIDEO_ID
        https://youtube.com/watch?v=VIDEO_ID
        https://m.youtube.com/watch?v=VIDEO_ID
        https://youtu.be/VIDEO_ID
        https://youtu.be/VIDEO_ID?si=...
        https://www.youtube.com/shorts/VIDEO_ID
        https://youtube.com/shorts/VIDEO_ID
        https://www.youtube.com/embed/VIDEO_ID

    Returns the video ID string (exactly 11 characters, base64url alphabet)
    or None if the URL is not a supported YouTube video URL.

    Does NOT attempt DNS resolution or network access.
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    if not url:
        return None

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    hostname = (parsed.hostname or "").lower()
    if hostname not in _YOUTUBE_HOSTS:
        return None

    path = parsed.path or ""

    # youtu.be/<VIDEO_ID>
    if hostname == "youtu.be":
        # Path is /<VIDEO_ID>; strip leading slash
        video_id = path.lstrip("/").split("/")[0]
        return _validate_video_id(video_id)

    # youtube.com paths:
    # /watch?v=VIDEO_ID
    if path in ("/watch", "/watch/") or path.startswith("/watch?"):
        qs = parse_qs(parsed.query, keep_blank_values=False)
        ids = qs.get("v", [])
        return _validate_video_id(ids[0] if ids else "")

    # /shorts/VIDEO_ID[/...]
    if path.startswith("/shorts/"):
        segment = path[len("/shorts/"):].split("/")[0].split("?")[0]
        return _validate_video_id(segment)

    # /embed/VIDEO_ID[/...]
    if path.startswith("/embed/"):
        segment = path[len("/embed/"):].split("/")[0].split("?")[0]
        return _validate_video_id(segment)

    # /v/VIDEO_ID (legacy embed)
    if path.startswith("/v/"):
        segment = path[len("/v/"):].split("/")[0].split("?")[0]
        return _validate_video_id(segment)

    return None


def _validate_video_id(candidate: str) -> str | None:
    """
    Return the candidate if it looks like a valid YouTube video ID, else None.

    YouTube video IDs are exactly 11 characters from the base64url alphabet
    (A-Z, a-z, 0-9, -, _).
    """
    candidate = (candidate or "").strip()
    if _VALID_VIDEO_ID_RE.match(candidate):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Service abstraction (dependency injection for testability)
# ---------------------------------------------------------------------------


class YouTubeApiService:
    """
    Thin wrapper around the YouTube Data API v3 commentThreads endpoint.

    Uses the ``requests`` library (already in requirements.txt) so that
    google-api-python-client is not required during the mocked/MVP phase.

    A fake/mock object implementing the same interface can be injected into
    YouTubeCommentCrawler for unit tests.  The interface is:

        service.list_comment_threads(
            video_id: str,
            max_results: int,
            page_token: str | None,
        ) -> dict

    The returned dict must mirror the YouTube API response shape:
        {
            "items": [...],
            "nextPageToken": "..." | missing,
        }
    Errors must be raised as YouTubeApiError.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def list_comment_threads(
        self,
        video_id: str,
        max_results: int = _YOUTUBE_MAX_RESULTS_PER_PAGE,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """
        Call commentThreads.list and return the parsed JSON response.

        Raises YouTubeApiError on non-200 responses or API-level errors.
        """
        import requests as _requests  # deferred; already in requirements.txt

        params: dict[str, Any] = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(max_results, _YOUTUBE_MAX_RESULTS_PER_PAGE),
            "textFormat": "plainText",
            "key": self._api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = _requests.get(
                YOUTUBE_COMMENT_THREADS_URL,
                params=params,
                timeout=15,
            )
        except Exception as exc:
            raise YouTubeApiError(f"Network error calling YouTube API: {exc}") from exc

        if resp.status_code != 200:
            _raise_from_response(resp.status_code, resp.text, video_id)

        try:
            data = resp.json()
        except Exception as exc:
            raise YouTubeApiError(
                f"Malformed JSON response from YouTube API: {exc}"
            ) from exc

        return data


class YouTubeApiError(RuntimeError):
    """
    Raised by YouTubeApiService (or a mock) for API-level errors.

    Attributes
    ----------
    reason : str | None
        Machine-readable error reason from the API (e.g. "commentsDisabled", "forbidden").
    status_code : int | None
        HTTP status code from the response.
    is_comments_disabled : bool
    is_forbidden : bool
    is_quota_exceeded : bool
    is_not_found : bool
    """

    def __init__(
        self,
        message: str,
        reason: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code
        self.is_comments_disabled = reason in _COMMENTS_DISABLED_REASONS
        self.is_forbidden = (
            reason in _FORBIDDEN_REASONS
            or (
                status_code == 403
                and not self.is_comments_disabled
                and reason not in _QUOTA_EXCEEDED_REASONS
            )
        )
        self.is_quota_exceeded = (
            reason in _QUOTA_EXCEEDED_REASONS or status_code == 429
        )
        self.is_not_found = (
            reason in _VIDEO_NOT_FOUND_REASONS or status_code == 404
        )


def _raise_from_response(
    status_code: int,
    body: str,
    video_id: str,
) -> None:
    """Parse a non-200 response and raise an appropriate YouTubeApiError."""
    reason: str | None = None
    try:
        payload = json.loads(body)
        errors = (
            payload.get("error", {}).get("errors") or []
        )
        if errors and isinstance(errors, list) and errors[0].get("reason"):
            reason = errors[0].get("reason")
        elif payload.get("error", {}).get("status"):
            reason = payload.get("error", {}).get("status")
    except Exception:
        pass

    if reason in _COMMENTS_DISABLED_REASONS:
        raise YouTubeApiError(
            f"Comments are disabled for video {video_id!r} (HTTP {status_code}, reason={reason!r}).",
            reason=reason,
            status_code=status_code,
        )
    if reason in _VIDEO_NOT_FOUND_REASONS or status_code == 404:
        raise YouTubeApiError(
            f"Video {video_id!r} not found or is unavailable (HTTP {status_code}, reason={reason!r}).",
            reason=reason or "videoNotFound",
            status_code=status_code,
        )
    if reason in _QUOTA_EXCEEDED_REASONS or status_code == 429:
        raise YouTubeApiError(
            f"YouTube API quota exceeded (HTTP {status_code}, reason={reason!r}).",
            reason=reason or "quotaExceeded",
            status_code=status_code,
        )
    if reason in _FORBIDDEN_REASONS or status_code == 403:
        raise YouTubeApiError(
            f"Access forbidden for video {video_id!r} (HTTP {status_code}, reason={reason!r}): {body[:300]}",
            reason=reason or "forbidden",
            status_code=status_code,
        )
    raise YouTubeApiError(
        f"YouTube API returned HTTP {status_code} for video {video_id!r} (reason={reason!r}): {body[:300]}",
        reason=reason,
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_comment_text(snippet: dict[str, Any]) -> str:
    """
    Extract plain text from a comment snippet dict.

    Preference order:
    1. textOriginal  — raw text submitted by the user (no HTML entities).
    2. textDisplay   — may contain HTML; unescape it.

    Returns an empty string if neither field is usable.
    """
    text = snippet.get("textOriginal")
    if text and isinstance(text, str):
        return text

    text = snippet.get("textDisplay")
    if text and isinstance(text, str):
        return html_lib.unescape(text)

    return ""


def _normalize_text_yt(text: str) -> str:
    """
    Minimal normalisation applied inside the YouTube adapter:
    - NFC Unicode normalisation (preserves Vietnamese diacritics)
    - Collapse internal whitespace

    This is intentionally lightweight.  The heavy cleaning
    (_clean_comment_text, _is_comment_like_text) is applied
    downstream by build_segments_jsonl / the shared pipeline.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main crawler class
# ---------------------------------------------------------------------------


class YouTubeCommentCrawler:
    """
    Collect public comments from a YouTube video via the YouTube Data API v3.

    Adapter-pattern: returns list[str] of comment texts for consumption
    by the shared build_segments_jsonl() / inference pipeline.

    Parameters
    ----------
    api_key : str
        YouTube Data API v3 key.  Ignored when ``service`` is provided.
    service : object | None
        Injectable API service object.  Must implement::

            service.list_comment_threads(video_id, max_results, page_token) -> dict

        When None, a live ``YouTubeApiService`` is created from ``api_key``.
        Pass a mock here in unit tests.
    max_comments : int
        Maximum number of comments to collect (0 = unlimited).
    include_replies : bool
        If True, inline replies (up to 5 per thread from the API response)
        are also included.  Full reply expansion (comments.list) is deferred.
        Default: False (top-level only) for MVP.
    """

    def __init__(
        self,
        api_key: str = "",
        service: object | None = None,
        max_comments: int = YOUTUBE_DEFAULT_MAX_COMMENTS,
        include_replies: bool = False,
    ) -> None:
        self._api_key = api_key
        self._service: Any
        if service is not None:
            self._service = service
        elif api_key:
            self._service = YouTubeApiService(api_key=api_key)
        else:
            self._service = None  # will fail fast on first call
        self.max_comments = max(0, int(max_comments))
        self.include_replies = bool(include_replies)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl_comments(self, url: str) -> dict[str, Any]:
        """
        Collect comments from a YouTube video URL.

        Returns
        -------
        dict with keys:
            comments    : list[str]  — raw comment texts (not yet cleaned)
            blocked     : bool
            block_reason: str | None
            warnings    : list[str]
        """
        result: dict[str, Any] = {
            "comments": [],
            "blocked": False,
            "block_reason": None,
            "warnings": [],
        }

        video_id = extract_youtube_video_id(url)
        if not video_id:
            msg = f"Could not extract a valid YouTube video ID from URL: {url!r}"
            result["blocked"] = True
            result["block_reason"] = msg
            logger.warning("YouTubeCommentCrawler: %s", msg)
            return result

        if self._service is None:
            msg = (
                "YOUTUBE_DATA_API_KEY is not configured and no service was injected. "
                "Set the environment variable or provide a service object."
            )
            result["blocked"] = True
            result["block_reason"] = msg
            logger.warning("YouTubeCommentCrawler: %s", msg)
            return result

        logger.info(
            "YouTubeCommentCrawler: fetching comments for videoId=%s (max=%s)",
            video_id,
            self.max_comments if self.max_comments > 0 else "unlimited",
        )

        try:
            comments = self._collect_comments(video_id, result["warnings"])
            result["comments"] = comments
            logger.info(
                "YouTubeCommentCrawler: finished with %d comment(s) for videoId=%s",
                len(comments),
                video_id,
            )
        except YouTubeApiError as exc:
            if exc.is_comments_disabled:
                msg = f"Comments are disabled for this video (videoId={video_id!r})."
                result["warnings"].append(msg)
                logger.warning("YouTubeCommentCrawler: %s", msg)
                # Not "blocked" — the video is accessible, comments are just off.
            elif exc.is_forbidden:
                status_part = f"HTTP {exc.status_code}, " if exc.status_code else ""
                reason_part = f"reason={exc.reason!r}, " if exc.reason else ""
                msg = (
                    f"YouTube API access forbidden ({status_part}{reason_part}videoId={video_id!r}): {exc}"
                )
                result["blocked"] = True
                result["block_reason"] = msg
                result["warnings"].append(msg)
                logger.warning("YouTubeCommentCrawler: %s", msg)
            elif exc.is_not_found:
                msg = f"Video not found or unavailable (videoId={video_id!r}): {exc}"
                result["blocked"] = True
                result["block_reason"] = msg
                result["warnings"].append(msg)
                logger.warning("YouTubeCommentCrawler: %s", msg)
            elif exc.is_quota_exceeded:
                msg = f"YouTube API quota exceeded; stopping collection: {exc}"
                result["warnings"].append(msg)
                logger.warning("YouTubeCommentCrawler: %s", msg)
            else:
                status_part = f"HTTP {exc.status_code}, " if exc.status_code else ""
                reason_part = f"reason={exc.reason!r}, " if exc.reason else ""
                msg = f"YouTube API error for videoId={video_id!r} ({status_part}{reason_part}): {exc}"
                result["blocked"] = True
                result["block_reason"] = msg
                result["warnings"].append(msg)
                logger.warning("YouTubeCommentCrawler: %s", msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"Unexpected error collecting YouTube comments for {url!r}: {exc}"
            result["warnings"].append(msg)
            logger.exception("YouTubeCommentCrawler: unexpected error for %s", url)

        return result

    # ------------------------------------------------------------------
    # Internal collection
    # ------------------------------------------------------------------

    def _collect_comments(
        self,
        video_id: str,
        warnings: list[str],
    ) -> list[str]:
        """Paginate through commentThreads.list and build the comment list."""
        comments: list[str] = []
        seen: set[str] = set()
        page_token: str | None = None
        page_num = 0

        cap = self.max_comments
        max_results = (
            min(_YOUTUBE_MAX_RESULTS_PER_PAGE, cap)
            if cap > 0
            else _YOUTUBE_MAX_RESULTS_PER_PAGE
        )

        while True:
            page_num += 1
            response = self._service.list_comment_threads(
                video_id=video_id,
                max_results=max_results,
                page_token=page_token,
            )

            items = response.get("items") or []
            logger.info(
                "YouTubeCommentCrawler: page %d → %d thread(s) for videoId=%s",
                page_num,
                len(items),
                video_id,
            )

            for item in items:
                if cap > 0 and len(comments) >= cap:
                    break
                self._process_thread(item, comments, seen, warnings)

            page_token = response.get("nextPageToken")

            # Stop conditions
            if not page_token:
                break
            if cap > 0 and len(comments) >= cap:
                logger.info(
                    "YouTubeCommentCrawler: comment cap (%d) reached, stopping pagination",
                    cap,
                )
                break
            if not items:
                break

        return comments

    def _process_thread(
        self,
        item: Any,
        comments: list[str],
        seen: set[str],
        warnings: list[str],
    ) -> None:
        """Extract top-level comment (and optionally inline replies) from one thread item."""
        if not isinstance(item, dict):
            return

        snippet = item.get("snippet")
        if not isinstance(snippet, dict):
            return

        top_level = snippet.get("topLevelComment")
        if isinstance(top_level, dict):
            self._append_comment(
                top_level.get("snippet") or {},
                comments,
                seen,
                warnings,
            )

        if not self.include_replies:
            return

        # Inline replies — up to 5, already included in the response.
        replies_wrapper = snippet.get("replies") or {}
        if isinstance(replies_wrapper, dict):
            for reply in replies_wrapper.get("comments") or []:
                if self.max_comments > 0 and len(comments) >= self.max_comments:
                    break
                if isinstance(reply, dict):
                    self._append_comment(
                        reply.get("snippet") or {},
                        comments,
                        seen,
                        warnings,
                    )

    def _append_comment(
        self,
        snippet: Any,
        comments: list[str],
        seen: set[str],
        warnings: list[str],
    ) -> None:
        """
        Extract text from a comment snippet dict, normalise it,
        and append to the list if it is non-empty and not a duplicate.
        """
        if not isinstance(snippet, dict):
            return

        try:
            raw = _extract_comment_text(snippet)
        except Exception as exc:
            warnings.append(f"Skipped malformed comment item: {exc}")
            return

        text = _normalize_text_yt(raw)
        if not text:
            return

        if text in seen:
            return

        seen.add(text)
        comments.append(text)


# ---------------------------------------------------------------------------
# CLI (standalone usage — requires live API key)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Collect YouTube comments via the Data API v3."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--max-comments",
        type=int,
        default=YOUTUBE_DEFAULT_MAX_COMMENTS,
        help=f"Maximum comments to collect (default: {YOUTUBE_DEFAULT_MAX_COMMENTS}, 0=unlimited)",
    )
    parser.add_argument(
        "--include-replies",
        action="store_true",
        help="Also collect inline replies (up to 5 per thread)",
    )
    args = parser.parse_args()

    api_key = os.getenv("YOUTUBE_DATA_API_KEY", "")
    if not api_key:
        parser.error(
            "YOUTUBE_DATA_API_KEY environment variable is not set. "
            "Obtain a key from the Google Cloud Console and set it before running."
        )

    crawler = YouTubeCommentCrawler(
        api_key=api_key,
        max_comments=args.max_comments,
        include_replies=args.include_replies,
    )
    result = crawler.crawl_comments(args.url)

    print("\n" + "=" * 60)
    print(f"URL:       {args.url}")
    print(f"Comments:  {len(result['comments'])}")
    if result["blocked"]:
        print(f"BLOCKED:   {result['block_reason']}")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"WARNING:   {w}")
    print("=" * 60)
    for i, c in enumerate(result["comments"][:10], 1):
        print(f"  [{i:2d}] {c[:120]}")
    if len(result["comments"]) > 10:
        print(f"  ... and {len(result['comments']) - 10} more")


if __name__ == "__main__":
    main()
