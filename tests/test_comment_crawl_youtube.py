"""
tests/test_comment_crawl_youtube.py

Unit tests for the YouTube comment crawler adapter.

All tests use mocked/fake API services — no network access, no real API key.

Coverage:
  URL parsing
    1.  Standard watch?v= URL
    2.  youtu.be short URL
    3.  YouTube Shorts URL
    4.  URL with additional query parameters (t=, si=, feature=)
    5.  Invalid YouTube URL (wrong path, no video ID)
    6.  Non-YouTube URL
    7.  Malformed/empty input
  Crawler behaviour
    8.  Single-page comment collection
    9.  Multi-page comment collection (nextPageToken)
    10. max_comments cap respected (stops pagination early)
    11. Zero comments (empty items list)
    12. Comments disabled (YouTubeApiError with is_comments_disabled)
    13. Video not found / unavailable
    14. API quota exceeded
    15. Generic API exception
    16. Malformed comment item (missing snippet)
    17. Vietnamese Unicode preserved through normalisation
    18. Emoji preserved
    19. Duplicate comment text (exact normalised duplicate skipped)
    20. inline-reply collection (include_replies=True)
  Integration with comment_crawl.py
    21. detect_url_type() returns 'youtube' for valid YouTube URLs
    22. detect_url_type() returns 'news' / 'unknown' for non-YouTube URLs
    23. detect_url_type() returns 'unknown' for YouTube channel/playlist URL (no video ID)
    24. crawl_comments_from_url() dispatches to YouTubeCommentCrawler
    25. crawl_comments_from_url() writes segments.jsonl when comments returned
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from comment_crawl_youtube import (
    YouTubeApiError,
    YouTubeCommentCrawler,
    _normalize_text_yt,
    _raise_from_response,
    extract_youtube_video_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thread(
    text_original: str = "Hello",
    text_display: str | None = None,
    comment_id: str = "abc123",
    replies: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal commentThreads item dict."""
    top_level_snippet: dict[str, Any] = {"textOriginal": text_original}
    if text_display is not None:
        top_level_snippet["textDisplay"] = text_display

    item: dict[str, Any] = {
        "id": f"thread_{comment_id}",
        "snippet": {
            "topLevelComment": {
                "id": comment_id,
                "snippet": top_level_snippet,
            },
        },
    }

    if replies:
        item["snippet"]["replies"] = {
            "comments": [
                {
                    "id": f"reply_{i}",
                    "snippet": {"textOriginal": r},
                }
                for i, r in enumerate(replies)
            ]
        }

    return item


def _make_service(pages: list[list[dict]]) -> MagicMock:
    """
    Build a fake API service whose list_comment_threads() returns successive pages.

    Each element of ``pages`` is a list of thread items for that page.
    The last page has no nextPageToken.
    """
    service = MagicMock()
    responses = []
    for idx, page_items in enumerate(pages):
        resp: dict[str, Any] = {"items": page_items}
        if idx < len(pages) - 1:
            resp["nextPageToken"] = f"token_page_{idx + 1}"
        responses.append(resp)

    service.list_comment_threads.side_effect = responses
    return service


def _make_error_service(exc: Exception) -> MagicMock:
    """Build a fake service that raises on the first call."""
    service = MagicMock()
    service.list_comment_threads.side_effect = exc
    return service


# ===========================================================================
# 1–7  URL parsing
# ===========================================================================


class TestExtractYouTubeVideoId:
    """extract_youtube_video_id() — pure function, no network."""

    # 1. Standard watch?v= URL
    def test_standard_watch_url(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_standard_watch_url_without_www(self) -> None:
        assert extract_youtube_video_id(
            "https://youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_mobile_watch_url(self) -> None:
        assert extract_youtube_video_id(
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    # 2. youtu.be short URL
    def test_youtu_be_url(self) -> None:
        assert extract_youtube_video_id(
            "https://youtu.be/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    # 3. YouTube Shorts URL
    def test_shorts_url_with_www(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_shorts_url_without_www(self) -> None:
        assert extract_youtube_video_id(
            "https://youtube.com/shorts/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    # 4. Extra query parameters (t=, si=, feature=)
    def test_watch_url_with_timestamp_param(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s"
        ) == "dQw4w9WgXcQ"

    def test_watch_url_with_si_param(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&si=ABCDE12345"
        ) == "dQw4w9WgXcQ"

    def test_youtu_be_with_si_and_t_params(self) -> None:
        assert extract_youtube_video_id(
            "https://youtu.be/dQw4w9WgXcQ?si=XYZ&t=10"
        ) == "dQw4w9WgXcQ"

    def test_shorts_url_with_extra_params(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share"
        ) == "dQw4w9WgXcQ"

    # 5. Invalid YouTube URL (wrong path, no video ID)
    def test_youtube_channel_url_returns_none(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/c/SomeChannel"
        ) is None

    def test_youtube_homepage_returns_none(self) -> None:
        assert extract_youtube_video_id("https://www.youtube.com/") is None

    def test_youtube_playlist_url_returns_none(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/playlist?list=PLxxxxxxxx"
        ) is None

    def test_youtube_shorts_no_id_returns_none(self) -> None:
        assert extract_youtube_video_id(
            "https://www.youtube.com/shorts/"
        ) is None

    # 6. Non-YouTube URL
    def test_vnexpress_url_returns_none(self) -> None:
        assert extract_youtube_video_id(
            "https://vnexpress.net/article-123.html"
        ) is None

    def test_facebook_url_returns_none(self) -> None:
        assert extract_youtube_video_id(
            "https://www.facebook.com/watch?v=123456789"
        ) is None

    # 7. Malformed / empty input
    def test_empty_string_returns_none(self) -> None:
        assert extract_youtube_video_id("") is None

    def test_none_input_returns_none(self) -> None:
        assert extract_youtube_video_id(None) is None  # type: ignore[arg-type]

    def test_random_garbage_returns_none(self) -> None:
        assert extract_youtube_video_id("not a url at all!!!") is None

    def test_too_short_video_id_returns_none(self) -> None:
        # IDs shorter than 11 chars are invalid
        assert extract_youtube_video_id(
            "https://youtu.be/short"
        ) is None

    def test_too_long_video_id_returns_none(self) -> None:
        assert extract_youtube_video_id(
            "https://youtu.be/thisistoolongforavideoid"
        ) is None


# ===========================================================================
# 8–20  YouTubeCommentCrawler behaviour
# ===========================================================================


class TestYouTubeCommentCrawler:
    VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    VIDEO_ID = "dQw4w9WgXcQ"

    # 8. Single-page comment collection
    def test_single_page_returns_comments(self) -> None:
        items = [
            _make_thread("First comment", comment_id="c1"),
            _make_thread("Second comment", comment_id="c2"),
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service, max_comments=500)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["blocked"] is False
        assert result["block_reason"] is None
        assert result["comments"] == ["First comment", "Second comment"]
        assert result["warnings"] == []
        service.list_comment_threads.assert_called_once_with(
            video_id=self.VIDEO_ID,
            max_results=100,
            page_token=None,
        )

    # 9. Multi-page comment collection
    def test_multi_page_collects_all_comments(self) -> None:
        page1 = [_make_thread(f"Comment {i}", comment_id=f"c{i}") for i in range(3)]
        page2 = [_make_thread(f"Comment {i+3}", comment_id=f"c{i+3}") for i in range(2)]
        service = _make_service([page1, page2])
        crawler = YouTubeCommentCrawler(service=service, max_comments=0)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert len(result["comments"]) == 5
        assert result["comments"][0] == "Comment 0"
        assert result["comments"][4] == "Comment 4"
        assert service.list_comment_threads.call_count == 2

    def test_multi_page_second_call_uses_page_token(self) -> None:
        page1 = [_make_thread("A", comment_id="c1")]
        page2 = [_make_thread("B", comment_id="c2")]
        service = _make_service([page1, page2])
        crawler = YouTubeCommentCrawler(service=service, max_comments=0)

        crawler.crawl_comments(self.VIDEO_URL)

        calls = service.list_comment_threads.call_args_list
        assert calls[0].kwargs["page_token"] is None
        assert calls[1].kwargs["page_token"] == "token_page_1"

    # 10. max_comments cap
    def test_max_comments_stops_within_first_page(self) -> None:
        items = [_make_thread(f"C{i}", comment_id=f"c{i}") for i in range(10)]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service, max_comments=3)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert len(result["comments"]) == 3
        # Only one page should have been fetched
        assert service.list_comment_threads.call_count == 1

    def test_max_comments_stops_pagination_between_pages(self) -> None:
        page1 = [_make_thread(f"C{i}", comment_id=f"c{i}") for i in range(5)]
        page2 = [_make_thread(f"D{i}", comment_id=f"d{i}") for i in range(5)]
        service = _make_service([page1, page2])
        crawler = YouTubeCommentCrawler(service=service, max_comments=5)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert len(result["comments"]) == 5
        # Cap hit after first page — second page must NOT be requested
        assert service.list_comment_threads.call_count == 1

    # 11. Zero comments
    def test_zero_comments_returns_empty_list(self) -> None:
        service = _make_service([[]])  # one empty page
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == []
        assert result["blocked"] is False

    # 12. Comments disabled
    def test_comments_disabled_returns_empty_with_warning(self) -> None:
        exc = YouTubeApiError(
            "Comments are disabled.",
            reason="commentsDisabled",
            status_code=403,
        )
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == []
        assert result["blocked"] is False  # video accessible, comments off
        assert any("disabled" in w.lower() for w in result["warnings"])

    def test_comments_disabled_by_user_returns_empty_with_warning(self) -> None:
        exc = YouTubeApiError(
            "Comments disabled by user.",
            reason="commentsDisabledByUser",
            status_code=403,
        )
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == []
        assert result["blocked"] is False

    # 12b. Generic forbidden (access/auth/API failure)
    def test_forbidden_error_sets_blocked_and_preserves_status_and_reason(self) -> None:
        exc = YouTubeApiError(
            "Access forbidden for video.",
            reason="forbidden",
            status_code=403,
        )
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["blocked"] is True
        assert result["block_reason"] is not None
        assert "forbidden" in result["block_reason"]
        assert "HTTP 403" in result["block_reason"]
        assert result["comments"] == []

    def test_access_not_configured_forbidden_sets_blocked(self) -> None:
        exc = YouTubeApiError(
            "Access not configured.",
            reason="accessNotConfigured",
            status_code=403,
        )
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["blocked"] is True
        assert "accessNotConfigured" in result["block_reason"]

    # 13. Video not found / unavailable
    def test_video_not_found_sets_blocked(self) -> None:
        exc = YouTubeApiError(
            "Video not found.",
            reason="videoNotFound",
            status_code=404,
        )
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["blocked"] is True
        assert result["block_reason"] is not None
        assert result["comments"] == []

    # 14. API quota exceeded
    def test_quota_exceeded_adds_warning_not_blocked(self) -> None:
        exc = YouTubeApiError(
            "Quota exceeded.",
            reason="quotaExceeded",
            status_code=403,
        )
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["blocked"] is False
        assert any("quota" in w.lower() for w in result["warnings"])

    def test_http_429_treated_as_quota_exceeded(self) -> None:
        exc = YouTubeApiError(
            "Too Many Requests",
            reason="rateLimitExceeded",
            status_code=429,
        )
        assert exc.is_quota_exceeded is True
        service = _make_error_service(exc)
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)
        assert result["blocked"] is False
        assert any("quota" in w.lower() for w in result["warnings"])

    # 15. Generic API exception
    def test_generic_exception_adds_warning(self) -> None:
        service = _make_error_service(RuntimeError("Unexpected failure"))
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == []
        assert result["warnings"]

    # 16. Malformed comment item
    def test_malformed_item_missing_snippet_is_skipped(self) -> None:
        items = [
            {"id": "bad_item"},  # no snippet key
            _make_thread("Good comment", comment_id="good"),
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        # Only the good comment should be collected
        assert result["comments"] == ["Good comment"]

    def test_malformed_item_none_snippet_is_skipped(self) -> None:
        items = [
            {"id": "bad", "snippet": None},
            _make_thread("OK", comment_id="ok"),
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert "OK" in result["comments"]

    def test_malformed_item_empty_text_is_skipped(self) -> None:
        items = [
            _make_thread("", comment_id="empty"),  # empty textOriginal
            _make_thread("Real", comment_id="real"),
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == ["Real"]

    # 17. Vietnamese Unicode preserved
    def test_vietnamese_unicode_preserved(self) -> None:
        viet_text = "Phim này thật sự hay, cảm ơn đạo diễn! 🎬"
        items = [_make_thread(viet_text, comment_id="viet")]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert len(result["comments"]) == 1
        collected = result["comments"][0]
        # All Vietnamese diacritics must survive
        assert "hay" in collected
        assert "cảm ơn" in collected
        assert "đạo diễn" in collected

    def test_nfc_normalisation_applied(self) -> None:
        # NFD-encoded Vietnamese character decomposed (e + combining hook above)
        nfd_text = "ca\u0309m o\u01a1n"  # cảm ơn in NFD
        items = [_make_thread(nfd_text, comment_id="nfd")]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"]
        # After NFC, the combining characters are merged
        import unicodedata
        normalised = unicodedata.normalize("NFC", result["comments"][0])
        assert normalised == result["comments"][0]

    # 18. Emoji preserved
    def test_emoji_preserved(self) -> None:
        text_with_emoji = "Great video! 🔥💯👍"
        items = [_make_thread(text_with_emoji, comment_id="emoji")]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert len(result["comments"]) == 1
        assert "🔥" in result["comments"][0]
        assert "💯" in result["comments"][0]
        assert "👍" in result["comments"][0]

    # 19. Duplicate text deduplication
    def test_duplicate_comment_text_deduplicated(self) -> None:
        items = [
            _make_thread("Same comment", comment_id="c1"),
            _make_thread("Same comment", comment_id="c2"),  # identical text
            _make_thread("Different comment", comment_id="c3"),
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        # Exact duplicates filtered out — only one "Same comment"
        assert result["comments"].count("Same comment") == 1
        assert "Different comment" in result["comments"]
        assert len(result["comments"]) == 2

    # 20. Inline reply collection (include_replies=True)
    def test_include_replies_false_ignores_replies(self) -> None:
        items = [
            _make_thread(
                "Top comment",
                comment_id="top",
                replies=["Reply A", "Reply B"],
            )
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service, include_replies=False)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == ["Top comment"]

    def test_include_replies_true_collects_inline_replies(self) -> None:
        items = [
            _make_thread(
                "Top comment",
                comment_id="top",
                replies=["Reply A", "Reply B"],
            )
        ]
        service = _make_service([items])
        crawler = YouTubeCommentCrawler(service=service, include_replies=True)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert "Top comment" in result["comments"]
        assert "Reply A" in result["comments"]
        assert "Reply B" in result["comments"]
        assert len(result["comments"]) == 3


# ===========================================================================
# Edge cases — missing/no API key
# ===========================================================================


class TestCrawlerMisconfiguration:
    VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_no_service_and_no_api_key_returns_blocked(self) -> None:
        crawler = YouTubeCommentCrawler(api_key="", service=None)
        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["blocked"] is True
        assert result["block_reason"] is not None
        assert "YOUTUBE_DATA_API_KEY" in result["block_reason"]

    def test_invalid_youtube_url_returns_blocked(self) -> None:
        service = MagicMock()
        crawler = YouTubeCommentCrawler(service=service)
        result = crawler.crawl_comments("https://www.youtube.com/c/SomeChannel")

        assert result["blocked"] is True
        assert service.list_comment_threads.call_count == 0

    def test_non_youtube_url_returns_blocked(self) -> None:
        service = MagicMock()
        crawler = YouTubeCommentCrawler(service=service)
        result = crawler.crawl_comments("https://vnexpress.net/article.html")

        assert result["blocked"] is True
        assert service.list_comment_threads.call_count == 0


# ===========================================================================
# textDisplay fallback (HTML unescape)
# ===========================================================================


class TestTextDisplayFallback:
    VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_uses_text_original_over_text_display(self) -> None:
        item = {
            "id": "thread1",
            "snippet": {
                "topLevelComment": {
                    "id": "c1",
                    "snippet": {
                        "textOriginal": "Raw original",
                        "textDisplay": "Display version &amp; more",
                    },
                }
            },
        }
        service = _make_service([[item]])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == ["Raw original"]

    def test_falls_back_to_text_display_when_no_text_original(self) -> None:
        item = {
            "id": "thread1",
            "snippet": {
                "topLevelComment": {
                    "id": "c1",
                    "snippet": {
                        # no textOriginal
                        "textDisplay": "Hello &amp; world",
                    },
                }
            },
        }
        service = _make_service([[item]])
        crawler = YouTubeCommentCrawler(service=service)

        result = crawler.crawl_comments(self.VIDEO_URL)

        assert result["comments"] == ["Hello & world"]


# ===========================================================================
# 21–25  Integration with comment_crawl.py
# ===========================================================================


class TestCommentCrawlIntegration:
    """Tests that verify comment_crawl.py dispatch and detect_url_type."""

    # 21. detect_url_type returns 'youtube' for valid video URLs
    def test_detect_url_type_youtube_watch(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "youtube"

    def test_detect_url_type_youtu_be(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type("https://youtu.be/dQw4w9WgXcQ") == "youtube"

    def test_detect_url_type_shorts(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        ) == "youtube"

    # 22. Existing news / unknown detection is unchanged
    def test_detect_url_type_vnexpress_still_news(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type(
            "https://vnexpress.net/article-123.html"
        ) == "news"

    def test_detect_url_type_facebook_unchanged(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type(
            "https://www.facebook.com/photo?fbid=123"
        ) == "facebook"

    def test_detect_url_type_x_twitter_unchanged(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type("https://twitter.com/user/status/123") == "x_twitter"

    def test_detect_url_type_unknown_domain_unchanged(self) -> None:
        from comment_crawl import detect_url_type

        assert detect_url_type("https://example.com/some-page") == "unknown"

    # 23. YouTube channel URL (no video ID) is NOT classified as youtube
    def test_detect_url_type_youtube_channel_is_unknown(self) -> None:
        from comment_crawl import detect_url_type

        result = detect_url_type("https://www.youtube.com/c/SomeChannel")
        assert result != "youtube"

    def test_detect_url_type_youtube_homepage_is_unknown(self) -> None:
        from comment_crawl import detect_url_type

        result = detect_url_type("https://www.youtube.com/")
        assert result != "youtube"

    # 24. crawl_comments_from_url dispatches to YouTubeCommentCrawler
    def test_crawl_comments_from_url_dispatches_youtube(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from comment_crawl import crawl_comments_from_url

        items = [_make_thread("Xin chào!", comment_id="yt1")]
        fake_service = _make_service([items])

        # Patch YouTubeCommentCrawler inside comment_crawl's import namespace
        with patch(
            "comment_crawl_youtube.YouTubeCommentCrawler",
            return_value=YouTubeCommentCrawler(service=fake_service),
        ):
            monkeypatch.setenv("YOUTUBE_DATA_API_KEY", "fake-key-for-test")
            result = crawl_comments_from_url(
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                output_base_dir=str(tmp_path),
            )

        assert result["source_type"] == "youtube"
        assert result["total_comments"] == 1

    # 25. segments.jsonl written correctly for YouTube comments
    def test_crawl_comments_from_url_writes_segments_jsonl(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from comment_crawl import crawl_comments_from_url

        items = [
            _make_thread("Bình luận một", comment_id="yt1"),
            _make_thread("Bình luận hai", comment_id="yt2"),
        ]
        fake_service = _make_service([items])

        with patch(
            "comment_crawl_youtube.YouTubeCommentCrawler",
            return_value=YouTubeCommentCrawler(service=fake_service),
        ):
            monkeypatch.setenv("YOUTUBE_DATA_API_KEY", "fake-key")
            result = crawl_comments_from_url(
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                output_base_dir=str(tmp_path),
            )

        seg_path = pathlib.Path(result["output_dir"]) / "segments.jsonl"
        assert seg_path.exists(), "segments.jsonl must be written"

        rows = [json.loads(line) for line in seg_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2

        for row in rows:
            assert "text" in row
            assert "segment_hash" in row
            assert "url_hash" in row
            assert row["html_tag_effective"] == "comment"

        texts = {r["text"] for r in rows}
        assert "Bình luận một" in texts
        assert "Bình luận hai" in texts


# ===========================================================================
# _normalize_text_yt unit tests
# ===========================================================================


class TestNormalizeTextYt:
    def test_nfc_normalisation(self) -> None:
        # Input: NFD — combining diacritic separated from base char
        nfd = "ca\u0309m"   # cảm in NFD
        result = _normalize_text_yt(nfd)
        import unicodedata
        assert unicodedata.is_normalized("NFC", result)

    def test_whitespace_collapsed(self) -> None:
        assert _normalize_text_yt("hello   world") == "hello world"

    def test_leading_trailing_stripped(self) -> None:
        assert _normalize_text_yt("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert _normalize_text_yt("") == ""

    def test_preserves_emoji(self) -> None:
        text = "Great 🔥"
        assert "🔥" in _normalize_text_yt(text)

    def test_preserves_vietnamese(self) -> None:
        text = "Tiếng Việt rất đẹp"
        result = _normalize_text_yt(text)
        assert "Tiếng Việt" in result
        assert "đẹp" in result


# ===========================================================================
# _raise_from_response raw JSON error parsing tests
# ===========================================================================


class TestRaiseFromResponse:
    def test_comments_disabled_error_parsing(self) -> None:
        body = json.dumps({
            "error": {
                "errors": [{"reason": "commentsDisabled", "message": "Comments are disabled."}],
                "code": 403,
                "message": "Comments are disabled."
            }
        })
        with pytest.raises(YouTubeApiError) as exc_info:
            _raise_from_response(403, body, "dQw4w9WgXcQ")
        err = exc_info.value
        assert err.is_comments_disabled is True
        assert err.is_forbidden is False
        assert err.status_code == 403
        assert err.reason == "commentsDisabled"

    def test_generic_forbidden_error_parsing(self) -> None:
        body = json.dumps({
            "error": {
                "errors": [{"reason": "forbidden", "message": "Access forbidden."}],
                "code": 403,
                "message": "Access forbidden."
            }
        })
        with pytest.raises(YouTubeApiError) as exc_info:
            _raise_from_response(403, body, "dQw4w9WgXcQ")
        err = exc_info.value
        assert err.is_forbidden is True
        assert err.is_comments_disabled is False
        assert err.status_code == 403
        assert err.reason == "forbidden"

    def test_video_not_found_error_parsing(self) -> None:
        body = json.dumps({
            "error": {
                "errors": [{"reason": "videoNotFound", "message": "Video not found."}],
                "code": 404,
                "message": "Video not found."
            }
        })
        with pytest.raises(YouTubeApiError) as exc_info:
            _raise_from_response(404, body, "dQw4w9WgXcQ")
        err = exc_info.value
        assert err.is_not_found is True
        assert err.status_code == 404

    def test_quota_exceeded_error_parsing(self) -> None:
        body = json.dumps({
            "error": {
                "errors": [{"reason": "quotaExceeded", "message": "Quota exceeded."}],
                "code": 403,
                "message": "Quota exceeded."
            }
        })
        with pytest.raises(YouTubeApiError) as exc_info:
            _raise_from_response(403, body, "dQw4w9WgXcQ")
        err = exc_info.value
        assert err.is_quota_exceeded is True
        assert err.status_code == 403
