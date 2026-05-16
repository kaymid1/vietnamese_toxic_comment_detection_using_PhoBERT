from __future__ import annotations

from typing import Any, Dict, List, Optional

from comment_crawl import crawl_urls as crawl_comment_urls


def crawl_urls(
    urls: List[str],
    out_dir: str,
    timeout: int = 90,
    enable_video: bool = False,
    enable_asr: bool = False,
    keep_artifacts: bool = False,
    asr_max_seconds: int = 600,
    asr_language: str = "vi",
    allow_selenium_fallback: bool = True,
    fallback_decisions: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    return crawl_comment_urls(
        urls=urls,
        out_dir=out_dir,
        timeout=timeout,
        enable_video=enable_video,
        enable_asr=enable_asr,
        keep_artifacts=keep_artifacts,
        asr_max_seconds=asr_max_seconds,
        asr_language=asr_language,
        allow_selenium_fallback=allow_selenium_fallback,
        fallback_decisions=fallback_decisions,
    )
