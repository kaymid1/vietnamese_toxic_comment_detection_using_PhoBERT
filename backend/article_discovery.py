"""RSS discovery helpers for scheduled VnExpress collection."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_VNEXPRESS_RSS_URL = "https://vnexpress.net/rss/tin-moi-nhat.rss"
_ARTICLE_ID_RE = re.compile(r"-(\d+)\.html?$", re.IGNORECASE)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "ref", "source"}


@dataclass(frozen=True)
class DiscoveredArticle:
    discovery_key: str
    source: str
    canonical_url: str
    article_id: str | None
    article_title: str | None
    published_at: str | None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(parent: ET.Element, name: str) -> str | None:
    for child in list(parent):
        if _local_name(child.tag) == name:
            value = "".join(child.itertext()).strip()
            return value or None
    return None


def canonicalize_vnexpress_url(raw_url: str) -> str:
    """Return a stable HTTPS article URL without fragments/tracking params."""
    value = (raw_url or "").strip()
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host == "www.vnexpress.net":
        host = "vnexpress.net"
    if host != "vnexpress.net" and not host.endswith(".vnexpress.net"):
        raise ValueError("RSS item is not a VnExpress URL")
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("RSS item URL must use HTTP or HTTPS")

    query = [
        (key, query_value)
        for key, query_value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_KEYS and not key.lower().startswith("utm_")
    ]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, urlencode(query), ""))


def extract_vnexpress_article_id(canonical_url: str) -> str | None:
    match = _ARTICLE_ID_RE.search(urlsplit(canonical_url).path)
    return match.group(1) if match else None


def make_discovery_key(canonical_url: str) -> tuple[str, str | None]:
    article_id = extract_vnexpress_article_id(canonical_url)
    if article_id:
        return f"vnexpress:{article_id}", article_id
    digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    return f"vnexpress:url:{digest}", None


def parse_vnexpress_rss(xml_text: str) -> list[DiscoveredArticle]:
    """Parse RSS XML, preserving order while deduplicating feed items."""
    root = ET.fromstring(xml_text)
    articles: list[DiscoveredArticle] = []
    seen: set[str] = set()
    for item in root.iter():
        if _local_name(item.tag) != "item":
            continue
        link = _child_text(item, "link")
        if not link:
            continue
        try:
            canonical_url = canonicalize_vnexpress_url(link)
        except ValueError:
            continue
        discovery_key, article_id = make_discovery_key(canonical_url)
        if discovery_key in seen:
            continue
        seen.add(discovery_key)
        articles.append(
            DiscoveredArticle(
                discovery_key=discovery_key,
                source="vnexpress_rss",
                canonical_url=canonical_url,
                article_id=article_id,
                article_title=_child_text(item, "title"),
                published_at=_child_text(item, "pubDate") or _child_text(item, "published"),
            )
        )
    return articles


def fetch_vnexpress_rss(rss_url: str = DEFAULT_VNEXPRESS_RSS_URL, *, timeout: int = 20) -> list[DiscoveredArticle]:
    request = Request(
        rss_url,
        headers={
            "User-Agent": "VietComment-ScheduledDiscovery/1.0",
            "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset, errors="replace")
    return parse_vnexpress_rss(payload)
