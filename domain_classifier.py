#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
domain_classifier.py
Classify a URL's domain into a content category (news, social, forum, unknown)
and return the corresponding seg_threshold to use during inference.

Usage (standalone):
    from domain_classifier import DomainClassifier
    clf = DomainClassifier()
    category, threshold = clf.classify("https://vnexpress.net/bai-viet/abc.html")
"""

import re
from urllib.parse import urlparse
from typing import Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Category → seg_threshold mapping
# Tune these values based on your validation set
# ---------------------------------------------------------------------------
CATEGORY_THRESHOLDS: Dict[str, float] = {
    "news":    0.72,   # Báo chí formal → raise threshold để giảm false positive
    "social":  0.50,   # Social media → default, model train trên đây
    "forum":   0.60,   # Forum → semi-formal
    "unknown": 0.62,   # Chưa xác định → conservative
}

# ---------------------------------------------------------------------------
# Hard-coded whitelist (Vietnamese + popular international)
# Extend this list as you encounter new domains
# ---------------------------------------------------------------------------
DOMAIN_WHITELIST: Dict[str, str] = {
    # === Báo chí Việt Nam ===
    "vnexpress.net":       "news",
    "tuoitre.vn":          "news",
    "thanhnien.vn":        "news",
    "dantri.com.vn":       "news",
    "nhandan.vn":          "news",
    "vietnamnet.vn":       "news",
    "vtv.vn":              "news",
    "vov.vn":              "news",
    "baochinhphu.vn":      "news",
    "tienphong.vn":        "news",
    "laodong.vn":          "news",
    "nld.com.vn":          "news",
    "sggp.org.vn":         "news",
    "zingnews.vn":         "news",
    "znews.vn":            "news",
    "kenh14.vn":           "news",
    "24h.com.vn":          "news",
    "soha.vn":             "news",
    "cafef.vn":            "news",
    "cafebiz.vn":          "news",
    "baomoi.com":          "news",
    "plo.vn":              "news",
    "anninhthudo.vn":      "news",
    "cand.com.vn":         "news",
    "qdnd.vn":             "news",
    "baoquangninh.com.vn": "news",
    "baothanhhoa.vn":      "news",
    "hanoitv.vn":          "news",
    "vietnamplus.vn":      "news",
    "vietnambiz.vn":       "news",
    "theleader.vn":        "news",
    "nhipsongkinhdoanh.vn":"news",
    "tapchicongthuong.vn": "news",

    # === Mạng xã hội ===
    "facebook.com":        "social",
    "fb.com":              "social",
    "m.facebook.com":      "social",
    "instagram.com":       "social",
    "tiktok.com":          "social",
    "twitter.com":         "social",
    "x.com":               "social",
    "youtube.com":         "social",
    "youtu.be":            "social",
    "zalo.me":             "social",
    "tiktok.com":          "social",
    "threads.net":         "social",
    "linkedin.com":        "social",

    # === Forum / Cộng đồng ===
    "voz.vn":              "forum",
    "vozforums.com":       "forum",
    "webtretho.com":       "forum",
    "tinhte.vn":           "forum",
    "otofun.net":          "forum",
    "gamevn.com":          "forum",
    "reddit.com":          "forum",
    "linhtinh.vn":         "forum",
    "vatgia.com":          "forum",
    "diendantinhoc.vn":    "forum",
}

# ---------------------------------------------------------------------------
# Heuristic patterns — applied when domain NOT in whitelist
# Order matters: first match wins
# ---------------------------------------------------------------------------
# (pattern_in_domain_name, category)
DOMAIN_HEURISTICS = [
    # News patterns
    (r"bao|news|tin|press|media|journal|vov|vtv|vov|thoi|bao|phapluat|"
     r"anninhthudo|cand|qdnd|phunuvietnam|congly",   "news"),

    # Social patterns  
    (r"social|zalo|fanpage|community",               "social"),

    # Forum patterns
    (r"forum|board|discuss|community|thread|cong.?dong", "forum"),
]

# URL path signals → bump toward news if path looks like article
NEWS_PATH_PATTERNS = [
    r"/tin-tuc/", r"/bai-viet/", r"/tin/", r"/news/",
    r"/article/", r"/post\d", r"/\d{4}/\d{2}/\d{2}/",
    r"-post\d+\.", r"\.html$",
]

FORUM_PATH_PATTERNS = [
    r"/threads/", r"/topic/", r"/showthread", r"/viewtopic",
    r"/forum/", r"/f\d+/", r"/t\d+",
]


class DomainClassifier:
    """
    Classify a URL and return (category, seg_threshold).

    Parameters
    ----------
    extra_whitelist : dict, optional
        Additional domain → category mappings to merge with defaults.
    threshold_overrides : dict, optional
        Override default thresholds per category, e.g. {"news": 0.80}.
    """

    def __init__(
        self,
        extra_whitelist: Optional[Dict[str, str]] = None,
        threshold_overrides: Optional[Dict[str, float]] = None,
    ):
        self.whitelist: Dict[str, str] = {**DOMAIN_WHITELIST}
        if extra_whitelist:
            self.whitelist.update(extra_whitelist)

        self.thresholds: Dict[str, float] = {**CATEGORY_THRESHOLDS}
        if threshold_overrides:
            self.thresholds.update(threshold_overrides)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, url: str) -> Tuple[str, float]:
        """
        Returns (category, seg_threshold) for the given URL.

        category ∈ {"news", "social", "forum", "unknown"}
        seg_threshold ∈ float
        """
        category = self._classify_category(url)
        threshold = self.thresholds.get(category, self.thresholds["unknown"])
        return category, threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_registered_domain(self, url: str) -> Tuple[str, str]:
        """Returns (registered_domain, path) e.g. ('vnexpress.net', '/tin-tuc/abc.html')"""
        try:
            parsed = urlparse(url if url.startswith("http") else "https://" + url)
            host = parsed.hostname or ""
            path = parsed.path or ""

            # Strip www / m / mobile prefixes
            host = re.sub(r"^(www\d?|m|mobile|wap)\.", "", host)
            return host, path
        except Exception:
            return "", ""

    def _classify_category(self, url: str) -> str:
        domain, path = self._extract_registered_domain(url)
        if not domain:
            return "unknown"

        # 1. Exact whitelist match
        if domain in self.whitelist:
            return self.whitelist[domain]

        # 2. Subdomain strip — try parent domain
        # e.g. "sports.vnexpress.net" → "vnexpress.net"
        parts = domain.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            if parent in self.whitelist:
                return self.whitelist[parent]
            # Vietnamese 3-part TLD: e.g. dantri.com.vn → already handled above
            if len(parts) > 3:
                parent3 = ".".join(parts[-3:])
                if parent3 in self.whitelist:
                    return self.whitelist[parent3]

        # 3. Heuristic regex on domain name
        domain_core = parts[0] if parts else domain
        for pattern, category in DOMAIN_HEURISTICS:
            if re.search(pattern, domain_core, re.IGNORECASE):
                return category

        # 4. Path-based signals as tiebreaker
        for pat in NEWS_PATH_PATTERNS:
            if re.search(pat, path, re.IGNORECASE):
                return "news"
        for pat in FORUM_PATH_PATTERNS:
            if re.search(pat, path, re.IGNORECASE):
                return "forum"

        return "unknown"


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clf = DomainClassifier()
    test_urls = [
        "https://znews.vn/chang-trai-lao-xuong-song-post1626551.html",
        "https://vnexpress.net/tin-tuc/xa-hoi/abc-123.html",
        "https://www.facebook.com/groups/abc",
        "https://voz.vn/threads/topic-123.456/",
        "https://sports.vnexpress.net/bong-da/abc.html",
        "https://somerandomblog.io/bai-viet/abc",
        "https://m.facebook.com/watch?v=123",
    ]
    print(f"{'URL':<60} {'Category':<10} {'Threshold'}")
    print("-" * 80)
    for url in test_urls:
        cat, thr = clf.classify(url)
        print(f"{url[:58]:<60} {cat:<10} {thr}")