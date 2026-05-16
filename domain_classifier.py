#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
domain_classifier.py
Hybrid thresholding for crawled URLs:
1) HTML metadata (schema.org / OpenGraph + header tags)
2) Text formality score (always runs)
"""

import json
import re
from typing import Tuple, Dict, Optional, List
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Category → base seg_threshold mapping
# ---------------------------------------------------------------------------
CATEGORY_THRESHOLDS: Dict[str, float] = {
    "news":    0.72,   # Báo chí formal → raise threshold để giảm false positive
    "social":  0.50,   # Social media → default, model train trên đây
    "forum":   0.60,   # Forum → semi-formal
    "unknown": 0.62,   # Chưa xác định → conservative
}

# ---------------------------------------------------------------------------
# Schema.org + OpenGraph types
# ---------------------------------------------------------------------------
NEWS_TYPES = {
    "NewsArticle", "ReportageNewsArticle", "AnalysisNewsArticle",
    "OpinionNewsArticle", "ReviewNewsArticle", "BackgroundNewsArticle",
    "AskPublicNewsArticle",
}
FORUM_TYPES = {"DiscussionForumPosting", "BlogPosting", "Article"}
SOCIAL_TYPES = {"SocialMediaPosting", "Comment"}

# ---------------------------------------------------------------------------
# HTML tag fallbacks
# ---------------------------------------------------------------------------
FALLBACK_HTML_TAGS = ["unknown", "attention"]

# ---------------------------------------------------------------------------
# Formality helpers
# ---------------------------------------------------------------------------
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]",
    flags=re.UNICODE,
)

TEENCODE_VOCAB = {
    "vcl", "đmm", "dmm", "cx", "ko", "k", "dc", "đc", "ntn", "sao",
    "mn", "mng", "ib", "ns", "nt",
}

EXPECTED_FORMALITY = {
    "news": 0.75,
    "forum": 0.50,
    "social": 0.25,
    "unknown": 0.50,
}


class HybridDomainClassifier:
    """
    Hybrid classifier that fuses:
    - Layer 1: HTML metadata (schema.org / OpenGraph + header tags)
    - Layer 2: text formality score (always runs)
    """

    def __init__(
        self,
        threshold_news: float = CATEGORY_THRESHOLDS["news"],
        threshold_social: float = CATEGORY_THRESHOLDS["social"],
        threshold_forum: float = CATEGORY_THRESHOLDS["forum"],
        threshold_unknown: float = CATEGORY_THRESHOLDS["unknown"],
        formality_range: float = 0.15,
        override_threshold: float = 0.30,
    ):
        self.thresholds: Dict[str, float] = {
            "news": threshold_news,
            "social": threshold_social,
            "forum": threshold_forum,
            "unknown": threshold_unknown,
        }
        self.formality_range = formality_range
        self.override_threshold = override_threshold

    # ------------------------------------------------------------------
    # Layer 1: HTML metadata
    # ------------------------------------------------------------------

    def classify_from_html(self, html: str, quiet: bool = False) -> Optional[Tuple[str, float, str]]:
        if not html:
            return None

        schema_types, og_types, _ = self.extract_html_tags(html, quiet=quiet)
        return self.classify_from_tags(schema_types, og_types)

    @staticmethod
    def classify_from_tags(
        schema_types: List[str],
        og_types: List[str],
    ) -> Optional[Tuple[str, float, str]]:
        if any(t in NEWS_TYPES for t in schema_types):
            return "news", 0.95, "schema.org"
        if any(t in FORUM_TYPES for t in schema_types):
            return "forum", 0.90, "schema.org"
        if any(t in SOCIAL_TYPES for t in schema_types):
            return "social", 0.90, "schema.org"

        if "article" in {t.lower() for t in og_types}:
            return "forum", 0.65, "opengraph"

        return None

    def extract_html_tags(self, html: str, quiet: bool = False) -> Tuple[List[str], List[str], List[str]]:
        if not html:
            return [], [], []

        soup = BeautifulSoup(html, "html.parser")
        schema_types: List[str] = []
        og_types: List[str] = []
        header_tags: List[str] = []

        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = tag.string or tag.get_text() or ""
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            self._collect_jsonld_types(payload, schema_types)

        for tag in soup.find_all("meta"):
            prop = (tag.get("property") or tag.get("name") or "").lower()
            if prop != "og:type":
                continue
            content = (tag.get("content") or "").strip()
            if content:
                og_types.append(content)

        for level in ("h1", "h2", "h3"):
            if soup.find(level):
                header_tags.append(level)

        schema_types = self._dedupe(schema_types)
        og_types = self._dedupe(og_types)
        header_tags = self._dedupe(header_tags)

        self._log_unknown_types(schema_types, og_types, quiet=quiet)

        return schema_types, og_types, header_tags

    def _collect_jsonld_types(self, obj: object, out: List[str]) -> None:
        if isinstance(obj, dict):
            raw_type = obj.get("@type")
            if raw_type:
                if isinstance(raw_type, list):
                    for t in raw_type:
                        norm = self._normalize_type(t)
                        if norm:
                            out.append(norm)
                else:
                    norm = self._normalize_type(raw_type)
                    if norm:
                        out.append(norm)

            graph = obj.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    self._collect_jsonld_types(item, out)

            for value in obj.values():
                if isinstance(value, (dict, list)):
                    self._collect_jsonld_types(value, out)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_jsonld_types(item, out)

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen: set[str] = set()
        output: List[str] = []
        for v in values:
            if v not in seen:
                output.append(v)
                seen.add(v)
        return output

    @staticmethod
    def _log_unknown_types(schema_types: List[str], og_types: List[str], quiet: bool) -> None:
        if quiet:
            return
        unknown_types = [
            t for t in schema_types
            if t not in NEWS_TYPES and t not in FORUM_TYPES and t not in SOCIAL_TYPES
        ]
        unknown_types += [t for t in og_types if t.lower() != "article"]
        for t in unknown_types:
            print(f"[INFO] Unknown HTML type detected: {t}")

    @staticmethod
    def _normalize_type(raw: object) -> Optional[str]:
        if not raw:
            return None
        if not isinstance(raw, str):
            return None
        raw = raw.strip()
        if not raw:
            return None
        if "/" in raw:
            raw = raw.rsplit("/", 1)[-1]
        if ":" in raw:
            raw = raw.rsplit(":", 1)[-1]
        return raw


    # ------------------------------------------------------------------
    # Layer 3: Formality scoring
    # ------------------------------------------------------------------

    def compute_formality_score(self, text: str) -> float:
        text = text or ""
        if not text.strip():
            return 0.5

        words = text.split()
        word_count = max(len(words), 1)

        emoji_density = self._count_emoji(text) / max(len(text), 1)
        repeated_punct = len(re.findall(r"[!?]{2,}", text)) / word_count

        alpha_total = 0
        alpha_upper = 0
        for ch in text:
            if ch.isalpha():
                alpha_total += 1
                if ch.isupper():
                    alpha_upper += 1
        all_caps_ratio = alpha_upper / max(alpha_total, 1)

        teencode_density = self._count_teencode_hits(text) / word_count

        avg_word_length = self._safe_mean([len(w) for w in words]) / 10

        sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]
        avg_sent_length = len(words) / max(len(sentences), 1) / 30

        has_proper_punct = float(bool(re.search(
            r"[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẶ][^.!?]*[.!?]",
            text,
        )))

        number_with_unit = len(re.findall(
            r"\d+\s*(tỷ|triệu|nghìn|%|km|kg|USD|VND|đồng)",
            text,
            flags=re.IGNORECASE,
        )) / word_count

        quote_usage = float("\"" in text or "\u201c" in text or "\u201d" in text)

        informal_score = (
            emoji_density * 3.0 +
            repeated_punct * 2.0 +
            all_caps_ratio * 1.5 +
            teencode_density * 2.5
        )
        formal_score = (
            avg_word_length * 1.0 +
            avg_sent_length * 1.0 +
            has_proper_punct * 0.5 +
            number_with_unit * 1.0 +
            quote_usage * 0.5
        )

        raw = formal_score / (formal_score + informal_score + 1e-6)
        return self._clamp(raw, 0.0, 1.0)

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _count_emoji(text: str) -> int:
        return len(EMOJI_RE.findall(text))

    @staticmethod
    def _count_teencode_hits(text: str) -> int:
        tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
        return sum(1 for t in tokens if t in TEENCODE_VOCAB)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    # ------------------------------------------------------------------
    # Threshold fusion
    # ------------------------------------------------------------------

    def compute_effective_threshold(
        self,
        layer1_result: Optional[Tuple[str, float, str]],
        formality_score: float,
    ) -> Dict[str, object]:
        # Step 1: pick structural signal (layer1 > unknown)
        if layer1_result:
            category, struct_confidence, struct_source = layer1_result
        else:
            category, struct_confidence, struct_source = "unknown", 0.0, "none"

        base = self.thresholds.get(category, self.thresholds["unknown"])

        # Step 2: formality adjustment and direct formality-based threshold
        formality_adjustment = (formality_score - 0.5) * 2 * self.formality_range
        formality_based_threshold = (
            self.thresholds["social"] +
            formality_score * (self.thresholds["news"] - self.thresholds["social"])
        )

        # Step 3: override if text formality diverges from structural expectation
        expected_formality = EXPECTED_FORMALITY.get(category, 0.50)
        formality_delta = abs(formality_score - expected_formality)
        layer3_overrides = formality_delta > self.override_threshold

        if layer3_overrides:
            effective_threshold = formality_based_threshold
            decision_source = f"layer3_override (delta={formality_delta:.2f})"
        else:
            blend_weight = struct_confidence
            effective_threshold = (
                blend_weight * (base + formality_adjustment) +
                (1 - blend_weight) * formality_based_threshold
            )
            decision_source = f"blend (struct={struct_confidence:.2f}, formality={formality_score:.2f})"

        effective_threshold = self._clamp(effective_threshold, 0.40, 0.85)

        return {
            "effective_threshold": round(float(effective_threshold), 4),
            "domain_category": category,
            "struct_confidence": round(float(struct_confidence), 3),
            "struct_source": struct_source,
            "formality_score": round(float(formality_score), 3),
            "formality_delta": round(float(formality_delta), 3),
            "layer3_overrides": bool(layer3_overrides),
            "decision_source": decision_source,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_threshold(
        self,
        url: str,
        segments: List[str],
        html: Optional[str],
        quiet: bool = False,
    ) -> Dict[str, object]:
        # Layer 1: HTML metadata (optional)
        schema_types, og_types, header_tags = self.extract_html_tags(html or "", quiet=quiet)
        layer1 = self.classify_from_tags(schema_types, og_types)

        # Layer 3: formality score (always runs)
        text_sample = self._sample_text(segments, max_words=500)
        formality_score = self.compute_formality_score(text_sample)

        threshold_info = self.compute_effective_threshold(layer1, formality_score)
        primary_tag = None
        if og_types:
            primary_tag = og_types[0]
        elif schema_types:
            primary_tag = schema_types[0]
        elif header_tags:
            primary_tag = header_tags[0]

        if not primary_tag:
            html_tags = FALLBACK_HTML_TAGS[:]
            primary_tag = html_tags[0]
        else:
            html_tags = [primary_tag]

        threshold_info.update({
            "schema_types": schema_types,
            "og_types": og_types,
            "header_tags": header_tags,
            "html_tags": html_tags,
        })
        return threshold_info

    @staticmethod
    def _sample_text(segments: List[str], max_words: int = 500) -> str:
        if not segments:
            return ""
        words: List[str] = []
        for seg in segments:
            if len(words) >= max_words:
                break
            for w in seg.split():
                words.append(w)
                if len(words) >= max_words:
                    break
        return " ".join(words)


if __name__ == "__main__":
    clf = HybridDomainClassifier()
    test_urls = [
        "https://znews.vn/chang-trai-lao-xuong-song-post1626551.html",
        "https://vnexpress.net/tin-tuc/xa-hoi/abc-123.html",
        "https://www.facebook.com/groups/abc",
        "https://voz.vn/threads/topic-123.456/",
        "https://sports.vnexpress.net/bong-da/abc.html",
        "https://somerandomblog.io/bai-viet/abc",
        "https://m.facebook.com/watch?v=123",
    ]
    for url in test_urls:
        info = clf.get_threshold(url, ["test"], html=None)
        print(url, "=>", info)
