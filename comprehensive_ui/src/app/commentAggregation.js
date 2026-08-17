/**
 * Summarize final per-comment classifications for one URL.
 *
 * This is deliberately an aggregation, not webpage/article inference.
 * The backend compatibility field `page_toxic` uses the same strict `>` rule.
 */
export function deriveCommentAggregation(segments, pageThreshold) {
  const normalizedThreshold = Number.isFinite(pageThreshold) ? pageThreshold : 0.5;
  const totalCommentCount = Array.isArray(segments) ? segments.length : 0;
  const toxicCommentCount = (segments || []).filter((segment) => segment?.toxic_label === 1).length;
  const toxicCommentRate = totalCommentCount > 0 ? toxicCommentCount / totalCommentCount : 0;
  const aggregateAlert = toxicCommentRate > normalizedThreshold;
  const state =
    toxicCommentCount === 0
      ? "none"
      : aggregateAlert
        ? "elevated"
        : "below_threshold";

  return {
    totalCommentCount,
    toxicCommentCount,
    cleanCommentCount: totalCommentCount - toxicCommentCount,
    toxicCommentRate,
    aggregateAlert,
    state,
  };
}
