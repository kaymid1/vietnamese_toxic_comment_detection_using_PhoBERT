import subprocess
from pathlib import Path


HELPER = (
    Path(__file__).resolve().parents[1]
    / "comprehensive_ui"
    / "src"
    / "app"
    / "commentAggregation.js"
)
APP = Path(__file__).resolve().parents[1] / "comprehensive_ui" / "src" / "app" / "App.tsx"


def _aggregate(labels: list[int], threshold: float) -> dict:
    script = """
      const { deriveCommentAggregation } = await import(process.argv[1]);
      const [labels, threshold] = JSON.parse(process.argv[2]);
      console.log(JSON.stringify(deriveCommentAggregation(
        labels.map((toxic_label) => ({ toxic_label })), threshold
      )));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, HELPER.as_uri(), __import__("json").dumps([labels, threshold])],
        check=True,
        capture_output=True,
        text=True,
    )
    return __import__("json").loads(result.stdout)


def test_comment_aggregation_states_and_strict_threshold_boundary():
    no_toxic = _aggregate([0] * 10, 0.25)
    assert no_toxic["toxicCommentRate"] == 0
    assert no_toxic["state"] == "none"

    below = _aggregate([1, 1] + [0] * 8, 0.25)
    assert below["toxicCommentRate"] == 0.2
    assert below["state"] == "below_threshold"
    assert below["aggregateAlert"] is False

    elevated = _aggregate([1, 1, 1] + [0] * 7, 0.25)
    assert elevated["toxicCommentRate"] == 0.3
    assert elevated["state"] == "elevated"
    assert elevated["aggregateAlert"] is True

    boundary = _aggregate([1, 0, 0, 0], 0.25)
    assert boundary["toxicCommentRate"] == 0.25
    assert boundary["aggregateAlert"] is False


def test_scan_more_aggregation_uses_final_comment_labels_not_average_scores():
    merged = _aggregate([1, 0, 0, 0, 0], 0.25)
    assert merged["toxicCommentRate"] == 0.2
    assert merged["aggregateAlert"] is False

    app_source = APP.read_text(encoding="utf-8-sig")
    assert "deriveCommentAggregation(mergedSegments, pageThreshold)" in app_source
    assert "page_toxic: aggregation.aggregateAlert ? 1 : 0" in app_source
