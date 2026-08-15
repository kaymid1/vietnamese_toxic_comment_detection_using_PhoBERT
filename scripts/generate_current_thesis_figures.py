#!/usr/bin/env python3
"""Generate verified, current thesis figures from fixed repository artifacts.

This script intentionally accepts no dataset or model path overrides. It validates
the processed ViCTSD gold partitions before creating any figure, then evaluates the
current PhoBERT v2 checkpoint with the repository's official inference helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_crawled_local import (  # noqa: E402
    predict_scores,
    resolve_phobert_tokenizer_source,
    validate_model_artifacts,
)


DATA_DIR = REPO_ROOT / "data" / "processed" / "victsd_gold"
DATA_PATHS = {
    "train": DATA_DIR / "train.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl",
}
MODEL_DIR = REPO_ROOT / "models" / "options" / "phobert" / "phobert_lora_4.7"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
RUN_CONFIG_PATH = MODEL_DIR / "run_config.json"
OFFICIAL_INFERENCE_PATH = REPO_ROOT / "infer_crawled_local.py"

FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results" / "current_thesis"
PREDICTIONS_PATH = RESULTS_DIR / "phobert_v2_test_predictions_threshold_0p44.jsonl"
EVALUATION_PATH = RESULTS_DIR / "phobert_v2_test_evaluation_threshold_0p44.json"

EXPECTED_COUNTS = {
    "train": {"total": 6_946, "toxic": 756, "constructive": 2_497},
    "validation": {"total": 1_967, "toxic": 232, "constructive": 727},
    "test": {"total": 984, "toxic": 108, "constructive": 363},
}
EXPECTED_THRESHOLD = 0.44
EXPECTED_METRICS = {
    "macro_f1": 0.7380,
    "f1_toxic": 0.5410,
    "precision": 0.4853,
    "recall": 0.6111,
    "accuracy": 0.8862,
}
EXPECTED_CONFUSION_MATRIX = [[806, 70], [42, 66]]
METRIC_TOLERANCE = 5e-5
MODEL_ID = "phobert/phobert_v2_finetuned"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}")
            rows.append(row)
    return rows


def binary_label(row: dict[str, Any], key: str, path: Path, row_number: int) -> int:
    if key not in row:
        raise ValueError(f"Missing '{key}' in {path} row {row_number}")
    try:
        value = int(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-integer '{key}' in {path} row {row_number}") from exc
    if value not in (0, 1):
        raise ValueError(f"Non-binary '{key}'={value} in {path} row {row_number}")
    return value


def validate_dataset_counts(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    observed: dict[str, dict[str, int]] = {}
    discrepancies: list[str] = []

    for split, rows in split_rows.items():
        path = DATA_PATHS[split]
        toxic = sum(
            binary_label(row, "toxicity", path, index)
            for index, row in enumerate(rows, start=1)
        )
        constructive = sum(
            binary_label(row, "constructiveness", path, index)
            for index, row in enumerate(rows, start=1)
        )
        observed[split] = {
            "total": len(rows),
            "toxic": toxic,
            "clean": len(rows) - toxic,
            "constructive": constructive,
            "non_constructive": len(rows) - constructive,
        }
        for key, expected_value in EXPECTED_COUNTS[split].items():
            actual_value = observed[split][key]
            if actual_value != expected_value:
                discrepancies.append(
                    f"{split}.{key}: expected {expected_value:,}, observed {actual_value:,}"
                )

    if discrepancies:
        detail = "\n  - ".join(discrepancies)
        raise RuntimeError(
            "Dataset count verification failed; no figures were generated:\n"
            f"  - {detail}"
        )
    return observed


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_grouped_bars(
    axis: plt.Axes,
    bars: Iterable[Any],
    totals: list[int],
) -> None:
    for bar, total in zip(bars, totals):
        count = int(round(float(bar.get_height())))
        percentage = 100.0 * count / total
        axis.annotate(
            f"{count:,}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2.0, count),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def generate_dataset_distribution(
    counts: dict[str, dict[str, int]],
) -> list[Path]:
    splits = ["train", "validation", "test"]
    display_splits = ["Train", "Validation", "Test"]
    totals = [counts[split]["total"] for split in splits]
    x = np.arange(len(splits))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25))
    palette = {
        "negative": "#4C78A8",
        "positive": "#D55E00",
    }

    panels = [
        ("(a) Toxicity labels", "clean", "toxic", "Clean", "Toxic"),
        (
            "(b) Constructiveness labels",
            "non_constructive",
            "constructive",
            "Non-constructive",
            "Constructive",
        ),
    ]

    for axis, (panel_title, negative_key, positive_key, negative_name, positive_name) in zip(
        axes, panels
    ):
        negative_values = [counts[split][negative_key] for split in splits]
        positive_values = [counts[split][positive_key] for split in splits]
        negative_bars = axis.bar(
            x - width / 2,
            negative_values,
            width,
            label=negative_name,
            color=palette["negative"],
            edgecolor="black",
            linewidth=0.7,
            hatch="///",
        )
        positive_bars = axis.bar(
            x + width / 2,
            positive_values,
            width,
            label=positive_name,
            color=palette["positive"],
            edgecolor="black",
            linewidth=0.7,
            hatch="...",
        )
        annotate_grouped_bars(axis, negative_bars, totals)
        annotate_grouped_bars(axis, positive_bars, totals)
        axis.set_title(panel_title, loc="left", fontweight="semibold")
        axis.set_ylabel("Number of comments")
        axis.set_xticks(x)
        axis.set_xticklabels(display_splits)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="upper right")
        max_value = max(negative_values + positive_values)
        axis.set_ylim(0, max_value * 1.18)

    fig.tight_layout(w_pad=2.2)
    return save_figure(fig, "dataset_label_distribution")


def comment_lengths(rows: list[dict[str, Any]]) -> np.ndarray:
    values: list[int] = []
    path = DATA_PATHS["train"]
    for index, row in enumerate(rows, start=1):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Missing or non-string 'text' in {path} row {index}")
        values.append(len(text.strip().split()) if text.strip() else 0)
    return np.asarray(values, dtype=np.int64)


def select_histogram_bins(lengths: np.ndarray) -> int:
    if len(lengths) < 2:
        return 1
    q25, q75 = np.percentile(lengths, [25, 75])
    iqr = float(q75 - q25)
    if iqr <= 0:
        return max(1, int(math.ceil(math.log2(len(lengths)) + 1)))
    width = 2.0 * iqr / np.cbrt(len(lengths))
    if width <= 0:
        return 1
    bins = int(math.ceil((int(lengths.max()) - int(lengths.min())) / width))
    return max(10, min(80, bins))


def generate_comment_length_distribution(
    lengths: np.ndarray,
) -> tuple[list[Path], dict[str, float | int]]:
    bins = select_histogram_bins(lengths)
    stats: dict[str, float | int] = {
        "bins": bins,
        "sample_count": int(len(lengths)),
        "median_words": float(np.median(lengths)),
        "p95_words": float(np.percentile(lengths, 95)),
        "maximum_words": int(lengths.max()),
    }

    fig, axis = plt.subplots(figsize=(7.2, 4.35))
    axis.hist(
        lengths,
        bins=bins,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.45,
    )
    axis.set_xlabel("Comment length (words)")
    axis.set_ylabel("Number of comments")
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    axis.set_axisbelow(True)
    fig.tight_layout()
    return save_figure(fig, "comment_length_distribution"), stats


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(device: torch.device) -> tuple[Any, Any, int]:
    validate_model_artifacts("phobert", MODEL_DIR)
    run_config = read_json(RUN_CONFIG_PATH)
    hyperparameters = run_config.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError(f"Missing hyperparameters object: {RUN_CONFIG_PATH}")
    max_length = int(hyperparameters.get("MAX_LENGTH", 256))

    tokenizer_source = resolve_phobert_tokenizer_source(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=Path(tokenizer_source).is_dir(),
    )
    config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        config=config,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, max_length


def evaluate_test_partition(
    rows: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold_payload = read_json(THRESHOLD_PATH)
    stored_threshold = float(threshold_payload.get("threshold"))
    if not math.isclose(stored_threshold, EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Stored threshold mismatch: expected {EXPECTED_THRESHOLD}, "
            f"observed {stored_threshold} in {THRESHOLD_PATH}"
        )
    if threshold_payload.get("deployment_mode") != "raw_threshold":
        raise RuntimeError(
            f"Expected raw_threshold deployment mode in {THRESHOLD_PATH}, "
            f"observed {threshold_payload.get('deployment_mode')!r}"
        )

    tokenizer, model, max_length = load_model(device)
    labels: list[int] = []
    probabilities: list[float] = []
    constructiveness_probabilities: list[float | None] = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [str(row["text"]) for row in batch_rows]
        score_rows = predict_scores(texts, tokenizer, model, device, max_length)
        for source_row, score_row in zip(batch_rows, score_rows):
            labels.append(int(source_row["toxicity"]))
            probabilities.append(float(score_row["toxic_prob"]))
            constructiveness_probabilities.append(score_row.get("constructiveness_prob"))
        completed = min(start + batch_size, len(rows))
        print(f"  Evaluated {completed:,}/{len(rows):,} test comments", flush=True)

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= stored_threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_toxic": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    prediction_records: list[dict[str, Any]] = []
    for index, (row, probability, constructive_probability, prediction) in enumerate(
        zip(rows, probabilities, constructiveness_probabilities, y_pred.tolist()),
        start=1,
    ):
        text = str(row["text"])
        prediction_records.append(
            {
                "source_file": str(DATA_PATHS["test"]),
                "source_row": index,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
                "toxicity_true": int(row["toxicity"]),
                "toxicity_probability": probability,
                "toxicity_predicted": int(prediction),
                "constructiveness_true": int(row["constructiveness"]),
                "constructiveness_probability": (
                    float(constructive_probability)
                    if constructive_probability is not None
                    else None
                ),
                "model_id": MODEL_ID,
                "threshold": stored_threshold,
            }
        )

    evaluation = {
        "model_id": MODEL_ID,
        "model_path": str(MODEL_DIR),
        "official_inference_implementation": str(OFFICIAL_INFERENCE_PATH),
        "dataset_path": str(DATA_PATHS["test"]),
        "dataset_sha256": file_sha256(DATA_PATHS["test"]),
        "sample_count": len(rows),
        "max_length": max_length,
        "batch_size": batch_size,
        "device": str(device),
        "deployment_mode": "raw_threshold",
        "threshold": stored_threshold,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "expected_metrics_rounded_4dp": EXPECTED_METRICS,
        "expected_confusion_matrix": EXPECTED_CONFUSION_MATRIX,
    }
    return prediction_records, evaluation


def write_evaluation_artifacts(
    records: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with EVALUATION_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return [PREDICTIONS_PATH, EVALUATION_PATH]


def verify_evaluation(evaluation: dict[str, Any]) -> tuple[bool, list[str]]:
    discrepancies: list[str] = []
    metrics = evaluation["metrics"]
    for name, expected_value in EXPECTED_METRICS.items():
        actual_value = float(metrics[name])
        if abs(actual_value - expected_value) >= METRIC_TOLERANCE:
            discrepancies.append(
                f"{name}: expected {expected_value:.4f}, observed {actual_value:.8f}"
            )
    if evaluation["confusion_matrix"] != EXPECTED_CONFUSION_MATRIX:
        discrepancies.append(
            "confusion_matrix: expected "
            f"{EXPECTED_CONFUSION_MATRIX}, observed {evaluation['confusion_matrix']}"
        )
    return not discrepancies, discrepancies


def generate_confusion_matrix(evaluation: dict[str, Any]) -> list[Path]:
    matrix = np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    cmap = LinearSegmentedColormap.from_list(
        "academic_blues",
        ["#F7FBFF", "#9ECAE1", "#3182BD", "#08519C"],
    )

    fig, axis = plt.subplots(figsize=(4.8, 4.35))
    normalization = Normalize(vmin=float(matrix.min()), vmax=float(matrix.max()))
    for row_index in range(2):
        for column_index in range(2):
            axis.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(normalization(matrix[row_index, column_index])),
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(1.5, -0.5)
    axis.set_aspect("equal")
    class_names = ["Clean", "Toxic"]
    axis.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
    )
    threshold = (float(matrix.max()) + float(matrix.min())) / 2.0
    for row_index in range(2):
        for column_index in range(2):
            count = int(matrix[row_index, column_index])
            percentage = 100.0 * row_percentages[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
                fontsize=11,
            )
    fig.tight_layout()
    return save_figure(fig, "phobert_v2_test_confusion_matrix")


def print_summary(
    counts: dict[str, dict[str, int]],
    length_stats: dict[str, float | int],
    output_paths: list[Path],
    evaluation: dict[str, Any],
    reproducible: bool,
    discrepancies: list[str],
) -> None:
    print("\nInput paths:")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {DATA_PATHS[split]}")
    print(f"  model: {MODEL_DIR}")
    print(f"  threshold: {THRESHOLD_PATH}")
    print(f"  official inference: {OFFICIAL_INFERENCE_PATH}")

    print("\nVerified dataset counts:")
    for split in ("train", "validation", "test"):
        row = counts[split]
        print(
            f"  {split}: total={row['total']:,}, toxic={row['toxic']:,}, "
            f"clean={row['clean']:,}, constructive={row['constructive']:,}, "
            f"non_constructive={row['non_constructive']:,}"
        )

    print("\nTraining comment-length statistics (whitespace-separated words):")
    print(f"  bins: {length_stats['bins']}")
    print(f"  sample_count: {length_stats['sample_count']:,}")
    print(f"  median: {length_stats['median_words']:.2f}")
    print(f"  95th percentile: {length_stats['p95_words']:.2f}")
    print(f"  maximum: {length_stats['maximum_words']}")

    print("\nPhoBERT v2 test evaluation:")
    print(f"  threshold: {evaluation['threshold']}")
    for name, value in evaluation["metrics"].items():
        print(f"  {name}: {value:.8f}")
    print(f"  confusion_matrix: {evaluation['confusion_matrix']}")
    print(f"  reproducibility_status: {'PASS' if reproducible else 'FAIL'}")
    for discrepancy in discrepancies:
        print(f"  discrepancy: {discrepancy}")

    print("\nOutput paths:")
    for path in output_paths:
        print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate current, count-gated thesis figures and PhoBERT v2 test predictions."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' uses CUDA when available.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    return args


def main() -> int:
    args = parse_args()
    split_rows = {split: read_jsonl(path) for split, path in DATA_PATHS.items()}

    # This hard gate occurs before output directories are created or figures saved.
    counts = validate_dataset_counts(split_rows)
    print("Dataset count verification: PASS", flush=True)

    configure_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    output_paths.extend(generate_dataset_distribution(counts))

    lengths = comment_lengths(split_rows["train"])
    length_outputs, length_stats = generate_comment_length_distribution(lengths)
    output_paths.extend(length_outputs)

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(device_name)
    print(f"Evaluating {MODEL_ID} on {device}...", flush=True)
    prediction_records, evaluation = evaluate_test_partition(
        split_rows["test"],
        batch_size=args.batch_size,
        device=device,
    )
    output_paths.extend(write_evaluation_artifacts(prediction_records, evaluation))
    reproducible, discrepancies = verify_evaluation(evaluation)
    evaluation["reproducibility_status"] = "PASS" if reproducible else "FAIL"
    evaluation["discrepancies"] = discrepancies
    with EVALUATION_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if reproducible:
        output_paths.extend(generate_confusion_matrix(evaluation))
    else:
        print(
            "PhoBERT v2 predictions did not reproduce the verification targets; "
            "the confusion-matrix figure was not generated.",
            file=sys.stderr,
        )

    print_summary(
        counts,
        length_stats,
        output_paths,
        evaluation,
        reproducible,
        discrepancies,
    )
    return 0 if reproducible else 2


if __name__ == "__main__":
    raise SystemExit(main())
