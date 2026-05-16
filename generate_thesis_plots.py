#!/usr/bin/env python3
"""
LIGHTWEIGHT PLOT GENERATION FOR THESIS
- No training
- No model loading
- Reads JSONL + existing metrics if available
- Saves:
  plots/label_distribution.png
  plots/comment_length_hist.png
  plots/confusion_matrix.png

CLI example:
python generate_thesis_plots.py --data-dir "./data/victsd" --results-dir "./results/phobert_v2" --plots-dir "./plots"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SEABORN = False

LABEL_MAP = {0: "clean", 1: "toxic"}
REQUIRED_COLUMNS = ("text", "label")
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_number}")
            rows.append(payload)

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    return pd.DataFrame(rows)


def validate_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns {missing_columns} in {path}")

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"Column 'label' contains non-numeric values in {path}")

    df = df.copy()
    df["label"] = labels.astype(int)

    invalid = sorted(set(df["label"].tolist()) - set(LABEL_MAP.keys()))
    if invalid:
        raise ValueError(f"Invalid labels {invalid} in {path}. Allowed labels: {sorted(LABEL_MAP.keys())}")

    return df


def _compute_length(text: Any, unit: str) -> int:
    if pd.isna(text):
        return 0
    normalized = str(text).strip()
    if not normalized:
        return 0
    if unit == "chars":
        return len(normalized)
    return len(normalized.split())


def load_all_splits(data_dir: Path, length_unit: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dfs: dict[str, pd.DataFrame] = {}

    for filename in SPLIT_FILES:
        split_name = filename.replace(".jsonl", "")
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required split file not found: {file_path}")
        df = validate_df(read_jsonl(file_path), file_path)
        df["split"] = split_name
        split_dfs[split_name] = df

    df_all = pd.concat(
        [split_dfs["train"], split_dfs["validation"], split_dfs["test"]],
        ignore_index=True,
    )
    df_all["length_value"] = df_all["text"].apply(lambda value: _compute_length(value, length_unit))

    return split_dfs["train"], split_dfs["validation"], split_dfs["test"], df_all


def plot_label_distribution(df_all: pd.DataFrame, plots_dir: Path) -> Path:
    counts = df_all["label"].value_counts().sort_index()
    x_labels = [LABEL_MAP.get(int(label), str(label)) for label in counts.index.tolist()]
    y_values = counts.values.tolist()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x_labels, y_values)
    plt.title("Label Distribution of ViCTSD")
    plt.xlabel("Label")
    plt.ylabel("Number of Comments")

    for bar, value in zip(bars, y_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")

    out_path = plots_dir / "label_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_comment_length_hist(df_all: pd.DataFrame, plots_dir: Path, length_unit: str) -> Path:
    length_label = "Words" if length_unit == "words" else "Characters"
    plt.figure(figsize=(8, 5))
    plt.hist(df_all["length_value"], bins=40)
    plt.title(f"Comment Length Distribution ({length_label})")
    plt.xlabel(f"Number of {length_label}")
    plt.ylabel("Frequency")

    out_path = plots_dir / "comment_length_hist.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def _sort_metrics_candidates(paths: list[Path], anchor: Path) -> list[Path]:
    unique_paths = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)

    def path_score(path: Path) -> tuple[int, int, float]:
        try:
            rel_depth = len(path.relative_to(anchor).parts)
        except ValueError:
            rel_depth = 10_000
        return (0 if path.parent == anchor else 1, rel_depth, -path.stat().st_mtime)

    return sorted(unique_paths, key=path_score)


def find_metrics_files(results_dir: Path, metrics_path: Path | None) -> list[Path]:
    if metrics_path is not None:
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics file not found: {metrics_path}")
        return [metrics_path]

    candidates: list[Path] = []
    preferred = results_dir / "metrics.json"
    if preferred.exists():
        candidates.append(preferred)

    if results_dir.exists():
        candidates.extend(path for path in results_dir.rglob("metrics.json") if path.is_file())

    cwd = Path.cwd()
    candidates.extend(path for path in cwd.rglob("metrics.json") if path.is_file())

    return _sort_metrics_candidates(candidates, results_dir)


def _is_valid_confusion_matrix(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            return False
        if not all(isinstance(cell, (int, float)) for cell in row):
            return False
    return True


def _recursive_find_confusion_matrix(payload: Any) -> list[list[int]] | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "confusion_matrix" and _is_valid_confusion_matrix(value):
                return [[int(value[0][0]), int(value[0][1])], [int(value[1][0]), int(value[1][1])]]
            nested = _recursive_find_confusion_matrix(value)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _recursive_find_confusion_matrix(item)
            if nested is not None:
                return nested
    return None


def extract_confusion_matrix_from_metrics(metrics: dict[str, Any]) -> list[list[int]] | None:
    preferred_paths = [
        ("test_tuned_threshold_rich", "confusion_matrix"),
        ("test_threshold_0p5_rich", "confusion_matrix"),
        ("test_argmax_basic", "confusion_matrix"),
    ]

    for section, key in preferred_paths:
        section_obj = metrics.get(section)
        if isinstance(section_obj, dict):
            candidate = section_obj.get(key)
            if _is_valid_confusion_matrix(candidate):
                return [[int(candidate[0][0]), int(candidate[0][1])], [int(candidate[1][0]), int(candidate[1][1])]]

    return _recursive_find_confusion_matrix(metrics)


def load_confusion_matrix(metrics_files: list[Path], error_analysis_path: Path | None = None) -> list[list[int]] | None:
    for metrics_file in metrics_files:
        try:
            with metrics_file.open("r", encoding="utf-8") as handle:
                metrics_obj = json.load(handle)
            if not isinstance(metrics_obj, dict):
                logging.warning("Skipped metrics file because root JSON is not an object: %s", metrics_file)
                continue
            cm = extract_confusion_matrix_from_metrics(metrics_obj)
            if cm is not None:
                logging.info("Loaded confusion matrix from: %s", metrics_file)
                return cm
        except json.JSONDecodeError as exc:
            logging.warning("Skipped invalid JSON file %s (%s)", metrics_file, exc)
        except OSError as exc:
            logging.warning("Could not read metrics file %s (%s)", metrics_file, exc)

    if error_analysis_path is not None and error_analysis_path.exists():
        try:
            with error_analysis_path.open("r", encoding="utf-8") as handle:
                error_obj = json.load(handle)
            cm = _recursive_find_confusion_matrix(error_obj)
            if cm is not None:
                logging.info("Loaded confusion matrix from error analysis file: %s", error_analysis_path)
                return cm
        except json.JSONDecodeError as exc:
            logging.warning("Skipped invalid JSON error analysis file %s (%s)", error_analysis_path, exc)
        except OSError as exc:
            logging.warning("Could not read error analysis file %s (%s)", error_analysis_path, exc)

    logging.warning("No usable confusion matrix found from metrics/error analysis.")
    return None


def plot_confusion_matrix(cm: list[list[int]] | None, plots_dir: Path) -> Path | None:
    if cm is None:
        logging.warning("Skip confusion matrix plot because confusion matrix data is missing.")
        return None

    labels = [LABEL_MAP[0], LABEL_MAP[1]]
    plt.figure(figsize=(6, 5))

    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], labels)
        plt.yticks([0, 1], labels)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.title("Confusion Matrix for PhoBERT")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    out_path = plots_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate lightweight thesis plots from ViCTSD JSONL and metrics.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing train/validation/test jsonl files.")
    parser.add_argument("--results-dir", required=True, type=Path, help="Directory containing metrics artifacts.")
    parser.add_argument("--plots-dir", required=True, type=Path, help="Output directory for generated plots.")
    parser.add_argument("--metrics-path", type=Path, default=None, help="Optional explicit metrics JSON path.")
    parser.add_argument(
        "--error-analysis-path",
        type=Path,
        default=None,
        help="Optional error_analysis JSON path used as fallback for confusion matrix lookup.",
    )
    parser.add_argument(
        "--length-unit",
        choices=("words", "chars"),
        default="words",
        help="Length unit for comment histogram (default: words).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    args = _build_arg_parser().parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    metrics_path = args.metrics_path.expanduser().resolve() if args.metrics_path else None
    error_analysis_path = args.error_analysis_path.expanduser().resolve() if args.error_analysis_path else None

    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        logging.info("Loading data splits from: %s", data_dir)
        df_train, df_val, df_test, df_all = load_all_splits(data_dir=data_dir, length_unit=args.length_unit)
        logging.info(
            "Loaded rows -> train=%d, validation=%d, test=%d, total=%d",
            len(df_train),
            len(df_val),
            len(df_test),
            len(df_all),
        )

        label_plot = plot_label_distribution(df_all=df_all, plots_dir=plots_dir)
        logging.info("Saved: %s", label_plot)

        length_plot = plot_comment_length_hist(df_all=df_all, plots_dir=plots_dir, length_unit=args.length_unit)
        logging.info("Saved: %s", length_plot)

        metrics_files = find_metrics_files(results_dir=results_dir, metrics_path=metrics_path)
        if metrics_files:
            logging.info("Metrics candidates (%d): %s", len(metrics_files), ", ".join(str(path) for path in metrics_files))
        else:
            logging.warning("No metrics.json files found in %s or %s", results_dir, Path.cwd())

        cm = load_confusion_matrix(metrics_files=metrics_files, error_analysis_path=error_analysis_path)
        cm_plot = plot_confusion_matrix(cm=cm, plots_dir=plots_dir)
        if cm_plot is not None:
            logging.info("Saved: %s", cm_plot)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Unexpected error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
