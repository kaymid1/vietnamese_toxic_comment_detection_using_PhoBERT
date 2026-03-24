from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_METRIC_KEYS = {
    "f1": ("macro_f1", "f1", "f1_score"),
    "f1_toxic": ("f1_toxic", "f1_toxic_score"),
    "precision": ("precision", "precision_toxic"),
    "recall": ("recall", "recall_toxic"),
    "accuracy": ("accuracy", "acc"),
}

NESTED_METRIC_SECTIONS = (
    "final_test_rich",
    "test_argmax_basic",
    "test_threshold_0p5_rich",
    "test_tuned_raw_threshold_rich",
    "test_tuned_scaled_threshold_rich",
)


def safe_load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def file_mtime_iso(path: Path) -> Optional[str]:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts).isoformat()


def to_relative(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir))
    except Exception:
        return str(path)


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def normalize_metrics(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}

    candidates: List[Dict[str, Any]] = [raw]
    nested_metrics = raw.get("metrics")
    if isinstance(nested_metrics, dict):
        candidates.append(nested_metrics)
    for key in NESTED_METRIC_SECTIONS:
        section = raw.get(key)
        if isinstance(section, dict):
            candidates.append(section)

    normalized: Dict[str, float] = {}
    for out_key, keys in DEFAULT_METRIC_KEYS.items():
        value = None
        for candidate in candidates:
            for key in keys:
                raw_value = candidate.get(key)
                if isinstance(raw_value, (int, float)):
                    value = float(raw_value)
                    break
            if value is not None:
                break
        if value is not None:
            normalized[out_key] = value
    return normalized


def read_training_curve(model_dir: Path) -> Optional[List[Dict[str, Any]]]:
    curve_path = model_dir / "training_curve.json"
    curve = safe_load_json(curve_path, None)
    if isinstance(curve, list):
        return curve
    return None


def build_registry_from_models(
    model_root: Path,
    base_dir: Path,
    legacy_registry: Optional[Dict[str, Any]] = None,
    merge_legacy: bool = True,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    run_ids: set[str] = set()

    if model_root.exists() and model_root.is_dir():
        for model_type_dir in sorted(p for p in model_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for model_dir in sorted(p for p in model_type_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
                run_config_path = model_dir / "run_config.json"
                metrics_path = model_dir / "metrics.json"

                run_config = safe_load_json(run_config_path, {})
                metrics_raw = safe_load_json(metrics_path, {})
                training_curve = read_training_curve(model_dir)

                model_name = run_config.get("model_name") if isinstance(run_config, dict) else None
                model_id = model_name or f"{model_type_dir.name}/{model_dir.name}"
                run_id = run_config.get("run_id") if isinstance(run_config, dict) else None
                run_id = run_id or model_id

                created_at = run_config.get("created_at") if isinstance(run_config, dict) else None
                if not created_at:
                    created_at = (
                        file_mtime_iso(run_config_path)
                        or file_mtime_iso(metrics_path)
                        or file_mtime_iso(model_dir)
                    )

                dataset_version = run_config.get("dataset_version") if isinstance(run_config, dict) else None
                dataset_version = dataset_version or "unknown"

                hyperparameters = run_config.get("hyperparameters") if isinstance(run_config, dict) else None
                hyperparameters = hyperparameters if isinstance(hyperparameters, dict) else {}
                if training_curve:
                    hyperparameters = {**hyperparameters, "training_curve": training_curve}

                metrics = normalize_metrics(metrics_raw)

                run = {
                    "run_id": run_id,
                    "model_name": model_id,
                    "dataset_version": dataset_version,
                    "created_at": created_at,
                    "checkpoint_path": to_relative(model_dir, base_dir),
                    "hyperparameters": hyperparameters,
                    "metrics": metrics,
                    "is_baseline": bool(run_config.get("is_baseline")) if isinstance(run_config, dict) else False,
                }

                runs.append(run)
                run_ids.add(run_id)

    if merge_legacy and isinstance(legacy_registry, dict):
        legacy_runs = legacy_registry.get("runs")
        if isinstance(legacy_runs, list):
            for run in legacy_runs:
                if not isinstance(run, dict):
                    continue
                legacy_id = run.get("run_id")
                if legacy_id and legacy_id in run_ids:
                    continue
                runs.append(run)

    def run_sort_key(run: Dict[str, Any]) -> tuple:
        dt = parse_iso(run.get("created_at")) if isinstance(run, dict) else None
        return (dt is None, dt or datetime.max, run.get("run_id") or "")

    runs.sort(key=run_sort_key)
    return {"runs": runs}
