#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone promotion gate for newly trained PhoBERT checkpoints.

Usage:
  python scripts/promote_model.py \
    --candidate models/options/phobert/<run_id> \
    --eval-data data/processed/victsd_gold/test.jsonl \
    [--current-serving phobert/v1] \
    [--force] \
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_crawled_local import (  # noqa: E402
    MODEL_OPTIONS_DIR,
    pick_device,
    predict_probs,
    resolve_model_path,
    validate_model_artifacts,
)

GATE_F1_TOXIC_DELTA = 0.0
GATE_MACRO_F1_DELTA = -0.01
GATE_MIN_F1_TOXIC_ABS = 0.45


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_delta(new_value: float, old_value: Optional[float]) -> Optional[float]:
    if old_value is None:
        return None
    return float(new_value - old_value)


def _symbol(passed: bool) -> str:
    return "✓" if passed else "✗"


def _format_float(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_eval_dataset(eval_data_path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    with eval_data_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {eval_data_path}:{line_idx}: {exc}") from exc

            text = obj.get("text")
            toxicity = obj.get("toxicity")
            if not isinstance(text, str) or not text.strip():
                continue
            if toxicity not in (0, 1):
                raise ValueError(f"Invalid toxicity label at {eval_data_path}:{line_idx}: {toxicity}")

            texts.append(text.strip())
            labels.append(int(toxicity))

    if not texts:
        raise ValueError(f"No valid rows found in eval data: {eval_data_path}")

    return texts, labels


@torch.inference_mode()
def evaluate_phobert_checkpoint(
    model_path: Path,
    texts: List[str],
    labels: List[int],
    batch_size: int,
    max_length: int,
) -> Dict[str, float]:
    validate_model_artifacts("phobert", model_path)

    device = pick_device("auto")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    probs: List[float] = []
    for start in range(0, len(texts), batch_size):
        probs.extend(
            predict_probs(
                texts=texts[start : start + batch_size],
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            )
        )

    preds = [1 if p >= 0.5 else 0 for p in probs]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "f1_toxic": float(f1[1]),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_toxic": float(precision[1]),
        "recall_toxic": float(recall[1]),
        "n_eval": int(len(labels)),
        "threshold": 0.5,
    }


def evaluate_gate(
    candidate_metrics: Dict[str, float],
    current_metrics: Optional[Dict[str, float]],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    checks: Dict[str, Any] = {}
    reasons: List[str] = []

    check_abs = candidate_metrics["f1_toxic"] >= GATE_MIN_F1_TOXIC_ABS
    checks["f1_toxic_abs"] = {
        "passed": check_abs,
        "threshold": GATE_MIN_F1_TOXIC_ABS,
    }
    if not check_abs:
        reasons.append(
            f"f1_toxic={candidate_metrics['f1_toxic']:.4f} < min_abs={GATE_MIN_F1_TOXIC_ABS:.4f}"
        )

    if current_metrics is None:
        return check_abs, reasons, checks

    delta_f1_toxic = candidate_metrics["f1_toxic"] - current_metrics["f1_toxic"]
    delta_macro_f1 = candidate_metrics["macro_f1"] - current_metrics["macro_f1"]

    check_f1_delta = delta_f1_toxic >= GATE_F1_TOXIC_DELTA
    check_macro_delta = delta_macro_f1 >= GATE_MACRO_F1_DELTA

    checks["f1_toxic_delta"] = {
        "passed": check_f1_delta,
        "delta": float(delta_f1_toxic),
        "threshold": GATE_F1_TOXIC_DELTA,
    }
    checks["macro_f1_delta"] = {
        "passed": check_macro_delta,
        "delta": float(delta_macro_f1),
        "threshold": GATE_MACRO_F1_DELTA,
    }

    if not check_f1_delta:
        reasons.append(
            f"f1_toxic delta={delta_f1_toxic:+.4f} < required={GATE_F1_TOXIC_DELTA:+.4f}"
        )
    if not check_macro_delta:
        reasons.append(
            f"macro_f1 delta={delta_macro_f1:+.4f} < allowed={GATE_MACRO_F1_DELTA:+.4f}"
        )

    passed = check_abs and check_f1_delta and check_macro_delta
    return passed, reasons, checks


def print_report(
    candidate_path: Path,
    manifest: Dict[str, Any],
    candidate_metrics: Dict[str, float],
    current_model_name: Optional[str],
    current_metrics: Optional[Dict[str, float]],
    gate_passed: bool,
    gate_reasons: List[str],
    gate_checks: Dict[str, Any],
    force: bool,
) -> None:
    training_type = manifest.get("training_type") or "unknown"
    pseudo_info = manifest.get("pseudo_labels") or {}
    pseudo_batch = pseudo_info.get("batch_id") or "none"
    seed_model = pseudo_info.get("seed_model") or "none"

    delta_f1_toxic = _safe_delta(candidate_metrics["f1_toxic"], None if not current_metrics else current_metrics["f1_toxic"])
    delta_macro_f1 = _safe_delta(candidate_metrics["macro_f1"], None if not current_metrics else current_metrics["macro_f1"])

    f1_line_ok = gate_checks.get("f1_toxic_abs", {}).get("passed", True)
    macro_line_ok = True
    if current_metrics is not None:
        f1_line_ok = f1_line_ok and gate_checks.get("f1_toxic_delta", {}).get("passed", False)
        macro_line_ok = gate_checks.get("macro_f1_delta", {}).get("passed", False)

    print("\n=== Promotion Report ===")
    print(f"Candidate: {candidate_path}")
    print(f"Training type: {training_type}")
    print(f"Pseudo batch: {pseudo_batch}")
    print(f"Seed model: {seed_model}")
    print("")
    print(f"Evaluation (victsd_gold test set, n={candidate_metrics['n_eval']}):")
    if current_metrics is not None:
        print(
            f"f1_toxic:   {candidate_metrics['f1_toxic']:.4f}  "
            f"(current: {current_metrics['f1_toxic']:.4f}, delta: {delta_f1_toxic:+.4f}) {_symbol(f1_line_ok)}"
        )
        print(
            f"macro_f1:   {candidate_metrics['macro_f1']:.4f}  "
            f"(current: {current_metrics['macro_f1']:.4f}, delta: {delta_macro_f1:+.4f}) {_symbol(macro_line_ok)}"
        )
    else:
        print(
            f"f1_toxic:   {candidate_metrics['f1_toxic']:.4f}  "
            f"(min abs: {GATE_MIN_F1_TOXIC_ABS:.4f}) {_symbol(f1_line_ok)}"
        )
        print("macro_f1:   N/A (no --current-serving provided)")

    print(f"precision_toxic: {candidate_metrics['precision_toxic']:.4f}")
    print(f"recall_toxic:    {candidate_metrics['recall_toxic']:.4f}")

    effective_pass = gate_passed or force
    print("")
    print(f"Gate result: {'PASS' if effective_pass else 'FAIL'}")
    if force and not gate_passed:
        print("Reason: Numeric gate failed but bypassed by --force")
        print("Details:")
        for reason in gate_reasons:
            print(f"- {reason}")
    elif gate_passed:
        print("Reason: All gate checks passed")
    else:
        print("Reason: " + "; ".join(gate_reasons))

    if current_model_name:
        print(f"Current serving: {current_model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and promote candidate model with hard gate")
    parser.add_argument("--candidate", required=True, type=str, help="Path to candidate model directory")
    parser.add_argument(
        "--eval-data",
        required=True,
        type=str,
        help="Gold evaluation set path (must be data/processed/victsd_gold/test.jsonl)",
    )
    parser.add_argument(
        "--current-serving",
        default=None,
        type=str,
        help="Current serving model id for delta comparison (e.g. phobert/v1)",
    )
    parser.add_argument("--force", action="store_true", help="Bypass numeric gate and promote anyway")
    parser.add_argument("--dry-run", action="store_true", help="Only evaluate and print report")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).expanduser().resolve()
    if not candidate_path.exists() or not candidate_path.is_dir():
        _err(f"Candidate path not found or not a directory: {candidate_path}")
        raise SystemExit(2)

    training_manifest_path = candidate_path / "training_manifest.json"
    if not training_manifest_path.exists():
        _err("Not a valid candidate: missing training_manifest.json")
        raise SystemExit(2)

    manifest = _read_json(training_manifest_path)

    expected_eval = (REPO_ROOT / "data" / "processed" / "victsd_gold" / "test.jsonl").resolve()
    eval_data_path = Path(args.eval_data).expanduser().resolve()
    if eval_data_path != expected_eval:
        _err(
            "--eval-data must be data/processed/victsd_gold/test.jsonl "
            f"(resolved expected: {expected_eval}, got: {eval_data_path})"
        )
        raise SystemExit(2)

    texts, labels = load_eval_dataset(eval_data_path)

    _info(f"Evaluating candidate: {candidate_path}")
    candidate_metrics = evaluate_phobert_checkpoint(
        model_path=candidate_path,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    current_metrics: Optional[Dict[str, float]] = None
    if args.current_serving:
        _info(f"Evaluating current serving model: {args.current_serving}")
        model_root = Path(os.getenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(MODEL_OPTIONS_DIR))).expanduser().resolve()
        model_type, _, current_model_path = resolve_model_path(model_root, args.current_serving)
        if model_type != "phobert":
            _err(f"Current serving must be a phobert model, got: {args.current_serving}")
            raise SystemExit(2)
        current_metrics = evaluate_phobert_checkpoint(
            model_path=current_model_path,
            texts=texts,
            labels=labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    gate_passed, gate_reasons, gate_checks = evaluate_gate(candidate_metrics, current_metrics)

    print_report(
        candidate_path=candidate_path,
        manifest=manifest,
        candidate_metrics=candidate_metrics,
        current_model_name=args.current_serving,
        current_metrics=current_metrics,
        gate_passed=gate_passed,
        gate_reasons=gate_reasons,
        gate_checks=gate_checks,
        force=args.force,
    )

    run_id = str(manifest.get("run_id") or candidate_path.name)
    promoted_at = _iso_now()

    eval_results_payload = {
        "eval_set": "victsd_gold/test.jsonl",
        "threshold": 0.5,
        "n_eval": int(candidate_metrics["n_eval"]),
        "candidate": {
            "f1_toxic": float(candidate_metrics["f1_toxic"]),
            "macro_f1": float(candidate_metrics["macro_f1"]),
            "precision_toxic": float(candidate_metrics["precision_toxic"]),
            "recall_toxic": float(candidate_metrics["recall_toxic"]),
        },
        "current": None
        if current_metrics is None
        else {
            "model": args.current_serving,
            "f1_toxic": float(current_metrics["f1_toxic"]),
            "macro_f1": float(current_metrics["macro_f1"]),
            "precision_toxic": float(current_metrics["precision_toxic"]),
            "recall_toxic": float(current_metrics["recall_toxic"]),
        },
        "delta": None
        if current_metrics is None
        else {
            "f1_toxic": float(candidate_metrics["f1_toxic"] - current_metrics["f1_toxic"]),
            "macro_f1": float(candidate_metrics["macro_f1"] - current_metrics["macro_f1"]),
        },
        "gate": {
            "passed": bool(gate_passed),
            "checks": gate_checks,
            "reasons": gate_reasons,
        },
    }

    effective_promote = gate_passed or args.force

    if args.dry_run:
        print("\n[DRY-RUN] No files updated.")
        print(
            "To serve this model, set VIETTOXIC_MODEL_OPTIONS_DIR or copy to "
            "models/options/phobert/ and restart backend"
        )
        return

    if effective_promote:
        manifest["promotion_status"] = "promoted"
        manifest["promoted_at"] = promoted_at
        manifest["promoted_by"] = "promote_model.py"
        manifest["eval_results"] = eval_results_payload
        _write_json(training_manifest_path, manifest)

        run_record_path = REPO_ROOT / "experiments" / "retrain_runs" / f"{run_id}.json"
        existing_run_record: Dict[str, Any] = {}
        if run_record_path.exists():
            try:
                existing_run_record = _read_json(run_record_path)
            except Exception:
                existing_run_record = {"run_id": run_id}

        retrain_config_rel = f"experiments/retrain_runs/{run_id}.json"
        lineage = {
            "promoted_model": run_id,
            "seed_model": (manifest.get("pseudo_labels") or {}).get("seed_model"),
            "pseudo_label_batch": (manifest.get("pseudo_labels") or {}).get("batch_id"),
            "retrain_config": retrain_config_rel,
            "eval_set": "victsd_gold/test.jsonl",
            "eval_results": {
                "f1_toxic": float(candidate_metrics["f1_toxic"]),
                "macro_f1": float(candidate_metrics["macro_f1"]),
            },
            "gate_passed": bool(gate_passed),
            "force_promoted": bool(args.force and not gate_passed),
        }

        existing_run_record["promoted_at"] = promoted_at
        existing_run_record["lineage"] = lineage
        _write_json(run_record_path, existing_run_record)

        print(f"\nModel promoted. Available as phobert/{run_id}")
        print(
            "To serve this model, set VIETTOXIC_MODEL_OPTIONS_DIR or copy to "
            "models/options/phobert/ and restart backend"
        )
        return

    manifest["promotion_status"] = "rejected"
    manifest["rejected_at"] = promoted_at
    manifest["rejected_by"] = "promote_model.py"
    manifest["rejection_reason"] = "; ".join(gate_reasons) if gate_reasons else "Gate failed"
    manifest["eval_results"] = eval_results_payload
    _write_json(training_manifest_path, manifest)

    print("\nModel rejected by gate.")
    print(
        "To serve this model, set VIETTOXIC_MODEL_OPTIONS_DIR or copy to "
        "models/options/phobert/ and restart backend"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
