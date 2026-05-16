#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate pseudo-labels from existing crawl artifacts.

IMPORTANT:
- Independent from eval pipeline.
- Never writes to data/processed/victsd_gold/ or data/victsd/.
"""

import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_crawled_local import (  # noqa: E402
    build_segment_hash,
    pick_device,
    predict_probs,
    resolve_model_path,
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_segments(segments_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with segments_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                _warn(f"Invalid JSON at {segments_path}:{line_idx}; skipping")
                continue
            rows.append(obj)
    return rows


def collect_crawl_segments(crawl_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    records: List[Dict[str, Any]] = []

    if not crawl_dir.exists() or not crawl_dir.is_dir():
        raise FileNotFoundError(f"CRAWL_DATA_DIR not found or not a directory: {crawl_dir}")

    for folder in sorted([p for p in crawl_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        segments_path = folder / "segments.jsonl"
        meta_path = folder / "meta.json"

        if not segments_path.exists() or not meta_path.exists():
            warnings.append(
                f"Skipping {folder.name}: missing {'segments.jsonl' if not segments_path.exists() else ''}"
                f"{' and ' if (not segments_path.exists() and not meta_path.exists()) else ''}"
                f"{'meta.json' if not meta_path.exists() else ''}"
            )
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Skipping {folder.name}: cannot read meta.json ({exc})")
            continue

        url = (meta.get("url") or "").strip()
        domain_category = (meta.get("domain_category") or "unknown").strip().lower()

        segments = load_segments(segments_path)
        if not segments:
            warnings.append(f"Skipping {folder.name}: segments.jsonl has no valid rows")
            continue

        for idx, row in enumerate(segments):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            html_tag_effective = (row.get("html_tag_effective") or "body").strip().lower()
            segment_hash = build_segment_hash(text, html_tag_effective)

            records.append(
                {
                    "text": text.strip(),
                    "url_hash": folder.name,
                    "url": url,
                    "domain_category": domain_category or "unknown",
                    "segment_index": row.get("segment_index", idx),
                    "html_tag_effective": html_tag_effective,
                    "segment_hash": segment_hash,
                }
            )

    return records, warnings


def resolve_seed_checkpoint(model_root: Path, preferred_model: str) -> Tuple[str, Path]:
    attempts = [preferred_model]
    if "phobert/v1" not in attempts:
        attempts.append("phobert/v1")

    errors: List[str] = []
    for model_id in attempts:
        try:
            model_type, model_name, model_path = resolve_model_path(model_root, model_id)
            if model_type != "phobert":
                raise ValueError(f"Expected phobert model, got: {model_type}/{model_name}")
            return f"{model_type}/{model_name}", model_path.resolve()
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    msg = "Failed to resolve seed checkpoint. Attempts:\n- " + "\n- ".join(errors)
    raise RuntimeError(msg)


def run_inference(
    records: List[Dict[str, Any]],
    model_path: Path,
    batch_size: int,
    max_length: int,
) -> List[float]:
    device = pick_device("auto")
    _info(f"Device: {device}")

    tokenizer_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    texts = [r["text"] for r in records]
    probs: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        probs.extend(
            predict_probs(
                texts=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            )
        )

    return probs


def gate_segments(
    records: List[Dict[str, Any]],
    probs: List[float],
    gate_high: float,
    gate_low: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    accepted_pre_dedup: List[Dict[str, Any]] = []

    stats = {
        "n_source_segments": len(records),
        "n_candidate_middle": 0,
        "n_accepted_toxic_pre": 0,
        "n_accepted_clean_pre": 0,
    }

    for rec, prob in zip(records, probs):
        if prob >= gate_high:
            gate_status = "accepted_toxic"
            toxicity = 1
            stats["n_accepted_toxic_pre"] += 1
        elif prob <= gate_low:
            gate_status = "accepted_clean"
            toxicity = 0
            stats["n_accepted_clean_pre"] += 1
        else:
            gate_status = "candidate_middle"
            toxicity = None
            stats["n_candidate_middle"] += 1

        candidate_row = {
            "text": rec["text"],
            "pseudo_prob": round(float(prob), 6),
            "segment_hash": rec["segment_hash"],
            "url_hash": rec["url_hash"],
            "domain_category": rec["domain_category"],
            "gate_status": gate_status,
        }
        if toxicity is not None:
            candidate_row["toxicity"] = toxicity
        candidates.append(candidate_row)

        if gate_status in ("accepted_toxic", "accepted_clean"):
            accepted_pre_dedup.append({**candidate_row, "toxicity": toxicity})

    return accepted_pre_dedup, candidates, stats


def dedup_accepted(accepted_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    deduped_removed = 0

    seen_domains_by_hash: Dict[str, set] = {}
    warned_hashes: set = set()

    for row in accepted_rows:
        seg_hash = row["segment_hash"]
        domain = (row.get("domain_category") or "unknown").strip().lower()

        seen_domains = seen_domains_by_hash.setdefault(seg_hash, set())
        if domain in seen_domains:
            deduped_removed += 1
            continue

        if seen_domains and seg_hash not in warned_hashes:
            _warn(
                "Same segment_hash appears across multiple domains; "
                f"keeping all cross-domain rows (segment_hash={seg_hash})"
            )
            warned_hashes.add(seg_hash)

        seen_domains.add(domain)
        kept.append(row)

    return kept, deduped_removed


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_batch_id() -> str:
    return f"pseudo_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"


def print_summary(manifest: Dict[str, Any], dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("PSEUDO-LABEL SUMMARY")
    print("=" * 60)
    for key in [
        "batch_id",
        "generated_at",
        "seed_model",
        "seed_model_path",
        "gate_high",
        "gate_low",
        "source_crawl_dir",
        "n_source_segments",
        "n_candidate_middle",
        "n_accepted_toxic",
        "n_accepted_clean",
        "n_deduped_removed",
        "accepted_toxic_ratio",
        "schema_version",
    ]:
        print(f"- {key}: {manifest[key]}")

    if manifest["accepted_toxic_ratio"] < 0.05:
        _warn("accepted_toxic_ratio < 0.05 (skewed heavily clean)")
    if manifest["n_accepted_toxic"] < 50:
        _warn("n_accepted_toxic < 50 (too few toxic samples)")

    print("[NOTE] Spot-check ~50 accepted samples trước khi dùng")
    if dry_run:
        print("[NOTE] Dry-run mode enabled: no files were written")


def main() -> None:
    default_crawl_dir = os.getenv("CRAWL_DATA_DIR", "data/raw/crawled_urls")
    default_model_options_dir = os.getenv(
        "VIETTOXIC_MODEL_OPTIONS_DIR",
        os.getenv("MODEL_OPTIONS_DIR", "models/options"),
    )
    default_seed_model_name = os.getenv("SEED_MODEL_NAME", "phobert/phobert_lora_latest")
    default_output_dir = os.getenv("PSEUDO_OUTPUT_DIR", "data/pseudo_labels")

    default_batch_size = _env_int("BATCH_SIZE", 16)
    default_max_length = _env_int("MAX_LENGTH", 256)
    default_gate_high = _env_float("GATE_HIGH", 0.80)
    default_gate_low = _env_float("GATE_LOW", 0.20)
    default_seed = _env_int("SEED", 42)

    parser = argparse.ArgumentParser(description="Generate pseudo-labels from crawl artifacts")
    parser.add_argument("--crawl-dir", type=str, default=default_crawl_dir, help="Override CRAWL_DATA_DIR")
    parser.add_argument("--dry-run", action="store_true", help="Run full pipeline without writing files")
    args = parser.parse_args()

    if not (0.0 <= default_gate_low < default_gate_high <= 1.0):
        raise ValueError("Invalid gate thresholds: require 0.0 <= GATE_LOW < GATE_HIGH <= 1.0")

    set_seed(default_seed)

    crawl_dir = Path(args.crawl_dir).expanduser().resolve()
    model_root = Path(default_model_options_dir).expanduser().resolve()
    output_root = Path(default_output_dir).expanduser().resolve()

    _info(f"CRAWL_DATA_DIR: {crawl_dir}")
    _info(f"MODEL_OPTIONS_DIR: {model_root}")
    _info(f"SEED_MODEL_NAME: {default_seed_model_name}")

    records, load_warnings = collect_crawl_segments(crawl_dir)
    for w in load_warnings:
        _warn(w)

    if not records:
        raise RuntimeError("No valid segments found in crawl artifacts")

    seed_model_name, seed_model_path = resolve_seed_checkpoint(model_root, default_seed_model_name)
    _info(f"Resolved seed model: {seed_model_name}")
    _info(f"Resolved seed model path: {seed_model_path}")

    probs = run_inference(
        records=records,
        model_path=seed_model_path,
        batch_size=default_batch_size,
        max_length=default_max_length,
    )

    accepted_pre_dedup, candidates, gate_stats = gate_segments(
        records=records,
        probs=probs,
        gate_high=default_gate_high,
        gate_low=default_gate_low,
    )

    accepted_deduped, n_deduped_removed = dedup_accepted(accepted_pre_dedup)

    batch_id = build_batch_id()
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    accepted_rows: List[Dict[str, Any]] = []
    for row in accepted_deduped:
        accepted_rows.append(
            {
                "text": row["text"],
                "toxicity": int(row["toxicity"]),
                "pseudo_prob": float(row["pseudo_prob"]),
                "segment_hash": row["segment_hash"],
                "url_hash": row["url_hash"],
                "domain_category": row["domain_category"],
                "meta": {
                    "source": "pseudo_label",
                    "batch_id": batch_id,
                    "seed_model": seed_model_name,
                },
            }
        )

    n_accepted_toxic = sum(1 for r in accepted_rows if r["toxicity"] == 1)
    n_accepted_clean = sum(1 for r in accepted_rows if r["toxicity"] == 0)
    denom = n_accepted_toxic + n_accepted_clean
    accepted_toxic_ratio = round(n_accepted_toxic / denom, 6) if denom else 0.0

    manifest = {
        "batch_id": batch_id,
        "generated_at": generated_at,
        "seed_model": seed_model_name,
        "seed_model_path": str(seed_model_path),
        "gate_high": float(default_gate_high),
        "gate_low": float(default_gate_low),
        "source_crawl_dir": str(crawl_dir),
        "n_source_segments": int(gate_stats["n_source_segments"]),
        "n_candidate_middle": int(gate_stats["n_candidate_middle"]),
        "n_accepted_toxic": int(n_accepted_toxic),
        "n_accepted_clean": int(n_accepted_clean),
        "n_deduped_removed": int(n_deduped_removed),
        "accepted_toxic_ratio": accepted_toxic_ratio,
        "schema_version": "v1",
    }

    batch_dir = output_root / batch_id
    accepted_path = batch_dir / "accepted.jsonl"
    candidates_path = batch_dir / "candidates.jsonl"
    manifest_path = batch_dir / "manifest.json"

    if not args.dry_run:
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(accepted_path, accepted_rows)
        write_jsonl(candidates_path, candidates)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _info(f"Wrote accepted: {accepted_path}")
        _info(f"Wrote candidates: {candidates_path}")
        _info(f"Wrote manifest: {manifest_path}")

    print_summary(manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
