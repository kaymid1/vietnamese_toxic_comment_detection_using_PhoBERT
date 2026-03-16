#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_crawled_local.py
Run PhoBERT sequence-classification inference on crawled segments.
Domain-aware thresholding: seg_threshold is adjusted per URL category
(news / social / forum / unknown) to reduce domain-bias false positives.

Example:
  python infer_crawled_local.py \
    --model_name v2 \
    --model_base_dir /models_2/phobert \
    --data_dir data/raw/crawled_urls \
    --out_dir data/processed \
    --batch_size 8 \
    --max_length 256 \
    --page_threshold 0.25 \
    --seg_threshold 0.5

Threshold overrides per category (optional):
  --threshold_news 0.75 --threshold_social 0.50 --threshold_forum 0.62 --threshold_unknown 0.65

Notes:
- Local only: model loaded from local checkpoint
- Tokenizer from HF ("vinai/phobert-base") unless you pass --tokenizer_name
- Device auto: CUDA > MPS (Apple Silicon) > CPU
- domain_classifier.py must be in the same directory as this script
"""

import json
import argparse
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Domain-aware threshold classifier (same directory)
from domain_classifier import DomainClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL_DIR = Path(os.getenv("VIETTOXIC_MODEL_BASE_DIR", "/models_2/phobert")).expanduser()
if not BASE_MODEL_DIR.exists():
    repo_local_model_dir = SCRIPT_DIR / "models_2" / "phobert"
    if repo_local_model_dir.exists():
        BASE_MODEL_DIR = repo_local_model_dir


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_segments_jsonl(path: Path) -> List[str]:
    segs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("text", "")
                if isinstance(t, str) and t.strip():
                    segs.append(t.strip())
            except json.JSONDecodeError:
                continue
    return segs


def iter_url_folders(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return [p for p in root_dir.iterdir() if p.is_dir()]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def list_available_models(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Model base path must be a directory: {base_dir}")
    try:
        return sorted(
            [
                p.name
                for p in base_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
        )
    except PermissionError as exc:
        raise PermissionError(f"Permission denied while reading model directory: {base_dir}") from exc


def get_default_model(base_dir: Path) -> Optional[str]:
    models = list_available_models(base_dir)
    if not models:
        return None
    if "v2" in models:
        return "v2"
    return models[0]


def resolve_model_path(base_dir: Path, model_name: Optional[str]) -> Path:
    models = list_available_models(base_dir)
    if not models:
        raise ValueError(f"No models found under {base_dir}")

    if not model_name:
        model_name = get_default_model(base_dir)

    if model_name is None:
        raise ValueError("No default model available")

    if any(x in model_name for x in ("..", "/", "\\")):
        raise ValueError(f"Invalid model name: {model_name}")

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {models}")

    model_path = base_dir / model_name
    if not model_path.is_dir():
        raise ValueError(f"Model '{model_name}' is not a directory under {base_dir}")
    return model_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_probs(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> List[float]:
    """Returns toxic prob for class 1."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1].detach().float().cpu().tolist()


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_crawled(
    model_path: Optional[str],
    data_dir: str,
    out_dir: str,
    batch_size: int = 8,
    max_length: int = 256,
    page_threshold: float = 0.25,
    seg_threshold: float = 0.5,          # fallback / default threshold
    tokenizer_name: str = "vinai/phobert-base",
    device: str = "auto",
    fp16: bool = False,
    limit_pages: int = 0,
    quiet: bool = False,
    only_url_hashes: Optional[List[str]] = None,
    debug_force_prob: Optional[float] = None,
    model_name: Optional[str] = None,
    model_base_dir: Optional[str] = None,
    # Per-category threshold overrides (None = use DomainClassifier defaults)
    threshold_news:    Optional[float] = None,
    threshold_social:  Optional[float] = None,
    threshold_forum:   Optional[float] = None,
    threshold_unknown: Optional[float] = None,
) -> Dict[str, Any]:

    # ── Validate model path ──────────────────────────────────────────────
    if model_path:
        resolved_model_path = Path(model_path).expanduser().resolve()
        selected_model_name = resolved_model_path.name
    else:
        base_dir = Path(model_base_dir).expanduser() if model_base_dir else BASE_MODEL_DIR
        resolved_model_path = resolve_model_path(base_dir, model_name).resolve()
        selected_model_name = resolved_model_path.name

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model path not found: {resolved_model_path}")
    if not resolved_model_path.is_dir():
        raise NotADirectoryError(f"Model path must be a directory: {resolved_model_path}")

    config_path      = resolved_model_path / "config.json"
    safetensors_path = resolved_model_path / "model.safetensors"
    pt_path          = resolved_model_path / "pytorch_model.bin"
    if not config_path.exists() or not (safetensors_path.exists() or pt_path.exists()):
        files = sorted([p.name for p in resolved_model_path.iterdir() if p.is_file()])
        missing = []
        if not config_path.exists():
            missing.append("config.json")
        if not (safetensors_path.exists() or pt_path.exists()):
            missing.append("model.safetensors or pytorch_model.bin")
        raise FileNotFoundError(
            f"Checkpoint folder missing: {', '.join(missing)}. Files found: {files}"
        )

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Domain classifier ────────────────────────────────────────────────
    threshold_overrides: Dict[str, float] = {}
    if threshold_news    is not None: threshold_overrides["news"]    = threshold_news
    if threshold_social  is not None: threshold_overrides["social"]  = threshold_social
    if threshold_forum   is not None: threshold_overrides["forum"]   = threshold_forum
    if threshold_unknown is not None: threshold_overrides["unknown"] = threshold_unknown

    # Pass the CLI seg_threshold as the default unknown fallback when no override given
    if "unknown" not in threshold_overrides:
        threshold_overrides["unknown"] = seg_threshold

    domain_clf = DomainClassifier(threshold_overrides=threshold_overrides)

    if not quiet:
        print("[INFO] Domain-aware thresholds:")
        for cat, thr in domain_clf.thresholds.items():
            marker = " ← CLI default (--seg_threshold)" if cat == "unknown" and threshold_unknown is None else ""
            print(f"         {cat:<10} → {thr}{marker}")
        print(f"[INFO] Selected model: {selected_model_name}")
        print(f"[INFO] Model path: {resolved_model_path}")

    # ── Device ───────────────────────────────────────────────────────────
    device = pick_device(device)
    if not quiet:
        print(f"[INFO] Device: {device}")

    # ── Load tokenizer & model (skip in debug force-prob mode) ───────────
    tokenizer = None
    model = None
    if debug_force_prob is None:
        if not quiet:
            print("[INFO] Loading tokenizer:", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if not quiet:
            print("[INFO] Loading model from:", str(resolved_model_path))
        config = AutoConfig.from_pretrained(str(resolved_model_path), local_files_only=True)
        model  = AutoModelForSequenceClassification.from_pretrained(
            str(resolved_model_path), config=config, local_files_only=True
        )
        model.to(device)
        if fp16 and device.type == "cuda":
            model.half()
            if not quiet:
                print("[INFO] Using FP16 on CUDA")
        model.eval()
    else:
        if debug_force_prob < 0.0 or debug_force_prob > 1.0:
            raise ValueError("debug_force_prob must be in [0.0, 1.0]")
        if not quiet:
            print(f"[INFO] Debug mode: force toxic_prob={debug_force_prob} for all segments")

    # ── Output paths ─────────────────────────────────────────────────────
    seg_out_path  = out_dir / "crawled_predictions.jsonl"
    page_out_path = out_dir / "page_level_results.json"
    csv_out_path  = out_dir / "page_level_results.csv"

    # ── Scan folders ─────────────────────────────────────────────────────
    url_folders = iter_url_folders(data_dir)
    if only_url_hashes:
        allow = set(only_url_hashes)
        url_folders = [p for p in url_folders if p.name in allow]
    if limit_pages and limit_pages > 0:
        url_folders = url_folders[:limit_pages]

    if not url_folders:
        print(f"[WARN] No url folders found in: {data_dir}")
        return {
            "page_results": [], "segment_results": [],
            "out_dir": str(out_dir),
            "page_out_path": str(page_out_path),
            "seg_out_path":  str(seg_out_path),
            "csv_out_path":  str(csv_out_path),
        }

    predictions_all: List[Dict[str, Any]] = []
    page_results:    List[Dict[str, Any]] = []

    # Category stats for end-of-run summary
    category_counts: Dict[str, int] = {}

    # ── Per-page processing ──────────────────────────────────────────────
    for folder in tqdm(url_folders, desc="Processing pages"):
        meta_path = folder / "meta.json"
        segs_path = folder / "segments.jsonl"

        if not (meta_path.exists() and segs_path.exists()):
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        url = meta.get("url", "unknown")

        # ── Domain-aware threshold ────────────────────────────────────
        domain_category, effective_threshold = domain_clf.classify(url)
        category_counts[domain_category] = category_counts.get(domain_category, 0) + 1

        segments = load_segments_jsonl(segs_path)
        if not segments:
            continue

        page_preds: List[Dict[str, Any]] = []

        # ── Batch inference ───────────────────────────────────────────
        for i in range(0, len(segments), batch_size):
            batch_texts = segments[i: i + batch_size]
            if debug_force_prob is None:
                probs = predict_probs(batch_texts, tokenizer, model, device=device, max_length=max_length)
            else:
                probs = [float(debug_force_prob)] * len(batch_texts)
            for text, prob in zip(batch_texts, probs):
                label = 1 if prob > effective_threshold else 0   # ← domain-aware threshold
                page_preds.append({
                    "text":        text,
                    "toxic_prob":  round(float(prob), 4),
                    "toxic_label": label,
                })

        # ── Aggregate page-level ──────────────────────────────────────
        total_segs  = len(page_preds)
        toxic_count = sum(1 for p in page_preds if p["toxic_label"] == 1)
        toxic_ratio = toxic_count / total_segs if total_segs else 0.0
        page_toxic  = 1 if toxic_ratio > page_threshold else 0
        avg_prob    = sum(p["toxic_prob"] for p in page_preds) / total_segs if total_segs else 0.0

        top5 = sorted(page_preds, key=lambda x: x["toxic_prob"], reverse=True)[:5]

        url_hash = folder.name
        page_results.append({
            "url_hash":             url_hash,
            "url":                  url,
            "domain_category":      domain_category,          # ← NEW
            "seg_threshold_used":   effective_threshold,      # ← NEW
            "method":               meta.get("method",  "unknown"),
            "status":               meta.get("status",  "unknown"),
            "total_segments":       total_segs,
            "toxic_segments":       toxic_count,
            "toxic_ratio":          round(toxic_ratio, 4),
            "page_toxic":           page_toxic,
            "avg_toxic_prob":       round(float(avg_prob), 4),
            "top5_toxic_segments":  [s["text"][:150] + "..." for s in top5],
        })

        for p in page_preds:
            predictions_all.append({
                "url_hash":           url_hash,
                "url":                url,
                "domain_category":    domain_category,        # ← NEW
                "seg_threshold_used": effective_threshold,    # ← NEW
                **p,
            })

    # ── Save outputs ─────────────────────────────────────────────────────
    with seg_out_path.open("w", encoding="utf-8") as f:
        for pred in predictions_all:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    df_pages = pd.DataFrame(page_results)
    page_out_path.write_text(
        df_pages.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    df_pages.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    # ── Summary ──────────────────────────────────────────────────────────
    if not quiet:
        print("\n" + "=" * 65)
        print(f"PAGE-LEVEL SUMMARY (max_length={max_length})")
        print("=" * 65)

        if len(df_pages) > 0:
            cols = ["url", "domain_category", "seg_threshold_used",
                    "total_segments", "toxic_ratio", "page_toxic", "avg_toxic_prob"]
            cols = [c for c in cols if c in df_pages.columns]
            print(df_pages[cols].to_string(index=False))

            print("\nMetrics tổng hợp:")
            print(f"  - Số trang processed    : {len(df_pages)}")
            print(f"  - Avg toxic_ratio        : {df_pages['toxic_ratio'].mean():.4f}")
            print(f"  - % trang toxic          : {(df_pages['page_toxic'] == 1).mean() * 100:.2f}%")
            print(f"  - Avg segments/page      : {df_pages['total_segments'].mean():.1f}")
            print("\n  Domain category breakdown:")
            for cat, cnt in sorted(category_counts.items()):
                thr = domain_clf.thresholds.get(cat, "?")
                print(f"    {cat:<10} : {cnt:>4} pages  (threshold={thr})")
        else:
            print("[WARN] No pages produced results.")

        print("=" * 65)
        print("\nKết quả đã lưu tại:")
        print(f"  - {seg_out_path}  (segment-level)")
        print(f"  - {page_out_path}  (page-level json)")
        print(f"  - {csv_out_path}   (page-level csv)")

    return {
        "page_results":     page_results,
        "segment_results":  predictions_all,
        "out_dir":          str(out_dir),
        "page_out_path":    str(page_out_path),
        "seg_out_path":     str(seg_out_path),
        "csv_out_path":     str(csv_out_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model directory name under --model_base_dir (e.g. v2, v1, lstm)",
    )
    ap.add_argument(
        "--model_base_dir",
        type=str,
        default=str(BASE_MODEL_DIR),
        help="Base dir that contains model subdirectories",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Explicit model directory path. If provided, overrides --model_name",
    )
    ap.add_argument("--tokenizer_name",  type=str, default="vinai/phobert-base")
    ap.add_argument("--data_dir",        type=str, default="data/raw/crawled_urls")
    ap.add_argument("--out_dir",         type=str, default="data/processed")
    ap.add_argument("--max_length",      type=int, default=256)
    ap.add_argument("--batch_size",      type=int, default=8)

    # Threshold args
    ap.add_argument("--seg_threshold",   type=float, default=0.5,
                    help="Default/fallback seg threshold (also used for 'unknown' domains unless --threshold_unknown set)")
    ap.add_argument("--page_threshold",  type=float, default=0.25)
    ap.add_argument("--threshold_news",    type=float, default=None, help="Override threshold for news domains")
    ap.add_argument("--threshold_social",  type=float, default=None, help="Override threshold for social domains")
    ap.add_argument("--threshold_forum",   type=float, default=None, help="Override threshold for forum domains")
    ap.add_argument("--threshold_unknown", type=float, default=None, help="Override threshold for unknown domains")

    # Device & perf
    ap.add_argument("--device",      type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--fp16",        action="store_true")
    ap.add_argument("--limit_pages", type=int, default=0)
    ap.add_argument("--quiet",       action="store_true")
    ap.add_argument("--debug_force_prob", type=float, default=None,
                    help="Debug only: force toxic_prob for every segment (0..1), bypass model inference")

    args = ap.parse_args()

    infer_crawled(
        model_path      = args.model_path,
        model_name      = args.model_name,
        model_base_dir  = args.model_base_dir,
        data_dir        = args.data_dir,
        out_dir         = args.out_dir,
        batch_size      = args.batch_size,
        max_length      = args.max_length,
        page_threshold  = args.page_threshold,
        seg_threshold   = args.seg_threshold,
        tokenizer_name  = args.tokenizer_name,
        device          = args.device,
        fp16            = args.fp16,
        limit_pages     = args.limit_pages,
        quiet           = args.quiet,
        debug_force_prob = args.debug_force_prob,
        threshold_news    = args.threshold_news,
        threshold_social  = args.threshold_social,
        threshold_forum   = args.threshold_forum,
        threshold_unknown = args.threshold_unknown,
    )


if __name__ == "__main__":
    main()
