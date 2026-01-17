#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_crawled_local.py
Run PhoBERT sequence-classification inference on crawled segments.

Example:
  python infer_crawled_local.py \
    --model_path ./experiments/phobert_exp-01/checkpoint-best \
    --data_dir data/raw/crawled_urls \
    --out_dir data/processed \
    --batch_size 8 \
    --max_length 256 \
    --page_threshold 0.25 \
    --seg_threshold 0.5

Notes:
- Local only: model loaded from local checkpoint
- Tokenizer from HF ("vinai/phobert-base") unless you pass --tokenizer_name
- Device auto: CUDA > MPS (Apple Silicon) > CPU
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def pick_device(prefer: str = "auto") -> torch.device:
    """
    prefer: auto|cuda|mps|cpu
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


@torch.inference_mode()
def predict_probs(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> List[float]:
    """
    Returns toxic prob for class 1.
    """
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
    toxic_probs = probs[:, 1].detach().float().cpu().tolist()
    return toxic_probs


def iter_url_folders(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return [p for p in root_dir.iterdir() if p.is_dir()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Local checkpoint dir (e.g. ./experiments/.../checkpoint-best)")
    ap.add_argument("--tokenizer_name", type=str, default="vinai/phobert-base", help="HF tokenizer name or local tokenizer path")
    ap.add_argument("--data_dir", type=str, default="data/raw/crawled_urls", help="Root of crawled url folders")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)

    # Thresholds
    ap.add_argument("--seg_threshold", type=float, default=0.5, help="Segment toxic label threshold (prob > seg_threshold => toxic)")
    ap.add_argument("--page_threshold", type=float, default=0.25, help="Page toxic threshold over ratio of toxic segments")

    # Device & perf
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA (ignored on CPU/MPS)")
    ap.add_argument("--limit_pages", type=int, default=0, help="Debug: process only first N pages (0 = all)")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    model_path = Path(args.model_path).expanduser()
    resolved_model_path = model_path.resolve()
    if not resolved_model_path.exists():
        cwd = Path.cwd()
        print("[ERROR] Model path does not exist.")
        print(f"- CWD: {cwd}")
        print(f"- Resolved model_path: {resolved_model_path}")
        raise FileNotFoundError(f"Model path not found: {resolved_model_path}")
    if not resolved_model_path.is_dir():
        raise NotADirectoryError(f"Model path must be a directory: {resolved_model_path}")

    config_path = resolved_model_path / "config.json"
    safetensors_path = resolved_model_path / "model.safetensors"
    pt_path = resolved_model_path / "pytorch_model.bin"
    if not config_path.exists() or not (safetensors_path.exists() or pt_path.exists()):
        files = sorted([p.name for p in resolved_model_path.iterdir() if p.is_file()])
        missing = []
        if not config_path.exists():
            missing.append("config.json")
        if not (safetensors_path.exists() or pt_path.exists()):
            missing.append("model.safetensors or pytorch_model.bin")
        missing_str = ", ".join(missing) if missing else "unknown"
        raise FileNotFoundError(
            "Checkpoint folder missing required files: "
            f"{missing_str}. Files found: {files}"
        )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    if not args.quiet:
        print(f"[INFO] Device: {device}")

    # Load tokenizer & model
    if not args.quiet:
        print("[INFO] Loading tokenizer:", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if not args.quiet:
        print("[INFO] Loading model from local path:", str(resolved_model_path))
    config = AutoConfig.from_pretrained(
        str(resolved_model_path),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(resolved_model_path),
        config=config,
        local_files_only=True,
    )

    # dtype / device placement
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()
        if not args.quiet:
            print("[INFO] Using FP16 on CUDA")
    model.eval()

    # Output files
    seg_out_path = out_dir / "crawled_predictions.jsonl"
    page_out_path = out_dir / "page_level_results.json"
    csv_out_path = out_dir / "page_level_results.csv"

    # Scan folders
    url_folders = iter_url_folders(data_dir)
    if args.limit_pages and args.limit_pages > 0:
        url_folders = url_folders[: args.limit_pages]

    if not url_folders:
        print(f"[WARN] No url folders found in: {data_dir}")
        return

    predictions_all: List[Dict[str, Any]] = []
    page_results: List[Dict[str, Any]] = []

    for folder in tqdm(url_folders, desc="Processing pages"):
        meta_path = folder / "meta.json"
        segs_path = folder / "segments.jsonl"

        if not (meta_path.exists() and segs_path.exists()):
            continue

        # meta
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        segments = load_segments_jsonl(segs_path)
        if not segments:
            continue

        page_preds: List[Dict[str, Any]] = []

        # Batch inference
        for i in range(0, len(segments), args.batch_size):
            batch_texts = segments[i : i + args.batch_size]
            probs = predict_probs(
                batch_texts, tokenizer, model, device=device, max_length=args.max_length
            )
            for text, prob in zip(batch_texts, probs):
                label = 1 if prob > args.seg_threshold else 0
                page_preds.append(
                    {
                        "text": text,
                        "toxic_prob": round(float(prob), 4),
                        "toxic_label": label,
                    }
                )

        # Aggregate page-level
        total_segs = len(page_preds)
        toxic_count = sum(1 for p in page_preds if p["toxic_label"] == 1)
        toxic_ratio = toxic_count / total_segs if total_segs else 0.0
        page_toxic = 1 if toxic_ratio > args.page_threshold else 0
        avg_prob = sum(p["toxic_prob"] for p in page_preds) / total_segs if total_segs else 0.0

        top5 = sorted(page_preds, key=lambda x: x["toxic_prob"], reverse=True)[:5]

        url_hash = folder.name
        page_results.append(
            {
                "url_hash": url_hash,
                "url": meta.get("url", "unknown"),
                "method": meta.get("method", "unknown"),
                "status": meta.get("status", "unknown"),
                "total_segments": total_segs,
                "toxic_segments": toxic_count,
                "toxic_ratio": round(toxic_ratio, 4),
                "page_toxic": page_toxic,
                "avg_toxic_prob": round(float(avg_prob), 4),
                "top5_toxic_segments": [s["text"][:150] + "..." for s in top5],
            }
        )

        # NOTE: segment-level output currently does NOT include url_hash.
        # If you want joinable data for error analysis, add url_hash/url fields here.
        for p in page_preds:
            predictions_all.append(
                {
                    "url_hash": url_hash,
                    "url": meta.get("url", "unknown"),
                    **p,
                }
            )

    # Save segment-level jsonl
    with seg_out_path.open("w", encoding="utf-8") as f:
        for pred in predictions_all:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    # Save page-level
    df_pages = pd.DataFrame(page_results)
    page_out_path.write_text(
        df_pages.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    df_pages.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    # Summary
    print("\n" + "=" * 60)
    print(f"PAGE-LEVEL SUMMARY (max_length={args.max_length})")
    print("=" * 60)
    if len(df_pages) > 0:
        cols = ["url", "total_segments", "toxic_ratio", "page_toxic", "avg_toxic_prob", "method"]
        cols = [c for c in cols if c in df_pages.columns]
        print(df_pages[cols].to_string(index=False))

        print("\nMetrics tổng hợp:")
        print(f"- Số trang processed: {len(df_pages)}")
        print(f"- Avg toxic_ratio trên tất cả pages: {df_pages['toxic_ratio'].mean():.4f}")
        print(f"- % trang toxic (page_threshold {args.page_threshold}): {(df_pages['page_toxic'] == 1).mean() * 100:.2f}%")
        print(f"- Avg segments/page: {df_pages['total_segments'].mean():.1f}")
    else:
        print("[WARN] No pages produced results. Check your data_dir structure & segments.jsonl existence.")
    print("=" * 60)

    print("\nKết quả đã lưu tại:")
    print(f"- {seg_out_path}  (segment-level)")
    print(f"- {page_out_path}  (page-level json)")
    print(f"- {csv_out_path}   (page-level csv)")


if __name__ == "__main__":
    main()
