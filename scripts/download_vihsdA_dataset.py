#!/usr/bin/env python3
"""
Download ura-hcmut/UIT-ViHSD dataset from Hugging Face
and save each split (train/dev/test) as JSONL files.

Dataset: Vietnamese Hate Speech Detection (ViHSD)
Labels: 0=CLEAN, 1=OFFENSIVE, 2=HATE

Usage:
    pip install datasets
    python download_vihsd.py [--output-dir ./vihsd_data] [--token YOUR_HF_TOKEN]
"""

import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL_MAP = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
DATASET_ID = "ura-hcmut/UIT-ViHSD"


def detect_columns(first_row: dict) -> tuple[str, str]:
    """
    Auto-detect column names for text and label from the first row.
    Returns (text_col, label_col).
    """
    keys = list(first_row.keys())
    log.info(f"  Dataset columns: {keys}")

    # Detect text column
    text_col = next(
        (k for k in keys if k in ("free_text", "text", "sentence", "comment", "content")),
        keys[0],  # fallback: first column
    )

    # Detect label column — prefer 'label_id' over 'label'
    label_col = next(
        (k for k in keys if k == "label_id"),
        next(
            (k for k in keys if "label" in k.lower() or "class" in k.lower()),
            keys[-1],  # fallback: last column
        ),
    )

    log.info(f"  Using → text_col='{text_col}', label_col='{label_col}'")
    return text_col, label_col


def save_jsonl(dataset_split, output_path: Path, split_name: str) -> int:
    """Write a dataset split to a .jsonl file. Returns number of records written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect columns from the first row
    first_row = dataset_split[0]
    text_col, label_col = detect_columns(first_row)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset_split:
            raw_label = row.get(label_col)

            # raw_label may be int or string — normalise to int for lookup
            try:
                label_int = int(raw_label) if raw_label is not None else None
            except (ValueError, TypeError):
                label_int = None

            record = {
                "free_text": row.get(text_col, ""),
                "label_id": label_int,
                "label": LABEL_MAP.get(label_int, "UNKNOWN"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    log.info(f"  ✔ [{split_name}] {count:,} records → {output_path}")
    return count


def download(output_dir: str, hf_token: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "❌ Thiếu thư viện 'datasets'. Chạy: pip install datasets"
        )

    log.info(f"Đang tải dataset: {DATASET_ID}")

    load_kwargs = dict(path=DATASET_ID, trust_remote_code=True)
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = load_dataset(**load_kwargs)

    log.info(f"Splits có sẵn: {list(ds.keys())}")

    out = Path(output_dir)
    total = 0

    for split_name, split_data in ds.items():
        output_file = out / f"{split_name}.jsonl"
        total += save_jsonl(split_data, output_file, split_name)

    # Write metadata
    meta = {
        "dataset": DATASET_ID,
        "label_map": LABEL_MAP,
        "splits": {
            split: {"records": len(ds[split]), "file": f"{split}.jsonl"}
            for split in ds.keys()
        },
        "total_records": total,
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"  ✔ Metadata → {meta_path}")

    log.info(f"\n✅ Hoàn tất! Tổng cộng {total:,} records → thư mục: {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Download UIT-ViHSD → JSONL")
    parser.add_argument(
        "--output-dir", default="./vihsd_data",
        help="Thư mục lưu file JSONL (default: ./vihsd_data)"
    )
    parser.add_argument(
        "--token", default=None,
        help="Hugging Face token (nếu dataset yêu cầu xác thực)"
    )
    args = parser.parse_args()

    download(output_dir=args.output_dir, hf_token=args.token)


if __name__ == "__main__":
    main()