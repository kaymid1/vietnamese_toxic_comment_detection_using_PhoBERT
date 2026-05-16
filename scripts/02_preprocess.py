import argparse
import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

SPLITS = ["train", "validation", "test"]


def clean_text(text: str) -> str:
    text = text.strip()

    # 1. Normalize unicode (rất quan trọng với tiếng Việt từ Facebook/Youtube)
    # Ví dụ: "tên" → "tên", "học" → "học"
    text = unicodedata.normalize("NFC", text)  # chuẩn hóa về dạng composed

    # 2. Normalize whitespace (loại bỏ thừa, tab, multiple spaces)
    text = " ".join(text.split())

    # 3. Optional: lowercase → KHÔNG NÊN làm với PhoBERT
    # PhoBERT là case-sensitive và được pretrain trên text gốc tiếng Việt (có chữ hoa đầu câu)
    # → giữ nguyên case để tận dụng subword tốt hơn

    # 4. Không remove punctuation mạnh
    # Giữ lại ! ? . , " ' để bảo toàn sarcasm, cảm xúc (e.g. "đẹp vãi!!!" vs "đẹp")

    # 5. Optional nhẹ: xử lý emoji → giữ nguyên
    # Emoji thường mang ý nghĩa toxic (😂 khi troll, 😡 khi chửi)
    # PhoBERT tokenizer xử lý được emoji

    # 6. Optional: normalize teencode thường gặp (nếu muốn experiment)
    # Chỉ áp dụng nếu error analysis cho thấy nhiều teencode bị miss
    # Ví dụ: thay "k" → "không", "dc" → "được", "ns" → "nói" ...
    # Nhưng giai đoạn đầu → chưa cần, sẽ test ablation sau

    return text


def dedup_key(text: str) -> str:
    return clean_text(text)


def process_file(input_dir: Path, output_dir: Path, split: str) -> Dict[str, int]:
    input_path = input_dir / f"{split}.jsonl"
    output_path = output_dir / f"{split}.jsonl"

    written = 0
    skipped_empty = 0
    unique_keys: Set[str] = set()

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            raw_text = item.get("Comment", "") or item.get("comment", "")  # phòng trường hợp key khác
            toxicity = int(item["Toxicity"])

            text = clean_text(raw_text)
            if not text:  # bỏ comment rỗng sau clean
                skipped_empty += 1
                continue

            record = {
                "text": text,
                "toxicity": toxicity,
                "meta": {
                    "source": "ViCTSD",
                    "original_length": len(raw_text),
                    "processed_length": len(text),
                    # Nếu dataset có thêm topic/constructiveness thì giữ lại ở đây
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            unique_keys.add(dedup_key(text))

    print(f"✅ Processed {split}.jsonl")
    return {
        "rows": written,
        "skipped_empty": skipped_empty,
        "unique_exact_keys": len(unique_keys),
    }


def load_split_keys(output_dir: Path, split: str) -> Set[str]:
    keys: Set[str] = set()
    with (output_dir / f"{split}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = dedup_key(row.get("text", ""))
            if key:
                keys.add(key)
    return keys


def overlap_count(a: Set[str], b: Set[str]) -> int:
    return len(a & b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ViCTSD and run cross-split leakage check")
    parser.add_argument("--input-dir", default="data/raw/victsd")
    parser.add_argument("--output-dir", default="data/processed/victsd_v1")
    parser.add_argument("--leakage-gate", choices=["off", "warn", "fail"], default="warn")
    parser.add_argument("--leakage-report-path", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_stats: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        split_stats[split] = process_file(input_dir=input_dir, output_dir=output_dir, split=split)

    split_keys = {split: load_split_keys(output_dir=output_dir, split=split) for split in SPLITS}
    overlap_exact = {
        "train_validation": overlap_count(split_keys["train"], split_keys["validation"]),
        "train_test": overlap_count(split_keys["train"], split_keys["test"]),
        "validation_test": overlap_count(split_keys["validation"], split_keys["test"]),
    }

    leakage_found = any(v > 0 for v in overlap_exact.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "leakage_gate": args.leakage_gate,
        },
        "splits": split_stats,
        "overlap_exact": overlap_exact,
        "leakage_found": leakage_found,
    }

    report_path = Path(args.leakage_report_path) if args.leakage_report_path else (output_dir / "preprocess_leakage_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if leakage_found and args.leakage_gate == "warn":
        print(f"⚠️ Leakage detected by exact normalized-text overlap: {overlap_exact}")
    elif leakage_found and args.leakage_gate == "fail":
        raise SystemExit(f"Leakage gate failed: {overlap_exact}")

    print(f"✅ Leakage report written to: {report_path}")
    print("✅ Preprocessing victsd_v1 hoàn tất – version này sẽ dùng cho baseline + PhoBERT đầu tiên")


if __name__ == "__main__":
    main()
