import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

SPLITS = ["train", "validation", "test"]


def clean_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def parse_toxicity(value: Any) -> int:
    toxicity = int(value)
    if toxicity not in (0, 1):
        raise ValueError(f"Toxicity must be 0/1, got: {value}")
    return toxicity


def process_split(input_path: Path, output_path: Path) -> Dict[str, Any]:
    input_rows = 0
    output_rows = 0
    dropped_empty = 0
    dropped_duplicate = 0
    toxic_count = 0

    seen_texts: Set[str] = set()
    output_texts: Set[str] = set()

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            input_rows += 1
            item = json.loads(line)

            raw_text = item.get("Comment", "")
            text = clean_text(raw_text if isinstance(raw_text, str) else str(raw_text))

            if not text:
                dropped_empty += 1
                continue

            if text in seen_texts:
                dropped_duplicate += 1
                continue

            seen_texts.add(text)
            output_texts.add(text)

            toxicity = parse_toxicity(item.get("Toxicity"))
            if toxicity == 1:
                toxic_count += 1

            record = {
                "text": text,
                "toxicity": toxicity,
                "meta": {
                    "source": "ViCTSD",
                    "original_length": len(raw_text if isinstance(raw_text, str) else str(raw_text)),
                    "processed_length": len(text),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_rows += 1

    toxic_ratio = (toxic_count / output_rows) if output_rows else 0.0
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "dropped": {
            "empty": dropped_empty,
            "duplicate": dropped_duplicate,
            "total": dropped_empty + dropped_duplicate,
        },
        "toxic": {
            "count": toxic_count,
            "ratio": toxic_ratio,
        },
        "normalized_texts": output_texts,
    }


def print_summary(summary: Dict[str, Any], overlaps: Dict[str, int]) -> None:
    print("=" * 72)
    print("ViCTSD Gold Dataset Build Summary")
    print("=" * 72)

    for split in SPLITS:
        stats = summary[split]
        print(f"[{split}]")
        print(f"  input_rows       : {stats['input_rows']}")
        print(f"  output_rows      : {stats['output_rows']}")
        print(f"  dropped_empty    : {stats['dropped']['empty']}")
        print(f"  dropped_duplicate: {stats['dropped']['duplicate']}")
        print(f"  dropped_total    : {stats['dropped']['total']}")
        print(f"  toxic_ratio      : {stats['toxic']['ratio']:.6f} ({stats['toxic']['count']}/{stats['output_rows']})")
        print()

    print("[cross-split overlap by normalized text]")
    print(f"  train_validation: {overlaps['train_validation']}")
    print(f"  train_test      : {overlaps['train_test']}")
    print(f"  validation_test : {overlaps['validation_test']}")
    print("=" * 72)


def main() -> None:
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw/victsd"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "data/processed/victsd_gold"))

    if output_dir.as_posix().rstrip("/") == "data/processed/victsd_v1":
        raise SystemExit("Refusing to write to data/processed/victsd_v1. Use OUTPUT_DIR for victsd_gold.")

    output_dir.mkdir(parents=True, exist_ok=True)

    split_stats: Dict[str, Dict[str, Any]] = {}
    split_texts: Dict[str, Set[str]] = {}

    for split in SPLITS:
        input_path = raw_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"

        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")

        stats = process_split(input_path=input_path, output_path=output_path)
        split_stats[split] = {
            "input_rows": stats["input_rows"],
            "output_rows": stats["output_rows"],
            "dropped": stats["dropped"],
            "toxic": stats["toxic"],
        }
        split_texts[split] = stats["normalized_texts"]

    overlaps = {
        "train_validation": len(split_texts["train"] & split_texts["validation"]),
        "train_test": len(split_texts["train"] & split_texts["test"]),
        "validation_test": len(split_texts["validation"] & split_texts["test"]),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data_raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "splits": SPLITS,
            "rules": {
                "text_field": "Comment",
                "toxicity_field": "Toxicity",
                "strip": True,
                "unicode_normalization": "NFC",
                "whitespace_normalization": True,
                "preserve_case": True,
                "preserve_punctuation": True,
                "preserve_emoji": True,
                "preserve_toxic_keywords": True,
                "drop_empty_after_clean": True,
                "dedup_exact_after_normalization": True,
                "dedup_scope": "within_split_keep_first",
            },
        },
        "splits": split_stats,
        "overlap_normalized_text": overlaps,
    }

    report_path = output_dir / "build_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_summary(split_stats, overlaps)
    print(f"Build report saved to: {report_path}")


if __name__ == "__main__":
    main()
