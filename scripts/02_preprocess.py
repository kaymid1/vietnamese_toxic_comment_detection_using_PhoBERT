import argparse
import json
import re
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


def canonical_basic(text: str) -> str:
    return clean_text(text)


def canonical_strong(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = " ".join(text.split())
    return text


def parse_binary_field(item: Dict[str, Any], keys: List[str], field_name: str) -> int:
    found = False
    value: Any = None
    for key in keys:
        if key in item:
            value = item.get(key)
            found = True
            break
    if not found:
        raise ValueError(f"Missing required field '{field_name}' (tried keys: {keys})")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{field_name}': {value}") from exc
    if parsed not in (0, 1):
        raise ValueError(f"'{field_name}' must be 0/1, got: {parsed}")
    return parsed


def load_split_records(input_dir: Path, split: str) -> Dict[str, Any]:
    input_path = input_dir / f"{split}.jsonl"
    rows: List[Dict[str, Any]] = []

    input_rows = 0
    kept_rows = 0
    dropped_empty = 0
    dropped_duplicate_within_split = 0

    seen_basic: Set[str] = set()

    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            input_rows += 1
            item = json.loads(line)

            raw_text = item.get("Comment", "") or item.get("comment", "") or item.get("text", "")
            raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)
            text = clean_text(raw_text)
            if not text:
                dropped_empty += 1
                continue

            basic_key = canonical_basic(text)
            strong_key = canonical_strong(text)
            if basic_key in seen_basic:
                dropped_duplicate_within_split += 1
                continue
            seen_basic.add(basic_key)

            try:
                toxicity = parse_binary_field(item, ["Toxicity", "toxicity", "label"], "toxicity")
                constructiveness = parse_binary_field(item, ["Constructiveness", "constructiveness"], "constructiveness")
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_no}: {exc}") from exc

            topic = item.get("Topic") or item.get("topic") or item.get("Domain") or item.get("domain")
            title = item.get("Title") or item.get("title")

            record = {
                "text": text,
                "toxicity": toxicity,
                "label": toxicity,
                "constructiveness": constructiveness,
                "meta": {
                    "source": "victsd",
                    "split": split,
                    "original_length": len(raw_text),
                    "processed_length": len(text),
                    "original_comment": raw_text,
                },
            }
            if isinstance(topic, str) and topic.strip():
                record["meta"]["topic"] = topic.strip()
            if isinstance(title, str) and title.strip():
                record["meta"]["title"] = title.strip()

            rows.append({
                "record": record,
                "basic_key": basic_key,
                "strong_key": strong_key,
            })
            kept_rows += 1

    return {
        "rows": rows,
        "stats": {
            "input_rows": input_rows,
            "kept_before_cross_split": kept_rows,
            "dropped_empty": dropped_empty,
            "dropped_duplicate_within_split": dropped_duplicate_within_split,
        },
    }


def apply_cross_split_dedup(records_by_split: Dict[str, List[Dict[str, Any]]], mode: str) -> Dict[str, int]:
    dropped = {split: 0 for split in SPLITS}
    if mode == "off":
        return dropped

    seen_basic: Set[str] = set()
    seen_strong: Set[str] = set()

    for split in SPLITS:
        kept: List[Dict[str, Any]] = []
        for item in records_by_split[split]:
            basic_key = item["basic_key"]
            strong_key = item["strong_key"]
            overlap = basic_key in seen_basic or (mode == "strong" and strong_key in seen_strong)
            if overlap:
                dropped[split] += 1
                continue
            kept.append(item)
            seen_basic.add(basic_key)
            seen_strong.add(strong_key)
        records_by_split[split] = kept

    return dropped


def write_split(output_dir: Path, split: str, rows: List[Dict[str, Any]]) -> int:
    output_path = output_dir / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as fout:
        for item in rows:
            fout.write(json.dumps(item["record"], ensure_ascii=False) + "\n")
    return len(rows)


def load_keys(output_dir: Path, split: str) -> Dict[str, Set[str]]:
    basic: Set[str] = set()
    strong: Set[str] = set()
    with (output_dir / f"{split}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                basic.add(canonical_basic(text))
                strong.add(canonical_strong(text))
    return {"basic": basic, "strong": strong}


def overlap_count(a: Set[str], b: Set[str]) -> int:
    return len(a & b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ViCTSD and run cross-split leakage check")
    parser.add_argument("--input-dir", default="data/raw/victsd")
    parser.add_argument("--output-dir", default="data/processed/victsd_gold")
    parser.add_argument("--cross-split-dedup", choices=["off", "basic", "strong"], default="strong")
    parser.add_argument("--leakage-gate", choices=["off", "warn", "fail"], default="warn")
    parser.add_argument("--leakage-report-path", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, Dict[str, Any]] = {}
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        loaded[split] = load_split_records(input_dir=input_dir, split=split)
        records_by_split[split] = loaded[split]["rows"]

    cross_split_dropped = apply_cross_split_dedup(records_by_split=records_by_split, mode=args.cross_split_dedup)

    split_stats: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        written = write_split(output_dir=output_dir, split=split, rows=records_by_split[split])
        base = loaded[split]["stats"]
        split_stats[split] = {
            "input_rows": base["input_rows"],
            "rows": written,
            "dropped_empty": base["dropped_empty"],
            "dropped_duplicate_within_split": base["dropped_duplicate_within_split"],
            "dropped_overlap_previous_splits": cross_split_dropped[split],
        }
        print(f"Processed {split}.jsonl: rows={written}")

    split_keys = {split: load_keys(output_dir=output_dir, split=split) for split in SPLITS}
    overlap_exact_basic = {
        "train_validation": overlap_count(split_keys["train"]["basic"], split_keys["validation"]["basic"]),
        "train_test": overlap_count(split_keys["train"]["basic"], split_keys["test"]["basic"]),
        "validation_test": overlap_count(split_keys["validation"]["basic"], split_keys["test"]["basic"]),
    }
    overlap_exact_strong = {
        "train_validation": overlap_count(split_keys["train"]["strong"], split_keys["validation"]["strong"]),
        "train_test": overlap_count(split_keys["train"]["strong"], split_keys["test"]["strong"]),
        "validation_test": overlap_count(split_keys["validation"]["strong"], split_keys["test"]["strong"]),
    }

    leakage_found = any(v > 0 for v in overlap_exact_basic.values()) or any(v > 0 for v in overlap_exact_strong.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "cross_split_dedup": args.cross_split_dedup,
            "leakage_gate": args.leakage_gate,
            "text_field_priority": ["Comment", "comment", "text"],
            "toxicity_field_priority": ["Toxicity", "toxicity", "label"],
            "constructiveness_field_priority": ["Constructiveness", "constructiveness"],
        },
        "splits": split_stats,
        "overlap_exact_basic": overlap_exact_basic,
        "overlap_exact_strong": overlap_exact_strong,
        "leakage_found": leakage_found,
    }

    report_path = Path(args.leakage_report_path) if args.leakage_report_path else (output_dir / "preprocess_leakage_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if leakage_found and args.leakage_gate == "warn":
        print(
            "Warning: leakage detected by overlap checks: "
            f"basic={overlap_exact_basic}, strong={overlap_exact_strong}"
        )
    elif leakage_found and args.leakage_gate == "fail":
        raise SystemExit(
            "Leakage gate failed: "
            f"basic={overlap_exact_basic}, strong={overlap_exact_strong}"
        )

    print(f"Leakage report written to: {report_path}")
    print("Preprocessing victsd_gold completed.")


if __name__ == "__main__":
    main()
