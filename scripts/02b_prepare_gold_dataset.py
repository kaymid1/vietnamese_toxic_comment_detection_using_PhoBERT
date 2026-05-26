import json
import os
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


def load_split_rows(input_path: Path, split: str) -> Dict[str, Any]:
    input_rows = 0
    dropped_empty = 0
    dropped_duplicate_within_split = 0
    toxic_count = 0
    constructive_count = 0

    seen_basic: Set[str] = set()
    rows: List[Dict[str, Any]] = []

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

            if toxicity == 1:
                toxic_count += 1
            if constructiveness == 1:
                constructive_count += 1

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

    return {
        "rows": rows,
        "stats": {
            "input_rows": input_rows,
            "dropped_empty": dropped_empty,
            "dropped_duplicate_within_split": dropped_duplicate_within_split,
            "toxicity_count_before_cross_split": toxic_count,
            "constructiveness_count_before_cross_split": constructive_count,
        },
    }


def apply_cross_split_dedup(rows_by_split: Dict[str, List[Dict[str, Any]]], mode: str) -> Dict[str, int]:
    dropped = {split: 0 for split in SPLITS}
    if mode == "off":
        return dropped

    seen_basic: Set[str] = set()
    seen_strong: Set[str] = set()

    for split in SPLITS:
        kept: List[Dict[str, Any]] = []
        for row in rows_by_split[split]:
            basic_key = row["basic_key"]
            strong_key = row["strong_key"]
            overlap = basic_key in seen_basic or (mode == "strong" and strong_key in seen_strong)
            if overlap:
                dropped[split] += 1
                continue
            kept.append(row)
            seen_basic.add(basic_key)
            seen_strong.add(strong_key)
        rows_by_split[split] = kept

    return dropped


def write_split(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row["record"], ensure_ascii=False) + "\n")


def count_label(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row["record"].get(key) == 1)


def print_summary(summary: Dict[str, Any], overlaps: Dict[str, Dict[str, int]]) -> None:
    print("=" * 72)
    print("ViCTSD Gold Dataset Build Summary")
    print("=" * 72)

    for split in SPLITS:
        stats = summary[split]
        print(f"[{split}]")
        print(f"  input_rows                     : {stats['input_rows']}")
        print(f"  output_rows                    : {stats['output_rows']}")
        print(f"  dropped_empty                  : {stats['dropped_empty']}")
        print(f"  dropped_duplicate_within_split : {stats['dropped_duplicate_within_split']}")
        print(f"  dropped_overlap_previous_split : {stats['dropped_overlap_previous_splits']}")
        print(f"  toxicity_ratio                 : {stats['toxicity_ratio']:.6f} ({stats['toxicity_count']}/{stats['output_rows']})")
        print(
            f"  constructiveness_ratio         : {stats['constructiveness_ratio']:.6f} "
            f"({stats['constructiveness_count']}/{stats['output_rows']})"
        )
        print()

    print("[cross-split overlap by normalized text]")
    print(f"  basic  train_validation: {overlaps['basic']['train_validation']}")
    print(f"  basic  train_test      : {overlaps['basic']['train_test']}")
    print(f"  basic  validation_test : {overlaps['basic']['validation_test']}")
    print(f"  strong train_validation: {overlaps['strong']['train_validation']}")
    print(f"  strong train_test      : {overlaps['strong']['train_test']}")
    print(f"  strong validation_test : {overlaps['strong']['validation_test']}")
    print("=" * 72)


def collect_keys(output_dir: Path, split: str) -> Dict[str, Set[str]]:
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


def main() -> None:
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw/victsd"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "data/processed/victsd_gold"))
    cross_split_dedup = os.getenv("CROSS_SPLIT_DEDUP", "strong").strip().lower()
    if cross_split_dedup not in {"off", "basic", "strong"}:
        raise ValueError("CROSS_SPLIT_DEDUP must be one of: off, basic, strong")

    if output_dir.as_posix().rstrip("/") in {"data/processed/victsd_v1", "data/victsd"}:
        raise SystemExit("Refusing to write to a legacy dataset path. Use OUTPUT_DIR for victsd_gold.")

    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, Dict[str, Any]] = {}
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split in SPLITS:
        input_path = raw_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        loaded[split] = load_split_rows(input_path=input_path, split=split)
        rows_by_split[split] = loaded[split]["rows"]

    dropped_cross = apply_cross_split_dedup(rows_by_split=rows_by_split, mode=cross_split_dedup)

    split_stats: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        output_path = output_dir / f"{split}.jsonl"
        rows = rows_by_split[split]
        write_split(output_path=output_path, rows=rows)

        out_rows = len(rows)
        tox_count = count_label(rows, "toxicity")
        cons_count = count_label(rows, "constructiveness")

        split_stats[split] = {
            "input_rows": loaded[split]["stats"]["input_rows"],
            "output_rows": out_rows,
            "dropped_empty": loaded[split]["stats"]["dropped_empty"],
            "dropped_duplicate_within_split": loaded[split]["stats"]["dropped_duplicate_within_split"],
            "dropped_overlap_previous_splits": dropped_cross[split],
            "toxicity_count": tox_count,
            "toxicity_ratio": (tox_count / out_rows) if out_rows else 0.0,
            "constructiveness_count": cons_count,
            "constructiveness_ratio": (cons_count / out_rows) if out_rows else 0.0,
        }

    keys = {split: collect_keys(output_dir=output_dir, split=split) for split in SPLITS}
    overlaps = {
        "basic": {
            "train_validation": len(keys["train"]["basic"] & keys["validation"]["basic"]),
            "train_test": len(keys["train"]["basic"] & keys["test"]["basic"]),
            "validation_test": len(keys["validation"]["basic"] & keys["test"]["basic"]),
        },
        "strong": {
            "train_validation": len(keys["train"]["strong"] & keys["validation"]["strong"]),
            "train_test": len(keys["train"]["strong"] & keys["test"]["strong"]),
            "validation_test": len(keys["validation"]["strong"] & keys["test"]["strong"]),
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data_raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "cross_split_dedup": cross_split_dedup,
            "splits": SPLITS,
            "rules": {
                "text_field_priority": ["Comment", "comment", "text"],
                "toxicity_field_priority": ["Toxicity", "toxicity", "label"],
                "constructiveness_field_priority": ["Constructiveness", "constructiveness"],
                "strip": True,
                "unicode_normalization": "NFC",
                "whitespace_normalization": True,
                "preserve_case": True,
                "preserve_punctuation": True,
                "drop_empty_after_clean": True,
                "dedup_within_split": "basic",
                "dedup_cross_split": cross_split_dedup,
                "dedup_priority": SPLITS,
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
