import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ["train", "validation", "test"]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = " ".join(text.strip().split())
    return text


def normalize_text_loose(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def load_rows(input_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for split in SPLITS:
        path = input_dir / f"{split}_augmented.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "toxicity" not in obj:
                    if "label" in obj:
                        obj["toxicity"] = obj.pop("label")
                    else:
                        raise ValueError(f"Row missing toxicity/label in {path}")
                obj["toxicity"] = int(obj["toxicity"])
                obj["__source_split"] = split
                rows.append(obj)
    return rows


def build_groups(rows: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = normalize_text(row.get("text", ""))
        if not key:
            continue
        groups[key].append(row)
    return groups


def group_label(group_rows: List[dict]) -> int:
    c = Counter(int(r["toxicity"]) for r in group_rows)
    if len(c) == 1:
        return next(iter(c))
    # Fallback if conflicting labels in one normalized-text group.
    if c[1] == c[0]:
        return 1
    return 1 if c[1] > c[0] else 0


def allocate_groups(
    groups: Dict[str, List[dict]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[dict]]:
    random.seed(seed)

    label_to_items: Dict[int, List[Tuple[str, List[dict]]]] = {0: [], 1: []}
    for key, rows in groups.items():
        g_label = group_label(rows)
        label_to_items[g_label].append((key, rows))

    total_label_counts = {
        label: sum(len(rows) for _, rows in items)
        for label, items in label_to_items.items()
    }

    targets_by_label: Dict[int, Dict[str, int]] = {0: {}, 1: {}}
    for label in [0, 1]:
        total = total_label_counts[label]
        train_target = int(round(total * ratios["train"]))
        val_target = int(round(total * ratios["validation"]))
        test_target = total - train_target - val_target
        targets_by_label[label] = {
            "train": train_target,
            "validation": val_target,
            "test": test_target,
        }

    assigned: Dict[str, List[dict]] = {"train": [], "validation": [], "test": []}

    for label in [0, 1]:
        items = label_to_items[label]
        random.shuffle(items)

        # Place bigger groups first for tighter target fitting.
        items.sort(key=lambda x: len(x[1]), reverse=True)

        running = {"train": 0, "validation": 0, "test": 0}
        target = targets_by_label[label]

        for _, rows in items:
            size = len(rows)
            # Choose split with largest remaining capacity; if all negative, pick smallest overflow.
            candidates = []
            for split in SPLITS:
                remaining = target[split] - running[split]
                overflow_after = running[split] + size - target[split]
                candidates.append((remaining >= size, remaining, -overflow_after, split))

            candidates.sort(reverse=True)
            chosen = candidates[0][3]

            assigned[chosen].extend(rows)
            running[chosen] += size

    random.shuffle(assigned["train"])
    random.shuffle(assigned["validation"])
    random.shuffle(assigned["test"])
    return assigned


def summarize(rows_by_split: Dict[str, List[dict]]) -> dict:
    summary = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        total = len(rows)
        toxic = sum(1 for r in rows if int(r["toxicity"]) == 1)
        clean = total - toxic
        texts = [normalize_text(r.get("text", "")) for r in rows]
        loose_texts = [normalize_text_loose(r.get("text", "")) for r in rows]
        summary[split] = {
            "total": total,
            "clean": clean,
            "toxic": toxic,
            "toxic_ratio": (toxic / total) if total else 0.0,
            "unique_norm_text": len(set(texts)),
            "unique_loose_text": len(set(loose_texts)),
        }

    def overlap_count(extract):
        a = set(extract("train"))
        b = set(extract("validation"))
        c = set(extract("test"))
        return {
            "train_validation": len(a & b),
            "train_test": len(a & c),
            "validation_test": len(b & c),
        }

    norm_overlap = overlap_count(lambda s: [normalize_text(r.get("text", "")) for r in rows_by_split[s]])
    loose_overlap = overlap_count(lambda s: [normalize_text_loose(r.get("text", "")) for r in rows_by_split[s]])

    return {
        "splits": summary,
        "overlap_normalized_text": norm_overlap,
        "overlap_loose_text": loose_overlap,
    }


def write_splits(output_dir: Path, rows_by_split: Dict[str, List[dict]]) -> None:
    for split in SPLITS:
        path = output_dir / f"{split}_augmented.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows_by_split[split]:
                out = {k: v for k, v in row.items() if k != "__source_split"}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe resplit for augmented ViCTSD data")
    parser.add_argument("--input-dir", default="data/victsd")
    parser.add_argument("--output-dir", default="data/victsd")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", default="data/victsd/resplit_report.json")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    all_rows = load_rows(input_dir)
    groups = build_groups(all_rows)

    rows_by_split = allocate_groups(
        groups=groups,
        ratios={
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        seed=args.seed,
    )

    report = summarize(rows_by_split)
    report["config"] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    report["global"] = {
        "total_rows": len(all_rows),
        "unique_normalized_text_groups": len(groups),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if not args.dry_run:
        write_splits(output_dir, rows_by_split)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
