import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

SPLITS = ["train", "validation", "test"]


def clean_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def basic_key(text: str) -> str:
    return clean_text(text)


def strong_key(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = " ".join(text.split())
    return text


def validate_split(path: Path, split: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    rows = 0
    missing_text = 0
    missing_toxicity = 0
    missing_label = 0
    missing_constructiveness = 0
    invalid_toxicity = 0
    invalid_label = 0
    invalid_constructiveness = 0
    mismatch_label_toxicity = 0
    toxicity_dist: Counter[int] = Counter()
    constructiveness_dist: Counter[int] = Counter()
    label_dist: Counter[int] = Counter()

    basic_seen: Set[str] = set()
    strong_seen: Set[str] = set()
    dup_basic = 0
    dup_strong = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1

            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                missing_text += 1
            else:
                bk = basic_key(text)
                sk = strong_key(text)
                if bk in basic_seen:
                    dup_basic += 1
                else:
                    basic_seen.add(bk)
                if sk in strong_seen:
                    dup_strong += 1
                else:
                    strong_seen.add(sk)

            toxicity = row.get("toxicity")
            label = row.get("label")
            constructiveness = row.get("constructiveness")

            if toxicity is None:
                missing_toxicity += 1
            if label is None:
                missing_label += 1
            if constructiveness is None:
                missing_constructiveness += 1

            if toxicity in (0, 1):
                toxicity_dist[int(toxicity)] += 1
            else:
                invalid_toxicity += 1

            if label in (0, 1):
                label_dist[int(label)] += 1
            else:
                invalid_label += 1

            if constructiveness in (0, 1):
                constructiveness_dist[int(constructiveness)] += 1
            else:
                invalid_constructiveness += 1

            if toxicity in (0, 1) and label in (0, 1) and int(toxicity) != int(label):
                mismatch_label_toxicity += 1

            if rows == 1 and split not in ("train", "validation", "test"):
                raise RuntimeError(f"Unexpected split token at {path}:{line_no}")

    return {
        "rows": rows,
        "missing_text": missing_text,
        "missing_toxicity": missing_toxicity,
        "missing_label": missing_label,
        "missing_constructiveness": missing_constructiveness,
        "invalid_toxicity": invalid_toxicity,
        "invalid_label": invalid_label,
        "invalid_constructiveness": invalid_constructiveness,
        "mismatch_label_toxicity": mismatch_label_toxicity,
        "toxicity_distribution": dict(toxicity_dist),
        "label_distribution": dict(label_dist),
        "constructiveness_distribution": dict(constructiveness_dist),
        "dup_basic": dup_basic,
        "dup_strong": dup_strong,
        "basic_keys": basic_seen,
        "strong_keys": strong_seen,
    }


def sample_records(path: Path, n: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def overlap_count(a: Set[str], b: Set[str]) -> int:
    return len(a & b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed ViCTSD gold JSONL splits")
    parser.add_argument("--data-dir", default="data/processed/victsd_gold")
    parser.add_argument("--sample-n", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results: Dict[str, Dict[str, Any]] = {}

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        results[split] = validate_split(path, split)

    basic_overlap = {
        "train_validation": overlap_count(results["train"]["basic_keys"], results["validation"]["basic_keys"]),
        "train_test": overlap_count(results["train"]["basic_keys"], results["test"]["basic_keys"]),
        "validation_test": overlap_count(results["validation"]["basic_keys"], results["test"]["basic_keys"]),
    }
    strong_overlap = {
        "train_validation": overlap_count(results["train"]["strong_keys"], results["validation"]["strong_keys"]),
        "train_test": overlap_count(results["train"]["strong_keys"], results["test"]["strong_keys"]),
        "validation_test": overlap_count(results["validation"]["strong_keys"], results["test"]["strong_keys"]),
    }

    print("Validation summary")
    for split in SPLITS:
        stats = results[split]
        print(
            f"{split}: rows={stats['rows']}, "
            f"toxicity={stats['toxicity_distribution']}, "
            f"constructiveness={stats['constructiveness_distribution']}, "
            f"label={stats['label_distribution']}, "
            f"missing_text={stats['missing_text']}, "
            f"missing_toxicity={stats['missing_toxicity']}, "
            f"missing_label={stats['missing_label']}, "
            f"missing_constructiveness={stats['missing_constructiveness']}, "
            f"invalid_toxicity={stats['invalid_toxicity']}, "
            f"invalid_label={stats['invalid_label']}, "
            f"invalid_constructiveness={stats['invalid_constructiveness']}, "
            f"mismatch_label_toxicity={stats['mismatch_label_toxicity']}, "
            f"dup_basic={stats['dup_basic']}, dup_strong={stats['dup_strong']}"
        )

    print(
        "Cross-split overlap: "
        f"basic={basic_overlap}, strong={strong_overlap}"
    )

    samples = sample_records(data_dir / "train.jsonl", args.sample_n)
    print(f"Sample records ({len(samples)} from train)")
    for row in samples:
        print(json.dumps(row, ensure_ascii=True))

    has_error = False
    for split in SPLITS:
        stats = results[split]
        if (
            stats["missing_text"] > 0
            or stats["missing_toxicity"] > 0
            or stats["missing_label"] > 0
            or stats["missing_constructiveness"] > 0
            or stats["invalid_toxicity"] > 0
            or stats["invalid_label"] > 0
            or stats["invalid_constructiveness"] > 0
            or stats["mismatch_label_toxicity"] > 0
            or stats["dup_basic"] > 0
            or stats["dup_strong"] > 0
        ):
            has_error = True
            break
    if any(v > 0 for v in basic_overlap.values()) or any(v > 0 for v in strong_overlap.values()):
        has_error = True

    if has_error:
        raise SystemExit("Validation failed: missing/invalid fields or overlap/duplicate detected.")

    print("Validation passed.")


if __name__ == "__main__":
    main()
