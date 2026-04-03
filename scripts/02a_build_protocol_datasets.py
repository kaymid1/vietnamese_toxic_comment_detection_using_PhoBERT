import argparse
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ["train", "validation", "test"]


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def dedup_key(text: str) -> str:
    return clean_text(text)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_toxicity(row: dict) -> int:
    if "toxicity" in row:
        return int(row["toxicity"])
    if "label" in row:
        return int(row["label"])
    raise ValueError("Row missing toxicity/label field")


def normalize_victsd_row(row: dict, split: str) -> dict | None:
    text = clean_text(row.get("text", ""))
    if not text:
        return None
    toxicity = ensure_toxicity(row)
    meta = dict(row.get("meta") or {})
    meta.setdefault("source", "ViCTSD")
    meta["source_split"] = split
    return {"text": text, "toxicity": toxicity, "meta": meta}


def normalize_vihsd_offensive_rows(vihsd_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for split in SPLITS:
        path = vihsd_dir / f"{split}.jsonl"
        for row in read_jsonl(path):
            label_id = row.get("label_id")
            label = str(row.get("label", "")).upper()
            is_offensive = (label_id == 1) or (label == "OFFENSIVE")
            if not is_offensive:
                continue
            text = clean_text(row.get("free_text", ""))
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "toxicity": 1,
                    "meta": {
                        "source": "UIT-ViHSD",
                        "source_split": split,
                        "label_id": 1,
                        "label": "OFFENSIVE",
                        "mapping": "label_id==1 OR label==OFFENSIVE -> toxicity=1",
                    },
                }
            )
    return rows


def dedup_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for row in rows:
        key = dedup_key(row["text"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def overlap_count(a: List[dict], b: List[dict]) -> int:
    sa = {dedup_key(x["text"]) for x in a}
    sb = {dedup_key(x["text"]) for x in b}
    return len(sa & sb)


def split_stats(rows: List[dict]) -> dict:
    total = len(rows)
    toxic = sum(1 for r in rows if int(r["toxicity"]) == 1)
    source_counter = Counter((r.get("meta") or {}).get("source", "<none>") for r in rows)
    source_split_counter = Counter(
        f"{(r.get('meta') or {}).get('source','<none>')}:{(r.get('meta') or {}).get('source_split','<none>')}"
        for r in rows
    )
    return {
        "total": total,
        "toxic": toxic,
        "clean": total - toxic,
        "toxic_ratio": (toxic / total) if total else 0.0,
        "sources": dict(source_counter),
        "source_splits": dict(source_split_counter),
    }


def stratified_split(rows: List[dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row["toxicity"])].append(row)

    rng = random.Random(seed)
    out = {"train": [], "validation": [], "test": []}

    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        out["train"].extend(label_rows[:n_train])
        out["validation"].extend(label_rows[n_train:n_train + n_val])
        out["test"].extend(label_rows[n_train + n_val:n_train + n_val + n_test])

    for split in SPLITS:
        rng.shuffle(out[split])
    return out


def compute_required_additions(base_total: int, base_toxic: int, target_min: float, target_max: float) -> Tuple[int, int]:
    # For adding toxic-only samples x:
    # (base_toxic + x)/(base_total + x) in [target_min, target_max]
    lower = (target_min * base_total - base_toxic) / (1.0 - target_min)
    upper = (target_max * base_total - base_toxic) / (1.0 - target_max)
    min_x = max(0, math.ceil(lower))
    max_x = max(0, math.floor(upper))
    return min_x, max_x


def protocol_paths(output_root: Path, dataset_prefix: str, protocol: str) -> Dict[str, Path]:
    base = output_root / f"protocol_{protocol}"
    return {
        "train": base / f"{dataset_prefix}_protocol_{protocol}_train_augmented.jsonl",
        "validation": base / f"{dataset_prefix}_protocol_{protocol}_validation_augmented.jsonl",
        "test": base / f"{dataset_prefix}_protocol_{protocol}_test_augmented.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protocol A/B/C datasets from ViCTSD + ViHSD raw")
    parser.add_argument("--victsd-dir", default="data/processed/victsd_v1")
    parser.add_argument("--vihsd-dir", default="data/raw/vihsd")
    parser.add_argument("--output-root", default="data/victsd")
    parser.add_argument("--dataset-prefix", default="victsd_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b-toxic-min", type=float, default=0.30)
    parser.add_argument("--b-toxic-max", type=float, default=0.40)
    parser.add_argument("--c-train-ratio", type=float, default=0.7)
    parser.add_argument("--c-validation-ratio", type=float, default=0.2)
    parser.add_argument("--c-test-ratio", type=float, default=0.1)
    parser.add_argument("--protocols", default="a,b,c", help="Comma-separated list of protocols to build: a,b,c")
    parser.add_argument("--protocol-a-mode", choices=["legacy", "leakage_safe"], default="legacy")
    args = parser.parse_args()

    if abs(args.c_train_ratio + args.c_validation_ratio + args.c_test_ratio - 1.0) > 1e-8:
        raise ValueError("C split ratios must sum to 1.0")

    selected_protocols = {
        p.strip().lower()
        for p in args.protocols.split(",")
        if p.strip()
    }
    valid_protocols = {"a", "b", "c"}
    invalid_protocols = selected_protocols - valid_protocols
    if not selected_protocols:
        raise ValueError("--protocols cannot be empty")
    if invalid_protocols:
        raise ValueError(f"Invalid protocol ids in --protocols: {sorted(invalid_protocols)}")

    victsd_dir = Path(args.victsd_dir)
    vihsd_dir = Path(args.vihsd_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{args.dataset_prefix}_protocol_build_report.json"

    existing_report = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            existing_report = json.load(f)

    existing_protocols = existing_report.get("protocols") if isinstance(existing_report, dict) else {}
    existing_warnings = existing_report.get("warnings") if isinstance(existing_report, dict) else []

    # Load + normalize ViCTSD
    victsd: Dict[str, List[dict]] = {}
    for split in SPLITS:
        path = victsd_dir / f"{split}.jsonl"
        raw_rows = read_jsonl(path)
        norm_rows = []
        for row in raw_rows:
            nrow = normalize_victsd_row(row, split)
            if nrow is not None:
                norm_rows.append(nrow)
        victsd[split] = norm_rows

    # Load + preprocess ViHSD offensive only when needed
    vihsd_offensive: List[dict] = []
    if "b" in selected_protocols or "c" in selected_protocols:
        vihsd_offensive = dedup_rows(normalize_vihsd_offensive_rows(vihsd_dir))

    report = {
        "config": {
            "victsd_dir": str(victsd_dir),
            "vihsd_dir": str(vihsd_dir),
            "output_root": str(output_root),
            "dataset_prefix": args.dataset_prefix,
            "seed": args.seed,
            "b_toxic_min": args.b_toxic_min,
            "b_toxic_max": args.b_toxic_max,
            "c_ratios": {
                "train": args.c_train_ratio,
                "validation": args.c_validation_ratio,
                "test": args.c_test_ratio,
            },
            "protocols": sorted(selected_protocols),
            "protocol_a_mode": args.protocol_a_mode,
        },
        "warnings": list(existing_warnings) if isinstance(existing_warnings, list) else [],
        "protocols": {},
    }

    if selected_protocols != valid_protocols:
        report["warnings"].append(
            f"Partial rebuild executed for protocols={sorted(selected_protocols)}; untouched protocol report entries were preserved from existing report."
        )

    for protocol_id in sorted(valid_protocols - selected_protocols):
        if isinstance(existing_protocols, dict) and isinstance(existing_protocols.get(protocol_id), dict):
            report["protocols"][protocol_id] = existing_protocols[protocol_id]

    # ----------------
    # Protocol A
    # ----------------
    if "a" in selected_protocols:
        protocol_a_rows = {split: list(victsd[split]) for split in SPLITS}
        removed_train_rows = 0
        removed_train_unique_keys = 0
        removed_validation_rows = 0
        removed_validation_unique_keys = 0

        if args.protocol_a_mode == "leakage_safe":
            original_validation = protocol_a_rows["validation"]
            original_validation_keys = {dedup_key(x["text"]) for x in original_validation}
            test_keys = {dedup_key(x["text"]) for x in protocol_a_rows["test"]}
            filtered_validation = [r for r in original_validation if dedup_key(r["text"]) not in test_keys]
            filtered_validation_keys = {dedup_key(x["text"]) for x in filtered_validation}
            removed_validation_rows = len(original_validation) - len(filtered_validation)
            removed_validation_unique_keys = len(original_validation_keys - filtered_validation_keys)
            protocol_a_rows["validation"] = filtered_validation

            train_block_keys = test_keys | filtered_validation_keys
            original_train = protocol_a_rows["train"]
            original_train_keys = {dedup_key(x["text"]) for x in original_train}
            filtered_train = [r for r in original_train if dedup_key(r["text"]) not in train_block_keys]
            filtered_train_keys = {dedup_key(x["text"]) for x in filtered_train}
            removed_train_rows = len(original_train) - len(filtered_train)
            removed_train_unique_keys = len(original_train_keys - filtered_train_keys)
            protocol_a_rows["train"] = filtered_train

        a_paths = protocol_paths(output_root, args.dataset_prefix, "a")
        for split in SPLITS:
            write_jsonl(a_paths[split], protocol_a_rows[split])

        report["protocols"]["a"] = {
            "mode": args.protocol_a_mode,
            "leakage_filter": {
                "removed_train_rows": removed_train_rows,
                "removed_train_unique_keys": removed_train_unique_keys,
                "removed_validation_rows": removed_validation_rows,
                "removed_validation_unique_keys": removed_validation_unique_keys,
            },
            "files": {k: str(v) for k, v in a_paths.items()},
            "stats": {split: split_stats(protocol_a_rows[split]) for split in SPLITS},
            "overlap_exact": {
                "train_validation": overlap_count(protocol_a_rows["train"], protocol_a_rows["validation"]),
                "train_test": overlap_count(protocol_a_rows["train"], protocol_a_rows["test"]),
                "validation_test": overlap_count(protocol_a_rows["validation"], protocol_a_rows["test"]),
            },
        }

    # ----------------
    # Protocol B
    # ----------------
    if "b" in selected_protocols:
        b_train_base = list(victsd["train"])
        b_val = list(victsd["validation"])
        b_test = list(victsd["test"])

        val_test_keys = {dedup_key(x["text"]) for x in b_val + b_test}
        train_base_keys = {dedup_key(x["text"]) for x in b_train_base}

        candidate_pool = [
            r for r in vihsd_offensive
            if dedup_key(r["text"]) not in val_test_keys and dedup_key(r["text"]) not in train_base_keys
        ]

        base_total = len(b_train_base)
        base_toxic = sum(1 for r in b_train_base if int(r["toxicity"]) == 1)
        min_add, max_add = compute_required_additions(base_total, base_toxic, args.b_toxic_min, args.b_toxic_max)

        if min_add > max_add and max_add > 0:
            report["warnings"].append(
                f"Protocol B target window infeasible from base distribution: min_add={min_add}, max_add={max_add}. Using max_add."
            )

        desired_add = min_add
        if max_add > 0:
            desired_add = min(desired_add, max_add)

        if desired_add > len(candidate_pool):
            report["warnings"].append(
                f"Protocol B lacks enough ViHSD offensive rows: need {desired_add}, available {len(candidate_pool)}. Using all available."
            )
            desired_add = len(candidate_pool)

        rng = random.Random(args.seed)
        rng.shuffle(candidate_pool)
        selected_additions = candidate_pool[:desired_add]

        b_train = dedup_rows(b_train_base + selected_additions)

        # Final safety: no overlap train vs val/test
        b_val_test_keys = {dedup_key(x["text"]) for x in b_val + b_test}
        b_train = [r for r in b_train if dedup_key(r["text"]) not in b_val_test_keys]

        protocol_b_rows = {
            "train": b_train,
            "validation": b_val,
            "test": b_test,
        }

        b_paths = protocol_paths(output_root, args.dataset_prefix, "b")
        for split in SPLITS:
            write_jsonl(b_paths[split], protocol_b_rows[split])

        report["protocols"]["b"] = {
            "files": {k: str(v) for k, v in b_paths.items()},
            "stats": {split: split_stats(protocol_b_rows[split]) for split in SPLITS},
            "merge": {
                "vihsd_offensive_pool_after_dedup": len(vihsd_offensive),
                "candidate_pool_after_overlap_filter": len(candidate_pool),
                "selected_additions": len(selected_additions),
                "base_train_total": base_total,
                "base_train_toxic": base_toxic,
                "requested_additions_min": min_add,
                "requested_additions_max": max_add,
            },
            "overlap_exact": {
                "train_validation": overlap_count(protocol_b_rows["train"], protocol_b_rows["validation"]),
                "train_test": overlap_count(protocol_b_rows["train"], protocol_b_rows["test"]),
                "validation_test": overlap_count(protocol_b_rows["validation"], protocol_b_rows["test"]),
            },
        }

    # ----------------
    # Protocol C
    # ----------------
    if "c" in selected_protocols:
        c_pool = dedup_rows(victsd["train"] + victsd["validation"] + victsd["test"] + vihsd_offensive)
        c_split = stratified_split(c_pool, args.c_train_ratio, args.c_validation_ratio, args.seed)

        c_paths = protocol_paths(output_root, args.dataset_prefix, "c")
        for split in SPLITS:
            write_jsonl(c_paths[split], c_split[split])

        report["protocols"]["c"] = {
            "files": {k: str(v) for k, v in c_paths.items()},
            "stats": {split: split_stats(c_split[split]) for split in SPLITS},
            "pool_size_after_global_dedup": len(c_pool),
            "overlap_exact": {
                "train_validation": overlap_count(c_split["train"], c_split["validation"]),
                "train_test": overlap_count(c_split["train"], c_split["test"]),
                "validation_test": overlap_count(c_split["validation"], c_split["test"]),
            },
        }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
