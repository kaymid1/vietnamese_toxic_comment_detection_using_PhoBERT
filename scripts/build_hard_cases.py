import json
import re
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "hard_case_config.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_wordlist(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_dataset_rows() -> List[Dict[str, Any]]:
    preferred = BASE_DIR / "data" / "processed" / "victsd_v1"
    fallback = BASE_DIR / "data" / "victsd"
    rows: List[Dict[str, Any]] = []

    def read_jsonl(path: Path, source_label: str) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = item.get("text") or item.get("Comment") or item.get("comment") or ""
                label = item.get("label")
                if label is None:
                    label = item.get("Toxicity")
                if text is None:
                    continue
                try:
                    label_int = int(label)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "text": str(text),
                        "true_label": label_int,
                        "source_dataset": item.get("meta", {}).get("source", source_label),
                    }
                )

    if preferred.exists():
        for split in ["train", "validation", "test"]:
            read_jsonl(preferred / f"{split}.jsonl", "victsd")
    elif fallback.exists():
        for split in ["train", "validation", "test"]:
            read_jsonl(fallback / f"{split}.jsonl", "victsd")

    return rows


def build_candidates(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = load_dataset_rows()
    profanity_path = BASE_DIR / config.get("profanity_wordlist_path", "")
    profanity = load_wordlist(profanity_path)
    profanity_regex = re.compile("|".join(re.escape(word) for word in profanity), re.IGNORECASE) if profanity else None

    sarcasm_patterns = config.get("sarcasm_cue_patterns") or []
    sarcasm_regexes = [re.compile(pat, re.IGNORECASE) for pat in sarcasm_patterns if pat]

    candidates: List[Dict[str, Any]] = []

    for row in rows:
        text = row["text"]
        label = row["true_label"]
        if label != 1:
            continue

        has_profanity = bool(profanity_regex.search(text)) if profanity_regex else False
        if not has_profanity:
            candidates.append(
                {
                    "text": text,
                    "true_label": label,
                    "predicted_label": 0,
                    "confidence": 0.5,
                    "source_dataset": row["source_dataset"],
                    "subset_tag": "implicit",
                    "candidate_reason": "implicit_toxicity",
                }
            )
            continue

        if any(regex.search(text) for regex in sarcasm_regexes):
            candidates.append(
                {
                    "text": text,
                    "true_label": label,
                    "predicted_label": 0,
                    "confidence": 0.5,
                    "source_dataset": row["source_dataset"],
                    "subset_tag": "sarcasm",
                    "candidate_reason": "sarcasm_cue",
                }
            )

    return candidates


def main() -> None:
    config = load_json(DEFAULT_CONFIG_PATH, {})
    output_path = BASE_DIR / config.get("output_path", "data/processed/hard_case_candidates.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(config)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
    main()
