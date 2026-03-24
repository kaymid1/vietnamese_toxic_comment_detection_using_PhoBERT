import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from registry_builder import build_registry_from_models

BASE_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = BASE_DIR / "experiments" / "registry.json"


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"runs": []}
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: Dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append or build experiments/registry.json")
    parser.add_argument("--run-id")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-version")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--created-at", default=datetime.now().isoformat())
    parser.add_argument("--is-baseline", action="store_true")
    parser.add_argument("--hyperparameters", default="{}")
    parser.add_argument("--metrics", default="{}")
    parser.add_argument("--from-models", action="store_true", help="Build registry from models/options metadata")
    parser.add_argument("--model-root", default=str(BASE_DIR / "models" / "options"))
    parser.add_argument("--merge-legacy", action="store_true", help="Merge existing registry entries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_models:
        model_root = Path(args.model_root).expanduser()
        registry = load_registry() if args.merge_legacy else {"runs": []}
        built = build_registry_from_models(
            model_root=model_root,
            base_dir=BASE_DIR,
            legacy_registry=registry,
            merge_legacy=args.merge_legacy,
        )
        save_registry(built)
        print(f"Built registry from {model_root} -> {REGISTRY_PATH}")
        return

    if not all([args.run_id, args.model_name, args.dataset_version, args.checkpoint_path]):
        raise SystemExit("--run-id, --model-name, --dataset-version, --checkpoint-path are required")

    registry = load_registry()
    runs = registry.get("runs") if isinstance(registry.get("runs"), list) else []

    try:
        hyperparameters = json.loads(args.hyperparameters)
    except json.JSONDecodeError:
        raise SystemExit("Invalid --hyperparameters JSON")

    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError:
        raise SystemExit("Invalid --metrics JSON")

    run = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "dataset_version": args.dataset_version,
        "created_at": args.created_at,
        "checkpoint_path": args.checkpoint_path,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "is_baseline": bool(args.is_baseline),
    }

    runs.append(run)
    registry["runs"] = runs
    save_registry(registry)
    print(f"Added run {args.run_id} to {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
