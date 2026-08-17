#!/usr/bin/env python3
"""Migrate checksum-verified historical Kaggle artifact URIs.

The default mode is read-only.  Pass --apply only after reviewing the plan.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.env_loader import load_env_files

load_env_files()

from backend.portability_migration import (  # noqa: E402
    apply_kaggle_registry_migration,
    plan_kaggle_registry_migration,
)
from backend.runtime_paths import get_feedback_db_path, get_model_registry_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_feedback_db_path())
    parser.add_argument("--registry-root", type=Path, default=get_model_registry_dir())
    parser.add_argument("--apply", action="store_true", help="create a backup and apply eligible updates")
    args = parser.parse_args()

    if args.apply:
        backup, plan = apply_kaggle_registry_migration(args.db, registry_root=args.registry_root)
        if backup:
            print(f"backup={backup}")
    else:
        plan = plan_kaggle_registry_migration(args.db, registry_root=args.registry_root)

    for item in plan:
        state = "ELIGIBLE" if item["eligible"] else "SKIP"
        print(
            f"{state} {item['table']} record={item['record_id']} field={item['field']} "
            f"checksum_verified={item['checksum_verified']} reason={item['reason']}"
        )
        if item["new_uri"] and item["new_uri"] != item["old_uri"]:
            print(f"  OLD: {item['old_uri']}")
            print(f"  NEW: {item['new_uri']}")


if __name__ == "__main__":
    main()
