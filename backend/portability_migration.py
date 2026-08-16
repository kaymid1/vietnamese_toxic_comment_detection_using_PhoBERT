"""Explicit, dry-run-first migration for application-owned artifact references."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.artifact_refs import encode_artifact_ref, inspect_artifact_ref
from backend.runtime_paths import get_feedback_db_path

DB_FIELDS = {
    "mlflow_training_artifact": ("artifact_path",),
    "mlflow_model_version": ("artifact_path", "artifact_uri", "bundle_path"),
    "mlflow_do_run": ("artifact_uri", "bundle_path", "bundle_url"),
}
ROW_KEYS = {"mlflow_training_artifact": "id", "mlflow_model_version": "id", "mlflow_do_run": "run_id"}
LOG_PATH_KEYS = {"artifact_path", "artifact_uri", "bundle_path", "work_dir", "path"}


def _integrity(conn: sqlite3.Connection) -> None:
    result = conn.execute("PRAGMA integrity_check").fetchone()
    if not result or str(result[0]).lower() != "ok":
        raise RuntimeError(f"SQLite integrity_check failed: {result[0] if result else 'no result'}")


def _transform_logs(raw: str) -> tuple[str, str]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw, "invalid_json"
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        return raw, "unknown_json"
    changed = False
    updated: list[dict[str, Any]] = []
    for item in payload:
        copy = dict(item)
        for key in LOG_PATH_KEYS.intersection(copy):
            if isinstance(copy[key], str):
                encoded = encode_artifact_ref(copy[key])
                if encoded != copy[key]:
                    copy[key] = encoded
                    changed = True
        updated.append(copy)
    return (json.dumps(updated, ensure_ascii=False, separators=(",", ":")) if changed else raw), "convertible" if changed else "unchanged"


def plan_migration(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as conn:
        _integrity(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        plan: list[dict[str, Any]] = []
        for table, fields in DB_FIELDS.items():
            if table not in tables:
                continue
            row_key = ROW_KEYS[table]
            for row in conn.execute(f"SELECT {row_key}, {', '.join(fields)} FROM {table}"):
                for field in fields:
                    old = row[fields.index(field) + 1]
                    if old is None:
                        continue
                    inspected = inspect_artifact_ref(str(old))
                    new = encode_artifact_ref(str(old))
                    plan.append({"table": table, "id": row[0], "field": field, "old": str(old), "new": new, "classification": inspected.classification})
            if table == "mlflow_do_run":
                for row in conn.execute("SELECT run_id, logs_json FROM mlflow_do_run WHERE logs_json IS NOT NULL"):
                    new, classification = _transform_logs(str(row[1]))
                    plan.append({"table": table, "id": row[0], "field": "logs_json", "old": str(row[1]), "new": new, "classification": classification})
    return plan


def _backup(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"{db_path.name}.pre-portability-{stamp}.bak")
    if backup.exists():
        raise RuntimeError(f"Backup already exists: {backup}")
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as source, sqlite3.connect(backup) as destination:
        _integrity(source)
        source.backup(destination)
        _integrity(destination)
    return backup


def apply_migration(db_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    plan = plan_migration(db_path)
    backup = _backup(db_path)
    updates = [item for item in plan if item["new"] != item["old"]]
    try:
        with sqlite3.connect(db_path) as conn:
            _integrity(conn)
            conn.execute("BEGIN IMMEDIATE")
            for item in updates:
                key = ROW_KEYS[item["table"]]
                conn.execute(f"UPDATE {item['table']} SET {item['field']} = ? WHERE {key} = ?", (item["new"], item["id"]))
            _integrity(conn)
            conn.commit()
    except Exception:
        raise
    return backup, plan


def _summary(plan: Iterable[dict[str, Any]]) -> dict[str, int]:
    summary = {"convertible": 0, "already_portable": 0, "external_unmanaged": 0, "missing_target": 0, "invalid": 0}
    for item in plan:
        if item["new"] != item["old"]:
            summary["convertible"] += 1
        elif item["classification"] == "portable":
            summary["already_portable"] += 1
        elif item["classification"] in {"external_absolute", "relative_unmanaged", "protected_uri", "unknown_json"}:
            summary["external_unmanaged"] += 1
        elif item["classification"] in {"invalid", "invalid_json"}:
            summary["invalid"] += 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert application artifact references; dry run by default")
    parser.add_argument("--db", type=Path, default=get_feedback_db_path())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="create backup and apply migration")
    mode.add_argument("--dry-run", action="store_true", help="explicitly request the default read-only mode")
    args = parser.parse_args()
    db_path = args.db.expanduser().resolve()
    plan = plan_migration(db_path)
    for item in plan:
        print(f"{item['table']} row id={item['id']} {item['field']}: {item['classification']}")
        if item["new"] != item["old"]:
            print(f"  OLD: {item['old']}\n  NEW: {item['new']}")
        else:
            print("  SKIP: preserved")
    print("Summary:", _summary(plan))
    if args.apply:
        backup, _ = apply_migration(db_path)
        print(f"APPLIED. SQLite-safe backup: {backup}")
    else:
        print("DRY RUN: no files were modified")


if __name__ == "__main__":
    main()
