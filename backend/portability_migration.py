"""Explicit, dry-run-first migration for application-owned artifact references."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
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
WINDOWS_KAGGLE_URI_RE = re.compile(
    r"^file:///?[A-Za-z]:/Code/Thesis/Thesis/\.runtime/kaggle_real_jobs/[^/]+/output/viettoxic/([A-Za-z0-9._-]+)\.zip$",
    re.IGNORECASE,
)


def _integrity(conn: sqlite3.Connection) -> None:
    result = conn.execute("PRAGMA integrity_check").fetchone()
    if not result or str(result[0]).lower() != "ok":
        raise RuntimeError(f"SQLite integrity_check failed: {result[0] if result else 'no result'}")


def _transform_logs(
    raw: str, *, roots: Mapping[str, Path] | None = None
) -> tuple[str, str]:
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
                encoded = encode_artifact_ref(copy[key], roots=roots)
                if encoded != copy[key]:
                    copy[key] = encoded
                    changed = True
        updated.append(copy)
    return (json.dumps(updated, ensure_ascii=False, separators=(",", ":")) if changed else raw), "convertible" if changed else "unchanged"


def plan_migration(
    db_path: Path, *, roots: Mapping[str, Path] | None = None
) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)) as conn:
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
                    inspected = inspect_artifact_ref(str(old), roots=roots)
                    new = encode_artifact_ref(str(old), roots=roots)
                    plan.append({"table": table, "id": row[0], "field": field, "old": str(old), "new": new, "classification": inspected.classification})
            if table == "mlflow_do_run":
                for row in conn.execute("SELECT run_id, logs_json FROM mlflow_do_run WHERE logs_json IS NOT NULL"):
                    new, classification = _transform_logs(str(row[1]), roots=roots)
                    plan.append({"table": table, "id": row[0], "field": "logs_json", "old": str(row[1]), "new": new, "classification": classification})
    return plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_registry_uri(model_family: str, run_id: str) -> str:
    if model_family not in {"tfidf_lr", "phobert"}:
        raise ValueError("unsupported model family")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("invalid source run ID")
    return f"runtime://model_registry/{model_family}/{run_id}/artifact.zip"


def plan_kaggle_registry_migration(
    db_path: Path,
    *,
    registry_root: Path,
) -> list[dict[str, Any]]:
    """Plan only checksum-verified rewrites of migrated Kaggle URI fields."""
    db_path = db_path.expanduser().resolve()
    registry_root = registry_root.expanduser().resolve()
    entries: list[dict[str, Any]] = []
    with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        runs = conn.execute(
            """
            SELECT run_id, artifact_uri, artifact_checksum
            FROM mlflow_do_run
            WHERE artifact_uri IS NOT NULL AND artifact_uri <> ''
            """
        ).fetchall()
        for run in runs:
            old_uri = str(run["artifact_uri"] or "")
            match = WINDOWS_KAGGLE_URI_RE.fullmatch(old_uri.replace("\\", "/"))
            if not match:
                continue
            run_id = str(run["run_id"] or "").strip()
            path_run_id = match.group(1)
            expected = str(run["artifact_checksum"] or "").strip().lower()
            base = {
                "run_id": run_id,
                "old_uri": old_uri,
                "new_uri": None,
                "checksum_verified": False,
                "eligible": False,
                "reason": None,
            }
            if path_run_id != run_id:
                base["reason"] = "URI run ID does not match persisted run ID"
                entries.append({"table": "mlflow_do_run", "record_id": run_id, "field": "artifact_uri", **base})
                continue
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                base["reason"] = "persisted artifact checksum is missing or invalid"
                entries.append({"table": "mlflow_do_run", "record_id": run_id, "field": "artifact_uri", **base})
                continue

            registry = conn.execute(
                """
                SELECT model_family, model_id, source_run_id, artifact_uri, artifact_checksum
                FROM mlflow_model_version
                WHERE source_run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if not registry:
                base["reason"] = "matching model registry record is missing"
                entries.append({"table": "mlflow_do_run", "record_id": run_id, "field": "artifact_uri", **base})
                continue
            family = str(registry["model_family"] or "").strip()
            try:
                new_uri = _canonical_registry_uri(family, run_id)
            except ValueError as exc:
                base["reason"] = str(exc)
                entries.append({"table": "mlflow_do_run", "record_id": run_id, "field": "artifact_uri", **base})
                continue
            target = (registry_root / family / run_id / "artifact.zip").resolve()
            try:
                target.relative_to(registry_root)
            except ValueError:
                base["reason"] = "registry target escapes configured registry root"
                entries.append({"table": "mlflow_do_run", "record_id": run_id, "field": "artifact_uri", **base})
                continue
            if str(registry["model_id"] or "") != f"{family}/{run_id}" or str(registry["source_run_id"] or "") != run_id:
                base["reason"] = "model registry family/run identity does not match"
            elif str(registry["artifact_uri"] or "") != new_uri:
                base["reason"] = "model registry artifact URI is not canonical"
            elif str(registry["artifact_checksum"] or "").strip().lower() != expected:
                base["reason"] = "model registry checksum metadata differs"
            elif not target.is_file():
                base["reason"] = "registry artifact file is missing"
            elif _sha256_file(target).lower() != expected:
                base["reason"] = "registry artifact checksum mismatch"
            else:
                base.update({"new_uri": new_uri, "checksum_verified": True, "eligible": True, "reason": "checksum verified"})
            entries.append({"table": "mlflow_do_run", "record_id": run_id, "field": "artifact_uri", **base})

            artifact = conn.execute(
                "SELECT id, artifact_path FROM mlflow_training_artifact WHERE source_run_id = ?",
                (run_id,),
            ).fetchone()
            if artifact and str(artifact["artifact_path"] or "") == old_uri:
                artifact_entry = dict(base)
                entries.append(
                    {
                        "table": "mlflow_training_artifact",
                        "record_id": int(artifact["id"]),
                        "field": "artifact_path",
                        **artifact_entry,
                    }
                )
    return entries


def apply_kaggle_registry_migration(
    db_path: Path,
    *,
    registry_root: Path,
) -> tuple[Optional[Path], list[dict[str, Any]]]:
    """Apply the scoped migration after creating a SQLite backup."""
    plan = plan_kaggle_registry_migration(db_path, registry_root=registry_root)
    updates = [item for item in plan if item["eligible"] and item["new_uri"] != item["old_uri"]]
    if not updates:
        return None, plan
    backup = _backup(db_path)
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        _integrity(conn)
        conn.execute("BEGIN IMMEDIATE")
        for item in updates:
            key = ROW_KEYS[item["table"]]
            conn.execute(
                f"UPDATE {item['table']} SET {item['field']} = ? WHERE {key} = ? AND {item['field']} = ?",
                (item["new_uri"], item["record_id"], item["old_uri"]),
            )
        _integrity(conn)
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
    return backup, plan


def _backup(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"{db_path.name}.pre-portability-{stamp}.bak")
    if backup.exists():
        raise RuntimeError(f"Backup already exists: {backup}")
    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    destination = sqlite3.connect(backup)
    try:
        _integrity(source)
        source.backup(destination)
        _integrity(destination)
        destination.commit()
    finally:
        destination.close()
        source.close()
    return backup


def apply_migration(
    db_path: Path, *, roots: Mapping[str, Path] | None = None
) -> tuple[Path, list[dict[str, Any]]]:
    plan = plan_migration(db_path, roots=roots)
    backup = _backup(db_path)
    updates = [item for item in plan if item["new"] != item["old"]]
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        _integrity(conn)
        conn.execute("BEGIN IMMEDIATE")
        for item in updates:
            key = ROW_KEYS[item["table"]]
            conn.execute(f"UPDATE {item['table']} SET {item['field']} = ? WHERE {key} = ?", (item["new"], item["id"]))
        _integrity(conn)
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
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
