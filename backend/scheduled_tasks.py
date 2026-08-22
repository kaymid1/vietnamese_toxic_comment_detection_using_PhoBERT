"""SQLite-backed scheduled task orchestration for the admin MVP."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from backend.article_discovery import DEFAULT_VNEXPRESS_RSS_URL


BUILTIN_TASK_ID = "vnexpress_auto_collection"
BUILTIN_TASK_NAME = "VnExpress Auto Collection"
BUILTIN_TASK_TYPE = "vnexpress_rss"
DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"
DEFAULT_INTERVAL_MINUTES = 60
DEFAULT_MAX_ARTICLES_PER_RUN = 10
MAX_ARTICLE_ATTEMPTS = 3
RETRY_DELAYS_MINUTES = {1: 30, 2: 120}
RUN_LEASE_HOURS = 6
RUN_STALL_TIMEOUT_MINUTES = 15
SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS = 24


class ScheduledTaskNotFound(KeyError):
    pass


class ScheduledTaskConflict(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_timezone(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timezone is required")
    try:
        ZoneInfo(text)
    except Exception as exc:
        raise ValueError(f"Unknown timezone: {text}") from exc
    return text


def _validate_interval(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("interval_minutes must be an integer") from exc
    if not 1 <= parsed <= 10080:
        raise ValueError("interval_minutes must be between 1 and 10080")
    return parsed


def _validate_max_articles(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_articles_per_run must be an integer") from exc
    if not 1 <= parsed <= 100:
        raise ValueError("max_articles_per_run must be between 1 and 100")
    return parsed


def _validate_recrawl_interval_hours(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("article_recrawl_interval_hours must be an integer") from exc
    if not 1 <= parsed <= 8760:
        raise ValueError("article_recrawl_interval_hours must be between 1 and 8760")
    return parsed


def _validate_rss_url(value: str) -> str:
    text = str(value or "").strip()
    if text != DEFAULT_VNEXPRESS_RSS_URL:
        raise ValueError("MVP supports only the VnExpress latest-news RSS URL")
    return text


def _validate_model_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > 160 or ".." in text or "\\" in text or text.count("/") > 1:
        raise ValueError("model_name is invalid")
    return text


def ensure_scheduled_task_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            interval_minutes INTEGER NOT NULL,
            timezone TEXT NOT NULL,
            config_json TEXT NOT NULL,
            next_run_at TEXT,
            last_run_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            updated_by TEXT,
            UNIQUE(task_type, name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_task_runs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            scheduled_for TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            status TEXT NOT NULL,
            lease_owner TEXT,
            lease_until TEXT,
            started_at TEXT,
            finished_at TEXT,
            discovered_count INTEGER NOT NULL DEFAULT 0,
            processed_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
            UNIQUE(task_id, scheduled_for)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS article_discovery_history (
            discovery_key TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            source TEXT NOT NULL,
            canonical_url TEXT NOT NULL,
            article_id TEXT,
            article_title TEXT,
            published_at TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            stage TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            crawl_attempt_count INTEGER NOT NULL DEFAULT 0,
            inference_attempt_count INTEGER NOT NULL DEFAULT 0,
            crawl_started_at TEXT,
            crawl_completed_at TEXT,
            inference_started_at TEXT,
            inference_completed_at TEXT,
            processed_at TEXT,
            retry_after TEXT,
            last_run_id TEXT,
            last_error TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_task_created "
        "ON scheduled_task_runs(task_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_running_lease "
        "ON scheduled_task_runs(status, lease_until)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_article_discovery_task_stage "
        "ON article_discovery_history(task_id, stage, retry_after)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_article_discovery_article_id "
        "ON article_discovery_history(source, article_id)"
    )


def _task_payload(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        config = json.loads(row["config_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        config = {}
    return {
        "id": row["id"],
        "name": row["name"],
        "task_type": row["task_type"],
        "enabled": bool(row["enabled"]),
        "interval_minutes": int(row["interval_minutes"]),
        "timezone": row["timezone"],
        "config": config,
        "rss_url": config.get("rss_url", DEFAULT_VNEXPRESS_RSS_URL),
        "max_articles_per_run": int(config.get("max_articles_per_run", DEFAULT_MAX_ARTICLES_PER_RUN)),
        "article_recrawl_interval_hours": int(
            config.get("article_recrawl_interval_hours", SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS)
        ),
        "model_name": config.get("model_name"),
        "next_run_at": row["next_run_at"],
        "last_run_at": row["last_run_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "updated_by": row["updated_by"],
    }


def _run_payload(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "scheduled_for": row["scheduled_for"],
        "trigger_type": row["trigger_type"],
        "status": row["status"],
        "lease_owner": row["lease_owner"],
        "lease_until": row["lease_until"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "discovered_count": int(row["discovered_count"] or 0),
        "processed_count": int(row["processed_count"] or 0),
        "failed_count": int(row["failed_count"] or 0),
        "error": row["error"],
        "metadata": metadata,
        "created_at": row["created_at"],
    }


class ScheduledTaskService:
    def __init__(
        self,
        db_path: Path,
        task_runner: Callable[[Dict[str, Any], str], Dict[str, Any]],
        *,
        poll_seconds: int = 30,
    ) -> None:
        self.db_path = db_path
        self.task_runner = task_runner
        self.poll_seconds = max(10, int(poll_seconds))
        self.owner = f"scheduler:{uuid.uuid4().hex}"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_scheduled_task_tables(conn)
        return conn

    def ensure_builtin_task(self) -> None:
        now = _iso(_now())
        config = json.dumps(
            {
                "rss_url": DEFAULT_VNEXPRESS_RSS_URL,
                "max_articles_per_run": DEFAULT_MAX_ARTICLES_PER_RUN,
                "article_recrawl_interval_hours": SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS,
            },
            ensure_ascii=False,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO scheduled_tasks (
                    id, name, task_type, enabled, interval_minutes, timezone, config_json,
                    next_run_at, last_run_at, created_at, updated_at, updated_by
                ) VALUES (?, ?, ?, 0, ?, ?, ?, NULL, NULL, ?, ?, NULL)
                """,
                (
                    BUILTIN_TASK_ID,
                    BUILTIN_TASK_NAME,
                    BUILTIN_TASK_TYPE,
                    DEFAULT_INTERVAL_MINUTES,
                    DEFAULT_TIMEZONE,
                    config,
                    now,
                    now,
                ),
            )
            conn.commit()

    def start(self) -> None:
        self.ensure_builtin_task()
        recovered = self.recover_stale_runs()
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scheduled-task-loop", daemon=True)
        self._thread.start()
        for task_id in recovered:
            # Recovery only resumes tasks that are still enabled. A user may
            # have disabled a task while the backend was down.
            if not self.get_task(task_id)["enabled"]:
                continue
            claim = self._claim_task(task_id, "recovery", force=True)
            if claim:
                self._start_claimed_run(claim)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception:
                # A scheduler failure must not terminate the backend process.
                continue

    def tick(self) -> None:
        self.recover_stale_runs()
        for task_id in self._due_task_ids():
            claim = self._claim_task(task_id, "scheduled")
            if claim:
                self._start_claimed_run(claim)

    def _due_task_ids(self) -> List[str]:
        now = _iso(_now())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM scheduled_tasks WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?",
                (now,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def list_tasks(self) -> List[Dict[str, Any]]:
        self.ensure_builtin_task()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM scheduled_tasks ORDER BY name").fetchall()
        return [_task_payload(row) for row in rows]

    def get_task(self, task_id: str) -> Dict[str, Any]:
        self.ensure_builtin_task()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise ScheduledTaskNotFound(task_id)
        return _task_payload(row)

    def update_task(self, task_id: str, values: Dict[str, Any], updated_by: str) -> Dict[str, Any]:
        task = self.get_task(task_id)
        enabled = bool(values.get("enabled", task["enabled"]))
        interval = _validate_interval(values.get("interval_minutes", task["interval_minutes"]))
        timezone_name = _validate_timezone(values.get("timezone", task["timezone"]))
        config = dict(task["config"])
        if "max_articles_per_run" in values:
            config["max_articles_per_run"] = _validate_max_articles(values["max_articles_per_run"])
        if "article_recrawl_interval_hours" in values:
            config["article_recrawl_interval_hours"] = _validate_recrawl_interval_hours(
                values["article_recrawl_interval_hours"]
            )
        if "rss_url" in values:
            config["rss_url"] = _validate_rss_url(values["rss_url"])
        if "model_name" in values:
            model_name = _validate_model_name(values["model_name"])
            if model_name:
                config["model_name"] = model_name
            else:
                config.pop("model_name", None)
        config.setdefault("rss_url", DEFAULT_VNEXPRESS_RSS_URL)
        config.setdefault("max_articles_per_run", DEFAULT_MAX_ARTICLES_PER_RUN)
        config.setdefault("article_recrawl_interval_hours", SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS)
        now = _now()
        next_run_at = task["next_run_at"]
        if not enabled:
            next_run_at = None
        elif not task["enabled"] or not next_run_at:
            next_run_at = _iso(now)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE scheduled_tasks
                SET enabled = ?, interval_minutes = ?, timezone = ?, config_json = ?,
                    next_run_at = ?, updated_at = ?, updated_by = ?
                WHERE id = ?
                """,
                (
                    int(enabled),
                    interval,
                    timezone_name,
                    json.dumps(config, ensure_ascii=False),
                    next_run_at,
                    _iso(now),
                    updated_by,
                    task_id,
                ),
            )
            conn.commit()
        return self.get_task(task_id)

    def recover_stale_runs(self) -> List[str]:
        now = _now()
        now_text = _iso(now)
        stale_before_text = _iso(now - timedelta(minutes=RUN_STALL_TIMEOUT_MINUTES))
        retry_after_30m = _iso(now + timedelta(minutes=30))
        retry_after_2h = _iso(now + timedelta(hours=2))
        recovered_tasks: set[str] = set()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            stale = conn.execute(
                """
                SELECT id, task_id FROM scheduled_task_runs
                WHERE status = 'running'
                  AND (
                    (lease_until IS NOT NULL AND lease_until < ?)
                    OR (started_at IS NOT NULL AND started_at < ?)
                  )
                """,
                (now_text, stale_before_text),
            ).fetchall()
            for row in stale:
                run_id = str(row["id"])
                task_id = str(row["task_id"])
                recovered_tasks.add(task_id)
                conn.execute(
                    "UPDATE scheduled_task_runs SET status = 'failed', finished_at = ?, error = ?, lease_owner = NULL, lease_until = NULL WHERE id = ? AND status = 'running'",
                    (now_text, "Run became stale and was marked failed; eligible work was requeued.", run_id),
                )
                conn.execute(
                    """
                    UPDATE article_discovery_history
                    SET stage = CASE
                            WHEN stage = 'inferring' THEN 'failed_inference'
                            WHEN stage = 'crawling' THEN 'failed_crawl'
                            ELSE stage
                        END,
                        retry_after = CASE
                            WHEN stage IN ('crawling', 'inferring')
                                 AND CASE WHEN stage = 'crawling' THEN crawl_attempt_count ELSE inference_attempt_count END = 1
                                THEN ?
                            WHEN stage IN ('crawling', 'inferring')
                                 AND CASE WHEN stage = 'crawling' THEN crawl_attempt_count ELSE inference_attempt_count END = 2
                                THEN ?
                            ELSE NULL
                        END,
                        last_error = ?, updated_at = ?
                    WHERE last_run_id = ? AND stage IN ('crawling', 'inferring')
                    """,
                    (retry_after_30m, retry_after_2h, "Run became stale before the article stage completed.", now_text, run_id),
                )
            for task_id in recovered_tasks:
                conn.execute(
                    "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ? AND enabled = 1",
                    (now_text, task_id),
                )
            conn.commit()
        return sorted(recovered_tasks)

    def _claim_task(self, task_id: str, trigger_type: str, *, force: bool = False) -> Optional[Dict[str, Any]]:
        now = _now()
        now_text = _iso(now)
        lease_until = _iso(now + timedelta(hours=RUN_LEASE_HOURS))
        run_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            task = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            if not task:
                conn.rollback()
                raise ScheduledTaskNotFound(task_id)
            if not force and (not task["enabled"] or not task["next_run_at"] or str(task["next_run_at"]) > now_text):
                conn.rollback()
                return None
            active = conn.execute(
                "SELECT id FROM scheduled_task_runs WHERE task_id = ? AND status = 'running' AND lease_until > ? LIMIT 1",
                (task_id, now_text),
            ).fetchone()
            if active:
                conn.rollback()
                if trigger_type == "manual":
                    raise ScheduledTaskConflict(f"Task is already running: {active[0]}")
                return None
            scheduled_for = str(task["next_run_at"] or now_text) if trigger_type == "scheduled" else now_text
            if trigger_type == "scheduled":
                next_run = _iso(now + timedelta(minutes=int(task["interval_minutes"])))
                conn.execute("UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?", (next_run, task_id))
            conn.execute(
                """
                INSERT INTO scheduled_task_runs (
                    id, task_id, scheduled_for, trigger_type, status, lease_owner, lease_until,
                    started_at, created_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (run_id, task_id, scheduled_for, trigger_type, f"{self.owner}:{run_id}", lease_until, now_text, now_text),
            )
            conn.commit()
            return {"run_id": run_id, "task": _task_payload(task), "trigger_type": trigger_type}

    def run_now(self, task_id: str) -> Dict[str, Any]:
        self.recover_stale_runs()
        claim = self._claim_task(task_id, "manual", force=True)
        if not claim:
            raise ScheduledTaskConflict("Task is already running")
        self._start_claimed_run(claim)
        return self.get_run(task_id, str(claim["run_id"]))

    def cancel_run(self, task_id: str, run_id: str, reason: str = "Canceled by administrator.") -> Dict[str, Any]:
        now = _now()
        now_text = _iso(now)
        retry_after_30m = _iso(now + timedelta(minutes=30))
        retry_after_2h = _iso(now + timedelta(hours=2))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE task_id = ? AND id = ?",
                (task_id, run_id),
            ).fetchone()
            if not run:
                conn.rollback()
                raise ScheduledTaskNotFound(run_id)
            if str(run["status"]) != "running":
                conn.rollback()
                raise ScheduledTaskConflict(f"Run is already {run['status']}")
            error = reason[:2000]
            updated = conn.execute(
                """
                UPDATE scheduled_task_runs
                SET status = 'canceled', finished_at = ?, error = ?,
                    metadata_json = ?, lease_owner = NULL, lease_until = NULL
                WHERE id = ? AND status = 'running'
                """,
                (now_text, error, json.dumps({"canceled": True, "reason": error}), run_id),
            )
            conn.execute(
                """
                UPDATE article_discovery_history
                SET stage = CASE
                        WHEN stage = 'inferring' THEN 'failed_inference'
                        WHEN stage = 'crawling' THEN 'failed_crawl'
                        ELSE stage
                    END,
                    retry_after = CASE
                        WHEN stage IN ('crawling', 'inferring')
                             AND CASE WHEN stage = 'crawling' THEN crawl_attempt_count ELSE inference_attempt_count END = 1
                            THEN ?
                        WHEN stage IN ('crawling', 'inferring')
                             AND CASE WHEN stage = 'crawling' THEN crawl_attempt_count ELSE inference_attempt_count END = 2
                            THEN ?
                        ELSE NULL
                    END,
                    last_error = ?, updated_at = ?
                WHERE task_id = ? AND last_run_id = ? AND stage IN ('crawling', 'inferring')
                """,
                (retry_after_30m, retry_after_2h, error, now_text, task_id, run_id),
            )
            conn.execute(
                """
                UPDATE scheduled_tasks
                SET last_run_at = ?, next_run_at = CASE WHEN enabled = 1 THEN ? ELSE NULL END
                WHERE id = ?
                """,
                (now_text, now_text, task_id),
            )
            conn.commit()
        return self.get_run(task_id, run_id)

    def _start_claimed_run(self, claim: Dict[str, Any]) -> None:
        threading.Thread(
            target=self._execute_run,
            args=(claim,),
            name=f"scheduled-task-run-{claim['run_id']}",
            daemon=True,
        ).start()

    def _execute_run(self, claim: Dict[str, Any]) -> None:
        run_id = str(claim["run_id"])
        task = claim["task"]
        try:
            summary = self.task_runner(task, run_id) or {}
            self._finish_run(run_id, "completed", summary, None)
        except Exception as exc:
            self._finish_run(run_id, "failed", {}, str(exc))

    def _finish_run(self, run_id: str, status: str, summary: Dict[str, Any], error: Optional[str]) -> None:
        now_text = _iso(_now())
        with self._connect() as conn:
            row = conn.execute("SELECT task_id FROM scheduled_task_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return
            updated = conn.execute(
                """
                UPDATE scheduled_task_runs
                SET status = ?, finished_at = ?, discovered_count = ?, processed_count = ?,
                    failed_count = ?, error = ?, metadata_json = ?, lease_owner = NULL, lease_until = NULL
                WHERE id = ? AND status = 'running'
                """,
                (
                    status,
                    now_text,
                    int(summary.get("discovered_count") or 0),
                    int(summary.get("processed_count") or 0),
                    int(summary.get("failed_count") or 0),
                    error or summary.get("error"),
                    json.dumps(summary, ensure_ascii=False),
                    run_id,
                ),
            )
            if updated.rowcount:
                conn.execute("UPDATE scheduled_tasks SET last_run_at = ? WHERE id = ?", (now_text, row["task_id"]))
            conn.commit()

    def get_run(self, task_id: str, run_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE task_id = ? AND id = ?",
                (task_id, run_id),
            ).fetchone()
        if not row:
            raise ScheduledTaskNotFound(run_id)
        return _run_payload(row)

    def list_runs(self, task_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        self.get_task(task_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                (task_id, max(1, min(200, int(limit)))),
            ).fetchall()
        return [_run_payload(row) for row in rows]

    def list_articles(self, task_id: str, *, run_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        self.get_task(task_id)
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM article_discovery_history WHERE task_id = ? AND last_run_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (task_id, run_id, max(1, min(500, int(limit)))),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM article_discovery_history WHERE task_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (task_id, max(1, min(500, int(limit)))),
                ).fetchall()
        return [dict(row) for row in rows]

    def upsert_articles(self, task_id: str, articles: Iterable[Any]) -> int:
        now_text = _iso(_now())
        count = 0
        with self._connect() as conn:
            for article in articles:
                conn.execute(
                    """
                    INSERT INTO article_discovery_history (
                        discovery_key, task_id, source, canonical_url, article_id, article_title,
                        published_at, first_seen_at, last_seen_at, stage, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'discovered', '{}', ?, ?)
                    ON CONFLICT(discovery_key) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        article_title = COALESCE(excluded.article_title, article_discovery_history.article_title),
                        published_at = COALESCE(excluded.published_at, article_discovery_history.published_at),
                        updated_at = excluded.updated_at
                    """,
                    (
                        article.discovery_key,
                        task_id,
                        article.source,
                        article.canonical_url,
                        article.article_id,
                        article.article_title,
                        article.published_at,
                        now_text,
                        now_text,
                        now_text,
                        now_text,
                    ),
                )
                count += 1
            conn.commit()
        return count

    def list_processable_articles(self, task_id: str, limit: int) -> List[Dict[str, Any]]:
        now = _now()
        now_text = _iso(now)
        with self._connect() as conn:
            task = conn.execute("SELECT config_json FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            if not task:
                raise ScheduledTaskNotFound(task_id)
            try:
                config = json.loads(task["config_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                config = {}
            recrawl_interval_hours = _validate_recrawl_interval_hours(
                config.get("article_recrawl_interval_hours", SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS)
            )
            recrawl_threshold = _iso(now - timedelta(hours=recrawl_interval_hours))
            rows = conn.execute(
                """
                SELECT * FROM article_discovery_history
                WHERE task_id = ?
                  AND stage NOT IN ('crawling', 'inferring')
                  AND (retry_after IS NULL OR retry_after <= ?)
                  AND (
                       (stage IN ('failed_crawl', 'failed_inference') AND
                        CASE WHEN stage = 'failed_crawl' THEN crawl_attempt_count ELSE inference_attempt_count END < ?)
                       OR stage IN ('discovered', 'queued', 'crawled')
                       OR (stage = 'completed'
                           AND processed_at IS NOT NULL
                           AND julianday(processed_at) <= julianday(?))
                  )
                ORDER BY CASE WHEN stage = 'failed_inference' THEN 0 ELSE 1 END, first_seen_at
                LIMIT ?
                """,
                (task_id, now_text, MAX_ARTICLE_ATTEMPTS, recrawl_threshold, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def begin_article_stage(self, discovery_key: str, run_id: str, stage: str) -> Optional[int]:
        if stage not in {"crawling", "inferring"}:
            raise ValueError(f"Unsupported active article stage: {stage}")
        now_text = _iso(_now())
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM article_discovery_history WHERE discovery_key = ?", (discovery_key,)).fetchone()
            if not row:
                return None
            active_run = conn.execute(
                "SELECT 1 FROM scheduled_task_runs WHERE id = ? AND task_id = ? AND status = 'running'",
                (run_id, row["task_id"]),
            ).fetchone()
            if not active_run:
                return None
            field = "crawl_attempt_count" if stage == "crawling" else "inference_attempt_count"
            is_recrawl_cycle = str(row["stage"] or "") == "completed" and stage == "crawling"
            attempt = 1 if is_recrawl_cycle else int(row[field] or 0) + 1
            total = int(row["attempt_count"] or 0) + 1
            if attempt > MAX_ARTICLE_ATTEMPTS:
                return None
            recrawl_reset = ", inference_attempt_count = 0" if is_recrawl_cycle else ""
            conn.execute(
                f"UPDATE article_discovery_history SET stage = :stage, attempt_count = :attempt_count, {field} = :stage_attempt{recrawl_reset}, retry_after = NULL, last_run_id = :last_run_id, last_error = NULL, updated_at = :updated_at, "
                + ("crawl_started_at = :started_at" if stage == "crawling" else "inference_started_at = :started_at")
                + " WHERE discovery_key = :discovery_key",
                {
                    "stage": stage,
                    "attempt_count": total,
                    "stage_attempt": attempt,
                    "last_run_id": run_id,
                    "updated_at": now_text,
                    "started_at": now_text,
                    "discovery_key": discovery_key,
                },
            )
            conn.commit()
            return attempt

    def mark_article_crawled(self, discovery_key: str, run_id: str) -> None:
        now_text = _iso(_now())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE article_discovery_history
                SET stage = 'crawled', crawl_completed_at = ?, last_run_id = ?, updated_at = ?
                WHERE discovery_key = ? AND last_run_id = ?
                  AND EXISTS (SELECT 1 FROM scheduled_task_runs WHERE id = ? AND status = 'running')
                """,
                (now_text, run_id, now_text, discovery_key, run_id, run_id),
            )
            conn.commit()

    def mark_article_completed(self, discovery_key: str, run_id: str) -> None:
        now_text = _iso(_now())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE article_discovery_history
                SET stage = 'completed', inference_completed_at = ?, processed_at = ?,
                    retry_after = NULL, last_run_id = ?, last_error = NULL, updated_at = ?
                WHERE discovery_key = ? AND last_run_id = ?
                  AND EXISTS (SELECT 1 FROM scheduled_task_runs WHERE id = ? AND status = 'running')
                """,
                (now_text, now_text, run_id, now_text, discovery_key, run_id, run_id),
            )
            conn.commit()

    def mark_article_failed(self, discovery_key: str, run_id: str, stage: str, error: str) -> None:
        now = _now()
        now_text = _iso(now)
        if stage not in {"crawl", "inference"}:
            raise ValueError(stage)
        failed_stage = "failed_crawl" if stage == "crawl" else "failed_inference"
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM article_discovery_history WHERE discovery_key = ?", (discovery_key,)).fetchone()
            if not row:
                return
            if row["last_run_id"] != run_id:
                return
            active_run = conn.execute(
                "SELECT 1 FROM scheduled_task_runs WHERE id = ? AND task_id = ? AND status = 'running'",
                (run_id, row["task_id"]),
            ).fetchone()
            if not active_run:
                return
            attempts = int(row["crawl_attempt_count"] if stage == "crawl" else row["inference_attempt_count"] or 0)
            delay = RETRY_DELAYS_MINUTES.get(attempts)
            retry_after = _iso(now + timedelta(minutes=delay)) if delay else None
            conn.execute(
                "UPDATE article_discovery_history SET stage = ?, retry_after = ?, last_run_id = ?, last_error = ?, updated_at = ? WHERE discovery_key = ?",
                (failed_stage, retry_after, run_id, error[:2000], now_text, discovery_key),
            )
            conn.commit()

    def article_metadata(self, discovery_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM article_discovery_history WHERE discovery_key = ?", (discovery_key,)).fetchone()
        return dict(row) if row else None
