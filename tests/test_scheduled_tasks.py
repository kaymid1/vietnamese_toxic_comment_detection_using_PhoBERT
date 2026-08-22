import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.article_discovery import (
    canonicalize_vnexpress_url,
    extract_vnexpress_article_id,
    make_discovery_key,
    parse_vnexpress_rss,
)
from backend.scheduled_tasks import (
    BUILTIN_TASK_ID,
    DEFAULT_INTERVAL_MINUTES,
    DEFAULT_MAX_ARTICLES_PER_RUN,
    DEFAULT_TIMEZONE,
    MAX_ARTICLE_ATTEMPTS,
    SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS,
    ScheduledTaskConflict,
    ScheduledTaskService,
)


RSS = """
<rss><channel>
  <item><title>One</title><link>https://www.vnexpress.net/example-4987123.html?utm_source=x#comments</link><pubDate>Mon, 17 Aug 2026 10:00:00 GMT</pubDate></item>
  <item><title>One duplicate</title><link>https://vnexpress.net/example-4987123.html</link></item>
  <item><title>No ID</title><link>https://vnexpress.net/chuyen-muc/bai-viet.html?fbclid=abc</link></item>
</channel></rss>
"""


def _article(key: str = "vnexpress:4987123"):
    from backend.article_discovery import DiscoveredArticle

    return DiscoveredArticle(key, "vnexpress_rss", "https://vnexpress.net/example-4987123.html", "4987123", "One", None)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _set_article_state(db_path: Path, discovery_key: str, *, stage: str, processed_at: str | None = None) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE article_discovery_history
            SET stage = ?, processed_at = ?, updated_at = ?
            WHERE discovery_key = ?
            """,
            (stage, processed_at, _utc_text(datetime.now(timezone.utc)), discovery_key),
        )
        conn.commit()


def test_rss_parsing_canonicalization_and_duplicate_identity():
    articles = parse_vnexpress_rss(RSS)

    assert len(articles) == 2
    assert articles[0].canonical_url == "https://vnexpress.net/example-4987123.html"
    assert articles[0].discovery_key == "vnexpress:4987123"
    assert articles[0].article_id == "4987123"
    assert extract_vnexpress_article_id(articles[1].canonical_url) is None
    assert make_discovery_key(articles[1].canonical_url)[0].startswith("vnexpress:url:")
    assert canonicalize_vnexpress_url("https://www.vnexpress.net/a-1.html?utm_medium=x") == "https://vnexpress.net/a-1.html"


def test_malformed_rss_is_rejected():
    with pytest.raises(Exception):
        parse_vnexpress_rss("<rss><channel>")


def test_scheduler_defaults_are_idempotent_and_lease_blocks_duplicate(tmp_path: Path):
    calls: list[str] = []
    service = ScheduledTaskService(tmp_path / "feedback.db", lambda task, run_id: calls.append(run_id) or {})
    service.ensure_builtin_task()
    task = service.get_task(BUILTIN_TASK_ID)

    assert task["enabled"] is False
    assert task["interval_minutes"] == DEFAULT_INTERVAL_MINUTES
    assert task["timezone"] == DEFAULT_TIMEZONE
    assert task["max_articles_per_run"] == DEFAULT_MAX_ARTICLES_PER_RUN

    updated = service.update_task(BUILTIN_TASK_ID, {"model_name": "phobert/test-model"}, "tester")
    assert updated["model_name"] == "phobert/test-model"

    service.update_task(BUILTIN_TASK_ID, {"enabled": True}, "tester")
    claim = service._claim_task(BUILTIN_TASK_ID, "scheduled")
    assert claim is not None
    with pytest.raises(ScheduledTaskConflict):
        service._claim_task(BUILTIN_TASK_ID, "manual", force=True)

    service._finish_run(claim["run_id"], "completed", {"discovered_count": 1, "processed_count": 1}, None)
    assert service.get_run(BUILTIN_TASK_ID, claim["run_id"])["status"] == "completed"


def test_run_now_uses_persisted_run_history(tmp_path: Path):
    calls: list[str] = []
    service = ScheduledTaskService(tmp_path / "feedback.db", lambda task, run_id: calls.append(run_id) or {"processed_count": 1})
    service.ensure_builtin_task()

    run = service.run_now(BUILTIN_TASK_ID)
    deadline = time.time() + 2
    while time.time() < deadline:
        current = service.get_run(BUILTIN_TASK_ID, run["id"])
        if current["status"] == "completed":
            break
        time.sleep(0.02)

    current = service.get_run(BUILTIN_TASK_ID, run["id"])
    assert current["status"] == "completed"
    assert current["processed_count"] == 1
    assert calls == [run["id"]]


def test_article_retry_due_and_stage_specific_attempts(tmp_path: Path):
    service = ScheduledTaskService(tmp_path / "feedback.db", lambda task, run_id: {})
    service.ensure_builtin_task()
    service.upsert_articles(BUILTIN_TASK_ID, [_article()])

    first_claim = service._claim_task(BUILTIN_TASK_ID, "manual", force=True)
    assert first_claim is not None
    assert service.begin_article_stage("vnexpress:4987123", first_claim["run_id"], "crawling") == 1
    service.mark_article_failed("vnexpress:4987123", first_claim["run_id"], "crawl", "temporary failure")
    assert service.list_processable_articles(BUILTIN_TASK_ID, 10) == []

    with sqlite3.connect(tmp_path / "feedback.db") as conn:
        conn.execute(
            "UPDATE article_discovery_history SET retry_after = ? WHERE discovery_key = ?",
            ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"), "vnexpress:4987123"),
        )
        conn.execute("UPDATE scheduled_task_runs SET status = 'failed' WHERE status = 'running'")
        conn.commit()
    assert len(service.list_processable_articles(BUILTIN_TASK_ID, 10)) == 1
    second_claim = service._claim_task(BUILTIN_TASK_ID, "manual", force=True)
    assert second_claim is not None
    assert service.begin_article_stage("vnexpress:4987123", second_claim["run_id"], "crawling") == 2


def test_completed_article_recrawl_eligibility_respects_success_timestamp_and_active_stages(tmp_path: Path):
    db_path = tmp_path / "feedback.db"
    service = ScheduledTaskService(db_path, lambda task, run_id: {})
    service.ensure_builtin_task()
    articles = [
        _article("vnexpress:recent-completed"),
        _article("vnexpress:old-completed"),
        _article("vnexpress:active-crawling"),
        _article("vnexpress:active-inferring"),
        _article("vnexpress:new-discovered"),
    ]
    service.upsert_articles(BUILTIN_TASK_ID, articles)
    now = datetime.now(timezone.utc)
    _set_article_state(
        db_path,
        "vnexpress:recent-completed",
        stage="completed",
        processed_at=_utc_text(now - timedelta(hours=SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS - 1)),
    )
    _set_article_state(
        db_path,
        "vnexpress:old-completed",
        stage="completed",
        processed_at=_utc_text(now - timedelta(hours=SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS + 1)),
    )
    _set_article_state(
        db_path,
        "vnexpress:active-crawling",
        stage="crawling",
        processed_at=_utc_text(now - timedelta(days=7)),
    )
    _set_article_state(
        db_path,
        "vnexpress:active-inferring",
        stage="inferring",
        processed_at=_utc_text(now - timedelta(days=7)),
    )

    processable = {row["discovery_key"] for row in service.list_processable_articles(BUILTIN_TASK_ID, 10)}

    assert "vnexpress:old-completed" in processable
    assert "vnexpress:new-discovered" in processable
    assert "vnexpress:recent-completed" not in processable
    assert "vnexpress:active-crawling" not in processable
    assert "vnexpress:active-inferring" not in processable


def test_recrawl_reuses_article_identity_and_preserves_running_slot_protection(tmp_path: Path):
    db_path = tmp_path / "feedback.db"
    service = ScheduledTaskService(db_path, lambda task, run_id: {})
    service.ensure_builtin_task()
    service.upsert_articles(BUILTIN_TASK_ID, [_article()])
    old_processed_at = _utc_text(
        datetime.now(timezone.utc) - timedelta(hours=SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS + 1)
    )
    _set_article_state(db_path, "vnexpress:4987123", stage="completed", processed_at=old_processed_at)
    before = service.article_metadata("vnexpress:4987123")

    service.upsert_articles(BUILTIN_TASK_ID, [_article()])
    processable = service.list_processable_articles(BUILTIN_TASK_ID, 10)

    assert [row["discovery_key"] for row in processable] == ["vnexpress:4987123"]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE article_discovery_history
            SET crawl_attempt_count = ?, inference_attempt_count = ?
            WHERE discovery_key = ?
            """,
            (MAX_ARTICLE_ATTEMPTS, MAX_ARTICLE_ATTEMPTS, "vnexpress:4987123"),
        )
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM article_discovery_history").fetchone()[0] == 1
    after = service.article_metadata("vnexpress:4987123")
    assert before and after
    assert after["first_seen_at"] == before["first_seen_at"]
    assert after["processed_at"] == old_processed_at

    claim = service._claim_task(BUILTIN_TASK_ID, "manual", force=True)
    assert claim is not None
    with pytest.raises(ScheduledTaskConflict):
        service._claim_task(BUILTIN_TASK_ID, "manual", force=True)
    assert service.begin_article_stage("vnexpress:4987123", claim["run_id"], "crawling") == 1
    article = service.article_metadata("vnexpress:4987123")
    assert article and article["crawl_attempt_count"] == 1
    assert article["inference_attempt_count"] == 0
    assert service.list_processable_articles(BUILTIN_TASK_ID, 10) == []


def test_article_recrawl_interval_override_is_task_configured(tmp_path: Path):
    db_path = tmp_path / "feedback.db"
    service = ScheduledTaskService(db_path, lambda task, run_id: {})
    service.ensure_builtin_task()
    task = service.get_task(BUILTIN_TASK_ID)
    assert task["article_recrawl_interval_hours"] == SCHEDULED_ARTICLE_RECRAWL_INTERVAL_HOURS

    service.update_task(BUILTIN_TASK_ID, {"article_recrawl_interval_hours": 1}, "tester")
    updated = service.get_task(BUILTIN_TASK_ID)
    assert updated["article_recrawl_interval_hours"] == 1
    service.upsert_articles(BUILTIN_TASK_ID, [_article("vnexpress:override")])
    _set_article_state(
        db_path,
        "vnexpress:override",
        stage="completed",
        processed_at=_utc_text(datetime.now(timezone.utc) - timedelta(hours=2)),
    )

    assert [row["discovery_key"] for row in service.list_processable_articles(BUILTIN_TASK_ID, 10)] == [
        "vnexpress:override"
    ]


def test_stale_run_recovery_requeues_article_stages(tmp_path: Path):
    service = ScheduledTaskService(tmp_path / "feedback.db", lambda task, run_id: {})
    service.ensure_builtin_task()
    service.update_task(BUILTIN_TASK_ID, {"enabled": True}, "tester")
    service.upsert_articles(BUILTIN_TASK_ID, [_article()])

    claim = service._claim_task(BUILTIN_TASK_ID, "scheduled")
    assert claim is not None
    assert service.begin_article_stage("vnexpress:4987123", claim["run_id"], "crawling") == 1
    with sqlite3.connect(tmp_path / "feedback.db") as conn:
        conn.execute(
            "UPDATE scheduled_task_runs SET lease_until = ?, started_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z"), "old", claim["run_id"]),
        )
        conn.commit()

    recovered = service.recover_stale_runs()
    assert BUILTIN_TASK_ID in recovered
    run = service.get_run(BUILTIN_TASK_ID, claim["run_id"])
    article = service.article_metadata("vnexpress:4987123")
    assert run["status"] == "failed"
    assert article and article["stage"] == "failed_crawl"
    assert article["retry_after"]


def test_cancel_run_marks_active_article_failed_and_cannot_be_overwritten(tmp_path: Path):
    service = ScheduledTaskService(tmp_path / "feedback.db", lambda task, run_id: {})
    service.ensure_builtin_task()
    service.upsert_articles(BUILTIN_TASK_ID, [_article()])
    claim = service._claim_task(BUILTIN_TASK_ID, "manual", force=True)
    assert claim is not None
    assert service.begin_article_stage("vnexpress:4987123", claim["run_id"], "crawling") == 1

    canceled = service.cancel_run(BUILTIN_TASK_ID, claim["run_id"])
    assert canceled["status"] == "canceled"
    article = service.article_metadata("vnexpress:4987123")
    assert article and article["stage"] == "failed_crawl"
    service.mark_article_crawled("vnexpress:4987123", claim["run_id"])
    service._finish_run(claim["run_id"], "completed", {"processed_count": 1}, None)
    assert service.get_run(BUILTIN_TASK_ID, claim["run_id"])["status"] == "canceled"
    assert service.article_metadata("vnexpress:4987123")["stage"] == "failed_crawl"


def test_disabled_task_is_not_auto_resumed_after_recovery(tmp_path: Path):
    service = ScheduledTaskService(tmp_path / "feedback.db", lambda task, run_id: {})
    service.ensure_builtin_task()
    service.update_task(BUILTIN_TASK_ID, {"enabled": True}, "tester")
    claim = service._claim_task(BUILTIN_TASK_ID, "scheduled")
    assert claim is not None
    service.update_task(BUILTIN_TASK_ID, {"enabled": False}, "tester")
    with sqlite3.connect(tmp_path / "feedback.db") as conn:
        conn.execute(
            "UPDATE scheduled_task_runs SET lease_until = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z"), claim["run_id"]),
        )
        conn.commit()

    try:
        service.start()
        time.sleep(0.05)
        assert service.get_run(BUILTIN_TASK_ID, claim["run_id"])["status"] == "failed"
        assert service.list_runs(BUILTIN_TASK_ID)[0]["trigger_type"] == "scheduled"
    finally:
        service.stop()
