"""Unit tests for heartbeat update and watchdog detection logic."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from heartbeat import is_heartbeat_stale, update_heartbeat, update_heartbeat_on_report
from manager_state import heartbeat_path
from models import HeartbeatState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minute: int = 0, second: int = 0) -> datetime:
    return datetime(2025, 6, 1, 12, minute, second, tzinfo=timezone.utc)


def _heartbeat(**kw) -> HeartbeatState:
    defaults = dict(
        updated_at=_ts(),
        manager_pid=1234,
        status="running",
        current_wip=["task-A"],
    )
    defaults.update(kw)
    return HeartbeatState(**defaults)


CID = "test-co"


# ---------------------------------------------------------------------------
# update_heartbeat
# ---------------------------------------------------------------------------

class TestUpdateHeartbeat:
    def test_creates_heartbeat_file(self, tmp_path: Path):
        hb = update_heartbeat(tmp_path, CID, "running", ["t1"], pid=42)
        assert heartbeat_path(tmp_path, CID).exists()
        assert hb.status == "running"
        assert hb.manager_pid == 42
        assert hb.current_wip == ["t1"]
        assert hb.last_report_at is None

    def test_updated_at_is_recent(self, tmp_path: Path):
        before = datetime.now(timezone.utc)
        hb = update_heartbeat(tmp_path, CID, "idle", [], pid=1)
        after = datetime.now(timezone.utc)
        assert before <= hb.updated_at <= after

    def test_with_last_report_at(self, tmp_path: Path):
        report_time = _ts(5)
        hb = update_heartbeat(
            tmp_path, CID, "running", ["x"], pid=10, last_report_at=report_time,
        )
        assert hb.last_report_at == report_time

    def test_overwrites_previous(self, tmp_path: Path):
        update_heartbeat(tmp_path, CID, "running", ["a"], pid=1)
        hb2 = update_heartbeat(tmp_path, CID, "idle", ["b"], pid=2)
        assert hb2.status == "idle"
        assert hb2.manager_pid == 2


# ---------------------------------------------------------------------------
# is_heartbeat_stale – boundary tests
# ---------------------------------------------------------------------------

class TestIsHeartbeatStale:
    def test_exactly_20_minutes_is_stale(self):
        """Exactly 20 minutes elapsed → stale (>= threshold)."""
        hb = _heartbeat(updated_at=_ts(0))
        now = _ts(20)
        assert is_heartbeat_stale(hb, now) is True

    def test_19_minutes_59_seconds_is_not_stale(self):
        """19 min 59 sec elapsed → not stale."""
        hb = _heartbeat(updated_at=_ts(0))
        now = _ts(19, 59)
        assert is_heartbeat_stale(hb, now) is False

    def test_20_minutes_1_second_is_stale(self):
        """20 min 1 sec elapsed → stale."""
        hb = _heartbeat(updated_at=_ts(0))
        now = _ts(20, 1)
        assert is_heartbeat_stale(hb, now) is True

    def test_zero_elapsed_is_not_stale(self):
        hb = _heartbeat(updated_at=_ts(0))
        assert is_heartbeat_stale(hb, _ts(0)) is False

    def test_custom_threshold(self):
        hb = _heartbeat(updated_at=_ts(0))
        # 5 minutes threshold, 5 min elapsed → stale
        assert is_heartbeat_stale(hb, _ts(5), threshold_minutes=5) is True
        # 4 min 59 sec → not stale
        assert is_heartbeat_stale(hb, _ts(4, 59), threshold_minutes=5) is False

    def test_large_gap_is_stale(self):
        hb = _heartbeat(updated_at=_ts(0))
        now = hb.updated_at + timedelta(hours=2)
        assert is_heartbeat_stale(hb, now) is True


# ---------------------------------------------------------------------------
# update_heartbeat_on_report (Req 3.5)
# ---------------------------------------------------------------------------

class TestUpdateHeartbeatOnReport:
    def test_sets_last_report_at(self, tmp_path: Path):
        hb = update_heartbeat_on_report(tmp_path, CID, "running", ["t1"], pid=99)
        assert hb.last_report_at is not None
        assert hb.last_report_at == hb.updated_at

    def test_updated_at_is_recent(self, tmp_path: Path):
        before = datetime.now(timezone.utc)
        hb = update_heartbeat_on_report(tmp_path, CID, "running", [], pid=1)
        after = datetime.now(timezone.utc)
        assert before <= hb.updated_at <= after
        assert before <= hb.last_report_at <= after

    def test_file_is_persisted(self, tmp_path: Path):
        update_heartbeat_on_report(tmp_path, CID, "idle", [], pid=5)
        assert heartbeat_path(tmp_path, CID).exists()
