"""Unit tests for NDJSON read/write module."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from models import LedgerEvent, DecisionLogEntry
from ndjson_store import ndjson_append, ndjson_read


@pytest.fixture
def ndjson_path(tmp_path: Path) -> Path:
    return tmp_path / "test.ndjson"


def _make_ledger_event(**overrides) -> LedgerEvent:
    defaults = dict(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        event_type="shell_exec",
        agent_id="mgr-1",
        task_id="task-1",
    )
    defaults.update(overrides)
    return LedgerEvent(**defaults)


def _make_decision_entry(**overrides) -> DecisionLogEntry:
    from datetime import date

    defaults = dict(
        date=date(2025, 1, 1),
        decision="テスト決定",
        why="テスト理由",
        scope="テスト範囲",
        revisit="1週間後",
    )
    defaults.update(overrides)
    return DecisionLogEntry(**defaults)


class TestNdjsonAppend:
    def test_creates_file_if_not_exists(self, ndjson_path: Path):
        event = _make_ledger_event()
        ndjson_append(ndjson_path, event)
        assert ndjson_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "events.ndjson"
        event = _make_ledger_event()
        ndjson_append(deep_path, event)
        assert deep_path.exists()

    def test_appends_single_line(self, ndjson_path: Path):
        event = _make_ledger_event()
        ndjson_append(ndjson_path, event)
        lines = ndjson_path.read_text().splitlines()
        assert len(lines) == 1

    def test_multiple_appends_accumulate(self, ndjson_path: Path):
        for i in range(3):
            ndjson_append(ndjson_path, _make_ledger_event(task_id=f"task-{i}"))
        lines = ndjson_path.read_text().splitlines()
        assert len(lines) == 3


class TestNdjsonRead:
    def test_returns_empty_for_nonexistent_file(self, tmp_path: Path):
        result = ndjson_read(tmp_path / "missing.ndjson", LedgerEvent)
        assert result == []

    def test_skips_blank_lines(self, ndjson_path: Path):
        event = _make_ledger_event()
        ndjson_append(ndjson_path, event)
        # Insert blank lines
        with open(ndjson_path, "a") as f:
            f.write("\n\n")
        ndjson_append(ndjson_path, event)

        result = ndjson_read(ndjson_path, LedgerEvent)
        assert len(result) == 2

    def test_skips_invalid_lines(self, ndjson_path: Path):
        ndjson_append(ndjson_path, _make_ledger_event(task_id="task-ok-1"))
        # Insert an invalid JSON line
        with open(ndjson_path, "a", encoding="utf-8") as f:
            f.write("{\n")
        ndjson_append(ndjson_path, _make_ledger_event(task_id="task-ok-2"))

        result = ndjson_read(ndjson_path, LedgerEvent)
        assert [e.task_id for e in result] == ["task-ok-1", "task-ok-2"]


class TestRoundTrip:
    def test_ledger_event_round_trip(self, ndjson_path: Path):
        original = _make_ledger_event()
        ndjson_append(ndjson_path, original)
        loaded = ndjson_read(ndjson_path, LedgerEvent)
        assert len(loaded) == 1
        assert loaded[0] == original

    def test_decision_log_round_trip(self, ndjson_path: Path):
        original = _make_decision_entry(status="proposed", request_id="req-001")
        ndjson_append(ndjson_path, original)
        loaded = ndjson_read(ndjson_path, DecisionLogEntry)
        assert len(loaded) == 1
        assert loaded[0] == original

    def test_multiple_events_round_trip(self, ndjson_path: Path):
        events = [_make_ledger_event(task_id=f"task-{i}") for i in range(5)]
        for e in events:
            ndjson_append(ndjson_path, e)
        loaded = ndjson_read(ndjson_path, LedgerEvent)
        assert loaded == events
