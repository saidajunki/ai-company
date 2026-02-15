"""Unit tests for Manager state persistence and recovery."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from models import (
    ConstitutionModel,
    DecisionLogEntry,
    HeartbeatState,
    LedgerEvent,
)
from manager_state import (
    ManagerState,
    append_decision,
    append_ledger_event,
    constitution_path,
    decision_log_path,
    heartbeat_path,
    ledger_path,
    persist_state,
    restore_state,
    save_heartbeat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minute: int = 0) -> datetime:
    return datetime(2025, 6, 1, 12, minute, tzinfo=timezone.utc)


def _ledger_event(**kw) -> LedgerEvent:
    defaults = dict(
        timestamp=_ts(),
        event_type="shell_exec",
        agent_id="mgr",
        task_id="t-1",
    )
    defaults.update(kw)
    return LedgerEvent(**defaults)


def _decision(**kw) -> DecisionLogEntry:
    defaults = dict(
        date=date(2025, 6, 1),
        decision="テスト決定",
        why="テスト理由",
        scope="全体",
        revisit="1週間後",
    )
    defaults.update(kw)
    return DecisionLogEntry(**defaults)


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
# Path helpers
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_ledger_path(self, tmp_path: Path):
        p = ledger_path(tmp_path, CID)
        assert p == tmp_path / "companies" / CID / "ledger" / "events.ndjson"

    def test_decision_log_path(self, tmp_path: Path):
        p = decision_log_path(tmp_path, CID)
        assert p == tmp_path / "companies" / CID / "decisions" / "log.ndjson"

    def test_heartbeat_path(self, tmp_path: Path):
        p = heartbeat_path(tmp_path, CID)
        assert p == tmp_path / "companies" / CID / "state" / "heartbeat.json"

    def test_constitution_path(self, tmp_path: Path):
        p = constitution_path(tmp_path, CID)
        assert p == tmp_path / "companies" / CID / "constitution.yaml"


# ---------------------------------------------------------------------------
# Restore from empty directory
# ---------------------------------------------------------------------------

class TestRestoreEmpty:
    def test_restore_from_empty_dir(self, tmp_path: Path):
        state = restore_state(tmp_path, CID)
        assert state.ledger_events == []
        assert state.decision_log == []
        assert state.constitution is None
        assert state.heartbeat is None
        assert state.wip == []


# ---------------------------------------------------------------------------
# Full round-trip: persist → restore
# ---------------------------------------------------------------------------

class TestPersistRestoreRoundTrip:
    def test_full_round_trip(self, tmp_path: Path):
        original = ManagerState(
            wip=["task-A", "task-B"],
            ledger_events=[_ledger_event(task_id="t-1"), _ledger_event(task_id="t-2")],
            decision_log=[_decision(decision="d1"), _decision(decision="d2")],
            constitution=ConstitutionModel(version=3, purpose="テスト組織"),
            heartbeat=_heartbeat(current_wip=["task-A", "task-B"]),
        )

        persist_state(tmp_path, CID, original)
        restored = restore_state(tmp_path, CID)

        assert restored.ledger_events == original.ledger_events
        assert restored.decision_log == original.decision_log
        assert restored.constitution == original.constitution
        assert restored.heartbeat == original.heartbeat
        # WIP is derived from heartbeat.current_wip
        assert restored.wip == ["task-A", "task-B"]

    def test_round_trip_empty_state(self, tmp_path: Path):
        original = ManagerState()
        persist_state(tmp_path, CID, original)
        restored = restore_state(tmp_path, CID)

        assert restored.ledger_events == []
        assert restored.decision_log == []
        assert restored.constitution is None
        assert restored.heartbeat is None

    def test_ledger_append_only_grows(self, tmp_path: Path):
        """Ledger line count is monotonically increasing (Req 6.2)."""
        append_ledger_event(tmp_path, CID, _ledger_event(task_id="t-1"))
        s1 = restore_state(tmp_path, CID)
        assert len(s1.ledger_events) == 1

        append_ledger_event(tmp_path, CID, _ledger_event(task_id="t-2"))
        s2 = restore_state(tmp_path, CID)
        assert len(s2.ledger_events) == 2
        assert len(s2.ledger_events) > len(s1.ledger_events)


# ---------------------------------------------------------------------------
# Partial state recovery
# ---------------------------------------------------------------------------

class TestPartialRecovery:
    def test_only_ledger_exists(self, tmp_path: Path):
        append_ledger_event(tmp_path, CID, _ledger_event())
        state = restore_state(tmp_path, CID)

        assert len(state.ledger_events) == 1
        assert state.decision_log == []
        assert state.constitution is None
        assert state.heartbeat is None

    def test_only_decisions_exist(self, tmp_path: Path):
        append_decision(tmp_path, CID, _decision())
        state = restore_state(tmp_path, CID)

        assert state.ledger_events == []
        assert len(state.decision_log) == 1
        assert state.constitution is None

    def test_only_heartbeat_exists(self, tmp_path: Path):
        save_heartbeat(tmp_path, CID, _heartbeat(current_wip=["x"]))
        state = restore_state(tmp_path, CID)

        assert state.ledger_events == []
        assert state.heartbeat is not None
        assert state.wip == ["x"]

    def test_ledger_and_heartbeat_no_constitution(self, tmp_path: Path):
        append_ledger_event(tmp_path, CID, _ledger_event())
        save_heartbeat(tmp_path, CID, _heartbeat())
        state = restore_state(tmp_path, CID)

        assert len(state.ledger_events) == 1
        assert state.heartbeat is not None
        assert state.constitution is None


# ---------------------------------------------------------------------------
# Incremental helpers
# ---------------------------------------------------------------------------

class TestIncrementalHelpers:
    def test_append_ledger_event_creates_file(self, tmp_path: Path):
        append_ledger_event(tmp_path, CID, _ledger_event())
        assert ledger_path(tmp_path, CID).exists()

    def test_append_decision_creates_file(self, tmp_path: Path):
        append_decision(tmp_path, CID, _decision())
        assert decision_log_path(tmp_path, CID).exists()

    def test_save_heartbeat_creates_file(self, tmp_path: Path):
        save_heartbeat(tmp_path, CID, _heartbeat())
        assert heartbeat_path(tmp_path, CID).exists()

    def test_save_heartbeat_overwrites(self, tmp_path: Path):
        save_heartbeat(tmp_path, CID, _heartbeat(status="running"))
        save_heartbeat(tmp_path, CID, _heartbeat(status="idle"))
        state = restore_state(tmp_path, CID)
        assert state.heartbeat is not None
        assert state.heartbeat.status == "idle"
