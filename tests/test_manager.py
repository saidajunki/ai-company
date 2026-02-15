"""Unit tests for Manager orchestration layer (Task 13.1)."""

from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest

from models import (
    ConstitutionModel,
    DecisionLogEntry,
    HeartbeatState,
    LedgerEvent,
)
from manager_state import (
    append_ledger_event,
    save_heartbeat,
    restore_state,
)
from manager import Manager, init_company_directory


CID = "test-co"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minute: int = 0) -> datetime:
    return datetime(2025, 6, 1, 12, minute, tzinfo=timezone.utc)


def _llm_event(minute: int = 0, cost: float = 1.0) -> LedgerEvent:
    return LedgerEvent(
        timestamp=_ts(minute),
        event_type="llm_call",
        agent_id="mgr",
        task_id="t-1",
        provider="openrouter",
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        unit_price_usd_per_1k_input_tokens=0.01,
        unit_price_usd_per_1k_output_tokens=0.03,
        price_retrieved_at=_ts(),
        estimated_cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# init_company_directory
# ---------------------------------------------------------------------------

class TestInitCompanyDirectory:
    def test_creates_all_subdirectories(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        root = tmp_path / "companies" / CID

        expected_dirs = [
            "ledger", "decisions", "state", "pricing",
            "templates", "schemas", "protocols",
        ]
        for d in expected_dirs:
            assert (root / d).is_dir(), f"Missing directory: {d}"

    def test_creates_default_constitution(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        constitution_file = tmp_path / "companies" / CID / "constitution.yaml"
        assert constitution_file.exists()

    def test_does_not_overwrite_existing_constitution(self, tmp_path: Path):
        # Create directory and constitution first
        init_company_directory(tmp_path, CID)
        constitution_file = tmp_path / "companies" / CID / "constitution.yaml"
        original_content = constitution_file.read_text()

        # Write custom content
        constitution_file.write_text("custom: true\n")

        # Re-init should not overwrite
        init_company_directory(tmp_path, CID)
        assert constitution_file.read_text() == "custom: true\n"

    def test_idempotent(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        init_company_directory(tmp_path, CID)  # no error
        root = tmp_path / "companies" / CID
        assert (root / "constitution.yaml").exists()


# ---------------------------------------------------------------------------
# Manager.__init__ and startup
# ---------------------------------------------------------------------------

class TestManagerStartup:
    def test_startup_from_empty_directory(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        action, desc = mgr.startup()

        # No WIP, no pending approvals → consult creator
        assert action == "consult_creator"
        assert mgr.state.heartbeat is not None
        assert mgr.state.heartbeat.status == "idle"

    def test_startup_restores_existing_state(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)

        # Pre-populate heartbeat with WIP
        hb = HeartbeatState(
            updated_at=_ts(),
            manager_pid=999,
            status="running",
            current_wip=["task-A"],
        )
        save_heartbeat(tmp_path, CID, hb)

        mgr = Manager(tmp_path, CID)
        action, desc = mgr.startup()

        assert action == "resume_wip"
        assert "task-A" in desc

    def test_startup_updates_heartbeat(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.startup()

        assert mgr.state.heartbeat is not None
        # Heartbeat should be very recent
        elapsed = datetime.now(timezone.utc) - mgr.state.heartbeat.updated_at
        assert elapsed.total_seconds() < 5

    def test_startup_with_pending_approvals(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)

        # Add a proposed decision
        from manager_state import append_decision
        entry = DecisionLogEntry(
            date=date(2025, 6, 1),
            decision="テスト提案",
            why="テスト",
            scope="全体",
            revisit="1週間後",
            status="proposed",
            request_id="req-123",
        )
        append_decision(tmp_path, CID, entry)

        mgr = Manager(tmp_path, CID)
        action, desc = mgr.startup()

        assert action == "report_pending_approvals"


# ---------------------------------------------------------------------------
# Budget check
# ---------------------------------------------------------------------------

class TestCheckBudget:
    def test_budget_not_exceeded_empty_ledger(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.check_budget() is False

    def test_budget_exceeded_with_events(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)

        # Add events totalling $11 within the last 60 minutes
        now = datetime.now(timezone.utc)
        for i in range(11):
            event = LedgerEvent(
                timestamp=now - timedelta(minutes=i),
                event_type="llm_call",
                agent_id="mgr",
                task_id=f"t-{i}",
                provider="openrouter",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                unit_price_usd_per_1k_input_tokens=0.01,
                unit_price_usd_per_1k_output_tokens=0.03,
                price_retrieved_at=now,
                estimated_cost_usd=1.0,
            )
            append_ledger_event(tmp_path, CID, event)

        mgr = Manager(tmp_path, CID)
        assert mgr.check_budget() is True

    def test_budget_not_exceeded_old_events(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)

        # Add expensive events but >60 minutes ago
        old_time = datetime.now(timezone.utc) - timedelta(minutes=120)
        for i in range(15):
            event = LedgerEvent(
                timestamp=old_time,
                event_type="llm_call",
                agent_id="mgr",
                task_id=f"t-{i}",
                provider="openrouter",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                unit_price_usd_per_1k_input_tokens=0.01,
                unit_price_usd_per_1k_output_tokens=0.03,
                price_retrieved_at=old_time,
                estimated_cost_usd=1.0,
            )
            append_ledger_event(tmp_path, CID, event)

        mgr = Manager(tmp_path, CID)
        assert mgr.check_budget() is False


# ---------------------------------------------------------------------------
# LLM call recording
# ---------------------------------------------------------------------------

class TestRecordLlmCall:
    def test_records_event_to_ledger(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)

        event = mgr.record_llm_call(
            provider="openrouter",
            model="test-model",
            input_tokens=1000,
            output_tokens=500,
            task_id="task-1",
        )

        assert event.event_type == "llm_call"
        assert event.input_tokens == 1000
        assert event.output_tokens == 500
        assert event.estimated_cost_usd is not None
        assert event.estimated_cost_usd > 0

        # Verify it's in memory
        assert len(mgr.state.ledger_events) == 1

        # Verify it's persisted
        restored = restore_state(tmp_path, CID)
        assert len(restored.ledger_events) == 1

    def test_records_fallback_metadata(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        # No pricing cache → fallback_default
        event = mgr.record_llm_call(
            provider="openrouter",
            model="unknown-model",
            input_tokens=100,
            output_tokens=50,
        )

        assert event.metadata is not None
        assert event.metadata["pricing_source"] == "fallback_default"

    def test_multiple_calls_accumulate(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)

        mgr.record_llm_call(
            provider="openrouter", model="m1",
            input_tokens=100, output_tokens=50,
        )
        mgr.record_llm_call(
            provider="openrouter", model="m2",
            input_tokens=200, output_tokens=100,
        )

        assert len(mgr.state.ledger_events) == 2


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_contains_required_sections(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        report = mgr.generate_report()

        required_sections = [
            "WIP", "Δ(10m)", "Next(10m)", "Blockers", "Cost(60m)", "Approvals",
        ]
        for section in required_sections:
            assert section in report, f"Missing section: {section}"

    def test_report_includes_company_id(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        report = mgr.generate_report()
        assert CID in report

    def test_report_updates_heartbeat(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.generate_report()

        assert mgr.state.heartbeat is not None
        assert mgr.state.heartbeat.last_report_at is not None
        # last_report_at should be very recent
        elapsed = datetime.now(timezone.utc) - mgr.state.heartbeat.last_report_at
        assert elapsed.total_seconds() < 5

    def test_report_cost_reflects_ledger(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)

        # Add a recent event
        now = datetime.now(timezone.utc)
        event = LedgerEvent(
            timestamp=now - timedelta(minutes=5),
            event_type="llm_call",
            agent_id="mgr",
            task_id="t-1",
            provider="openrouter",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            unit_price_usd_per_1k_input_tokens=0.01,
            unit_price_usd_per_1k_output_tokens=0.03,
            price_retrieved_at=now,
            estimated_cost_usd=2.50,
        )
        append_ledger_event(tmp_path, CID, event)

        mgr = Manager(tmp_path, CID)
        report = mgr.generate_report()

        assert "$2.50" in report
