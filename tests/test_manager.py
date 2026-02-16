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


# ---------------------------------------------------------------------------
# WIP management (Task 6.2)
# ---------------------------------------------------------------------------

class TestAddWip:
    def test_add_single_task(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.add_wip("task-a") is True
        assert mgr.state.wip == ["task-a"]

    def test_add_up_to_limit(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.add_wip("task-a") is True
        assert mgr.add_wip("task-b") is True
        assert mgr.add_wip("task-c") is True
        assert len(mgr.state.wip) == 3

    def test_add_beyond_limit_returns_false(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.add_wip("task-a")
        mgr.add_wip("task-b")
        mgr.add_wip("task-c")
        assert mgr.add_wip("task-d") is False
        assert len(mgr.state.wip) == 3
        assert "task-d" not in mgr.state.wip

    def test_add_preserves_order(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.add_wip("first")
        mgr.add_wip("second")
        assert mgr.state.wip == ["first", "second"]


class TestRemoveWip:
    def test_remove_existing_task(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.add_wip("task-a")
        assert mgr.remove_wip("task-a") is True
        assert mgr.state.wip == []

    def test_remove_nonexistent_returns_false(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.remove_wip("no-such-task") is False

    def test_remove_from_empty_returns_false(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.remove_wip("anything") is False

    def test_remove_allows_adding_again(self, tmp_path: Path):
        """After removing a task from a full WIP, a new task can be added."""
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.add_wip("a")
        mgr.add_wip("b")
        mgr.add_wip("c")
        assert mgr.add_wip("d") is False  # full
        mgr.remove_wip("b")
        assert mgr.add_wip("d") is True
        assert "d" in mgr.state.wip
        assert "b" not in mgr.state.wip


# ---------------------------------------------------------------------------
# Conversation memory integration (Task 2.5)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
from llm_client import LLMResponse
from models import ConversationEntry


def _make_manager_with_llm(tmp_path: Path) -> Manager:
    """Create a Manager with a mocked LLM client that returns a simple response."""
    init_company_directory(tmp_path, CID)
    mgr = Manager(tmp_path, CID)
    mock_llm = MagicMock()
    mock_llm.chat.return_value = LLMResponse(
        content="<reply>テスト応答</reply>",
        input_tokens=100,
        output_tokens=50,
        model="test-model",
        finish_reason="stop",
    )
    mgr.llm_client = mock_llm
    mgr.slack = MagicMock()
    return mgr


class TestProcessMessageConversationMemory:
    def test_saves_user_message(self, tmp_path: Path):
        mgr = _make_manager_with_llm(tmp_path)
        mgr.process_message("こんにちは", user_id="U123")

        entries = mgr.conversation_memory.recent()
        user_entries = [e for e in entries if e.role == "user"]
        assert len(user_entries) == 1
        assert user_entries[0].content == "こんにちは"
        assert user_entries[0].user_id == "U123"

    def test_saves_assistant_response(self, tmp_path: Path):
        mgr = _make_manager_with_llm(tmp_path)
        mgr.process_message("テスト", user_id="U123")

        entries = mgr.conversation_memory.recent()
        assistant_entries = [e for e in entries if e.role == "assistant"]
        assert len(assistant_entries) == 1
        assert "<reply>テスト応答</reply>" in assistant_entries[0].content

    def test_saves_both_user_and_assistant(self, tmp_path: Path):
        mgr = _make_manager_with_llm(tmp_path)
        mgr.process_message("質問です", user_id="U456")

        entries = mgr.conversation_memory.recent()
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[1].role == "assistant"

    def test_passes_conversation_history_to_prompt(self, tmp_path: Path):
        mgr = _make_manager_with_llm(tmp_path)

        # First message populates history
        mgr.process_message("最初のメッセージ", user_id="U123")

        # Second message should include history in the system prompt
        mgr.process_message("二番目のメッセージ", user_id="U123")

        # The LLM was called twice; check the second call's system prompt
        calls = mgr.llm_client.chat.call_args_list
        assert len(calls) == 2
        second_call_messages = calls[1][0][0]
        system_prompt = second_call_messages[0]["content"]
        # History from first message should be in the prompt
        assert "最初のメッセージ" in system_prompt

    def test_conversation_memory_failure_does_not_break_processing(self, tmp_path: Path):
        mgr = _make_manager_with_llm(tmp_path)

        # Break the conversation memory append
        mgr.conversation_memory.append = MagicMock(side_effect=OSError("disk full"))

        # Should still process the message without raising
        mgr.process_message("テスト", user_id="U123")

        # LLM was still called
        assert mgr.llm_client.chat.called

    def test_task_id_set_on_entries(self, tmp_path: Path):
        mgr = _make_manager_with_llm(tmp_path)
        mgr.process_message("テスト", user_id="U123")

        entries = mgr.conversation_memory.recent()
        for entry in entries:
            assert entry.task_id is not None
            assert entry.task_id.startswith("msg-")


# ---------------------------------------------------------------------------
# Research action integration (Task 6.4)
# ---------------------------------------------------------------------------

from web_searcher import SearchResult


def _make_manager_with_research_llm(tmp_path: Path, responses: list[LLMResponse]) -> Manager:
    """Create a Manager with a mocked LLM that returns sequential responses."""
    init_company_directory(tmp_path, CID)
    mgr = Manager(tmp_path, CID)
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = responses
    mgr.llm_client = mock_llm
    mgr.slack = MagicMock()
    return mgr


class TestResearchActionIntegration:
    """Tests for research action handling in Manager._execute_action_loop()."""

    def test_research_action_calls_web_searcher(self, tmp_path: Path):
        """Research action triggers WebSearcher.search() with the query."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            # Initial LLM response with research tag
            LLMResponse(
                content="<research>AI最新ニュース</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            # Follow-up after research results
            LLMResponse(
                content="<reply>調査完了</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = [
            SearchResult(title="AI News", url="https://example.com/ai", snippet="Latest AI news"),
        ]

        mgr.process_message("AIについて調べて", user_id="U123")

        mgr.web_searcher.search.assert_called_once_with("AI最新ニュース")

    def test_research_action_saves_notes(self, tmp_path: Path):
        """Research results are saved as ResearchNotes via the store."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            LLMResponse(
                content="<research>Python trends</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            LLMResponse(
                content="<reply>結果をまとめました</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = [
            SearchResult(title="Python 2025", url="https://example.com/py", snippet="Python trends"),
            SearchResult(title="FastAPI News", url="https://example.com/fa", snippet="FastAPI update"),
        ]

        mgr.process_message("Pythonのトレンドを調べて", user_id="U123")

        notes = mgr.research_note_store.load_all()
        assert len(notes) == 2
        assert notes[0].query == "Python trends"
        assert notes[0].source_url == "https://example.com/py"
        assert notes[1].title == "FastAPI News"

    def test_research_action_requeires_llm(self, tmp_path: Path):
        """After research, LLM is re-queried with the results summary."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            LLMResponse(
                content="<research>test query</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            LLMResponse(
                content="<reply>了解</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = [
            SearchResult(title="Result 1", url="https://example.com", snippet="Snippet 1"),
        ]

        mgr.process_message("調べて", user_id="U123")

        # LLM called twice: initial + follow-up after research
        assert mgr.llm_client.chat.call_count == 2
        # Second call should include research results in conversation
        second_call_messages = mgr.llm_client.chat.call_args_list[1][0][0]
        research_msg = [m for m in second_call_messages if m["role"] == "user" and "リサーチ結果" in m["content"]]
        assert len(research_msg) == 1
        assert "Result 1" in research_msg[0]["content"]

    def test_research_action_empty_results(self, tmp_path: Path):
        """When search returns no results, LLM still gets re-queried."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            LLMResponse(
                content="<research>obscure query</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            LLMResponse(
                content="<reply>見つかりませんでした</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = []

        mgr.process_message("調べて", user_id="U123")

        assert mgr.llm_client.chat.call_count == 2
        second_call_messages = mgr.llm_client.chat.call_args_list[1][0][0]
        research_msg = [m for m in second_call_messages if m["role"] == "user" and "検索結果なし" in m["content"]]
        assert len(research_msg) == 1

    def test_research_action_records_llm_cost(self, tmp_path: Path):
        """Follow-up LLM call after research is recorded in the ledger."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            LLMResponse(
                content="<research>cost test</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            LLMResponse(
                content="<reply>done</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = []

        mgr.process_message("テスト", user_id="U123")

        # Two LLM calls recorded in ledger
        llm_events = [e for e in mgr.state.ledger_events if e.event_type == "llm_call"]
        assert len(llm_events) == 2

    def test_research_notes_passed_to_system_prompt(self, tmp_path: Path):
        """Recent research notes are included in the system prompt."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            # First call: research
            LLMResponse(
                content="<research>first query</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            # Follow-up after research
            LLMResponse(
                content="<reply>ok</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
            # Second process_message call
            LLMResponse(
                content="<reply>参考にしました</reply>",
                input_tokens=150, output_tokens=60,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = [
            SearchResult(title="Saved Note", url="https://example.com/saved", snippet="Important info"),
        ]

        # First message triggers research and saves notes
        mgr.process_message("調べて", user_id="U123")

        # Second message should have research notes in system prompt
        mgr.process_message("結果を教えて", user_id="U123")

        third_call_messages = mgr.llm_client.chat.call_args_list[2][0][0]
        system_prompt = third_call_messages[0]["content"]
        assert "https://example.com/saved" in system_prompt

    def test_research_note_save_failure_does_not_break_flow(self, tmp_path: Path):
        """If saving a research note fails, processing continues."""
        mgr = _make_manager_with_research_llm(tmp_path, [
            LLMResponse(
                content="<research>fail save test</research>",
                input_tokens=100, output_tokens=50,
                model="test-model", finish_reason="stop",
            ),
            LLMResponse(
                content="<reply>ok</reply>",
                input_tokens=200, output_tokens=80,
                model="test-model", finish_reason="stop",
            ),
        ])
        mgr.web_searcher = MagicMock()
        mgr.web_searcher.search.return_value = [
            SearchResult(title="Test", url="https://example.com", snippet="test"),
        ]
        mgr.research_note_store = MagicMock()
        mgr.research_note_store.save.side_effect = OSError("disk full")
        mgr.research_note_store.recent.return_value = []

        # Should not raise
        mgr.process_message("テスト", user_id="U123")

        # LLM was still re-queried
        assert mgr.llm_client.chat.call_count == 2


# ---------------------------------------------------------------------------
# Recovery planner integration in startup (Task 6.4)
# ---------------------------------------------------------------------------

from unittest.mock import patch


class TestStartupRecoveryPlanner:
    """Tests for Manager.startup() calling RecoveryPlanner on consult_creator."""

    def test_startup_consult_creator_calls_recovery_planner(self, tmp_path: Path):
        """When startup action is consult_creator, recovery_planner.handle_idle() is called."""
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.recovery_planner = MagicMock()
        mgr.recovery_planner.handle_idle.return_value = "自律的にイニシアチブを計画しました"

        action, desc = mgr.startup()

        assert action == "consult_creator"
        assert desc == "自律的にイニシアチブを計画しました"
        mgr.recovery_planner.handle_idle.assert_called_once()

    def test_startup_recovery_planner_exception_falls_back(self, tmp_path: Path):
        """If recovery_planner.handle_idle() raises, startup falls back to original description."""
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.recovery_planner = MagicMock()
        mgr.recovery_planner.handle_idle.side_effect = RuntimeError("boom")

        action, desc = mgr.startup()

        assert action == "consult_creator"
        # Falls back to the original recovery description (not the exception)
        assert "boom" not in desc

    def test_startup_resume_wip_does_not_call_recovery_planner(self, tmp_path: Path):
        """When action is resume_wip, recovery_planner should NOT be called."""
        init_company_directory(tmp_path, CID)

        hb = HeartbeatState(
            updated_at=_ts(),
            manager_pid=999,
            status="running",
            current_wip=["task-A"],
        )
        save_heartbeat(tmp_path, CID, hb)

        mgr = Manager(tmp_path, CID)
        mgr.recovery_planner = MagicMock()

        action, desc = mgr.startup()

        assert action == "resume_wip"
        mgr.recovery_planner.handle_idle.assert_not_called()


# ---------------------------------------------------------------------------
# Initiative info in process_message (Task 6.4)
# ---------------------------------------------------------------------------


class TestProcessMessageInitiativeContext:
    """Tests for initiative info being passed to build_system_prompt in process_message."""

    def test_initiative_info_passed_to_system_prompt(self, tmp_path: Path):
        """process_message should include active initiatives and strategy in system prompt."""
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="<reply>了解</reply>",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        mgr.llm_client = mock_llm
        mgr.slack = MagicMock()

        mgr.process_message("テスト", user_id="U123")

        # Verify system prompt contains initiative section
        call_args = mock_llm.chat.call_args[0][0]
        system_prompt = call_args[0]["content"]
        # With no active initiatives, should show the empty message
        assert "アクティブなイニシアチブなし" in system_prompt

    def test_initiative_info_includes_active_initiatives(self, tmp_path: Path):
        """When initiatives exist, they should appear in the system prompt."""
        from models import InitiativeEntry
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="<reply>了解</reply>",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        mgr.llm_client = mock_llm
        mgr.slack = MagicMock()

        # Save an in_progress initiative
        now = datetime.now(timezone.utc)
        entry = InitiativeEntry(
            initiative_id="init-001",
            title="テストイニシアチブ",
            description="テスト用の施策",
            status="in_progress",
            created_at=now,
            updated_at=now,
        )
        mgr.initiative_store.save(entry)

        mgr.process_message("テスト", user_id="U123")

        call_args = mock_llm.chat.call_args[0][0]
        system_prompt = call_args[0]["content"]
        assert "テストイニシアチブ" in system_prompt

    def test_strategy_direction_in_system_prompt(self, tmp_path: Path):
        """Strategy direction should appear in the system prompt."""
        from models import CreatorReview
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="<reply>了解</reply>",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        mgr.llm_client = mock_llm
        mgr.slack = MagicMock()

        # Save a creator review so strategy analyzer has data
        now = datetime.now(timezone.utc)
        review = CreatorReview(
            timestamp=now,
            user_id="U123",
            score_total_100=80,
            score_interestingness_25=22,
            score_cost_efficiency_25=18,
            score_realism_25=20,
            score_evolvability_25=20,
        )
        mgr.creator_review_store.save(review)

        mgr.process_message("テスト", user_id="U123")

        call_args = mock_llm.chat.call_args[0][0]
        system_prompt = call_args[0]["content"]
        # Strategy section should be present (not "戦略方針未設定")
        assert "戦略方針" in system_prompt

    def test_initiative_load_failure_does_not_break_processing(self, tmp_path: Path):
        """If loading initiatives fails, process_message should still work."""
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="<reply>了解</reply>",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        mgr.llm_client = mock_llm
        mgr.slack = MagicMock()

        # Break the initiative store
        mgr.initiative_store = MagicMock()
        mgr.initiative_store.list_by_status.side_effect = OSError("disk error")

        # Should not raise
        mgr.process_message("テスト", user_id="U123")
        assert mock_llm.chat.call_count == 1
