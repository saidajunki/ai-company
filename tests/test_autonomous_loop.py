"""Unit tests for AutonomousLoop (Task 10.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autonomous_loop import AutonomousLoop, DEFAULT_WIP_LIMIT
from llm_client import LLMError, LLMResponse
from manager import Manager, init_company_directory
from models import TaskEntry
from task_queue import TaskQueue
from vision_loader import VisionLoader


CID = "test-co"


def _make_manager(tmp_path: Path) -> Manager:
    """Create a Manager with mocked components for autonomous loop testing."""
    init_company_directory(tmp_path, CID)
    mgr = Manager(tmp_path, CID)

    # TaskQueue (real)
    mgr.task_queue = TaskQueue(tmp_path, CID)

    # VisionLoader (real)
    mgr.vision_loader = VisionLoader(tmp_path, CID)

    # Mocked LLM client
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.chat.return_value = LLMResponse(
        content="<done>完了</done>",
        input_tokens=100,
        output_tokens=50,
        model="test-model",
        finish_reason="stop",
    )
    mgr.llm_client = mock_llm

    # Mocked Slack
    mgr.slack = MagicMock()

    return mgr


# ---------------------------------------------------------------------------
# tick() — budget exceeded skips
# ---------------------------------------------------------------------------

class TestTickBudgetExceeded:
    def test_tick_skips_when_budget_exceeded(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.task_queue.add("some task")
        loop = AutonomousLoop(mgr)

        with patch.object(mgr, "check_budget", return_value=True):
            loop.tick()

        # Task should remain pending — not executed
        tasks = mgr.task_queue.list_by_status("pending")
        assert len(tasks) == 1
        assert mgr.llm_client.chat.call_count == 0

    def test_tick_proceeds_when_budget_ok(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.task_queue.add("some task")
        loop = AutonomousLoop(mgr)

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Task should have been executed
        assert mgr.llm_client.chat.call_count >= 1


# ---------------------------------------------------------------------------
# tick() — WIP full skips
# ---------------------------------------------------------------------------

class TestTickWipFull:
    def test_tick_skips_when_wip_full(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Add running tasks up to WIP limit
        for i in range(DEFAULT_WIP_LIMIT):
            t = mgr.task_queue.add(f"running task {i}")
            mgr.task_queue.update_status(t.task_id, "running")

        # Add a pending task
        mgr.task_queue.add("pending task")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Pending task should remain pending
        pending = mgr.task_queue.list_by_status("pending")
        assert len(pending) == 1
        assert mgr.llm_client.chat.call_count == 0

    def test_tick_proceeds_when_wip_has_space(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Only 1 running task (under limit)
        t = mgr.task_queue.add("running task")
        mgr.task_queue.update_status(t.task_id, "running")

        mgr.task_queue.add("pending task")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        assert mgr.llm_client.chat.call_count >= 1


# ---------------------------------------------------------------------------
# tick() — executes pending task
# ---------------------------------------------------------------------------

class TestTickExecutesPending:
    def test_tick_executes_pending_task(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        mgr.task_queue.add("do something")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Task should be completed
        completed = mgr.task_queue.list_by_status("completed")
        assert len(completed) == 1
        assert completed[0].description == "do something"

    def test_tick_reports_completion_to_slack(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        mgr.task_queue.add("report task")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Slack should have been called
        assert mgr.slack.send_message.call_count >= 1

    def test_tick_handles_llm_error(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.llm_client.chat.return_value = LLMError(
            error_type="api_error",
            message="API failure",
        )
        loop = AutonomousLoop(mgr)

        mgr.task_queue.add("failing task")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Task should be failed
        failed = mgr.task_queue.list_by_status("failed")
        assert len(failed) == 1


# ---------------------------------------------------------------------------
# _propose_tasks — when no pending tasks
# ---------------------------------------------------------------------------

class TestProposeTasks:
    def test_propose_tasks_when_no_pending(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.initiative_planner = None
        mgr.llm_client.chat.return_value = LLMResponse(
            content="- OSS調査を行う\n- プロトタイプを作成する\n",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        loop = AutonomousLoop(mgr)

        tasks = loop._propose_tasks()
        assert len(tasks) == 2
        descriptions = [t.description for t in tasks]
        assert "OSS調査を行う" in descriptions
        assert "プロトタイプを作成する" in descriptions

    def test_propose_tasks_returns_empty_on_llm_error(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.initiative_planner = None
        mgr.llm_client.chat.return_value = LLMError(
            error_type="api_error",
            message="fail",
        )
        loop = AutonomousLoop(mgr)

        tasks = loop._propose_tasks()
        assert tasks == []

    def test_propose_tasks_returns_empty_without_llm(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.initiative_planner = None
        mgr.llm_client = None
        loop = AutonomousLoop(mgr)

        tasks = loop._propose_tasks()
        assert tasks == []

    def test_tick_proposes_when_no_pending(self, tmp_path: Path):
        """tick() should propose tasks and then execute one when queue is empty."""
        mgr = _make_manager(tmp_path)
        # Disable initiative planner so the LLM fallback path is tested
        mgr.initiative_planner = None
        call_count = 0

        def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: propose tasks
                return LLMResponse(
                    content="- 新しいタスク\n",
                    input_tokens=50,
                    output_tokens=30,
                    model="test-model",
                    finish_reason="stop",
                )
            # Subsequent calls: execute task
            return LLMResponse(
                content="<done>実行完了</done>",
                input_tokens=50,
                output_tokens=30,
                model="test-model",
                finish_reason="stop",
            )

        mgr.llm_client.chat.side_effect = mock_chat
        loop = AutonomousLoop(mgr)

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Should have proposed and then executed
        completed = mgr.task_queue.list_by_status("completed")
        assert len(completed) == 1


# ---------------------------------------------------------------------------
# _execute_task — shell command handling
# ---------------------------------------------------------------------------

class TestExecuteTaskShell:
    def test_shell_command_triggers_followup(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        call_count = 0

        def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="<shell>echo hello</shell>",
                    input_tokens=50,
                    output_tokens=30,
                    model="test-model",
                    finish_reason="stop",
                )
            return LLMResponse(
                content="<done>シェル実行完了</done>",
                input_tokens=50,
                output_tokens=30,
                model="test-model",
                finish_reason="stop",
            )

        mgr.llm_client.chat.side_effect = mock_chat
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("echo test")

        with patch.object(mgr, "check_budget", return_value=False):
            loop._execute_task(task)

        completed = mgr.task_queue.list_by_status("completed")
        assert len(completed) == 1
        assert call_count == 2


# ---------------------------------------------------------------------------
# _pick_task
# ---------------------------------------------------------------------------

class TestPickTask:
    def test_pick_returns_highest_priority(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        mgr.task_queue.add("low priority", priority=5)
        mgr.task_queue.add("high priority", priority=1)

        task = loop._pick_task()
        assert task is not None
        assert task.description == "high priority"

    def test_pick_returns_none_when_empty(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        task = loop._pick_task()
        assert task is None


# ---------------------------------------------------------------------------
# WIP limit from constitution
# ---------------------------------------------------------------------------

class TestWipLimitFromConstitution:
    def test_uses_constitution_wip_limit(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Set custom WIP limit via constitution
        mgr.state.constitution.work_principles.wip_limit = 1

        # Add 1 running task
        t = mgr.task_queue.add("running")
        mgr.task_queue.update_status(t.task_id, "running")

        mgr.task_queue.add("pending")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Should skip because WIP limit is 1 and 1 task is running
        pending = mgr.task_queue.list_by_status("pending")
        assert len(pending) == 1
        assert mgr.llm_client.chat.call_count == 0


# ---------------------------------------------------------------------------
# _propose_tasks — InitiativePlanner integration (Task 6.3)
# ---------------------------------------------------------------------------

class TestProposeTasksWithInitiativePlanner:
    """Tests for _propose_tasks() calling InitiativePlanner first."""

    def test_uses_initiative_planner_when_available(self, tmp_path: Path):
        """When initiative_planner is set and returns initiatives with tasks,
        _propose_tasks() should return those tasks without calling LLM."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Pre-add a task to the queue so we can reference its ID
        task_entry = mgr.task_queue.add("イニシアチブの最初の一手")

        # Create a mock initiative planner
        now = datetime.now(timezone.utc)
        mock_initiative = MagicMock()
        mock_initiative.task_ids = [task_entry.task_id]

        mock_planner = MagicMock()
        mock_planner.plan.return_value = [mock_initiative]
        mgr.initiative_planner = mock_planner

        tasks = loop._propose_tasks()

        assert len(tasks) == 1
        assert tasks[0].task_id == task_entry.task_id
        mock_planner.plan.assert_called_once()
        # LLM should NOT have been called
        assert mgr.llm_client.chat.call_count == 0

    def test_falls_back_to_llm_when_planner_returns_empty(self, tmp_path: Path):
        """When initiative_planner.plan() returns empty list,
        should fall back to LLM-based proposal."""
        mgr = _make_manager(tmp_path)
        mgr.llm_client.chat.return_value = LLMResponse(
            content="- LLMフォールバックタスク\n",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        loop = AutonomousLoop(mgr)

        mock_planner = MagicMock()
        mock_planner.plan.return_value = []
        mgr.initiative_planner = mock_planner

        tasks = loop._propose_tasks()

        assert len(tasks) == 1
        assert tasks[0].description == "LLMフォールバックタスク"
        mock_planner.plan.assert_called_once()
        assert mgr.llm_client.chat.call_count == 1

    def test_falls_back_to_llm_when_planner_raises(self, tmp_path: Path):
        """When initiative_planner.plan() raises an exception,
        should fall back to LLM-based proposal."""
        mgr = _make_manager(tmp_path)
        mgr.llm_client.chat.return_value = LLMResponse(
            content="- 例外後フォールバック\n",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        loop = AutonomousLoop(mgr)

        mock_planner = MagicMock()
        mock_planner.plan.side_effect = RuntimeError("planner broke")
        mgr.initiative_planner = mock_planner

        tasks = loop._propose_tasks()

        assert len(tasks) == 1
        assert tasks[0].description == "例外後フォールバック"

    def test_falls_back_when_no_initiative_planner_attr(self, tmp_path: Path):
        """When manager has no initiative_planner attribute,
        should use LLM-based proposal as before."""
        mgr = _make_manager(tmp_path)
        mgr.llm_client.chat.return_value = LLMResponse(
            content="- 通常のLLMタスク\n",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        loop = AutonomousLoop(mgr)

        # Ensure no initiative_planner attribute
        if hasattr(mgr, "initiative_planner"):
            delattr(mgr, "initiative_planner")

        tasks = loop._propose_tasks()

        assert len(tasks) == 1
        assert tasks[0].description == "通常のLLMタスク"

    def test_falls_back_when_tasks_not_in_queue(self, tmp_path: Path):
        """When initiative_planner returns initiatives but task_ids
        don't match any tasks in queue, should fall back to LLM."""
        mgr = _make_manager(tmp_path)
        mgr.llm_client.chat.return_value = LLMResponse(
            content="- キュー不一致フォールバック\n",
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            finish_reason="stop",
        )
        loop = AutonomousLoop(mgr)

        mock_initiative = MagicMock()
        mock_initiative.task_ids = ["nonexistent-id"]

        mock_planner = MagicMock()
        mock_planner.plan.return_value = [mock_initiative]
        mgr.initiative_planner = mock_planner

        tasks = loop._propose_tasks()

        # Should fall back to LLM
        assert len(tasks) == 1
        assert tasks[0].description == "キュー不一致フォールバック"

    def test_multiple_initiatives_multiple_tasks(self, tmp_path: Path):
        """When initiative_planner returns multiple initiatives with tasks,
        all tasks should be collected."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        task1 = mgr.task_queue.add("タスク1")
        task2 = mgr.task_queue.add("タスク2")

        ini1 = MagicMock()
        ini1.task_ids = [task1.task_id]
        ini2 = MagicMock()
        ini2.task_ids = [task2.task_id]

        mock_planner = MagicMock()
        mock_planner.plan.return_value = [ini1, ini2]
        mgr.initiative_planner = mock_planner

        tasks = loop._propose_tasks()

        assert len(tasks) == 2
        task_ids = {t.task_id for t in tasks}
        assert task1.task_id in task_ids
        assert task2.task_id in task_ids
        assert mgr.llm_client.chat.call_count == 0
