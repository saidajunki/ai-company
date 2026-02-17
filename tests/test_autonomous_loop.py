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
            if call_count == 2:
                return LLMResponse(
                    content="<done>シェル実行完了</done>",
                    input_tokens=50,
                    output_tokens=30,
                    model="test-model",
                    finish_reason="stop",
                )
            # 3rd call: quality verification (always active when shell used)
            return LLMResponse(
                content="score: 0.9\nnotes: good",
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
        assert call_count == 3  # shell + done + quality verification


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


# ---------------------------------------------------------------------------
# _retry_failed_tasks — retry logic
# ---------------------------------------------------------------------------


class TestRetryFailedTasks:
    def test_retries_failed_task_below_max(self, tmp_path: Path):
        """retry_count < max_retries の失敗タスクはpendingに戻る."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("retryable task")
        mgr.task_queue.update_status(task.task_id, "failed", error="一時エラー")

        loop._retry_failed_tasks()

        updated = mgr.task_queue.list_by_status("pending")
        assert len(updated) == 1
        assert updated[0].task_id == task.task_id
        assert updated[0].retry_count == 1

    def test_does_not_retry_at_max_retries(self, tmp_path: Path):
        """retry_count == max_retries のタスクはpendingに戻らない."""
        mgr = _make_manager(tmp_path)
        mgr.consultation_store = MagicMock()
        mgr.consultation_store.add.return_value = MagicMock(consultation_id="abc12345")
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("exhausted task")
        # Simulate reaching max_retries (default=3)
        mgr.task_queue.update_status(task.task_id, "failed", error="永続エラー")
        for i in range(3):
            mgr.task_queue.update_status_for_retry(task.task_id, retry_count=i + 1)
            mgr.task_queue.update_status(task.task_id, "failed", error="永続エラー")

        loop._retry_failed_tasks()

        # Should still be failed, not pending
        pending = mgr.task_queue.list_by_status("pending")
        assert len(pending) == 0
        failed = mgr.task_queue.list_by_status("failed")
        assert len(failed) == 1

    def test_retry_priority_order(self, tmp_path: Path):
        """高優先度（数値が小さい）タスクが先にリトライされる."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        t_low = mgr.task_queue.add("low priority", priority=5)
        t_high = mgr.task_queue.add("high priority", priority=1)
        mgr.task_queue.update_status(t_low.task_id, "failed", error="err")
        mgr.task_queue.update_status(t_high.task_id, "failed", error="err")

        loop._retry_failed_tasks()

        pending = mgr.task_queue.list_by_status("pending")
        assert len(pending) == 2
        # Both should be retried; verify retry_count incremented
        for t in pending:
            assert t.retry_count == 1

    def test_escalates_when_max_retries_reached(self, tmp_path: Path):
        """max_retries到達タスクはCreatorにエスカレーションされる."""
        mgr = _make_manager(tmp_path)
        mock_consult = MagicMock()
        mock_consult.add.return_value = MagicMock(consultation_id="esc12345")
        mgr.consultation_store = mock_consult
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("doomed task")
        mgr.task_queue.update_status(task.task_id, "failed", error="致命的エラー")
        for i in range(3):
            mgr.task_queue.update_status_for_retry(task.task_id, retry_count=i + 1)
            mgr.task_queue.update_status(task.task_id, "failed", error="致命的エラー")

        loop._retry_failed_tasks()

        # consultation_store.add should have been called
        mock_consult.add.assert_called_once()
        call_args = mock_consult.add.call_args
        assert task.task_id in call_args[0][0]

        # エスカレーション済みマーカーが付いていること
        updated = mgr.task_queue._get_latest(task.task_id)
        assert updated.error.startswith("[escalated]")

    def test_escalated_task_not_re_escalated(self, tmp_path: Path):
        """エスカレーション済みタスクは再度エスカレーションされない."""
        mgr = _make_manager(tmp_path)
        mock_consult = MagicMock()
        mock_consult.add.return_value = MagicMock(consultation_id="esc12345")
        mgr.consultation_store = mock_consult
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("doomed task")
        mgr.task_queue.update_status(task.task_id, "failed", error="致命的エラー")
        for i in range(3):
            mgr.task_queue.update_status_for_retry(task.task_id, retry_count=i + 1)
            mgr.task_queue.update_status(task.task_id, "failed", error="致命的エラー")

        # 1回目のエスカレーション
        loop._retry_failed_tasks()
        assert mock_consult.add.call_count == 1

        # 2回目のtickではエスカレーションされない
        loop._retry_failed_tasks()
        assert mock_consult.add.call_count == 1  # 変わらない

    def test_tick_calls_retry_before_pick(self, tmp_path: Path):
        """tick()が_pick_taskの前に_retry_failed_tasksを呼ぶ."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("retry in tick")
        mgr.task_queue.update_status(task.task_id, "failed", error="一時エラー")

        with patch.object(mgr, "check_budget", return_value=False):
            loop.tick()

        # Task should have been retried (pending→running→completed by mock LLM)
        completed = mgr.task_queue.list_by_status("completed")
        assert len(completed) == 1
        assert completed[0].task_id == task.task_id
        assert completed[0].retry_count == 1


# ---------------------------------------------------------------------------
# _check_parent_completion — 親タスク完了チェック
# ---------------------------------------------------------------------------

class TestCheckParentCompletion:
    def test_all_siblings_completed_marks_parent_completed(self, tmp_path: Path):
        """全サブタスク完了時に親タスクがcompletedになる."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        parent = mgr.task_queue.add("parent task")
        sub1 = mgr.task_queue.add_with_deps("sub1", depends_on=[], parent_task_id=parent.task_id)
        sub2 = mgr.task_queue.add_with_deps("sub2", depends_on=[], parent_task_id=parent.task_id)

        mgr.task_queue.update_status(sub1.task_id, "completed", result="done1")
        mgr.task_queue.update_status(sub2.task_id, "completed", result="done2")

        # Refresh sub2 to get updated status
        updated_sub2 = mgr.task_queue._get_latest(sub2.task_id)
        loop._check_parent_completion(updated_sub2)

        parent_updated = mgr.task_queue._get_latest(parent.task_id)
        assert parent_updated.status == "completed"
        assert parent_updated.result == "全サブタスク完了"

    def test_not_all_siblings_completed_no_change(self, tmp_path: Path):
        """一部サブタスクが未完了なら親タスクは変更されない."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        parent = mgr.task_queue.add("parent task")
        sub1 = mgr.task_queue.add_with_deps("sub1", depends_on=[], parent_task_id=parent.task_id)
        sub2 = mgr.task_queue.add_with_deps("sub2", depends_on=[], parent_task_id=parent.task_id)

        mgr.task_queue.update_status(sub1.task_id, "completed", result="done1")
        # sub2 is still pending

        updated_sub1 = mgr.task_queue._get_latest(sub1.task_id)
        loop._check_parent_completion(updated_sub1)

        parent_updated = mgr.task_queue._get_latest(parent.task_id)
        assert parent_updated.status == "pending"

    def test_permanently_failed_sibling_marks_parent_failed(self, tmp_path: Path):
        """永久失敗サブタスク（retry_count >= max_retries）があれば親がfailedになる."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        parent = mgr.task_queue.add("parent task")
        sub1 = mgr.task_queue.add_with_deps("sub1", depends_on=[], parent_task_id=parent.task_id)
        sub2 = mgr.task_queue.add_with_deps("sub2", depends_on=[], parent_task_id=parent.task_id)

        mgr.task_queue.update_status(sub1.task_id, "completed", result="done1")
        # sub2 fails permanently: retry_count reaches max_retries (3)
        mgr.task_queue.update_status(sub2.task_id, "failed", error="永続エラー")
        for i in range(3):
            mgr.task_queue.update_status_for_retry(sub2.task_id, retry_count=i + 1)
            mgr.task_queue.update_status(sub2.task_id, "failed", error="永続エラー")

        updated_sub2 = mgr.task_queue._get_latest(sub2.task_id)
        assert updated_sub2.retry_count == 3
        assert updated_sub2.max_retries == 3

        loop._check_parent_completion(updated_sub2)

        parent_updated = mgr.task_queue._get_latest(parent.task_id)
        assert parent_updated.status == "failed"
        assert parent_updated.error == "サブタスク永久失敗"

    def test_no_parent_task_id_does_nothing(self, tmp_path: Path):
        """parent_task_idがNoneのタスクでは何もしない."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("standalone task")
        mgr.task_queue.update_status(task.task_id, "completed", result="done")

        updated = mgr.task_queue._get_latest(task.task_id)
        # Should not raise
        loop._check_parent_completion(updated)

    def test_failed_but_not_permanent_no_parent_change(self, tmp_path: Path):
        """失敗サブタスクがあるがリトライ可能なら親は変更されない."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        parent = mgr.task_queue.add("parent task")
        sub1 = mgr.task_queue.add_with_deps("sub1", depends_on=[], parent_task_id=parent.task_id)
        sub2 = mgr.task_queue.add_with_deps("sub2", depends_on=[], parent_task_id=parent.task_id)

        mgr.task_queue.update_status(sub1.task_id, "completed", result="done1")
        # sub2 fails but retry_count (0) < max_retries (3) — still retryable
        mgr.task_queue.update_status(sub2.task_id, "failed", error="一時エラー")

        updated_sub2 = mgr.task_queue._get_latest(sub2.task_id)
        loop._check_parent_completion(updated_sub2)

        parent_updated = mgr.task_queue._get_latest(parent.task_id)
        assert parent_updated.status == "pending"

    def test_called_during_execute_task_completion(self, tmp_path: Path):
        """_execute_task完了時に_check_parent_completionが呼ばれる."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        parent = mgr.task_queue.add("parent task")
        sub = mgr.task_queue.add_with_deps("sub task", depends_on=[], parent_task_id=parent.task_id)

        with patch.object(loop, "_check_parent_completion") as mock_check:
            loop._execute_task(sub)
            mock_check.assert_called_once()
            call_arg = mock_check.call_args[0][0]
            assert call_arg.task_id == sub.task_id


# ---------------------------------------------------------------------------
# _execute_task() — task history context (Task 8.1)
# ---------------------------------------------------------------------------

class TestExecuteTaskHistoryContext:
    """_execute_task()がタスク履歴コンテキストをシステムプロンプトに含める."""

    def test_system_prompt_includes_task_history(self, tmp_path: Path):
        """完了・失敗タスクがシステムプロンプトに含まれる."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Add completed and failed tasks for history
        t_done = mgr.task_queue.add("completed task")
        mgr.task_queue.update_status(t_done.task_id, "completed", result="done")
        t_fail = mgr.task_queue.add("failed task")
        mgr.task_queue.update_status(t_fail.task_id, "failed", error="some error")

        # Task to execute
        task = mgr.task_queue.add("new task")

        loop._execute_task(task)

        # Check the system message sent to LLM
        call_args = mgr.llm_client.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "タスク履歴" in system_msg
        assert "completed task" in system_msg
        assert "failed task" in system_msg

    def test_system_prompt_limits_completed_to_10(self, tmp_path: Path):
        """完了タスクは最大10件に制限される."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Add 12 completed tasks
        for i in range(12):
            t = mgr.task_queue.add(f"done_task_{i:03d}")
            mgr.task_queue.update_status(t.task_id, "completed", result=f"result-{i}")

        task = mgr.task_queue.add("new task")
        loop._execute_task(task)

        system_msg = mgr.llm_client.chat.call_args[0][0][0]["content"]
        # First 2 (index 0,1) should be excluded, last 10 (index 2-11) included
        assert "done_task_000" not in system_msg
        assert "done_task_001" not in system_msg
        assert "done_task_002" in system_msg
        assert "done_task_011" in system_msg

    def test_system_prompt_limits_failed_to_5(self, tmp_path: Path):
        """失敗タスクは最大5件に制限される."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        # Add 7 failed tasks
        for i in range(7):
            t = mgr.task_queue.add(f"failed-{i}")
            mgr.task_queue.update_status(t.task_id, "failed", error=f"err-{i}")

        task = mgr.task_queue.add("new task")
        loop._execute_task(task)

        system_msg = mgr.llm_client.chat.call_args[0][0][0]["content"]
        # First 2 (index 0,1) should be excluded, last 5 (index 2-6) included
        assert "failed-0" not in system_msg
        assert "failed-1" not in system_msg
        assert "failed-2" in system_msg
        assert "failed-6" in system_msg

    def test_no_history_still_works(self, tmp_path: Path):
        """タスク履歴がない場合でも正常に動作する."""
        mgr = _make_manager(tmp_path)
        loop = AutonomousLoop(mgr)

        task = mgr.task_queue.add("new task")
        loop._execute_task(task)

        system_msg = mgr.llm_client.chat.call_args[0][0][0]["content"]
        assert "タスクを実行してください" in system_msg
