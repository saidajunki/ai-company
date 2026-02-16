"""Unit tests for TaskQueue.

Tests: add + list_all round-trip, update_status lifecycle,
next_pending priority ordering, list_by_status filtering, empty queue behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from task_queue import TaskQueue


@pytest.fixture
def tq(tmp_path: Path) -> TaskQueue:
    return TaskQueue(tmp_path, "test-co")


class TestAddAndListAll:
    def test_add_returns_task_entry(self, tq: TaskQueue) -> None:
        task = tq.add("do something")
        assert task.description == "do something"
        assert task.status == "pending"
        assert task.priority == 3
        assert task.agent_id == "ceo"
        assert len(task.task_id) == 8

    def test_add_custom_priority_and_agent(self, tq: TaskQueue) -> None:
        task = tq.add("urgent", priority=1, agent_id="worker-1")
        assert task.priority == 1
        assert task.agent_id == "worker-1"

    def test_list_all_round_trip(self, tq: TaskQueue) -> None:
        t1 = tq.add("task A")
        t2 = tq.add("task B")
        all_tasks = tq.list_all()
        assert len(all_tasks) == 2
        descs = {t.description for t in all_tasks}
        assert descs == {"task A", "task B"}

    def test_list_all_returns_latest_entry_per_task_id(self, tq: TaskQueue) -> None:
        task = tq.add("original")
        tq.update_status(task.task_id, "running")
        all_tasks = tq.list_all()
        assert len(all_tasks) == 1
        assert all_tasks[0].status == "running"


class TestUpdateStatus:
    def test_pending_to_running(self, tq: TaskQueue) -> None:
        task = tq.add("work item")
        tq.update_status(task.task_id, "running")
        updated = tq.list_all()
        assert len(updated) == 1
        assert updated[0].status == "running"

    def test_full_lifecycle(self, tq: TaskQueue) -> None:
        task = tq.add("lifecycle task")
        tq.update_status(task.task_id, "running")
        tq.update_status(task.task_id, "completed", result="done!")
        updated = tq.list_all()
        assert len(updated) == 1
        assert updated[0].status == "completed"
        assert updated[0].result == "done!"

    def test_failed_with_error(self, tq: TaskQueue) -> None:
        task = tq.add("failing task")
        tq.update_status(task.task_id, "running")
        tq.update_status(task.task_id, "failed", error="boom")
        updated = tq.list_all()
        assert len(updated) == 1
        assert updated[0].status == "failed"
        assert updated[0].error == "boom"

    def test_update_nonexistent_task_raises(self, tq: TaskQueue) -> None:
        with pytest.raises(ValueError, match="Task not found"):
            tq.update_status("no-such-id", "running")

    def test_updated_at_changes(self, tq: TaskQueue) -> None:
        task = tq.add("time check")
        original_updated = task.updated_at
        tq.update_status(task.task_id, "running")
        updated = tq.list_all()[0]
        assert updated.updated_at >= original_updated

    def test_created_at_preserved(self, tq: TaskQueue) -> None:
        task = tq.add("preserve created_at")
        tq.update_status(task.task_id, "running")
        tq.update_status(task.task_id, "completed", result="ok")
        updated = tq.list_all()[0]
        assert updated.created_at == task.created_at


class TestNextPending:
    def test_returns_highest_priority(self, tq: TaskQueue) -> None:
        tq.add("low", priority=5)
        tq.add("high", priority=1)
        tq.add("mid", priority=3)
        nxt = tq.next_pending()
        assert nxt is not None
        assert nxt.description == "high"
        assert nxt.priority == 1

    def test_skips_non_pending(self, tq: TaskQueue) -> None:
        t1 = tq.add("running task", priority=1)
        tq.update_status(t1.task_id, "running")
        tq.add("pending task", priority=5)
        nxt = tq.next_pending()
        assert nxt is not None
        assert nxt.description == "pending task"

    def test_empty_queue_returns_none(self, tq: TaskQueue) -> None:
        assert tq.next_pending() is None

    def test_all_completed_returns_none(self, tq: TaskQueue) -> None:
        t = tq.add("done task")
        tq.update_status(t.task_id, "completed", result="ok")
        assert tq.next_pending() is None


class TestListByStatus:
    def test_filter_pending(self, tq: TaskQueue) -> None:
        tq.add("a")
        t2 = tq.add("b")
        tq.update_status(t2.task_id, "running")
        pending = tq.list_by_status("pending")
        assert len(pending) == 1
        assert pending[0].description == "a"

    def test_filter_running(self, tq: TaskQueue) -> None:
        t1 = tq.add("x")
        tq.update_status(t1.task_id, "running")
        tq.add("y")
        running = tq.list_by_status("running")
        assert len(running) == 1
        assert running[0].description == "x"

    def test_filter_returns_empty_for_no_match(self, tq: TaskQueue) -> None:
        tq.add("only pending")
        assert tq.list_by_status("completed") == []


class TestEmptyQueue:
    def test_list_all_empty(self, tq: TaskQueue) -> None:
        assert tq.list_all() == []

    def test_list_by_status_empty(self, tq: TaskQueue) -> None:
        assert tq.list_by_status("pending") == []

    def test_next_pending_empty(self, tq: TaskQueue) -> None:
        assert tq.next_pending() is None
