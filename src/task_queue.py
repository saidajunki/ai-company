"""Task queue backed by NDJSON storage.

Provides append-only persistence and retrieval of autonomous tasks.
Same task_id's latest entry is the current status (append-only pattern).

Requirements: 3.1, 3.8
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import TaskEntry
from ndjson_store import ndjson_append, ndjson_read


class TaskQueue:
    """自律タスクの管理を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "tasks.ndjson"

    def _find_existing_active(self, description: str) -> TaskEntry | None:
        """同一descriptionのアクティブタスクがあれば返す（best-effort重複抑止）."""
        needle = (description or "").strip()
        if not needle:
            return None
        active_statuses = {"pending", "running", "paused", "canceled"}
        for t in self.list_all():
            if (t.description or "").strip() == needle and t.status in active_statuses:
                return t
        return None

    def add(self, description: str, priority: int = 3, agent_id: str = "ceo", source: str = "autonomous") -> TaskEntry:
        """新しいタスクを追加する. Returns the created TaskEntry."""
        if source != "creator":
            existing = self._find_existing_active(description)
            if existing is not None:
                return existing

        now = datetime.now(timezone.utc)
        entry = TaskEntry(
            task_id=uuid4().hex[:8],
            description=description,
            priority=priority,
            status="pending",
            created_at=now,
            updated_at=now,
            agent_id=agent_id,
            source=source,
        )
        ndjson_append(self._path, entry)
        return entry
    def add_with_deps(
        self,
        description: str,
        depends_on: list[str],
        priority: int = 3,
        agent_id: str = "ceo",
        parent_task_id: str | None = None,
        source: str = "autonomous",
    ) -> TaskEntry:
        """新しいタスクを依存関係・親タスクID付きで追加する."""
        if source != "creator":
            existing = self._find_existing_active(description)
            if existing is not None:
                return existing

        now = datetime.now(timezone.utc)
        entry = TaskEntry(
            task_id=uuid4().hex[:8],
            description=description,
            priority=priority,
            status="pending",
            created_at=now,
            updated_at=now,
            agent_id=agent_id,
            depends_on=depends_on,
            parent_task_id=parent_task_id,
            source=source,
        )
        ndjson_append(self._path, entry)
        return entry

    def next_pending(self) -> TaskEntry | None:
        """優先度順で次のpendingタスクを返す (数値が小さい方が高優先度).

        依存関係が未完了のタスクはスキップする。
        同一優先度時は作成日時が早いタスクを優先する。
        """
        all_tasks = self.list_all()
        completed_ids = {t.task_id for t in all_tasks if t.status == "completed"}
        pending = [t for t in all_tasks if t.status == "pending"]
        eligible = [
            t for t in pending
            if all(dep in completed_ids for dep in t.depends_on)
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda t: (t.priority, t.created_at))
        return eligible[0]

    def update_status(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
        error: str | None = None,
        quality_score: float | None = None,
        quality_notes: str | None = None,
    ) -> None:
        """タスクのステータスを更新する. Appends a new entry with updated status."""
        current = self._get_latest(task_id)
        if current is None:
            raise ValueError(f"Task not found: {task_id}")
        update_fields: dict = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
            "result": result,
            "error": error,
        }
        if quality_score is not None:
            update_fields["quality_score"] = quality_score
        if quality_notes is not None:
            update_fields["quality_notes"] = quality_notes
        updated = current.model_copy(update=update_fields)
        ndjson_append(self._path, updated)

    def list_all(self) -> list[TaskEntry]:
        """全タスクを返す (task_idごとに最新エントリのみ)."""
        entries = ndjson_read(self._path, TaskEntry)
        latest: dict[str, TaskEntry] = {}
        for entry in entries:
            latest[entry.task_id] = entry
        return list(latest.values())

    def list_by_status(self, status: str) -> list[TaskEntry]:
        """指定ステータスのタスクを返す."""
        return [t for t in self.list_all() if t.status == status]

    def list_by_parent(self, parent_task_id: str) -> list[TaskEntry]:
        """指定した親タスクIDのサブタスクを返す."""
        return [t for t in self.list_all() if t.parent_task_id == parent_task_id]

    def update_status_for_retry(self, task_id: str, retry_count: int) -> None:
        """ステータスをpendingに戻しretry_countを更新する."""
        current = self._get_latest(task_id)
        if current is None:
            raise ValueError(f"Task not found: {task_id}")
        updated = current.model_copy(update={
            "status": "pending",
            "retry_count": retry_count,
            "updated_at": datetime.now(timezone.utc),
        })
        ndjson_append(self._path, updated)

    def update_status_tree(
        self,
        root_task_id: str,
        status: str,
        *,
        error: str | None = None,
        result: str | None = None,
        only_statuses: set[str] | None = None,
    ) -> list[str]:
        """親タスク + その配下（parent_task_id）をまとめてステータス更新する.

        Returns:
            更新したtask_id一覧（存在しないものは除外）
        """
        tasks = self.list_all()
        by_id: dict[str, TaskEntry] = {t.task_id: t for t in tasks}
        children: dict[str, list[str]] = {}
        for t in tasks:
            if t.parent_task_id:
                children.setdefault(t.parent_task_id, []).append(t.task_id)

        # BFS to collect descendants
        queue: list[str] = [root_task_id]
        ordered: list[str] = []
        seen: set[str] = set()
        while queue:
            tid = queue.pop(0)
            if tid in seen:
                continue
            seen.add(tid)
            ordered.append(tid)
            queue.extend(children.get(tid, []))

        updated_ids: list[str] = []
        now = datetime.now(timezone.utc)
        effective_only = only_statuses or {"pending", "running", "paused", "canceled", "failed"}
        for tid in ordered:
            current = by_id.get(tid)
            if current is None:
                continue
            if current.status not in effective_only:
                continue
            updated = current.model_copy(update={
                "status": status,
                "updated_at": now,
                "result": result,
                "error": error,
            })
            ndjson_append(self._path, updated)
            updated_ids.append(tid)

        return updated_ids

    def _get_latest(self, task_id: str) -> TaskEntry | None:
        """task_idの最新エントリを返す."""
        for entry in self.list_all():
            if entry.task_id == task_id:
                return entry
        return None
