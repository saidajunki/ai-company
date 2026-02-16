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

    def add(self, description: str, priority: int = 3, agent_id: str = "ceo") -> TaskEntry:
        """新しいタスクを追加する. Returns the created TaskEntry."""
        now = datetime.now(timezone.utc)
        entry = TaskEntry(
            task_id=uuid4().hex[:8],
            description=description,
            priority=priority,
            status="pending",
            created_at=now,
            updated_at=now,
            agent_id=agent_id,
        )
        ndjson_append(self._path, entry)
        return entry
    def add_with_deps(
        self,
        description: str,
        depends_on: list[str],
        priority: int = 3,
        agent_id: str = "ceo",
    ) -> TaskEntry:
        """新しいタスクを依存関係付きで追加する."""
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
        )
        ndjson_append(self._path, entry)
        return entry

    def next_pending(self) -> TaskEntry | None:
        """優先度順で次のpendingタスクを返す (数値が小さい方が高優先度).

        依存関係が未完了のタスクはスキップする。
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
        eligible.sort(key=lambda t: t.priority)
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

    def _get_latest(self, task_id: str) -> TaskEntry | None:
        """task_idの最新エントリを返す."""
        for entry in self.list_all():
            if entry.task_id == task_id:
                return entry
        return None
