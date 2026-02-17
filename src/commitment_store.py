"""Commitment store backed by NDJSON storage.

Commitments are first-class "promises/TODOs" that survive restarts.
They are intentionally separate from TaskQueue:
- Tasks are execution units (pending/running/completed/failed)
- Commitments are organizational obligations (open/done/canceled)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import CommitmentEntry
from ndjson_store import ndjson_append, ndjson_read


MAX_OPEN_COMMITMENTS = 30


class CommitmentStore:
    """約束/TODO（open/done/canceled）の永続化と取得を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "commitments.ndjson"

    def add(
        self,
        content: str,
        *,
        title: str = "",
        owner: str = "ceo",
        due_date: date | None = None,
        related_task_id: str | None = None,
    ) -> CommitmentEntry:
        now = datetime.now(timezone.utc)
        entry = CommitmentEntry(
            commitment_id=uuid4().hex[:8],
            created_at=now,
            status="open",
            title=title or "",
            content=content,
            owner=owner,
            due_date=due_date,
            related_task_id=related_task_id,
        )
        ndjson_append(self._path, entry)
        return entry

    def ensure_open(
        self,
        content: str,
        *,
        title: str = "",
        owner: str = "ceo",
        due_date: date | None = None,
        related_task_id: str | None = None,
    ) -> tuple[CommitmentEntry, bool]:
        """Ensure an open commitment exists (dedupe identical open items)."""
        normalized = (content or "").strip()
        try:
            open_items = self.list_by_status("open")
            for c in open_items:
                if (
                    (c.content or "").strip() == normalized
                    and (c.related_task_id or None) == related_task_id
                ):
                    return c, False
        except Exception:
            open_items = []

        # Soft limit: cancel oldest open items if at/over the limit.
        if len(open_items) >= MAX_OPEN_COMMITMENTS:
            sorted_open = sorted(open_items, key=lambda c: c.created_at)
            to_cancel = sorted_open[: len(sorted_open) - MAX_OPEN_COMMITMENTS + 1]
            for old in to_cancel:
                try:
                    self.close(old.commitment_id, note="自動cancel（上限超過）", status="canceled")
                except Exception:
                    pass

        entry = self.add(
            normalized,
            title=title,
            owner=owner,
            due_date=due_date,
            related_task_id=related_task_id,
        )
        return entry, True

    def close(
        self,
        commitment_id: str,
        *,
        note: str = "",
        status: str = "done",
    ) -> CommitmentEntry:
        current = self.get_latest(commitment_id)
        if current is None:
            raise ValueError(f"Commitment not found: {commitment_id}")
        if status not in ("done", "canceled"):
            raise ValueError(f"Invalid close status: {status}")
        updated = current.model_copy(
            update={
                "status": status,
                "closed_at": datetime.now(timezone.utc),
                "close_note": note or None,
            }
        )
        ndjson_append(self._path, updated)
        return updated

    def list_all(self) -> list[CommitmentEntry]:
        entries = ndjson_read(self._path, CommitmentEntry)
        latest: dict[str, CommitmentEntry] = {}
        for entry in entries:
            latest[entry.commitment_id] = entry
        return list(latest.values())

    def list_by_status(self, status: str) -> list[CommitmentEntry]:
        return [c for c in self.list_all() if c.status == status]

    def get_latest(self, commitment_id: str) -> CommitmentEntry | None:
        for entry in self.list_all():
            if entry.commitment_id == commitment_id:
                return entry
        return None

