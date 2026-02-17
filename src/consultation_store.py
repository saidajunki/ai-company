"""Consultation queue store backed by NDJSON storage.

Used to keep track of pending questions for the Creator.
Append-only: updates are recorded as new entries (like TaskQueue).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import ConsultationEntry
from ndjson_store import ndjson_append, ndjson_read


MAX_PENDING_CONSULTATIONS = 5


class ConsultationStore:
    """相談事項（pending/resolved）の永続化と取得を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "consultations.ndjson"

    def add(self, content: str, *, related_task_id: str | None = None) -> ConsultationEntry:
        now = datetime.now(timezone.utc)
        entry = ConsultationEntry(
            consultation_id=uuid4().hex[:8],
            created_at=now,
            status="pending",
            content=content,
            related_task_id=related_task_id,
        )
        ndjson_append(self._path, entry)
        return entry

    def ensure_pending(
        self, content: str, *, related_task_id: str | None = None
    ) -> tuple[ConsultationEntry, bool]:
        """Ensure a pending consultation exists (dedupe identical pending items).

        If the number of pending consultations exceeds MAX_PENDING_CONSULTATIONS,
        the oldest ones are auto-resolved to make room.

        Returns:
            (entry, created) where created=True only when a new pending entry was added.
        """
        normalized = (content or "").strip()
        try:
            pending = self.list_by_status("pending")
            for c in pending:
                if c.related_task_id == related_task_id and (c.content or "").strip() == normalized:
                    return c, False
        except Exception:
            pending = []

        # Auto-resolve oldest consultations if at or over the limit
        if len(pending) >= MAX_PENDING_CONSULTATIONS:
            sorted_pending = sorted(pending, key=lambda c: c.created_at)
            to_resolve = sorted_pending[: len(sorted_pending) - MAX_PENDING_CONSULTATIONS + 1]
            for old in to_resolve:
                try:
                    self.resolve(old.consultation_id, resolution="自動resolve（上限超過）")
                except Exception:
                    pass

        entry = self.add(normalized, related_task_id=related_task_id)
        return entry, True

    def resolve(self, consultation_id: str, *, resolution: str = "") -> ConsultationEntry:
        current = self.get_latest(consultation_id)
        if current is None:
            raise ValueError(f"Consultation not found: {consultation_id}")
        updated = current.model_copy(
            update={
                "status": "resolved",
                "resolved_at": datetime.now(timezone.utc),
                "resolution": resolution or None,
            }
        )
        ndjson_append(self._path, updated)
        return updated

    def list_all(self) -> list[ConsultationEntry]:
        entries = ndjson_read(self._path, ConsultationEntry)
        latest: dict[str, ConsultationEntry] = {}
        for entry in entries:
            latest[entry.consultation_id] = entry
        return list(latest.values())

    def list_by_status(self, status: str) -> list[ConsultationEntry]:
        return [c for c in self.list_all() if c.status == status]

    def get_latest(self, consultation_id: str) -> ConsultationEntry | None:
        for entry in self.list_all():
            if entry.consultation_id == consultation_id:
                return entry
        return None
