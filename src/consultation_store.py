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

