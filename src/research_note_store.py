"""Research note store backed by NDJSON storage.

Provides persistence and retrieval of research notes collected by CEO AI.

Requirements: 2.3, 2.4
"""

from __future__ import annotations

from pathlib import Path

from models import ResearchNote
from ndjson_store import ndjson_append, ndjson_read


class ResearchNoteStore:
    """リサーチノートの永続化と取得を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "research_notes.ndjson"

    def save(self, note: ResearchNote) -> None:
        """リサーチノートを追記する."""
        ndjson_append(self._path, note)

    def load_all(self) -> list[ResearchNote]:
        """保存された全ノートをリストとして返す."""
        return ndjson_read(self._path, ResearchNote)

    def recent(self, limit: int = 10) -> list[ResearchNote]:
        """直近N件のリサーチノートを返す."""
        notes = ndjson_read(self._path, ResearchNote)
        return notes[-limit:]
