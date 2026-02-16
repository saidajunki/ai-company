"""Initiative store backed by NDJSON storage.

Provides persistence and retrieval of business initiatives with
same-ID deduplication (append-only, latest entry wins).

Requirements: 3.2, 3.5, 3.6
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from models import InitiativeEntry, InitiativeScores
from ndjson_store import ndjson_append, ndjson_read


class InitiativeStore:
    """イニシアチブの永続化と取得を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "initiatives.ndjson"

    def save(self, entry: InitiativeEntry) -> None:
        """イニシアチブを保存する（append-only）."""
        ndjson_append(self._path, entry)

    def get(self, initiative_id: str) -> InitiativeEntry | None:
        """IDで最新のイニシアチブを取得する."""
        entries = ndjson_read(self._path, InitiativeEntry)
        latest: InitiativeEntry | None = None
        for entry in entries:
            if entry.initiative_id == initiative_id:
                latest = entry
        return latest

    def list_all(self) -> list[InitiativeEntry]:
        """全イニシアチブの最新エントリを取得する（同一ID重複排除）."""
        entries = ndjson_read(self._path, InitiativeEntry)
        latest_by_id: dict[str, InitiativeEntry] = {}
        for entry in entries:
            latest_by_id[entry.initiative_id] = entry
        return list(latest_by_id.values())

    def list_by_status(self, status: str) -> list[InitiativeEntry]:
        """ステータスでフィルタリングする."""
        return [e for e in self.list_all() if e.status == status]

    def recent(self, limit: int = 10) -> list[InitiativeEntry]:
        """直近N件のイニシアチブを取得する（重複排除後）."""
        all_entries = self.list_all()
        return all_entries[-limit:]

    def update_status(
        self,
        initiative_id: str,
        status: str,
        retrospective: str | None = None,
        actual_score: dict[str, int] | None = None,
    ) -> None:
        """ステータスを更新する（新しいエントリを追記）."""
        existing = self.get(initiative_id)
        if existing is None:
            return

        actual_scores = None
        if actual_score is not None:
            actual_scores = InitiativeScores(**actual_score)

        updated = existing.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(timezone.utc),
                "retrospective": retrospective if retrospective is not None else existing.retrospective,
                "actual_scores": actual_scores if actual_scores is not None else existing.actual_scores,
            },
        )
        self.save(updated)
