"""Service registry backed by NDJSON storage.

Manages service/deliverable information with append-only persistence.
Same service name's latest entry is the current state.
Deduplication: register() with same name updates existing entry.

Requirements: 5.1, 5.3, 5.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from models import ServiceEntry
from ndjson_store import ndjson_append, ndjson_read


class ServiceRegistry:
    """成果物の管理を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "services.ndjson"

    def register(self, name: str, description: str, agent_id: str) -> ServiceEntry:
        """サービスを登録する。同名が存在すれば更新する."""
        now = datetime.now(timezone.utc)
        existing = self.get(name)
        if existing is not None:
            entry = existing.model_copy(
                update={
                    "description": description,
                    "agent_id": agent_id,
                    "updated_at": now,
                }
            )
        else:
            entry = ServiceEntry(
                name=name,
                description=description,
                status="active",
                created_at=now,
                updated_at=now,
                agent_id=agent_id,
            )
        ndjson_append(self._path, entry)
        return entry

    def list_all(self) -> list[ServiceEntry]:
        """全サービスを返す."""
        entries = ndjson_read(self._path, ServiceEntry)
        latest: dict[str, ServiceEntry] = {}
        for entry in entries:
            latest[entry.name] = entry
        return list(latest.values())

    def get(self, name: str) -> ServiceEntry | None:
        """名前でサービスを検索する."""
        for entry in self.list_all():
            if entry.name == name:
                return entry
        return None
