"""Agent registry backed by NDJSON storage.

Manages agent information (CEO_AI, Sub_Agents) with append-only persistence.
Same agent_id's latest entry is the current state.

Requirements: 4.1, 4.2, 4.3, 4.5
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from models import AgentEntry
from ndjson_store import ndjson_append, ndjson_read


class AgentRegistry:
    """エージェント情報の管理を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "agents.ndjson"

    def register(
        self,
        agent_id: str,
        name: str,
        role: str,
        model: str,
        budget_limit_usd: float,
    ) -> AgentEntry:
        """エージェントを登録する."""
        now = datetime.now(timezone.utc)
        entry = AgentEntry(
            agent_id=agent_id,
            name=name,
            role=role,
            model=model,
            budget_limit_usd=budget_limit_usd,
            status="active",
            created_at=now,
            updated_at=now,
        )
        ndjson_append(self._path, entry)
        return entry

    def update_status(self, agent_id: str, status: str) -> None:
        """エージェントのステータスを更新する. Appends a new entry with updated status."""
        current = self.get(agent_id)
        if current is None:
            raise ValueError(f"Agent not found: {agent_id}")
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        ndjson_append(self._path, updated)

    def get(self, agent_id: str) -> AgentEntry | None:
        """IDでエージェントを検索する."""
        for entry in self._list_all():
            if entry.agent_id == agent_id:
                return entry
        return None

    def list_active(self) -> list[AgentEntry]:
        """アクティブなエージェント一覧を返す."""
        return [a for a in self._list_all() if a.status == "active"]

    def ensure_ceo(self, model: str) -> AgentEntry:
        """CEO_AIがなければデフォルト登録する."""
        existing = self.get("ceo")
        if existing is not None:
            return existing
        return self.register(
            agent_id="ceo",
            name="CEO AI",
            role="ceo",
            model=model,
            budget_limit_usd=10.0,
        )

    def _list_all(self) -> list[AgentEntry]:
        """全エントリを読み込み、agent_idごとに最新エントリを返す."""
        entries = ndjson_read(self._path, AgentEntry)
        latest: dict[str, AgentEntry] = {}
        for entry in entries:
            latest[entry.agent_id] = entry
        return list(latest.values())
