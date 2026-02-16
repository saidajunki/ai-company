"""Conversation memory backed by NDJSON storage.

Provides append-only persistence and retrieval of conversation history entries.

Requirements: 1.1, 1.3, 1.5
"""

from __future__ import annotations

from pathlib import Path

from models import ConversationEntry
from ndjson_store import ndjson_append, ndjson_read


class ConversationMemory:
    """会話履歴の永続化と取得を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "conversations.ndjson"

    def append(self, entry: ConversationEntry) -> None:
        """会話エントリを追記する."""
        ndjson_append(self._path, entry)

    def recent(self, n: int = 20) -> list[ConversationEntry]:
        """直近N件の会話エントリを返す."""
        entries = ndjson_read(self._path, ConversationEntry)
        return entries[-n:]
