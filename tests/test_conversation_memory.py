"""Unit tests for ConversationMemory."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from conversation_memory import ConversationMemory
from models import ConversationEntry


def _make_entry(**overrides) -> ConversationEntry:
    defaults = dict(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        role="user",
        content="hello",
    )
    defaults.update(overrides)
    return ConversationEntry(**defaults)


@pytest.fixture
def memory(tmp_path: Path) -> ConversationMemory:
    return ConversationMemory(base_dir=tmp_path, company_id="test-co")


class TestAppendAndRecent:
    def test_round_trip_single_entry(self, memory: ConversationMemory):
        entry = _make_entry(content="テスト", user_id="U123")
        memory.append(entry)
        result = memory.recent()
        assert len(result) == 1
        assert result[0] == entry

    def test_round_trip_preserves_all_fields(self, memory: ConversationMemory):
        entry = _make_entry(
            role="assistant",
            content="応答テスト",
            user_id="U456",
            task_id="task-1",
        )
        memory.append(entry)
        loaded = memory.recent()[0]
        assert loaded.timestamp == entry.timestamp
        assert loaded.role == entry.role
        assert loaded.content == entry.content
        assert loaded.user_id == entry.user_id
        assert loaded.task_id == entry.task_id

    def test_multiple_entries(self, memory: ConversationMemory):
        entries = [
            _make_entry(content=f"msg-{i}", role="user" if i % 2 == 0 else "assistant")
            for i in range(5)
        ]
        for e in entries:
            memory.append(e)
        result = memory.recent()
        assert result == entries


class TestEmptyFile:
    def test_recent_returns_empty_when_no_file(self, memory: ConversationMemory):
        assert memory.recent() == []

    def test_recent_returns_empty_with_n(self, memory: ConversationMemory):
        assert memory.recent(n=5) == []


class TestRecentLimit:
    def test_recent_limits_to_n(self, memory: ConversationMemory):
        for i in range(10):
            memory.append(_make_entry(content=f"msg-{i}"))
        result = memory.recent(n=3)
        assert len(result) == 3
        assert result[0].content == "msg-7"
        assert result[2].content == "msg-9"

    def test_recent_returns_all_when_fewer_than_n(self, memory: ConversationMemory):
        for i in range(2):
            memory.append(_make_entry(content=f"msg-{i}"))
        result = memory.recent(n=10)
        assert len(result) == 2

    def test_default_n_is_20(self, memory: ConversationMemory):
        for i in range(25):
            memory.append(_make_entry(content=f"msg-{i}"))
        result = memory.recent()
        assert len(result) == 20
        assert result[0].content == "msg-5"
