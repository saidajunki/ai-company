"""Unit tests for ResearchNoteStore."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from models import ResearchNote
from research_note_store import ResearchNoteStore


def _make_note(**overrides) -> ResearchNote:
    defaults = dict(
        query="AI trends 2025",
        source_url="https://example.com/article",
        title="AI Trends",
        snippet="AI is evolving rapidly...",
        summary="Summary of AI trends in 2025.",
        published_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        retrieved_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ResearchNote(**defaults)


@pytest.fixture
def store(tmp_path: Path) -> ResearchNoteStore:
    return ResearchNoteStore(base_dir=tmp_path, company_id="test-co")


class TestSaveAndLoadAll:
    def test_round_trip_single_note(self, store: ResearchNoteStore):
        note = _make_note()
        store.save(note)
        result = store.load_all()
        assert len(result) == 1
        assert result[0] == note

    def test_round_trip_preserves_all_fields(self, store: ResearchNoteStore):
        note = _make_note(
            query="quantum computing",
            source_url="https://example.com/quantum",
            title="Quantum Advances",
            snippet="Quantum computing breakthroughs...",
            summary="Major quantum computing advances.",
            published_at=datetime(2025, 3, 10, tzinfo=timezone.utc),
            retrieved_at=datetime(2025, 6, 2, 8, 30, 0, tzinfo=timezone.utc),
        )
        store.save(note)
        loaded = store.load_all()[0]
        assert loaded.query == note.query
        assert loaded.source_url == note.source_url
        assert loaded.title == note.title
        assert loaded.snippet == note.snippet
        assert loaded.summary == note.summary
        assert loaded.published_at == note.published_at
        assert loaded.retrieved_at == note.retrieved_at

    def test_published_at_none(self, store: ResearchNoteStore):
        note = _make_note(published_at=None)
        store.save(note)
        loaded = store.load_all()[0]
        assert loaded.published_at is None

    def test_multiple_notes(self, store: ResearchNoteStore):
        notes = [_make_note(query=f"query-{i}") for i in range(5)]
        for n in notes:
            store.save(n)
        result = store.load_all()
        assert result == notes


class TestLoadAllEmpty:
    def test_returns_empty_when_no_file(self, store: ResearchNoteStore):
        assert store.load_all() == []


class TestRecent:
    def test_recent_limits_to_n(self, store: ResearchNoteStore):
        for i in range(15):
            store.save(_make_note(query=f"query-{i}"))
        result = store.recent(limit=5)
        assert len(result) == 5
        assert result[0].query == "query-10"
        assert result[4].query == "query-14"

    def test_recent_returns_all_when_fewer_than_limit(self, store: ResearchNoteStore):
        for i in range(3):
            store.save(_make_note(query=f"query-{i}"))
        result = store.recent(limit=10)
        assert len(result) == 3

    def test_default_limit_is_10(self, store: ResearchNoteStore):
        for i in range(15):
            store.save(_make_note(query=f"query-{i}"))
        result = store.recent()
        assert len(result) == 10
        assert result[0].query == "query-5"

    def test_recent_returns_empty_when_no_file(self, store: ResearchNoteStore):
        assert store.recent() == []
