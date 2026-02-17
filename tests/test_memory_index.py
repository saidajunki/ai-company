"""Unit tests for memory_index module.

Ensures the SQLite-backed long-term memory index works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from memory_index import MemoryIndex


class TestMemoryIndex:
    def test_upsert_insert_and_update(self, tmp_path: Path) -> None:
        idx = MemoryIndex(tmp_path / "memory.sqlite3")
        try:
            inserted = idx.upsert(
                doc_id="doc-1",
                text="hello world",
                source_type="test",
                source_id="1",
                importance=3,
                tags=["alpha"],
            )
            assert inserted is True

            updated = idx.upsert(
                doc_id="doc-1",
                text="hello world updated",
                source_type="test",
                source_id="1",
                importance=4,
                tags=["alpha", "updated"],
            )
            assert updated is False

            hits = idx.search("updated", limit=5)
            assert any(h.doc_id == "doc-1" for h in hits)
        finally:
            idx.close()

    def test_source_offsets_round_trip(self, tmp_path: Path) -> None:
        idx = MemoryIndex(tmp_path / "memory.sqlite3")
        try:
            assert idx.get_source_offset("tasks") == 0
            idx.set_source_offset("tasks", 123)
            assert idx.get_source_offset("tasks") == 123
        finally:
            idx.close()

