"""Unit tests for commitment_store."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from commitment_store import CommitmentStore


class TestCommitmentStore:
    def test_add_and_list_open(self, tmp_path: Path) -> None:
        store = CommitmentStore(tmp_path, "alpha")
        c = store.add("やること", title="TODO", owner="ceo")
        open_items = store.list_by_status("open")
        assert any(it.commitment_id == c.commitment_id for it in open_items)

    def test_ensure_open_dedupes(self, tmp_path: Path) -> None:
        store = CommitmentStore(tmp_path, "alpha")
        first, created1 = store.ensure_open("同じ内容", title="t1")
        second, created2 = store.ensure_open("同じ内容", title="t2")
        assert created1 is True
        assert created2 is False
        assert first.commitment_id == second.commitment_id

    def test_close_marks_done(self, tmp_path: Path) -> None:
        store = CommitmentStore(tmp_path, "alpha")
        c = store.add("やること", due_date=date(2026, 2, 17))
        updated = store.close(c.commitment_id, note="完了", status="done")
        assert updated.status == "done"
        assert updated.closed_at is not None
        assert updated.close_note == "完了"

