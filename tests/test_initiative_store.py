"""Tests for InitiativeStore.

Requirements: 3.2, 3.5, 3.6
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from initiative_store import InitiativeStore
from models import InitiativeEntry, InitiativeScores

NOW = datetime.now(timezone.utc)


def _make_entry(
    initiative_id: str = "init-001",
    title: str = "テスト施策",
    status: str = "planned",
    **kwargs,
) -> InitiativeEntry:
    defaults = dict(
        initiative_id=initiative_id,
        title=title,
        description="テスト用の施策",
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(kwargs)
    return InitiativeEntry(**defaults)


class TestSave:
    def test_save_creates_file(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        entry = _make_entry()
        store.save(entry)
        assert store._path.exists()

    def test_save_and_get_round_trip(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        entry = _make_entry()
        store.save(entry)
        result = store.get("init-001")
        assert result is not None
        assert result.initiative_id == "init-001"
        assert result.title == "テスト施策"


class TestGet:
    def test_get_nonexistent_returns_none(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        assert store.get("no-such-id") is None

    def test_get_returns_latest_for_same_id(self, tmp_path):
        """Req 3.6: 同一IDの最新エントリのみを返す."""
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry(title="v1", status="planned"))
        store.save(_make_entry(title="v2", status="in_progress"))
        result = store.get("init-001")
        assert result is not None
        assert result.title == "v2"
        assert result.status == "in_progress"


class TestListAll:
    def test_empty_store(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        assert store.list_all() == []

    def test_deduplication(self, tmp_path):
        """Req 3.6: list_all は同一IDの最新エントリのみ."""
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", title="first-v1"))
        store.save(_make_entry("id-2", title="second-v1"))
        store.save(_make_entry("id-1", title="first-v2"))
        result = store.list_all()
        assert len(result) == 2
        titles = {e.title for e in result}
        assert "first-v2" in titles
        assert "second-v1" in titles
        assert "first-v1" not in titles

    def test_multiple_distinct_ids(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        for i in range(5):
            store.save(_make_entry(f"id-{i}", title=f"initiative-{i}"))
        assert len(store.list_all()) == 5


class TestListByStatus:
    def test_filter_by_status(self, tmp_path):
        """Req 3.5: ステータスフィルタリング."""
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", status="planned"))
        store.save(_make_entry("id-2", status="in_progress"))
        store.save(_make_entry("id-3", status="planned"))
        result = store.list_by_status("planned")
        assert len(result) == 2
        assert all(e.status == "planned" for e in result)

    def test_filter_returns_empty_for_no_match(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", status="planned"))
        assert store.list_by_status("completed") == []

    def test_filter_uses_latest_status(self, tmp_path):
        """Req 3.5 + 3.6: フィルタは重複排除後のステータスを使う."""
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", status="planned"))
        store.save(_make_entry("id-1", status="in_progress"))
        assert store.list_by_status("planned") == []
        assert len(store.list_by_status("in_progress")) == 1


class TestRecent:
    def test_recent_with_limit(self, tmp_path):
        """Req 3.5: 直近N件の取得."""
        store = InitiativeStore(tmp_path, "test-co")
        for i in range(5):
            store.save(_make_entry(f"id-{i}"))
        result = store.recent(limit=3)
        assert len(result) == 3

    def test_recent_fewer_than_limit(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1"))
        result = store.recent(limit=10)
        assert len(result) == 1

    def test_recent_empty(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        assert store.recent() == []


class TestUpdateStatus:
    def test_update_status(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", status="planned"))
        store.update_status("id-1", "in_progress")
        result = store.get("id-1")
        assert result is not None
        assert result.status == "in_progress"

    def test_update_with_retrospective(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", status="in_progress"))
        store.update_status("id-1", "completed", retrospective="学びが多かった")
        result = store.get("id-1")
        assert result is not None
        assert result.status == "completed"
        assert result.retrospective == "学びが多かった"

    def test_update_with_actual_score(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        store.save(_make_entry("id-1", status="in_progress"))
        store.update_status(
            "id-1",
            "completed",
            actual_score={
                "interestingness": 20,
                "cost_efficiency": 15,
                "realism": 10,
                "evolvability": 25,
            },
        )
        result = store.get("id-1")
        assert result is not None
        assert result.actual_scores is not None
        assert result.actual_scores.total == 70

    def test_update_nonexistent_is_noop(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        store.update_status("no-such-id", "completed")
        assert store.list_all() == []

    def test_update_preserves_other_fields(self, tmp_path):
        store = InitiativeStore(tmp_path, "test-co")
        scores = InitiativeScores(
            interestingness=20, cost_efficiency=15, realism=10, evolvability=25,
        )
        store.save(_make_entry(
            "id-1",
            status="planned",
            estimated_scores=scores,
            first_step="最初のタスク",
            task_ids=["t1", "t2"],
        ))
        store.update_status("id-1", "in_progress")
        result = store.get("id-1")
        assert result is not None
        assert result.estimated_scores == scores
        assert result.first_step == "最初のタスク"
        assert result.task_ids == ["t1", "t2"]
