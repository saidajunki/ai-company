"""Unit tests for rolling_summary module."""

from __future__ import annotations

from pathlib import Path

from rolling_summary import RollingSummary


class TestRollingSummary:
    def test_placeholder_when_missing(self, tmp_path: Path) -> None:
        rs = RollingSummary(tmp_path / "rolling_summary.md")
        text = rs.format_for_prompt()
        assert "## 永続メモリ（要約）" in text
        assert "要約なし" in text

    def test_update_dedupe_and_persist(self, tmp_path: Path) -> None:
        path = tmp_path / "rolling_summary.md"
        rs = RollingSummary(path)

        rs.update(
            pinned_add=["目的: 有名で面白い存在になる", "目的: 有名で面白い存在になる"],
            recent_add=["a", "a", "b"],
        )

        assert path.exists()
        formatted = rs.format_for_prompt(max_recent=10)

        assert formatted.count("目的: 有名で面白い存在になる") == 1
        assert "- a" in formatted
        assert "- b" in formatted

