"""Unit tests for memory_vault."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from memory_vault import MemoryVault, curated_memory_path, daily_memory_path


class TestMemoryVault:
    def test_ensure_initialized_creates_file(self, tmp_path: Path) -> None:
        v = MemoryVault(tmp_path, "alpha")
        v.ensure_initialized()
        assert curated_memory_path(tmp_path, "alpha").exists()

    def test_append_and_load_tail(self, tmp_path: Path) -> None:
        v = MemoryVault(tmp_path, "alpha")
        v.append("大事な方針", title="方針", author="ceo")
        tail = v.load_tail(tail_chars=2000)
        assert tail is not None
        assert "大事な方針" in tail

    def test_append_daily_and_load_tail(self, tmp_path: Path) -> None:
        v = MemoryVault(tmp_path, "alpha")
        ts = datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc)
        v.append_daily("今日の学び", title="学び", author="ceo", timestamp=ts)
        path = daily_memory_path(tmp_path, "alpha", day="2026-02-17")
        assert path.exists()
        tail = v.load_daily_tail(day="2026-02-17", tail_chars=2000)
        assert tail is not None
        assert "今日の学び" in tail

