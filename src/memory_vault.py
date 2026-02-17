"""Memory Vault — file-first curated memory store.

Inspired by Human Storage System (HSS):
- A curated long-term memory file that is the "truth" on disk
- Append-only by default (avoid losing important direction/values)

This complements:
- constitution.yaml (structured rules)
- vision.md (narrative vision)
- rolling_summary.md (auto-maintained short summary)
- memory.sqlite3 (search index; derived)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_CURATED_MEMORY = """\
# MEMORY (Curated / LTM)

このファイルは「会社として絶対に忘れてはいけないこと」を保管する場所です。

- 価値観 / 目的 / 禁止事項 / 運用ルール / 大事な合意 を追記する
- 原則として **追記（append-only）** を推奨（過去のログは消さない）
- 変更した場合は「いつ/なぜ変えたか」を残す
"""


def curated_memory_path(base_dir: Path, company_id: str) -> Path:
    return base_dir / "companies" / company_id / "knowledge" / "MEMORY.md"


def daily_memory_path(base_dir: Path, company_id: str, *, day: str) -> Path:
    """Return ``companies/<company_id>/knowledge/daily/<YYYY-MM-DD>.md``."""
    return base_dir / "companies" / company_id / "knowledge" / "daily" / f"{day}.md"


class MemoryVault:
    """Curated memory file manager (best-effort, never raises)."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self.base_dir = base_dir
        self.company_id = company_id
        self._path = curated_memory_path(base_dir, company_id)

    @property
    def path(self) -> Path:
        return self._path

    def ensure_initialized(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._path.exists():
                self._path.write_text(DEFAULT_CURATED_MEMORY.rstrip() + "\n", encoding="utf-8")
        except Exception:
            logger.warning("Failed to initialize curated memory: %s", self._path, exc_info=True)

    def load_tail(self, *, tail_chars: int = 6000) -> str | None:
        self.ensure_initialized()
        try:
            text = self._path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("Failed to read curated memory: %s", self._path, exc_info=True)
            return None

        if tail_chars <= 0:
            return ""
        if len(text) <= tail_chars:
            return text.strip()
        return text[-tail_chars:].strip()

    def append(
        self,
        content: str,
        *,
        title: str | None = None,
        author: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.ensure_initialized()
        t = (content or "").strip()
        if not t:
            return

        ts = timestamp or datetime.now(timezone.utc)
        header = f"\n## {ts.isoformat()}\n"
        meta = ""
        if author:
            meta += f"- author: {author}\n"
        if title:
            meta += f"- title: {title.strip()}\n"

        body = t.rstrip() + "\n"

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(header)
                if meta:
                    f.write(meta)
                f.write("\n")
                f.write(body)
        except Exception:
            logger.warning("Failed to append curated memory: %s", self._path, exc_info=True)

    # ------------------------------------------------------------------
    # Daily memory (append-only)
    # ------------------------------------------------------------------

    def load_daily_tail(self, *, tail_chars: int = 3000, day: str | None = None) -> str | None:
        """Load tail of daily memory file for *day* (default: today UTC)."""
        self.ensure_initialized()
        target_day = day or datetime.now(timezone.utc).date().isoformat()
        path = daily_memory_path(self.base_dir, self.company_id, day=target_day)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("Failed to read daily memory: %s", path, exc_info=True)
            return None

        if tail_chars <= 0:
            return ""
        if len(text) <= tail_chars:
            return text.strip()
        return text[-tail_chars:].strip()

    def append_daily(
        self,
        content: str,
        *,
        title: str | None = None,
        author: str | None = None,
        timestamp: datetime | None = None,
        day: str | None = None,
    ) -> None:
        """Append a note to daily memory file (today UTC by default)."""
        self.ensure_initialized()
        t = (content or "").strip()
        if not t:
            return

        ts = timestamp or datetime.now(timezone.utc)
        target_day = day or ts.astimezone(timezone.utc).date().isoformat()
        path = daily_memory_path(self.base_dir, self.company_id, day=target_day)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = f"\n## {ts.isoformat()}\n"
        meta = ""
        if author:
            meta += f"- author: {author}\n"
        if title:
            meta += f"- title: {title.strip()}\n"

        body = t.rstrip() + "\n"

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(header)
                if meta:
                    f.write(meta)
                f.write("\n")
                f.write(body)
        except Exception:
            logger.warning("Failed to append daily memory: %s", path, exc_info=True)
