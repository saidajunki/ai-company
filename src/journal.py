"""Journal writer.

Appends compact, human-readable daily logs.
This complements NDJSON stores by providing a "never forget" narrative layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int = 800) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[:max_len] + "…"


class JournalWriter:
    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._dir = base_dir / "companies" / company_id / "journal"

    def append_interaction(
        self,
        *,
        timestamp: datetime | None = None,
        user_id: str | None = None,
        request_text: str,
        response_text: str,
        snapshot_lines: list[str] | None = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        day = ts.astimezone(timezone.utc).date().isoformat()
        path = self._dir / f"{day}.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        header = f"## {ts.isoformat()}\n"
        who = f"- user_id: {user_id}\n" if user_id else ""
        snap = ""
        if snapshot_lines:
            snap_lines = "\n".join(f"- {s}" for s in snapshot_lines if (s or '').strip())
            if snap_lines:
                snap = f"\n### Snapshot\n{snap_lines}\n"

        body = (
            f"\n{header}"
            f"{who}"
            f"\n### Request\n{_truncate(request_text)}\n"
            f"\n### Response\n{_truncate(response_text)}\n"
            f"{snap}\n"
        )

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(body)
        except Exception:
            logger.warning("Failed to append journal: %s", path, exc_info=True)

