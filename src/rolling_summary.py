"""Rolling summary store.

Keeps a compact, human-readable summary that survives restarts.
This is meant to prevent "important direction/value" loss even when
conversation history is truncated in prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER = "# Rolling Summary"
_PINNED = "## Pinned"
_RECENT = "## Recent"


@dataclass
class RollingSummaryState:
    pinned: list[str]
    recent: list[str]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


class RollingSummary:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> RollingSummaryState:
        if not self._path.exists():
            return RollingSummaryState(pinned=[], recent=[])
        try:
            text = self._path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("Failed to read rolling summary: %s", self._path, exc_info=True)
            return RollingSummaryState(pinned=[], recent=[])

        pinned: list[str] = []
        recent: list[str] = []
        section: str | None = None

        for raw in text.splitlines():
            line = raw.strip()
            if line == _PINNED:
                section = "pinned"
                continue
            if line == _RECENT:
                section = "recent"
                continue
            if not line.startswith("- "):
                continue
            item = line[2:].strip()
            if not item:
                continue
            if section == "pinned":
                pinned.append(item)
            elif section == "recent":
                recent.append(item)

        return RollingSummaryState(
            pinned=_dedupe_preserve_order(pinned),
            recent=_dedupe_preserve_order(recent),
        )

    def save(self, state: RollingSummaryState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            _HEADER,
            "",
            _PINNED,
        ]
        for it in state.pinned:
            lines.append(f"- {it}")
        lines.extend(["", _RECENT])
        for it in state.recent:
            lines.append(f"- {it}")
        content = "\n".join(lines).rstrip() + "\n"
        self._path.write_text(content, encoding="utf-8")

    def update(
        self,
        *,
        pinned_add: list[str] | None = None,
        recent_add: list[str] | None = None,
        max_pinned: int = 30,
        max_recent: int = 50,
    ) -> RollingSummaryState:
        state = self.load()
        pinned = state.pinned + list(pinned_add or [])
        recent = state.recent + list(recent_add or [])
        pinned = _dedupe_preserve_order(pinned)[-max_pinned:]
        recent = _dedupe_preserve_order(recent)[-max_recent:]

        new_state = RollingSummaryState(pinned=pinned, recent=recent)
        try:
            self.save(new_state)
        except Exception:
            logger.warning("Failed to save rolling summary: %s", self._path, exc_info=True)
        return new_state

    def format_for_prompt(self, *, max_recent: int = 10) -> str:
        state = self.load()
        pinned = state.pinned
        recent = state.recent[-max_recent:] if max_recent > 0 else []

        lines: list[str] = ["## 永続メモリ（要約）"]
        if not pinned and not recent:
            lines.append("要約なし")
            return "\n".join(lines)

        if pinned:
            lines.append("### Pinned（絶対に忘れない）")
            for it in pinned:
                lines.append(f"- {it}")

        if recent:
            lines.append("### Recent（直近の重要メモ）")
            for it in recent:
                lines.append(f"- {it}")

        return "\n".join(lines)

