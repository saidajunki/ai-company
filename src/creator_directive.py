"""Creator directive parser.

Parses Creator's free-form Japanese instruction like "保留" / "中止" / "再開"
and (best-effort) extracts the target task_id / consult_id from message body
or Slack thread context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


DirectiveKind = Literal["pause", "cancel", "resume"]


@dataclass(frozen=True)
class CreatorDirective:
    kind: DirectiveKind
    raw_text: str
    task_id: str | None = None
    consult_id: str | None = None
    initiative_id: str | None = None
    query: str | None = None


_TASK_ID_RE = re.compile(r"task_id\s*[:：]\s*([0-9a-f]{8})", re.IGNORECASE)
_CONSULT_ID_RE = re.compile(r"consult_id\s*[:：]\s*([0-9a-f]{8})", re.IGNORECASE)
_INITIATIVE_ID_RE = re.compile(r"initiative_id\s*[:：]\s*([0-9a-f]{8,})", re.IGNORECASE)


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text or "")
    if not matches:
        return None
    return str(matches[-1]).strip() or None


def _derive_query(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    t = re.sub(r"(一旦|とりあえず|保留解除|保留|凍結|中止|停止|ストップ|やめて|やめる|中断|再開)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= 3 else None


def _looks_like_directive_text(raw: str, normalized: str) -> bool:
    if _TASK_ID_RE.search(raw) or _CONSULT_ID_RE.search(raw) or _INITIATIVE_ID_RE.search(raw):
        return True

    head = normalized.strip().lower()
    if head.startswith(("pause", "hold", "cancel", "stop", "resume", "保留", "中止", "停止", "再開")):
        return True

    imperative_markers = (
        "してください", "して下さい", "してほしい", "して欲しい",
        "お願いします", "お願い", "頼む", "してくれ", "しろ",
        "保留で", "中止で", "停止で", "再開で",
        "保留して", "中止して", "停止して", "再開して",
        "やめて", "止めて",
    )
    return any(marker in normalized for marker in imperative_markers)


def parse_creator_directive(
    text: str,
    *,
    thread_context: str | None = None,
) -> CreatorDirective | None:
    """Parse a pause/cancel/resume directive from free-form text.

    Returns None when no directive intent is detected.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # --- Explicit commands (preferred, unambiguous) ---
    explicit_patterns: list[tuple[DirectiveKind, re.Pattern[str]]] = [
        ("pause", re.compile(r"^(?:pause|hold|保留)\s+([0-9a-f]{8})(?:\s*[:：]\s*(.*))?$", re.IGNORECASE)),
        ("cancel", re.compile(r"^(?:cancel|stop|中止|停止)\s+([0-9a-f]{8})(?:\s*[:：]\s*(.*))?$", re.IGNORECASE)),
        ("resume", re.compile(r"^(?:resume|再開)\s+([0-9a-f]{8})(?:\s*[:：]\s*(.*))?$", re.IGNORECASE)),
    ]
    for kind, pat in explicit_patterns:
        m = pat.match(raw)
        if not m:
            continue
        task_id = m.group(1).strip()
        reason = (m.group(2) or "").strip() or raw
        consult_id = _last_match(_CONSULT_ID_RE, raw) or _last_match(_CONSULT_ID_RE, thread_context or "")
        initiative_id = _last_match(_INITIATIVE_ID_RE, raw) or _last_match(_INITIATIVE_ID_RE, thread_context or "")
        return CreatorDirective(
            kind=kind,
            raw_text=reason,
            task_id=task_id,
            consult_id=consult_id,
            initiative_id=initiative_id,
            query=_derive_query(reason),
        )

    # --- Intent detection (free-form) ---
    normalized = raw.replace("　", " ")

    # Resume has higher priority when "保留解除" is present.
    if any(k in normalized for k in ("保留解除", "再開", "再開して")):
        kind: DirectiveKind = "resume"
    elif any(k in normalized for k in ("やめて", "中止", "停止", "ストップ", "廃止", "打ち切", "もうやらない", "やらなくていい")):
        kind = "cancel"
    elif any(k in normalized for k in ("保留", "凍結", "後回し", "あとで", "棚上げ", "一旦置いといて")):
        kind = "pause"
    else:
        return None

    if not _looks_like_directive_text(raw, normalized):
        return None

    task_id = _last_match(_TASK_ID_RE, raw) or _last_match(_TASK_ID_RE, thread_context or "")
    consult_id = _last_match(_CONSULT_ID_RE, raw) or _last_match(_CONSULT_ID_RE, thread_context or "")
    initiative_id = _last_match(_INITIATIVE_ID_RE, raw) or _last_match(_INITIATIVE_ID_RE, thread_context or "")

    return CreatorDirective(
        kind=kind,
        raw_text=raw,
        task_id=task_id,
        consult_id=consult_id,
        initiative_id=initiative_id,
        query=_derive_query(raw),
    )
