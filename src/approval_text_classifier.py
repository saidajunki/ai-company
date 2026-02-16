"""Approval text classifier.

Interprets Creator's free-form Japanese reply as approval/rejection.
Uses simple heuristics first, and optionally falls back to an LLM.
"""

from __future__ import annotations

import re
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from llm_client import LLMClient, LLMError

ApprovalDecision = Literal["approved", "rejected", "unknown"]


_APPROVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bOK\b",
        r"\bok\b",
        r"ok",  # Japanese text often attaches "OK" without word boundaries
        r"いいよ",
        r"進めて",
        r"やって",
        r"やろう",
        r"お願い",
        r"承認",
        r"可$",
        r"賛成",
        r"問題ない",
    ]
]

_REJECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bNG\b",
        r"\bng\b",
        r"却下",
        r"ダメ",
        r"やめて",
        r"やらない",
        r"中止",
        r"ストップ",
        r"見送り",
        r"反対",
    ]
]


def classify_approval_text(
    text: str,
    *,
    request_summary: str = "",
    llm_client: LLMClient | None = None,
) -> ApprovalDecision:
    """Return approved/rejected/unknown for Creator's free-form reply."""
    t = (text or "").strip()
    if not t:
        return "unknown"

    approve_hit = any(p.search(t) for p in _APPROVE_PATTERNS)
    reject_hit = any(p.search(t) for p in _REJECT_PATTERNS)

    if approve_hit and not reject_hit:
        return "approved"
    if reject_hit and not approve_hit:
        return "rejected"
    if approve_hit and reject_hit:
        return "unknown"

    if llm_client is None:
        return "unknown"

    # LLM fallback: ask for a strict label only.
    messages = [
        {
            "role": "system",
            "content": (
                "あなたは承認判定器です。Creatorの返信が承認か却下か不明かを判定してください。\n"
                "出力は必ず次のいずれか1語のみ: approved / rejected / unknown"
            ),
        },
        {
            "role": "user",
            "content": (
                f"承認リクエスト概要:\n{request_summary}\n\n"
                f"Creator返信:\n{t}"
            ),
        },
    ]

    result = llm_client.chat(messages)
    if isinstance(result, LLMError):
        return "unknown"

    label = (result.content or "").strip().lower()
    if "approved" in label:
        return "approved"
    if "rejected" in label:
        return "rejected"
    if "unknown" in label:
        return "unknown"
    return "unknown"
