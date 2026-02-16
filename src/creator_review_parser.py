"""Creator review parser.

Parses Creator's score feedback message into a structured CreatorReview model.
This is used as a short-term KPI loop (日報 → 採点 → 学習).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from models import CreatorReview


_FULLWIDTH_TRANSLATION_TABLE = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "：": ":",
        "／": "/",
        "，": ",",
        "　": " ",
    }
)


def parse_creator_review(
    text: str,
    *,
    user_id: str | None = None,
    timestamp: datetime | None = None,
) -> CreatorReview | None:
    """Parse a Creator scoring message. Returns None if the message isn't a review."""
    raw = (text or "").strip()
    if not raw:
        return None

    normalized = raw.translate(_FULLWIDTH_TRANSLATION_TABLE)

    interestingness = _extract_axis_score(normalized, ["面白さ", "おもしろさ", "興味"], max_score=25)
    cost_efficiency = _extract_axis_score(normalized, ["コスト効率", "費用対効果", "コスパ"], max_score=25)
    realism = _extract_axis_score(normalized, ["現実性", "実現性", "実行可能性"], max_score=25)
    evolvability = _extract_axis_score(normalized, ["進化性", "学習", "ナレッジ", "蓄積"], max_score=25)

    total = _extract_total_score(normalized)

    # If all axes exist, total is derived from them (source of truth)
    if all(v is not None for v in (interestingness, cost_efficiency, realism, evolvability)):
        total = int(interestingness + cost_efficiency + realism + evolvability)  # type: ignore[operator]

    if total is None:
        return None

    comment = _extract_comment(normalized)
    if comment is None:
        comment = raw

    try:
        return CreatorReview(
            timestamp=timestamp or datetime.now(timezone.utc),
            user_id=user_id,
            score_total_100=total,
            score_interestingness_25=interestingness,
            score_cost_efficiency_25=cost_efficiency,
            score_realism_25=realism,
            score_evolvability_25=evolvability,
            comment=comment.strip(),
            raw_text=raw,
        )
    except Exception:
        # If values are out of range or invalid, treat as non-review.
        return None


def _extract_axis_score(text: str, keywords: list[str], *, max_score: int) -> int | None:
    for kw in keywords:
        pattern = re.compile(
            rf"{re.escape(kw)}\s*[:=]?\s*(\d{{1,3}})\s*(?:/\s*{max_score})?",
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            continue
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _extract_total_score(text: str) -> int | None:
    patterns = [
        r"(?:合計|総合|total|score|スコア)\s*[:=]?\s*(\d{1,3})\s*(?:/\s*100)?",
        r"(\d{1,3})\s*/\s*100",
        r"(?:合計|総合|スコア)\s*[:=]?\s*(\d{1,3})\s*点",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _extract_comment(text: str) -> str | None:
    m = re.search(r"(?:コメント|comment)\s*[:=]\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

