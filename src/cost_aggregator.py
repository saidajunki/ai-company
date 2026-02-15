"""60-minute sliding window cost aggregation.

Provides functions to:
1. Compute total cost within a 60-minute sliding window (Req 5.3)
2. Check if the cost limit has been reached (Req 5.4)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from models import LedgerEvent

DEFAULT_WINDOW_MINUTES = 60
DEFAULT_LIMIT_USD = 10.0


def compute_window_cost(
    events: list[LedgerEvent],
    now: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> float:
    """Compute total estimated_cost_usd within the sliding window.

    Args:
        events: List of ledger events.
        now: Reference time (right edge of the window).
        window_minutes: Window size in minutes (default 60).

    Returns:
        Sum of estimated_cost_usd for events whose timestamp is
        within (now - window_minutes, now].
    """
    cutoff = now - timedelta(minutes=window_minutes)
    total = 0.0
    for ev in events:
        if ev.timestamp > cutoff and ev.estimated_cost_usd is not None:
            total += ev.estimated_cost_usd
    return total


def is_budget_exceeded(
    events: list[LedgerEvent],
    now: datetime,
    limit_usd: float = DEFAULT_LIMIT_USD,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> bool:
    """Return True if the 60-min sliding window cost >= limit.

    When True, new LLM / paid-API calls must be blocked (Req 5.4).
    """
    return compute_window_cost(events, now, window_minutes) >= limit_usd
