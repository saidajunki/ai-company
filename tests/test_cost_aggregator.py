"""Tests for cost_aggregator module.

Covers:
- compute_window_cost: 60-min sliding window aggregation (Req 5.3)
- is_budget_exceeded: $10 limit check (Req 5.4)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cost_aggregator import compute_window_cost, is_budget_exceeded
from models import LedgerEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    timestamp: datetime,
    cost: float,
    event_type: str = "llm_call",
) -> LedgerEvent:
    """Create a minimal LedgerEvent with the given timestamp and cost."""
    base = dict(
        timestamp=timestamp,
        event_type=event_type,
        agent_id="test-agent",
        task_id="test-task",
        estimated_cost_usd=cost,
    )
    if event_type == "llm_call":
        base.update(
            provider="openrouter",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            unit_price_usd_per_1k_input_tokens=0.01,
            unit_price_usd_per_1k_output_tokens=0.02,
            price_retrieved_at=timestamp,
        )
    return LedgerEvent(**base)


NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# compute_window_cost
# ---------------------------------------------------------------------------

class TestComputeWindowCost:
    """Tests for compute_window_cost (Req 5.3)."""

    def test_empty_events(self):
        assert compute_window_cost([], NOW) == 0.0

    def test_single_event_inside_window(self):
        ev = _make_event(NOW - timedelta(minutes=30), 2.5)
        assert compute_window_cost([ev], NOW) == 2.5

    def test_single_event_outside_window(self):
        ev = _make_event(NOW - timedelta(minutes=61), 2.5)
        assert compute_window_cost([ev], NOW) == 0.0

    def test_event_exactly_at_boundary(self):
        """Event exactly 60 minutes ago should NOT be included.

        The window is (now - 60min, now], so the cutoff itself is excluded.
        """
        ev = _make_event(NOW - timedelta(minutes=60), 5.0)
        assert compute_window_cost([ev], NOW) == 0.0

    def test_event_just_inside_boundary(self):
        ev = _make_event(NOW - timedelta(minutes=59, seconds=59), 5.0)
        assert compute_window_cost([ev], NOW) == 5.0

    def test_multiple_events_mixed(self):
        events = [
            _make_event(NOW - timedelta(minutes=10), 1.0),
            _make_event(NOW - timedelta(minutes=30), 2.0),
            _make_event(NOW - timedelta(minutes=59), 3.0),
            _make_event(NOW - timedelta(minutes=61), 4.0),  # outside
        ]
        assert compute_window_cost(events, NOW) == 6.0

    def test_events_with_none_cost_ignored(self):
        """Events without estimated_cost_usd should be ignored."""
        ev = _make_event(NOW - timedelta(minutes=5), 0.0)
        ev.estimated_cost_usd = None  # type: ignore[assignment]
        # Need to bypass validation for this test
        ev_good = _make_event(NOW - timedelta(minutes=5), 3.0)
        assert compute_window_cost([ev, ev_good], NOW) == 3.0

    def test_non_llm_events_with_cost(self):
        """Non-llm events (e.g. shell_exec) with cost should be counted."""
        ev = _make_event(NOW - timedelta(minutes=5), 0.5, event_type="shell_exec")
        assert compute_window_cost([ev], NOW) == 0.5


# ---------------------------------------------------------------------------
# is_budget_exceeded
# ---------------------------------------------------------------------------

class TestIsBudgetExceeded:
    """Tests for is_budget_exceeded (Req 5.4)."""

    def test_empty_events_not_exceeded(self):
        assert is_budget_exceeded([], NOW) is False

    def test_below_limit(self):
        events = [_make_event(NOW - timedelta(minutes=5), 9.99)]
        assert is_budget_exceeded(events, NOW) is False

    def test_exactly_at_limit(self):
        """$10.00 exactly should be considered exceeded (>= $10)."""
        events = [_make_event(NOW - timedelta(minutes=5), 10.0)]
        assert is_budget_exceeded(events, NOW) is True

    def test_above_limit(self):
        events = [_make_event(NOW - timedelta(minutes=5), 15.0)]
        assert is_budget_exceeded(events, NOW) is True

    def test_cumulative_exceeds_limit(self):
        events = [
            _make_event(NOW - timedelta(minutes=10), 4.0),
            _make_event(NOW - timedelta(minutes=20), 3.5),
            _make_event(NOW - timedelta(minutes=30), 3.0),
        ]
        # total = 10.5 >= 10
        assert is_budget_exceeded(events, NOW) is True

    def test_old_events_dont_count(self):
        events = [
            _make_event(NOW - timedelta(minutes=61), 9.0),  # outside
            _make_event(NOW - timedelta(minutes=5), 1.0),   # inside
        ]
        assert is_budget_exceeded(events, NOW) is False

    def test_custom_limit(self):
        events = [_make_event(NOW - timedelta(minutes=5), 5.0)]
        assert is_budget_exceeded(events, NOW, limit_usd=5.0) is True
        assert is_budget_exceeded(events, NOW, limit_usd=6.0) is False
