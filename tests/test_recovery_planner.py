"""Tests for RecoveryPlanner.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from models import CreatorReview
from recovery_planner import RecoveryPlanner, _AUTONOMOUS_SCORE_THRESHOLD


NOW = datetime.now(timezone.utc)


def _make_review(score: int) -> CreatorReview:
    return CreatorReview(timestamp=NOW, score_total_100=score)


def _make_planner(
    latest_review: CreatorReview | None = None,
    plan_result: list | None = None,
) -> tuple[RecoveryPlanner, MagicMock, MagicMock]:
    """Create a RecoveryPlanner with mocked manager and initiative_planner."""
    manager = MagicMock()
    manager.creator_review_store.latest.return_value = latest_review
    manager._slack_send = MagicMock()

    initiative_planner = MagicMock()
    initiative_planner.plan.return_value = plan_result or []

    rp = RecoveryPlanner(manager=manager, initiative_planner=initiative_planner)
    return rp, manager, initiative_planner


# ---------------------------------------------------------------------------
# should_plan_autonomously
# ---------------------------------------------------------------------------

class TestShouldPlanAutonomously:
    def test_no_reviews_returns_false(self):
        rp, _, _ = _make_planner(latest_review=None)
        assert rp.should_plan_autonomously() is False

    def test_score_below_threshold_returns_false(self):
        rp, _, _ = _make_planner(latest_review=_make_review(39))
        assert rp.should_plan_autonomously() is False

    def test_score_at_threshold_returns_true(self):
        rp, _, _ = _make_planner(latest_review=_make_review(40))
        assert rp.should_plan_autonomously() is True

    def test_score_above_threshold_returns_true(self):
        rp, _, _ = _make_planner(latest_review=_make_review(85))
        assert rp.should_plan_autonomously() is True

    def test_score_zero_returns_false(self):
        rp, _, _ = _make_planner(latest_review=_make_review(0))
        assert rp.should_plan_autonomously() is False

    def test_score_max_returns_true(self):
        rp, _, _ = _make_planner(latest_review=_make_review(100))
        assert rp.should_plan_autonomously() is True


# ---------------------------------------------------------------------------
# handle_idle — autonomous planning path
# ---------------------------------------------------------------------------

class TestHandleIdleAutonomous:
    def test_calls_initiative_planner_when_autonomous(self):
        entry = MagicMock()
        entry.title = "OSSツール公開"
        rp, manager, ip = _make_planner(
            latest_review=_make_review(60),
            plan_result=[entry],
        )
        result = rp.handle_idle()
        ip.plan.assert_called_once()
        assert "OSSツール公開" in result

    def test_slack_report_on_autonomous_plan(self):
        """Req 4.4: Slackに自律計画の報告を送信する."""
        rp, manager, _ = _make_planner(
            latest_review=_make_review(50),
            plan_result=[],
        )
        rp.handle_idle()
        manager._slack_send.assert_called_once_with("自律的にイニシアチブを計画しました")

    def test_returns_description_with_titles(self):
        e1 = MagicMock()
        e1.title = "施策A"
        e2 = MagicMock()
        e2.title = "施策B"
        rp, _, _ = _make_planner(
            latest_review=_make_review(70),
            plan_result=[e1, e2],
        )
        result = rp.handle_idle()
        assert "施策A" in result
        assert "施策B" in result

    def test_empty_plan_result(self):
        rp, _, _ = _make_planner(
            latest_review=_make_review(45),
            plan_result=[],
        )
        result = rp.handle_idle()
        assert "計画結果なし" in result


# ---------------------------------------------------------------------------
# handle_idle — consult creator path
# ---------------------------------------------------------------------------

class TestHandleIdleConsult:
    def test_consults_creator_when_no_reviews(self):
        rp, manager, ip = _make_planner(latest_review=None)
        result = rp.handle_idle()
        ip.plan.assert_not_called()
        manager._slack_send.assert_called_once()
        assert "Creator" in result

    def test_consults_creator_when_low_score(self):
        rp, manager, ip = _make_planner(latest_review=_make_review(30))
        result = rp.handle_idle()
        ip.plan.assert_not_called()
        manager._slack_send.assert_called_once()
        assert "Creator" in result

    def test_slack_message_on_consult(self):
        rp, manager, _ = _make_planner(latest_review=_make_review(10))
        rp.handle_idle()
        sent_msg = manager._slack_send.call_args[0][0]
        assert "Creator" in sent_msg or "方向性" in sent_msg


# ---------------------------------------------------------------------------
# handle_idle — error handling
# ---------------------------------------------------------------------------

class TestHandleIdleErrors:
    def test_falls_back_to_consult_on_plan_exception(self):
        """If initiative_planner.plan() raises, fall back to consulting Creator."""
        rp, manager, ip = _make_planner(latest_review=_make_review(80))
        ip.plan.side_effect = RuntimeError("LLM failure")
        result = rp.handle_idle()
        assert "Creator" in result
        # Should have sent a consult message (fallback)
        assert manager._slack_send.call_count >= 1
