"""Unit tests for post-wakeup recovery action priority logic.

Covers:
- WIP exists → resume_wip (Req 10.4)
- Pending approvals → report_pending_approvals
- No tasks → consult_creator (Req 10.5)
- Edge cases: empty state, WIP takes priority over pending approvals
"""

from __future__ import annotations

from datetime import date

import pytest

from manager_state import ManagerState
from models import DecisionLogEntry
from recovery import determine_recovery_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(status: str = "decided", decision: str = "test decision") -> DecisionLogEntry:
    return DecisionLogEntry(
        date=date(2025, 6, 1),
        decision=decision,
        why="test reason",
        scope="test scope",
        revisit="none",
        status=status,
    )


# ---------------------------------------------------------------------------
# Priority 1: WIP exists → resume_wip
# ---------------------------------------------------------------------------

class TestResumeWip:
    def test_single_wip_item(self):
        state = ManagerState(wip=["task-A"])
        action, desc = determine_recovery_action(state)
        assert action == "resume_wip"
        assert "task-A" in desc

    def test_multiple_wip_items(self):
        state = ManagerState(wip=["task-A", "task-B", "task-C"])
        action, desc = determine_recovery_action(state)
        assert action == "resume_wip"
        assert "task-A" in desc
        assert "task-B" in desc
        assert "task-C" in desc

    def test_wip_takes_priority_over_pending_approvals(self):
        """Req 10.4: WIP resume is highest priority, even with pending approvals."""
        state = ManagerState(
            wip=["task-X"],
            decision_log=[_decision(status="proposed", decision="change budget")],
        )
        action, _ = determine_recovery_action(state)
        assert action == "resume_wip"


# ---------------------------------------------------------------------------
# Priority 2: Pending approvals → report_pending_approvals
# ---------------------------------------------------------------------------

class TestReportPendingApprovals:
    def test_single_pending_approval(self):
        state = ManagerState(
            decision_log=[_decision(status="proposed", decision="deploy v2")],
        )
        action, desc = determine_recovery_action(state)
        assert action == "report_pending_approvals"
        assert "deploy v2" in desc
        assert "1" in desc

    def test_multiple_pending_approvals(self):
        state = ManagerState(
            decision_log=[
                _decision(status="proposed", decision="deploy v2"),
                _decision(status="proposed", decision="update config"),
            ],
        )
        action, desc = determine_recovery_action(state)
        assert action == "report_pending_approvals"
        assert "2" in desc

    def test_ignores_non_proposed_entries(self):
        """Only status='proposed' counts as pending approval."""
        state = ManagerState(
            decision_log=[
                _decision(status="approved", decision="old approval"),
                _decision(status="rejected", decision="old rejection"),
                _decision(status="decided", decision="old decision"),
            ],
        )
        action, _ = determine_recovery_action(state)
        assert action == "consult_creator"

    def test_mixed_statuses_picks_proposed(self):
        state = ManagerState(
            decision_log=[
                _decision(status="approved", decision="done"),
                _decision(status="proposed", decision="pending one"),
                _decision(status="rejected", decision="nope"),
            ],
        )
        action, desc = determine_recovery_action(state)
        assert action == "report_pending_approvals"
        assert "pending one" in desc


# ---------------------------------------------------------------------------
# Priority 3: No tasks → consult_creator
# ---------------------------------------------------------------------------

class TestConsultCreator:
    def test_empty_state(self):
        """Completely empty state → consult creator (Req 10.5)."""
        state = ManagerState()
        action, desc = determine_recovery_action(state)
        assert action == "consult_creator"
        assert "Creator" in desc

    def test_no_wip_no_pending(self):
        state = ManagerState(
            decision_log=[_decision(status="approved")],
        )
        action, _ = determine_recovery_action(state)
        assert action == "consult_creator"

    def test_empty_wip_list(self):
        """Explicit empty WIP list → consult creator."""
        state = ManagerState(wip=[])
        action, _ = determine_recovery_action(state)
        assert action == "consult_creator"
