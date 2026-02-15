"""Post-wakeup action priority logic.

After a wakeup (叩き起こし), the Manager determines its first action based on
the current state:

1. WIP exists → resume WIP (highest priority)  (Req 10.4)
2. Pending approvals exist → report to Creator   (Req 10.5 implied)
3. No tasks → consult Creator                    (Req 10.5)

Requirements: 10.4, 10.5
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from manager_state import ManagerState


RecoveryAction = Literal["resume_wip", "report_pending_approvals", "consult_creator"]


def determine_recovery_action(state: ManagerState) -> tuple[RecoveryAction, str]:
    """Determine the first action after a wakeup based on current state.

    Returns:
        A ``(action, description)`` tuple where *action* is one of
        ``"resume_wip"``, ``"report_pending_approvals"``, or
        ``"consult_creator"``, and *description* is a human-readable
        explanation of the situation.
    """
    # Priority 1: WIP exists → resume (Req 10.4)
    if state.wip:
        items = ", ".join(state.wip)
        return ("resume_wip", f"WIP再開: {items}")

    # Priority 2: Pending approvals (decision_log entries with status="proposed")
    pending = [
        entry for entry in state.decision_log if entry.status == "proposed"
    ]
    if pending:
        summaries = "; ".join(e.decision for e in pending)
        return (
            "report_pending_approvals",
            f"承認待ち {len(pending)} 件: {summaries}",
        )

    # Priority 3: Nothing to do → consult Creator (Req 10.5)
    return ("consult_creator", "タスクなし: Creatorに次のアクションを相談します")
