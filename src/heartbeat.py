"""Heartbeat update and watchdog detection logic.

Provides:
- update_heartbeat: Create/update heartbeat file with current timestamp
- is_heartbeat_stale: Check if heartbeat exceeds staleness threshold
- update_heartbeat_on_report: Update heartbeat with last_report_at (Req 3.5)

Requirements: 7.6, 10.1, 10.2, 3.5
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from manager_state import save_heartbeat
from models import HeartbeatState


def update_heartbeat(
    base_dir: Path,
    company_id: str,
    status: str,
    current_wip: list[str],
    pid: int,
    last_report_at: datetime | None = None,
) -> HeartbeatState:
    """Create or update the heartbeat file with the current timestamp.

    Args:
        base_dir: Root directory containing ``companies/`` folder.
        company_id: Company identifier.
        status: Manager state (``running``, ``idle``, ``waiting_approval``).
        current_wip: List of in-progress task descriptions (max 3).
        pid: Manager process ID.
        last_report_at: Timestamp of the last report sent (optional).

    Returns:
        The persisted ``HeartbeatState``.
    """
    hb = HeartbeatState(
        updated_at=datetime.now(timezone.utc),
        manager_pid=pid,
        status=status,
        current_wip=current_wip,
        last_report_at=last_report_at,
    )
    save_heartbeat(base_dir, company_id, hb)
    return hb


def is_heartbeat_stale(
    heartbeat: HeartbeatState,
    now: datetime,
    threshold_minutes: int = 20,
) -> bool:
    """Return ``True`` if the heartbeat is stale (Req 10.2).

    A heartbeat is considered stale when ``now - heartbeat.updated_at``
    is **greater than or equal to** *threshold_minutes*.

    Args:
        heartbeat: The heartbeat state to check.
        now: Current reference time (should be timezone-aware UTC).
        threshold_minutes: Minutes after which the heartbeat is stale.
            Defaults to 20 as per Req 10.2.

    Returns:
        ``True`` if stale, ``False`` otherwise.
    """
    elapsed = now - heartbeat.updated_at
    return elapsed.total_seconds() >= threshold_minutes * 60


def update_heartbeat_on_report(
    base_dir: Path,
    company_id: str,
    status: str,
    current_wip: list[str],
    pid: int,
) -> HeartbeatState:
    """Update heartbeat with ``last_report_at`` set to now (Req 3.5).

    Called when a Ten_Minute_Report is sent so that both ``updated_at``
    and ``last_report_at`` reflect the report time.

    Returns:
        The persisted ``HeartbeatState``.
    """
    now = datetime.now(timezone.utc)
    hb = HeartbeatState(
        updated_at=now,
        manager_pid=pid,
        status=status,
        current_wip=current_wip,
        last_report_at=now,
    )
    save_heartbeat(base_dir, company_id, hb)
    return hb
