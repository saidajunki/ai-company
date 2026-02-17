"""Data aggregation service for the AI Company Dashboard.

Reads all NDJSON/JSON data files from the shared volume and returns
a unified DashboardData object for the API layer.

Requirements: 1.1, 1.2, 2.1, 2.2, 2.4, 3.1, 3.2, 3.3, 3.4,
              4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 9.1, 9.2
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from models import (
    AgentEntry,
    ConsultationEntry,
    ConversationEntry,
    HeartbeatState,
    InitiativeEntry,
    LedgerEvent,
    TaskEntry,
)
from ndjson_reader import read_ndjson, resolve_latest

logger = logging.getLogger(__name__)

# Status ordering for initiative grouping (Req 2.4)
_INITIATIVE_STATUS_ORDER: dict[str, int] = {
    "in_progress": 0,
    "consulting": 1,
    "planned": 2,
    "completed": 3,
    "abandoned": 4,
}

_DEFAULT_BUDGET_LIMIT_USD = 10.0
_RECENT_TASKS_LIMIT = 50
_COST_WINDOW_MINUTES = 60

_DASHBOARD_AGENTS_MAX = int(os.environ.get("DASHBOARD_AGENTS_MAX", "50"))
_DASHBOARD_INACTIVE_MAX = int(os.environ.get("DASHBOARD_INACTIVE_MAX", "20"))
_DASHBOARD_INACTIVE_RECENT_HOURS = int(os.environ.get("DASHBOARD_INACTIVE_RECENT_HOURS", "24"))


@dataclass
class TasksSummary:
    """Task counts by status."""

    pending: int
    running: int
    completed: int
    failed: int
    total: int


@dataclass
class CostSummary:
    """Cost usage summary."""

    window_60min_usd: float
    total_usd: float
    budget_limit_usd: float
    budget_usage_percent: float


@dataclass
class DashboardData:
    """Aggregated dashboard data returned by DataService."""

    heartbeat: HeartbeatState | None
    agents: list[AgentEntry]
    initiatives: list[InitiativeEntry]
    tasks_summary: TasksSummary
    recent_tasks: list[TaskEntry]
    consultations: list[ConsultationEntry]
    cost: CostSummary
    conversations: list[ConversationEntry]


class DataService:
    """Reads all data sources and returns aggregated dashboard data."""

    def __init__(self, data_dir: Path, company_id: str = "alpha") -> None:
        self.data_dir = data_dir
        self.company_id = company_id

    # ── public API ──────────────────────────────────────────────

    def get_dashboard_data(self) -> DashboardData:
        """Read all data sources and return aggregated dashboard data."""
        state_dir = self.data_dir / "companies" / self.company_id / "state"
        ledger_dir = self.data_dir / "companies" / self.company_id / "ledger"

        heartbeat = self._read_heartbeat(state_dir / "heartbeat.json")
        agents = resolve_latest(
            read_ndjson(state_dir / "agents.ndjson", AgentEntry), "agent_id"
        )
        agents = self._filter_agents_for_dashboard(agents)
        initiatives = self._sort_initiatives(
            resolve_latest(
                read_ndjson(state_dir / "initiatives.ndjson", InitiativeEntry),
                "initiative_id",
            )
        )
        tasks = resolve_latest(
            read_ndjson(state_dir / "tasks.ndjson", TaskEntry), "task_id"
        )
        consultations = self._sort_consultations(
            resolve_latest(
                read_ndjson(state_dir / "consultations.ndjson", ConsultationEntry),
                "consultation_id",
            )
        )
        conversations = self._sort_conversations(
            read_ndjson(state_dir / "conversations.ndjson", ConversationEntry)
        )
        ledger_events = read_ndjson(ledger_dir / "events.ndjson", LedgerEvent)

        return DashboardData(
            heartbeat=heartbeat,
            agents=agents,
            initiatives=initiatives,
            tasks_summary=self._compute_tasks_summary(tasks),
            recent_tasks=self._recent_tasks(tasks),
            consultations=consultations,
            cost=self._compute_cost(ledger_events),
            conversations=conversations,
        )

    @staticmethod
    def _filter_agents_for_dashboard(agents: list[AgentEntry]) -> list[AgentEntry]:
        """Filter/sort agents to keep the dashboard usable even with many historical sub-agents."""
        if not agents:
            return []

        now = datetime.now(timezone.utc)

        def _ts(a: AgentEntry) -> datetime:
            ts = a.updated_at
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

        # Always keep key agents even if limits are exceeded.
        pinned_ids = {"ceo", "manager"}
        pinned = [a for a in agents if a.agent_id in pinned_ids]

        active = [a for a in agents if a.status == "active" and a.agent_id not in pinned_ids]

        recent_inactive: list[AgentEntry] = []
        cutoff_seconds = _DASHBOARD_INACTIVE_RECENT_HOURS * 3600
        for a in agents:
            if a.status != "inactive" or a.agent_id in pinned_ids:
                continue
            age = (now - _ts(a)).total_seconds()
            if age <= cutoff_seconds:
                recent_inactive.append(a)

        # Sort: active first (newest), then inactive (newest)
        active.sort(key=_ts, reverse=True)
        recent_inactive.sort(key=_ts, reverse=True)
        recent_inactive = recent_inactive[: max(0, _DASHBOARD_INACTIVE_MAX)]

        combined = pinned + active + recent_inactive

        # De-dup while preserving order (pinned may overlap).
        seen: set[str] = set()
        ordered: list[AgentEntry] = []
        for a in combined:
            if a.agent_id in seen:
                continue
            seen.add(a.agent_id)
            ordered.append(a)

        # As a final safety, cap to keep UI responsive (but keep pinned).
        if _DASHBOARD_AGENTS_MAX <= 0:
            return ordered

        if len(ordered) <= _DASHBOARD_AGENTS_MAX:
            return ordered

        keep: list[AgentEntry] = []
        keep_ids = set()
        for a in ordered:
            if a.agent_id in pinned_ids and a.agent_id not in keep_ids:
                keep.append(a)
                keep_ids.add(a.agent_id)

        for a in ordered:
            if a.agent_id in keep_ids:
                continue
            if len(keep) >= _DASHBOARD_AGENTS_MAX:
                break
            keep.append(a)
            keep_ids.add(a.agent_id)

        return keep

    # ── heartbeat ───────────────────────────────────────────────

    @staticmethod
    def _read_heartbeat(path: Path) -> HeartbeatState | None:
        """Read heartbeat.json (single JSON, not NDJSON). Returns None on error."""
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            return HeartbeatState.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"Failed to read heartbeat from {path}: {e}")
            return None

    # ── cost calculation (Req 3.1, 3.4) ─────────────────────────

    @staticmethod
    def _compute_cost(events: list[LedgerEvent]) -> CostSummary:
        now = datetime.now(timezone.utc)
        total_usd = 0.0
        window_usd = 0.0

        for ev in events:
            cost = ev.estimated_cost_usd
            if cost is None:
                continue
            total_usd += cost
            # Ensure we compare timezone-aware datetimes
            ev_ts = ev.timestamp if ev.timestamp.tzinfo else ev.timestamp.replace(tzinfo=timezone.utc)
            diff_seconds = (now - ev_ts).total_seconds()
            if diff_seconds <= _COST_WINDOW_MINUTES * 60:
                window_usd += cost

        budget_limit = _DEFAULT_BUDGET_LIMIT_USD
        usage_pct = (window_usd / budget_limit * 100) if budget_limit > 0 else 0.0

        return CostSummary(
            window_60min_usd=window_usd,
            total_usd=total_usd,
            budget_limit_usd=budget_limit,
            budget_usage_percent=usage_pct,
        )

    # ── consultation sorting (Req 4.2) ──────────────────────────

    @staticmethod
    def _sort_consultations(entries: list[ConsultationEntry]) -> list[ConsultationEntry]:
        """Sort consultations: pending first, then resolved. Stable within groups."""
        return sorted(entries, key=lambda c: 0 if c.status == "pending" else 1)

    # ── conversation sorting (Req 9.1) ──────────────────────────

    @staticmethod
    def _sort_conversations(entries: list[ConversationEntry]) -> list[ConversationEntry]:
        """Sort conversations by timestamp descending (newest first)."""
        return sorted(entries, key=lambda c: c.timestamp, reverse=True)

    # ── initiative grouping (Req 2.4) ───────────────────────────

    @staticmethod
    def _sort_initiatives(entries: list[InitiativeEntry]) -> list[InitiativeEntry]:
        """Group initiatives by status in defined order."""
        return sorted(
            entries,
            key=lambda i: _INITIATIVE_STATUS_ORDER.get(i.status, 99),
        )

    # ── task summary (Req 5.3) ──────────────────────────────────

    @staticmethod
    def _compute_tasks_summary(tasks: list[TaskEntry]) -> TasksSummary:
        counts: dict[str, int] = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for t in tasks:
            if t.status in counts:
                counts[t.status] += 1
        return TasksSummary(
            pending=counts["pending"],
            running=counts["running"],
            completed=counts["completed"],
            failed=counts["failed"],
            total=len(tasks),
        )

    # ── recent tasks (top 50 by updated_at desc) ────────────────

    @staticmethod
    def _recent_tasks(tasks: list[TaskEntry]) -> list[TaskEntry]:
        return sorted(tasks, key=lambda t: t.updated_at, reverse=True)[:_RECENT_TASKS_LIMIT]
