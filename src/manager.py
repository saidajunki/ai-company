"""Manager – orchestration layer tying all components together.

Provides:
- init_company_directory: Create the full directory structure and default constitution
- Manager class: Synchronous logic layer for event-driven operations

Requirements: 7.1, 7.2, 7.6
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from constitution_store import constitution_save
from cost_aggregator import compute_window_cost, is_budget_exceeded
from heartbeat import update_heartbeat, update_heartbeat_on_report
from manager_state import (
    ManagerState,
    append_ledger_event,
    restore_state,
)
from models import (
    ConstitutionModel,
    HeartbeatState,
    LedgerEvent,
)
from pricing import (
    get_pricing_with_fallback,
    load_pricing_cache,
    pricing_cache_path,
)
from recovery import determine_recovery_action, RecoveryAction
from report_formatter import CostSummary, ReportData, format_report


# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------

_SUBDIRS = [
    "ledger",
    "decisions",
    "state",
    "pricing",
    "templates",
    "schemas",
    "protocols",
]


def init_company_directory(base_dir: Path, company_id: str) -> None:
    """Create the full company directory structure and a default constitution.

    Directory layout::

        companies/<company_id>/
        ├── constitution.yaml
        ├── ledger/
        ├── decisions/
        ├── state/
        ├── pricing/
        ├── templates/
        ├── schemas/
        └── protocols/
    """
    company_root = base_dir / "companies" / company_id
    for sub in _SUBDIRS:
        (company_root / sub).mkdir(parents=True, exist_ok=True)

    # Write default constitution if it doesn't already exist
    constitution_file = company_root / "constitution.yaml"
    if not constitution_file.exists():
        constitution_save(constitution_file, ConstitutionModel())


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_LIMIT_USD = 10.0
DEFAULT_WINDOW_MINUTES = 60


class Manager:
    """Synchronous orchestration layer for the AI company.

    Ties together all components (state, heartbeat, cost, reports, recovery).
    This is the logic layer – actual Slack I/O and asyncio event loops are
    handled externally and call into these methods.

    Req 7.1: Single long-running process (this class is the core).
    Req 7.2: Event-driven – methods are invoked by external event sources.
    """

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self.base_dir = base_dir
        self.company_id = company_id
        self.pid = os.getpid()

        # Restore persisted state
        self.state: ManagerState = restore_state(base_dir, company_id)

        # Load pricing cache
        cache_path = pricing_cache_path(base_dir, company_id)
        self.pricing_cache = load_pricing_cache(cache_path)

    # ------------------------------------------------------------------
    # Startup (Req 7.1, 7.6)
    # ------------------------------------------------------------------

    def startup(self) -> tuple[RecoveryAction, str]:
        """Restore state, update heartbeat, and determine recovery action.

        Returns:
            ``(action, description)`` from recovery logic.
        """
        # Re-read state (idempotent – already done in __init__, but explicit
        # for callers who construct then call startup separately)
        self.state = restore_state(self.base_dir, self.company_id)

        # Update heartbeat to signal we're alive (Req 7.6)
        status = "running" if self.state.wip else "idle"
        self.state.heartbeat = update_heartbeat(
            self.base_dir,
            self.company_id,
            status=status,
            current_wip=self.state.wip,
            pid=self.pid,
        )

        # Determine what to do first after wakeup
        action, description = determine_recovery_action(self.state)
        return action, description

    # ------------------------------------------------------------------
    # Budget check (Req 5.3, 5.4)
    # ------------------------------------------------------------------

    def check_budget(self) -> bool:
        """Return ``True`` if the 60-min sliding window budget is exceeded."""
        now = datetime.now(timezone.utc)
        limit = DEFAULT_BUDGET_LIMIT_USD
        if self.state.constitution and self.state.constitution.budget:
            limit = self.state.constitution.budget.limit_usd
        return is_budget_exceeded(
            self.state.ledger_events,
            now,
            limit_usd=limit,
            window_minutes=DEFAULT_WINDOW_MINUTES,
        )

    # ------------------------------------------------------------------
    # LLM call recording (Req 5.1, 5.2, 9.3)
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: str = "manager",
        task_id: str = "",
    ) -> LedgerEvent:
        """Record an LLM call to the ledger with pricing lookup.

        Uses the pricing cache with fallback logic (Req 9.3).

        Returns:
            The persisted ``LedgerEvent``.
        """
        pricing, source = get_pricing_with_fallback(self.pricing_cache, model)

        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        estimated_cost = input_cost + output_cost

        now = datetime.now(timezone.utc)
        event = LedgerEvent(
            timestamp=now,
            event_type="llm_call",
            agent_id=agent_id,
            task_id=task_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unit_price_usd_per_1k_input_tokens=pricing.input_price_per_1k,
            unit_price_usd_per_1k_output_tokens=pricing.output_price_per_1k,
            price_retrieved_at=pricing.retrieved_at,
            estimated_cost_usd=estimated_cost,
            metadata={"pricing_source": source} if source != "cache" else None,
        )

        append_ledger_event(self.base_dir, self.company_id, event)
        self.state.ledger_events.append(event)
        return event

    # ------------------------------------------------------------------
    # Report generation (Req 3.1, 3.5, 7.6)
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a 10-min report and update heartbeat.

        Returns:
            Formatted Markdown report string.
        """
        now = datetime.now(timezone.utc)

        # Cost summary
        spent = compute_window_cost(self.state.ledger_events, now)
        limit = DEFAULT_BUDGET_LIMIT_USD
        if self.state.constitution and self.state.constitution.budget:
            limit = self.state.constitution.budget.limit_usd
        remaining = max(0.0, limit - spent)

        cost_summary = CostSummary(
            spent_usd=spent,
            remaining_usd=remaining,
        )

        data = ReportData(
            timestamp=now,
            company_id=self.company_id,
            wip=list(self.state.wip),
            cost=cost_summary,
        )

        report = format_report(data)

        # Update heartbeat with last_report_at (Req 3.5)
        self.state.heartbeat = update_heartbeat_on_report(
            self.base_dir,
            self.company_id,
            status="running" if self.state.wip else "idle",
            current_wip=self.state.wip,
            pid=self.pid,
        )

        return report
