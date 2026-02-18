"""Manager – orchestration layer tying all components together.

Provides:
- init_company_directory: Create the full directory structure and default constitution
- Manager class: Synchronous logic layer for event-driven operations

Requirements: 7.1, 7.2, 7.6
"""

from __future__ import annotations

import logging
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from agent_registry import AgentRegistry
from constitution_store import constitution_save
from context_builder import build_system_prompt, TaskHistoryContext
from conversation_memory import ConversationMemory
from cost_aggregator import compute_window_cost, is_budget_exceeded
from creator_review_parser import parse_creator_review
from creator_review_store import CreatorReviewStore
from creator_directive import CreatorDirective, parse_creator_directive
from daily_brief_formatter import DailyBriefData, DailyCostSummary, format_daily_brief
from heartbeat import update_heartbeat, update_heartbeat_on_report
from llm_client import LLMClient, LLMError, LLMResponse
from manager_state import (
    ManagerState,
    append_ledger_event,
    restore_state,
)
from models import (
    ConstitutionModel,
    ConversationEntry,
    HeartbeatState,
    LedgerEvent,
    ResearchNote,
)
from pricing import (
    get_pricing_with_fallback,
    load_pricing_cache,
    pricing_cache_path,
    refresh_openrouter_pricing_cache,
)
from recovery import determine_recovery_action, RecoveryAction
from report_formatter import CostSummary, ReportData, format_report
from response_parser import Action, parse_plan_content, parse_response
from research_note_store import ResearchNoteStore
from consultation_store import ConsultationStore
from commitment_store import CommitmentStore
from consultation_policy import assess_creator_consultation
from service_registry import ServiceRegistry
from shell_executor import ShellResult, execute_shell
from sub_agent_runner import SubAgentRunner
from git_publisher import GitPublisher
from initiative_store import InitiativeStore
from initiative_planner import InitiativePlanner
from strategy_analyzer import StrategyAnalyzer
from model_catalog import build_model_catalog, format_model_catalog_for_prompt
from memory_manager import MemoryManager
from memory_vault import DEFAULT_CURATED_MEMORY, curated_memory_path, MemoryVault
from policy_memory_store import PolicyMemoryStore
from adaptive_memory_store import AdaptiveMemoryStore
from procedure_store import ProcedureStore
from web_searcher import WebSearcher
from mcp_client import MCPClient
from task_queue import TaskQueue
from vision_loader import DEFAULT_VISION, VisionLoader
from alarm_scheduler import AlarmScheduler
from employee_store import EmployeeStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------

_SUBDIRS = [
    "ledger",
    "decisions",
    "state",
    "pricing",
    "knowledge",
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

    vision_file = company_root / "vision.md"
    if not vision_file.exists():
        vision_file.write_text(DEFAULT_VISION.rstrip() + "\n", encoding="utf-8")

    # Curated memory (file-first LTM)
    mem_path = curated_memory_path(base_dir, company_id)
    if not mem_path.exists():
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(DEFAULT_CURATED_MEMORY.rstrip() + "\n", encoding="utf-8")

    # MCP servers config (company SoT)
    mcp_servers_file = company_root / "protocols" / "mcp_servers.yaml"
    if not mcp_servers_file.exists():
        mcp_servers_file.write_text(
            "\n".join([
                "servers:",
                "  vps-monitor:",
                "    api_base: \"https://mcp.app.babl.tech\"",
                "    token_env: \"AI_COMPANY_MCP_VPS_MONITOR_TOKEN\"",
                "    desc: \"VPS監視MCPサーバ\"",
                "",
            ]),
            encoding="utf-8",
        )

    # Newsroom sources config (RSS SoT)
    newsroom_sources_file = company_root / "protocols" / "newsroom_sources.yaml"
    if not newsroom_sources_file.exists():
        newsroom_sources_file.write_text(
            "\n".join([
                "schedule:",
                "  interval_minutes: 60",
                "budgets:",
                "  post_usd: 0.5",
                "  research_usd: 0.2",
                "  writer_usd: 0.25",
                "wordpress:",
                "  categories: [\"ニュース\", \"AI\", \"テクノロジー\"]",
                "  status: publish",
                "sources:",
                "  - name: \"TechCrunch AI\"",
                "    rss: \"https://techcrunch.com/category/artificial-intelligence/feed/\"",
                "    keywords: [\"ai\", \"artificial intelligence\", \"llm\", \"model\", \"agent\"]",
                "  - name: \"VentureBeat AI\"",
                "    rss: \"https://venturebeat.com/category/ai/feed/\"",
                "    keywords: [\"ai\", \"llm\", \"model\", \"inference\", \"agent\"]",
                "  - name: \"The Verge AI\"",
                "    rss: \"https://www.theverge.com/rss/ai-artificial-intelligence/index.xml\"",
                "    keywords: [\"ai\", \"artificial intelligence\", \"model\", \"robot\", \"agent\"]",
                "",
            ]),
            encoding="utf-8",
        )

# ---------------------------------------------------------------------------
# TaskStep dataclass (Req 5.1, 5.3, 5.4)
# ---------------------------------------------------------------------------

@dataclass
class TaskStep:
    """Individual execution step within a task."""
    step_id: str
    description: str
    status: Literal["pending", "running", "completed", "failed"]
    command: str | None = None
    output: str | None = None
    error: str | None = None



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
        self._pricing_refresh_attempted_models: set[str] = set()
        self._pricing_api_key: str | None = None

        # Set externally after construction
        self.llm_client: LLMClient | None = None
        self.slack: "SlackBot | None" = None  # noqa: F821 — forward ref
        self._slack_reply_channel: str | None = None
        self._slack_reply_thread_ts: str | None = None
        self._trace_thread_ts: str | None = None
        self._trace_enabled = os.environ.get("SLACK_TRACE_ENABLED", "1").strip().lower() not in ("0", "false", "off", "no")
        self._activity_log_enabled = os.environ.get("SLACK_ACTIVITY_LOG_ENABLED", "0").strip().lower() not in ("0", "false", "off", "no")
        self._activity_log_channel = os.environ.get("SLACK_ACTIVITY_LOG_CHANNEL", "C0AFPAYTLP4").strip() or None
        self._slack_default_channel = (
            os.environ.get("SLACK_DEFAULT_CHANNEL_ID", "").strip()
            or os.environ.get("SLACK_CHANNEL_ID", "").strip()
            or None
        )
        self._slack_context_path = self.base_dir / "companies" / self.company_id / "state" / "slack_context.json"
        self._slack_last_channel: str | None = None
        self._slack_last_thread_ts: str | None = None
        self._load_slack_context()
        if self._slack_default_channel is None and self._slack_last_channel:
            self._slack_default_channel = self._slack_last_channel

        # Conversation memory (Req 1.1, 1.5)
        self.conversation_memory = ConversationMemory(base_dir, company_id)
        self.creator_review_store = CreatorReviewStore(base_dir, company_id)
        self.consultation_store = ConsultationStore(base_dir, company_id)
        self.commitment_store = CommitmentStore(base_dir, company_id)
        self.memory_vault = MemoryVault(base_dir, company_id)
        try:
            self.memory_manager: MemoryManager | None = MemoryManager(base_dir, company_id)
        except Exception:
            logger.warning("Failed to initialize memory manager", exc_info=True)
            self.memory_manager = None
        self.policy_memory = PolicyMemoryStore(base_dir, company_id)
        self.adaptive_memory = AdaptiveMemoryStore(base_dir, company_id)
        self.procedure_store = ProcedureStore(base_dir, company_id)

        # Autonomous growth components
        self.vision_loader = VisionLoader(base_dir, company_id)
        self.task_queue = TaskQueue(base_dir, company_id)
        self.agent_registry = AgentRegistry(base_dir, company_id)
        self.service_registry = ServiceRegistry(base_dir, company_id)
        self.sub_agent_runner = SubAgentRunner(self)
        self.autonomous_loop = None
        self.newsroom_team = None
        self.alarm_scheduler = AlarmScheduler(base_dir, company_id)
        self.employee_store = EmployeeStore(base_dir, company_id)
        self.web_searcher = WebSearcher()
        self.mcp_client = MCPClient(base_dir, company_id)
        self.research_note_store = ResearchNoteStore(base_dir, company_id)
        self.git_publisher = GitPublisher(work_dir=self.base_dir / "companies" / self.company_id)

        # Initiative components
        self.initiative_store = InitiativeStore(base_dir, company_id)
        self.strategy_analyzer = StrategyAnalyzer(self.creator_review_store, self.initiative_store)
        self.initiative_planner = InitiativePlanner(self, self.initiative_store, self.strategy_analyzer)
        self.recovery_planner = None

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

        # Register CEO agent (Req 4.2)
        try:
            model = self.llm_client.model if self.llm_client else "unknown"
            self.agent_registry.ensure_ceo(model)
        except Exception:
            logger.warning("Failed to ensure CEO agent registration", exc_info=True)

        # Ensure curated memory file exists (file-first LTM)
        try:
            self.memory_vault.ensure_initialized()
        except Exception:
            logger.warning("Failed to initialize memory vault", exc_info=True)

        # Bootstrap long-term memory (best-effort)
        try:
            if self.memory_manager is not None:
                self.memory_manager.bootstrap()
        except Exception:
            logger.warning("Memory bootstrap failed", exc_info=True)

        # Ensure policy memory and seed stable paths/rules
        try:
            repo_root = Path(os.environ.get("APP_REPO_PATH", "/opt/apps/ai-company")).expanduser().resolve()
            restart_flag = (self.base_dir / "companies" / self.company_id / "state" / "restart_manager.flag").resolve()
            self.policy_memory.ensure_initialized()
            self.policy_memory.seed_defaults(
                app_repo_path=str(repo_root),
                system_prompt_file=str(repo_root / "src" / "context_builder.py"),
                restart_flag_path=str(restart_flag),
            )
            self.policy_memory.compact()
        except Exception:
            logger.warning("Failed to initialize policy memory", exc_info=True)

        # Ensure adaptive memory (dynamic memory domains)
        try:
            self.adaptive_memory.ensure_initialized()
            self.adaptive_memory.compact_and_prune()
        except Exception:
            logger.warning("Failed to initialize adaptive memory", exc_info=True)

        # Ensure procedure SoT store (verbatim runbooks)
        try:
            self.procedure_store.ensure_initialized()
        except Exception:
            logger.warning("Failed to initialize procedure store", exc_info=True)

        # Determine what to do first after wakeup
        action, description = determine_recovery_action(self.state)

        if action == "consult_creator" and self.recovery_planner is not None:
            try:
                planned = self.recovery_planner.handle_idle()
                if planned:
                    description = planned
            except Exception:
                logger.warning("recovery_planner.handle_idle() failed", exc_info=True)

        return action, description

    # ------------------------------------------------------------------
    # Budget check (Req 5.3, 5.4)
    # ------------------------------------------------------------------

    def check_budget(self) -> bool:
        """Return True when the sliding-window LLM budget is exceeded."""
        now = datetime.now(timezone.utc)

        limit = DEFAULT_BUDGET_LIMIT_USD
        window = DEFAULT_WINDOW_MINUTES
        try:
            if self.state.constitution and self.state.constitution.budget:
                limit = float(self.state.constitution.budget.limit_usd)
                window = int(self.state.constitution.budget.window_minutes)
        except Exception:
            pass

        # Optional env overrides
        try:
            if os.environ.get("BUDGET_LIMIT_USD"):
                limit = float(os.environ["BUDGET_LIMIT_USD"])
            if os.environ.get("BUDGET_WINDOW_MINUTES"):
                window = int(os.environ["BUDGET_WINDOW_MINUTES"])
        except Exception:
            logger.warning("Invalid budget env vars; ignoring", exc_info=True)

        try:
            return is_budget_exceeded(
                self.state.ledger_events,
                now,
                limit_usd=limit,
                window_minutes=window,
            )
        except Exception:
            logger.warning("Budget check failed; treating as not exceeded", exc_info=True)
            return False

    def refresh_pricing_cache(self, *, api_key: str | None = None, force: bool = False) -> None:
        """Refresh OpenRouter pricing cache (best-effort)."""
        self._pricing_api_key = api_key or self._pricing_api_key
        path = pricing_cache_path(self.base_dir, self.company_id)
        refreshed = refresh_openrouter_pricing_cache(
            path,
            api_key=self._pricing_api_key,
            force=force,
        )
        if refreshed is not None:
            self.pricing_cache = refreshed

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
        """Record an LLM call as a ledger event with estimated cost."""
        now = datetime.now(timezone.utc)

        # Best-effort pricing refresh (missing cache/model)
        auto_refresh = os.environ.get("OPENROUTER_PRICING_AUTO_REFRESH", "1").strip().lower() not in (
            "0", "false", "off", "no"
        )
        prev_cache = self.pricing_cache
        try:
            missing = (self.pricing_cache is None) or (model not in (self.pricing_cache.models or {}))
        except Exception:
            missing = True

        if auto_refresh and missing and model not in self._pricing_refresh_attempted_models:
            self._pricing_refresh_attempted_models.add(model)
            try:
                api_key = self._pricing_api_key or os.environ.get("OPENROUTER_API_KEY") or None
                self.refresh_pricing_cache(api_key=api_key)
            except Exception:
                logger.warning("Failed to refresh pricing cache", exc_info=True)

        pricing, source = get_pricing_with_fallback(
            self.pricing_cache,
            model,
            previous_cache=prev_cache,
        )

        in_price = float(pricing.input_price_per_1k)
        out_price = float(pricing.output_price_per_1k)
        estimated = (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price

        event = LedgerEvent(
            timestamp=now,
            event_type="llm_call",
            agent_id=agent_id,
            task_id=task_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unit_price_usd_per_1k_input_tokens=in_price,
            unit_price_usd_per_1k_output_tokens=out_price,
            price_retrieved_at=pricing.retrieved_at,
            estimated_cost_usd=estimated,
            metadata={"pricing_source": source},
        )

        try:
            append_ledger_event(self.base_dir, self.company_id, event)
        except Exception:
            logger.warning("Failed to append ledger event", exc_info=True)

        try:
            self.state.ledger_events.append(event)
        except Exception:
            logger.warning("Failed to update in-memory ledger", exc_info=True)

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
        from datetime import timedelta

        # Cost summary
        spent = compute_window_cost(self.state.ledger_events, now)
        limit = DEFAULT_BUDGET_LIMIT_USD
        if self.state.constitution and self.state.constitution.budget:
            limit = self.state.constitution.budget.limit_usd
        remaining = max(0.0, limit - spent)

        cost_summary = CostSummary(
            spent_usd=spent,
            remaining_usd=remaining,
            limit_usd=limit,
        )

        # Autonomous growth data (Req 7.1, 7.2, 7.3)
        running_tasks: list[str] = []
        active_agents: list[str] = []
        recent_services: list[str] = []

        try:
            if hasattr(self, "task_queue") and self.task_queue:
                running_tasks = [
                    t.description for t in self.task_queue.list_by_status("running")
                ]
        except Exception:
            logger.warning("Failed to get running tasks for report", exc_info=True)

        try:
            if hasattr(self, "agent_registry") and self.agent_registry:
                active_agents = [
                    f"{a.name} ({a.role})" for a in self.agent_registry.list_active()
                ]
        except Exception:
            logger.warning("Failed to get active agents for report", exc_info=True)

        try:
            if hasattr(self, "service_registry") and self.service_registry:
                recent_services = [
                    f"{s.name}: {s.description}"
                    for s in self.service_registry.list_all()
                ]
        except Exception:
            logger.warning("Failed to get services for report", exc_info=True)

        # --- Build delta_description from recent activity ---
        delta_parts: list[str] = []
        window_start = now - timedelta(minutes=10)

        # Count recent LLM calls and shell execs
        recent_llm = 0
        recent_shell = 0
        for ev in self.state.ledger_events:
            if ev.timestamp >= window_start:
                if ev.event_type == "llm_call":
                    recent_llm += 1
                elif ev.event_type == "shell_exec":
                    recent_shell += 1

        if recent_llm > 0:
            delta_parts.append(f"LLM呼び出し {recent_llm}回")
        if recent_shell > 0:
            delta_parts.append(f"シェル実行 {recent_shell}回")

        # Recently completed tasks
        try:
            completed = self.task_queue.list_by_status("completed")
            recent_completed = [
                t for t in completed if t.updated_at >= window_start
            ]
            for t in recent_completed[:3]:
                delta_parts.append(f"完了: {t.description}")
        except Exception:
            logger.warning("Failed to get completed tasks for report", exc_info=True)

        # Recently failed tasks
        try:
            failed = self.task_queue.list_by_status("failed")
            recent_failed = [
                t for t in failed if t.updated_at >= window_start
            ]
            for t in recent_failed[:2]:
                reason = t.error or "不明"
                delta_parts.append(f"失敗: {t.description} ({reason})")
        except Exception:
            logger.warning("Failed to get failed tasks for report", exc_info=True)

        if not delta_parts:
            delta_description = "特筆すべき活動なし"
        else:
            delta_description = " / ".join(delta_parts)

        # --- Build next_plan from pending tasks ---
        next_parts: list[str] = []
        try:
            pending = self.task_queue.list_by_status("pending")
            pending.sort(key=lambda t: t.priority)
            for t in pending[:3]:
                next_parts.append(t.description)
        except Exception:
            logger.warning("Failed to get pending tasks for report", exc_info=True)

        if running_tasks:
            next_plan = "実行中タスクを継続"
            if next_parts:
                next_plan += f" → 次: {next_parts[0]}"
        elif next_parts:
            next_plan = " / ".join(next_parts)
        else:
            next_plan = "新規タスクの提案を検討"

        # --- Blockers: pending Creator consultations ---
        blockers: list[str] = []
        try:
            pending_consults = self.consultation_store.list_by_status("pending")
            pending_consults.sort(key=lambda c: c.created_at)
            for c in pending_consults[:5]:
                first_line = (c.content or "").strip().splitlines()[0] if c.content else ""
                if len(first_line) > 120:
                    first_line = first_line[:120] + "…"
                blockers.append(f"[consult_id: {c.consultation_id}] {first_line}")
        except Exception:
            logger.warning("Failed to load consultations for report blockers", exc_info=True)

        # --- Approvals: pending constitution amendment proposals ---
        approvals: list[str] = []
        try:
            processed: set[str] = {
                e.request_id
                for e in self.state.decision_log
                if e.request_id and e.status in ("approved", "rejected")
            }
            for entry in self.state.decision_log:
                if (
                    entry.status == "proposed"
                    and entry.request_id
                    and entry.request_id not in processed
                ):
                    decision = (entry.decision or "").strip()
                    if len(decision) > 120:
                        decision = decision[:120] + "…"
                    approvals.append(f"[request_id: {entry.request_id}] {decision}")
            approvals = approvals[:5]
        except Exception:
            logger.warning("Failed to load approvals for report", exc_info=True)

        # --- Cost allocation plan (lightweight, derived from current state) ---
        alloc_parts: list[str] = []
        if approvals:
            alloc_parts.append(f"承認待ち{len(approvals)}件")
        if blockers:
            alloc_parts.append(f"相談待ち{len(blockers)}件")
        if next_plan:
            alloc_next = next_plan
            if len(alloc_next) > 120:
                alloc_next = alloc_next[:120] + "…"
            alloc_parts.append(f"次: {alloc_next}")
        if remaining <= 0.0:
            alloc_parts.append("残予算0のためLLM/APIは最小化")
        cost_summary.allocation_plan = " / ".join(alloc_parts) if alloc_parts else ""

        data = ReportData(
            timestamp=now,
            company_id=self.company_id,
            wip=list(self.state.wip),
            delta_description=delta_description,
            next_plan=next_plan,
            blockers=blockers,
            cost=cost_summary,
            approvals=approvals,
            running_tasks=running_tasks,
            active_agents=active_agents,
            recent_services=recent_services,
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

    # ------------------------------------------------------------------
    # Creator daily brief (KPI loop)
    # ------------------------------------------------------------------

    def generate_daily_brief(self) -> str:
        """Generate a Creator日報 (施策/相談/コスト/スコア) for the KPI loop."""
        now = datetime.now(timezone.utc)

        # Planned initiatives (pending tasks)
        planned: list[str] = []
        try:
            pending = self.task_queue.list_by_status("pending")
            pending.sort(key=lambda t: t.priority)
            for t in pending[:7]:
                planned.append(f"[{t.task_id}] P{t.priority} {t.description}")
        except Exception:
            logger.warning("Failed to get planned initiatives for daily brief", exc_info=True)

        # Active initiatives (running tasks + WIP)
        active: list[str] = []
        try:
            running = self.task_queue.list_by_status("running")
            running.sort(key=lambda t: t.priority)
            for t in running[:7]:
                active.append(f"[{t.task_id}] P{t.priority} {t.description}")
        except Exception:
            logger.warning("Failed to get active initiatives for daily brief", exc_info=True)

        paused: list[str] = []
        try:
            paused_tasks = self.task_queue.list_by_status("paused")
            paused_tasks.sort(key=lambda t: t.updated_at, reverse=True)
            for t in paused_tasks[:7]:
                reason = (t.error or "").strip()
                if len(reason) > 60:
                    reason = reason[:60] + "…"
                suffix = f" — {reason}" if reason else ""
                paused.append(f"[{t.task_id}] P{t.priority} {t.description}{suffix}")
        except Exception:
            logger.warning("Failed to get paused tasks for daily brief", exc_info=True)

        canceled: list[str] = []
        try:
            canceled_tasks = self.task_queue.list_by_status("canceled")
            canceled_tasks.sort(key=lambda t: t.updated_at, reverse=True)
            for t in canceled_tasks[:7]:
                reason = (t.error or "").strip()
                if len(reason) > 60:
                    reason = reason[:60] + "…"
                suffix = f" — {reason}" if reason else ""
                canceled.append(f"[{t.task_id}] P{t.priority} {t.description}{suffix}")
        except Exception:
            logger.warning("Failed to get canceled tasks for daily brief", exc_info=True)

        for w in self.state.wip[:3]:
            if w and w not in active:
                active.append(f"(WIP) {w}")

        # Consultations (pending)
        consultations: list[str] = []

        try:
            pending_consults = self.consultation_store.list_by_status("pending")
            pending_consults.sort(key=lambda c: c.created_at)
            for c in pending_consults[:10]:
                first_line = (c.content or "").strip().splitlines()[0] if c.content else ""
                if len(first_line) > 120:
                    first_line = first_line[:120] + "…"
                consultations.append(f"[consult_id: {c.consultation_id}] {first_line}")
        except Exception:
            logger.warning("Failed to load consultations for daily brief", exc_info=True)

        # Include pending constitution amendment approvals as "consultations"
        try:
            processed: set[str] = {
                e.request_id
                for e in self.state.decision_log
                if e.request_id and e.status in ("approved", "rejected")
            }
            for entry in self.state.decision_log:
                if (
                    entry.status == "proposed"
                    and entry.request_id
                    and entry.request_id not in processed
                ):
                    consultations.append(
                        f"[request_id: {entry.request_id}] 憲法変更提案: {entry.decision}"
                    )
        except Exception:
            logger.warning("Failed to load proposed amendments for daily brief", exc_info=True)

        # Cost summary
        limit = DEFAULT_BUDGET_LIMIT_USD
        if self.state.constitution and self.state.constitution.budget:
            limit = self.state.constitution.budget.limit_usd

        spent_60m = compute_window_cost(self.state.ledger_events, now, window_minutes=60)
        spent_24h = compute_window_cost(self.state.ledger_events, now, window_minutes=60 * 24)

        cost = DailyCostSummary(
            spent_usd_60m=spent_60m,
            spent_usd_24h=spent_24h,
            budget_limit_usd_60m=limit,
        )

        # Latest creator score
        latest_review_text = ""
        try:
            latest = self.creator_review_store.latest()
            if latest:
                axis = []
                if latest.score_interestingness_25 is not None:
                    axis.append(f"面白さ{latest.score_interestingness_25}/25")
                if latest.score_cost_efficiency_25 is not None:
                    axis.append(f"コスト効率{latest.score_cost_efficiency_25}/25")
                if latest.score_realism_25 is not None:
                    axis.append(f"現実性{latest.score_realism_25}/25")
                if latest.score_evolvability_25 is not None:
                    axis.append(f"進化性{latest.score_evolvability_25}/25")
                axis_text = " ".join(axis) if axis else "軸スコアなし"
                ts = latest.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                comment = (latest.comment or "").strip()
                if comment:
                    latest_review_text = f"- [{ts}] {latest.score_total_100}/100 ({axis_text})\n  {comment}"
                else:
                    latest_review_text = f"- [{ts}] {latest.score_total_100}/100 ({axis_text})"
        except Exception:
            logger.warning("Failed to load latest creator review", exc_info=True)

        reply_format = ""
        try:
            if self.state.constitution and self.state.constitution.creator_score_policy:
                reply_format = self.state.constitution.creator_score_policy.creator_reply_format
        except Exception:
            pass

        data = DailyBriefData(
            timestamp=now,
            company_id=self.company_id,
            planned_initiatives=planned,
            active_initiatives=active,
            paused_initiatives=paused,
            canceled_initiatives=canceled,
            consultations=consultations,
            cost=cost,
            latest_creator_score=latest_review_text,
            creator_reply_format=reply_format,
        )

        return format_daily_brief(data)

    # ------------------------------------------------------------------
    # Message processing — Think → Act → Report (Req 3.1–3.4, 4.1–4.6)
    # ------------------------------------------------------------------

    _MAX_WIP = 3
    _MAX_ACTION_LOOP = 10
    _MAX_MEMORY_ACTIONS_PER_TASK = 2

    # ------------------------------------------------------------------
    # WIP management (Req 5.1, 5.2, 5.5)
    # ------------------------------------------------------------------

    def add_wip(self, task_name: str) -> bool:
        """Add a task to the WIP list.

        Returns ``True`` if the task was added, ``False`` if the WIP limit
        (3) has been reached.
        """
        if len(self.state.wip) >= self._MAX_WIP:
            logger.warning("WIP limit reached (%d), cannot add: %s", self._MAX_WIP, task_name)
            return False
        self.state.wip.append(task_name)
        return True

    def remove_wip(self, task_name: str) -> bool:
        """Remove a task from the WIP list.

        Returns ``True`` if the task was found and removed, ``False`` otherwise.
        """
        try:
            self.state.wip.remove(task_name)
            return True
        except ValueError:
            logger.warning("Task not in WIP, cannot remove: %s", task_name)
            return False

    # ------------------------------------------------------------------
    # Message processing — Think → Act → Report (Req 3.1–3.4, 4.1–4.6)
    # ------------------------------------------------------------------

    def process_message(
        self,
        text: str,
        user_id: str,
        *,
        slack_channel: str | None = None,
        slack_thread_ts: str | None = None,
        slack_thread_context: str | None = None,
    ) -> None:
        """Creatorメッセージを処理する（Think → Act → Report）.

        1. 予算チェック
        2. コンテキスト構築
        3. LLM呼び出し
        4. 応答パース
        5. アクション実行（ループ）
        6. 結果報告
        """
        task_id = f"msg-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        logger.info("process_message start: user=%s task=%s", user_id, task_id)

        prev_channel = self._slack_reply_channel
        prev_thread = self._slack_reply_thread_ts
        prev_trace_thread = self._trace_thread_ts
        self._slack_reply_channel = slack_channel
        self._slack_reply_thread_ts = slack_thread_ts
        self._trace_thread_ts = slack_thread_ts
        self._remember_slack_context(slack_channel, slack_thread_ts)
        try:
            stripped = (text or "").strip()
            self._bootstrap_trace_thread(stripped)
            self._trace_event("依頼を受信。意図解析を開始")
            self._activity_log(f"Creator→CEO: {self._summarize_for_activity_log(stripped, limit=500)}")

            # Ingest policy/rule/budget memories from incoming conversation
            try:
                ingest_result = self.policy_memory.ingest_text(
                    stripped,
                    source="creator_message",
                    user_id=user_id,
                    task_id=task_id,
                )
                if ingest_result.conflicts:
                    conflict_lines = []
                    for c in ingest_result.conflicts[:3]:
                        others = " / ".join([f"[{e.memory_id}] {e.content}" for e in c.conflicts_with[:2]])
                        conflict_lines.append(f"- 新規: {c.new_entry.content}\n  競合: {others}")
                    consult_text = (
                        "方針記憶に矛盾候補が検出されました。どちらを優先するか確認したいです。\n"
                        + "\n".join(conflict_lines)
                    )
                    entry, created = self.consultation_store.ensure_pending(
                        consult_text,
                        related_task_id=task_id,
                    )
                    if created:
                        self._slack_send(
                            f"🤝 方針衝突を検知しました [consult_id: {entry.consultation_id}]\n\n{consult_text}"
                        )
            except Exception:
                logger.warning("Failed to ingest policy memory from message", exc_info=True)

            # Ingest dynamic important memories (beyond fixed policy/budget/rules)
            try:
                self.adaptive_memory.ingest_text(
                    stripped,
                    source="creator_message",
                    user_id=user_id,
                    task_id=task_id,
                )
            except Exception:
                logger.warning("Failed to ingest adaptive memory from message", exc_info=True)

            # Ingest verbatim procedure/runbook blocks into dedicated SoT
            try:
                self.procedure_store.ingest_text(
                    stripped,
                    source="creator_message",
                    user_id=user_id,
                    task_id=task_id,
                )
            except Exception:
                logger.warning("Failed to ingest procedure SoT from message", exc_info=True)

            # --- Creator directive (pause/cancel/resume) ---
            try:
                directive = parse_creator_directive(stripped, thread_context=slack_thread_context)
            except Exception:
                directive = None
            if directive is not None:
                if self._apply_creator_directive(directive, user_id=user_id):
                    return

            # --- Fast paths (no LLM required) ---
            if stripped in ("日報", "creator日報", "daily", "daily brief", "daily report"):
                self._trace_event("fast-path: 日報生成")
                self._slack_send(self.generate_daily_brief())
                return

            normalized = stripped.replace(" ", "").replace("　", "")
            asks_prompt_location = (
                "システムプロンプト" in normalized
                and any(k in normalized for k in ("どこ", "どのファイル", "場所", "ファイル"))
            )
            asks_logic_location = (
                "ロジック" in normalized
                and any(k in normalized for k in ("どこ", "どのファイル", "場所", "確認"))
            )
            if asks_prompt_location or asks_logic_location:
                self._trace_event("fast-path: 実体ファイル照会")
                repo_root = Path(os.environ.get("APP_REPO_PATH", "/opt/apps/ai-company"))
                restart_flag = self.base_dir / "companies" / self.company_id / "state" / "restart_manager.flag"
                self._slack_send(
                    "私の実体ファイルは以下です。\n"
                    f"- システムプロンプト: `{repo_root}/src/context_builder.py` の `build_system_prompt()`\n"
                    f"- 読み込み元: `{repo_root}/src/manager.py` の `process_message()`\n"
                    f"- 主要ロジック: `{repo_root}/src/`\n"
                    f"- 手順SoT: `{repo_root}/data/companies/{self.company_id}/state/procedures.ndjson`\n"
                    f"- 再読込フラグ: `{restart_flag}`\n\n"
                    "必要なら私自身がコード編集→self_commit→再読込まで実行できます。"
                )
                return

            if self._is_max_turns_question(stripped):
                limits = self._read_turn_limit_settings()
                self._trace_event("設定値を確認: turn/time guards")
                self._slack_send(
                    "現在のターン/時間ガード設定は以下です。\n"
                    f"- 社員AI SUB_AGENT_MAX_TURNS: `{limits['sub_agent_max_turns_text']}`\n"
                    f"- 社員AI SUB_AGENT_MAX_WALL_SECONDS: `{limits['sub_agent_max_wall_text']}`\n"
                    f"- 自律タスク AUTONOMOUS_MAX_TURNS: `{limits['autonomous_max_turns_text']}`\n"
                    f"- 自律タスク AUTONOMOUS_MAX_WALL_SECONDS: `{limits['autonomous_max_wall_text']}`\n"
                    "0 は『上限なし』です。中断時は run_id 付き進捗レポートを返し、`employee resume <run_id>` で再開できます。"
                )
                return

            handled_control, control_reply = self._handle_runtime_control_command(
                stripped,
                actor_id=user_id or "creator",
                actor_role="ceo",
                actor_model=self.llm_client.model if self.llm_client else None,
            )
            if handled_control:
                if control_reply:
                    self._slack_send(control_reply)
                return

            if self._is_time_question(stripped):
                self._trace_event("fast-path: 現在時刻照会")
                self._slack_send(self.alarm_scheduler.now_text())
                return

            asks_web_search_impl = (
                ("web検索" in normalized.lower() or "検索エンジン" in normalized.lower())
                and any(k in normalized.lower() for k in ("何", "なに", "使", "仕組", "どのファイル", "どこ", "実装", "設定"))
            )
            if asks_web_search_impl:
                self._trace_event("fast-path: web検索実装照会")
                repo_root = Path(os.environ.get("APP_REPO_PATH", "/opt/apps/ai-company"))
                backend = (os.environ.get("AI_COMPANY_WEB_SEARCH_BACKEND") or "searxng").strip() or "searxng"
                searxng_url = (os.environ.get("AI_COMPANY_SEARXNG_URL") or "http://127.0.0.1:8088").strip() or "http://127.0.0.1:8088"
                self._slack_send(
                    "Web検索（<research>）は次の実装です。\n"
                    f"- 実行箇所: `{repo_root}/src/manager.py` の `action_type==\"research\"`\n"
                    f"- 検索実装: `{repo_root}/src/web_searcher.py` (WebSearcher)\n"
                    f"- backend: `{backend}` (env: AI_COMPANY_WEB_SEARCH_BACKEND)\n"
                    f"- SearxNG URL: `{searxng_url}` (env: AI_COMPANY_SEARXNG_URL)\n"
                    f"- SearxNG compose: `/opt/apps/services/searxng/docker-compose.yml`"
                )
                return

            if self._is_agent_list_request(stripped):
                self._trace_event("fast-path: 社員AI一覧を生成")
                self._slack_send(self._build_agent_list_reply(stripped))
                return

            if self._is_procedure_library_request(stripped):
                self._trace_event("fast-path: 手順SoTライブラリ照会")
                self._slack_send(self._build_procedure_library_reply())
                return
            recalled_procedure = self.procedure_store.find_best_for_request(stripped)
            if recalled_procedure is not None:
                self._trace_event("fast-path: 手順SoTを再掲")
                self._slack_send(self.procedure_store.render_reply(recalled_procedure))
                return

            # Creator score feedback (KPI loop)
            review = parse_creator_review(text, user_id=user_id)
            if review is not None:
                try:
                    self.creator_review_store.save(review)
                    axes = []
                    if review.score_interestingness_25 is not None:
                        axes.append(f"面白さ{review.score_interestingness_25}/25")
                    if review.score_cost_efficiency_25 is not None:
                        axes.append(f"コスト効率{review.score_cost_efficiency_25}/25")
                    if review.score_realism_25 is not None:
                        axes.append(f"現実性{review.score_realism_25}/25")
                    if review.score_evolvability_25 is not None:
                        axes.append(f"進化性{review.score_evolvability_25}/25")
                    axis_text = " ".join(axes) if axes else "軸スコアなし"
                    self._slack_send(
                        f"✅ Creatorスコアを記録しました: {review.score_total_100}/100 ({axis_text})"
                    )
                except Exception:
                    logger.warning("Failed to save creator review", exc_info=True)
                    self._slack_send("⚠️ Creatorスコアの保存に失敗しました")
                return

            # Resolve consultation (optional helper)
            try:
                import re

                m = re.match(r"^(?:resolve|解決)\s+([0-9a-f]{8})(?:\s*[:：]\s*(.*))?$", stripped, re.IGNORECASE)
                if m:
                    consult_id = m.group(1)
                    resolution = (m.group(2) or "").strip()
                    updated = self.consultation_store.resolve(consult_id, resolution=resolution)
                    self._slack_send(f"✅ 相談を解決として記録しました: {updated.consultation_id}")
                    return
            except Exception:
                logger.warning("Failed to resolve consultation command", exc_info=True)

            if self.llm_client is None:
                logger.error("LLM client not configured")
                self._slack_send("エラー: LLMクライアントが設定されていません")
                return

            # Budget guard (Creator/Autonomy共通)
            if self.check_budget():
                limit = DEFAULT_BUDGET_LIMIT_USD
                window = DEFAULT_WINDOW_MINUTES
                try:
                    if self.state.constitution and self.state.constitution.budget:
                        limit = float(self.state.constitution.budget.limit_usd)
                        window = int(self.state.constitution.budget.window_minutes)
                except Exception:
                    pass
                spent = compute_window_cost(
                    self.state.ledger_events,
                    datetime.now(timezone.utc),
                    window_minutes=window,
                )
                self._slack_send(
                    f"直近{window}分のLLMコストが上限に達したため、いまは処理を止めています（${spent:.2f}/${limit:.2f}）。"
                )
                return

            # 2. コンテキスト構築
            now = datetime.now(timezone.utc)

            # Memory recall/summarization (best-effort)
            rolling_summary_text: str | None = None
            recalled_memories: list[str] | None = None
            try:
                if self.memory_manager is not None:
                    self.memory_manager.ingest_all_sources()
                    rolling_summary_text = self.memory_manager.summary_for_prompt()
                    recalled_memories = self.memory_manager.recall_for_prompt(
                        stripped or text,
                        limit=8,
                    )
            except Exception:
                logger.warning("Failed to build memory context", exc_info=True)

            spent = compute_window_cost(self.state.ledger_events, now)
            limit = DEFAULT_BUDGET_LIMIT_USD
            if self.state.constitution and self.state.constitution.budget:
                limit = self.state.constitution.budget.limit_usd

            recent_decisions = self.state.decision_log[-5:]

            # Load recent conversation history (Req 1.2)
            try:
                conversation_history = self.conversation_memory.recent(n=60)
            except Exception:
                logger.warning("Failed to load conversation history", exc_info=True)
                conversation_history = None

            # Save user message to conversation memory (Req 1.1)
            try:
                self.conversation_memory.append(ConversationEntry(
                    timestamp=now,
                    role="user",
                    content=text,
                    user_id=user_id,
                    task_id=task_id,
                ))
            except Exception:
                logger.warning("Failed to save user message to conversation memory", exc_info=True)

            # Load vision text (Req 2.1)
            try:
                vision_text = self.vision_loader.load()
            except Exception:
                logger.warning("Failed to load vision", exc_info=True)
                vision_text = None

            # Load curated memory tail (file-first LTM)
            curated_memory_text = None
            try:
                curated_memory_text = self.memory_vault.load_tail(tail_chars=6000)
            except Exception:
                logger.warning("Failed to load curated memory", exc_info=True)
                curated_memory_text = None

            # Load daily memory tail (append-only daily memo)
            daily_memory_text = None
            try:
                daily_memory_text = self.memory_vault.load_daily_tail(tail_chars=3000)
            except Exception:
                logger.warning("Failed to load daily memory", exc_info=True)
                daily_memory_text = None

            # Load recent research notes
            try:
                research_notes = self.research_note_store.recent()
            except Exception:
                logger.warning("Failed to load research notes", exc_info=True)
                research_notes = None

            # Load recent creator reviews (KPI loop)
            try:
                creator_reviews = self.creator_review_store.recent(limit=3)
            except Exception:
                logger.warning("Failed to load creator reviews", exc_info=True)
                creator_reviews = None

            # Load task history for context
            task_history = None
            try:
                completed = self.task_queue.list_by_status("completed")
                completed.sort(key=lambda t: t.updated_at, reverse=True)
                failed = self.task_queue.list_by_status("failed")
                failed.sort(key=lambda t: t.updated_at, reverse=True)
                running = self.task_queue.list_by_status("running")
                paused = self.task_queue.list_by_status("paused")
                paused.sort(key=lambda t: t.updated_at, reverse=True)
                canceled = self.task_queue.list_by_status("canceled")
                canceled.sort(key=lambda t: t.updated_at, reverse=True)
                task_history = TaskHistoryContext(
                    completed=completed[:10],
                    failed=failed[:5],
                    running=running,
                    paused=paused[:5],
                    canceled=canceled[:5],
                )
            except Exception:
                logger.warning("Failed to load task history", exc_info=True)

            # Load open commitments (promises/TODOs)
            open_commitments = None
            try:
                open_commitments = self.commitment_store.list_by_status("open")
            except Exception:
                logger.warning("Failed to load open commitments", exc_info=True)
                open_commitments = None

            # モデルカタログ生成
            model_catalog_text = None
            try:
                catalog = build_model_catalog(self.pricing_cache)
                model_catalog_text = format_model_catalog_for_prompt(catalog) or None
            except Exception:
                logger.warning("Failed to build model catalog", exc_info=True)

            # Load policy memory context (direction/rules/budget)
            policy_memory_text = None
            policy_timeline_text = None
            policy_conflicts_text = None
            try:
                policy_memory_text = self.policy_memory.format_active(limit=24)
                policy_timeline_text = self.policy_memory.format_timeline(limit=30)
                policy_conflicts_text = self.policy_memory.format_conflicts(limit=10)
            except Exception:
                logger.warning("Failed to load policy memory context", exc_info=True)

            # Load adaptive memory context (dynamic domains + forgetting)
            adaptive_memory_text = None
            adaptive_domains_text = None
            try:
                adaptive_memory_text = self.adaptive_memory.format_active(limit=24)
                adaptive_domains_text = self.adaptive_memory.format_domains(limit=16)
            except Exception:
                logger.warning("Failed to load adaptive memory context", exc_info=True)

            # Load procedure SoT context (verbatim runbooks + shared docs)
            procedure_library_text = None
            shared_procedure_text = None
            try:
                procedure_library_text = self.procedure_store.format_library(limit=12, include_steps=True)
                shared_procedure_text = self.procedure_store.format_shared(limit=12)
            except Exception:
                logger.warning("Failed to load procedure SoT context", exc_info=True)

            sot_policy_text = (
                "- SoT優先順: 会社内SoT(手順/共有ドキュメント/方針) → Web一次情報。\n"
                "- VPSや社内運用に関わる判断は、まず保存済み手順SoTを参照する。\n"
                "- 外部仕様（git/サービスAPI/OSS仕様）で鮮度が必要ならWebで確認し、必要ならSoTへ反映する。"
            )

            mcp_servers_text = None
            try:
                mcp_servers_text = self.mcp_client.format_servers_for_prompt()
            except Exception:
                logger.warning("Failed to load MCP servers for prompt", exc_info=True)
                mcp_servers_text = None

            employee_roster_text = None
            try:
                employee_roster_text = self.employee_store.format_roster(limit=24)
            except Exception:
                logger.warning("Failed to load employee roster for prompt", exc_info=True)
                employee_roster_text = None

            system_prompt = build_system_prompt(
                constitution=self.state.constitution,
                wip=self.state.wip,
                recent_decisions=recent_decisions,
                budget_spent=spent,
                budget_limit=limit,
                conversation_history=conversation_history,
                vision_text=vision_text,
                curated_memory_text=curated_memory_text,
                daily_memory_text=daily_memory_text,
                creator_reviews=creator_reviews,
                research_notes=research_notes,
                rolling_summary=rolling_summary_text,
                recalled_memories=recalled_memories,
                slack_thread_context=slack_thread_context,
                task_history=task_history,
                active_initiatives=self._load_active_initiatives(),
                strategy_direction=self._load_strategy_direction(),
                model_catalog_text=model_catalog_text,
                open_commitments=open_commitments,
                policy_memory_text=policy_memory_text,
                policy_timeline_text=policy_timeline_text,
                policy_conflicts_text=policy_conflicts_text,
                adaptive_memory_text=adaptive_memory_text,
                adaptive_domains_text=adaptive_domains_text,
                procedure_library_text=procedure_library_text,
                shared_procedure_text=shared_procedure_text,
                sot_policy_text=sot_policy_text,
                mcp_servers_text=mcp_servers_text,
                current_time_text=self.alarm_scheduler.now_text(now),
                employee_roster_text=employee_roster_text,
            )

            conversation: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]

            prefetch_urls: set[str] | None = None
            if self._should_prefetch_web_search(stripped):
                self._trace_event("事前Web検索を実行（最新/期間指定の外部情報）")
                self._activity_log(
                    f"CEOツール利用: web_search (prefetch: {self._summarize_for_activity_log(stripped, limit=120)})"
                )
                pre_results = self.web_searcher.search(stripped)
                prefetch_urls = {self._normalize_url(sr.url) for sr in pre_results}

                if pre_results:
                    guard = (
                        "重要: あなたは次の事前リサーチ結果に含まれるURL以外を、返信の出典として提示してはいけません。"
                        "URLを捏造しないでください。足りない場合は『見つからない』と答えるか、<research>で追加調査してください。"
                    )
                    conversation.insert(1, {"role": "system", "content": guard})
                    summary_parts = [f"事前リサーチ結果 (query={stripped}):"]
                    for i, sr in enumerate(pre_results, 1):
                        summary_parts.append(f"{i}. {sr.title}\n   {sr.url}\n   {sr.snippet}")
                    conversation.append({"role": "user", "content": "\n".join(summary_parts)})
                else:
                    guard = (
                        "重要: 事前リサーチ結果は空でした。URLを捏造せず『見つからない』と答えるか、<research>で別クエリ検索してください。"
                    )
                    conversation.insert(1, {"role": "system", "content": guard})

            # 3. LLM呼び出し
            model_name = getattr(self.llm_client, "model", "unknown")
            self._trace_event(f"LLM推論を開始 (model={model_name})")
            llm_result = self.llm_client.chat(conversation)

            if isinstance(llm_result, LLMError):
                logger.error("LLM call failed: %s", llm_result.message)
                self._slack_send(f"エラー: LLM呼び出しに失敗しました — {llm_result.message}")
                return

            # 4. Ledger記録
            self.record_llm_call(
                provider="openrouter",
                model=llm_result.model,
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                task_id=task_id,
            )
            logger.info(
                "LLM call recorded: in=%d out=%d model=%s",
                llm_result.input_tokens,
                llm_result.output_tokens,
                llm_result.model,
            )

            # 5. 応答パース
            actions = parse_response(llm_result.content)

            # Anti-hallucination guard: if we prefetched, block fabricated URLs in replies (best-effort)
            if prefetch_urls is not None:
                has_research = any(a.action_type == "research" for a in actions)
                if not has_research:
                    reply_text = "\n".join([a.content for a in actions if a.action_type == "reply"])
                    reply_urls = {self._normalize_url(u) for u in self._extract_http_urls(reply_text)}
                    invalid = [u for u in reply_urls if u and u not in prefetch_urls]

                    dead: list[str] = []
                    for u in list(reply_urls)[:2]:
                        if not u:
                            continue
                        cmd = (
                            'curl -s -I -L -m 10 -A "Mozilla/5.0" '
                            + f'"{u}" | head -n 1'
                        )
                        try:
                            head = execute_shell(command=cmd, timeout=15)
                            m = re.search(r"(\d{3})", head.stdout or "")
                            code = int(m.group(1)) if m else 0
                            if code >= 400 or code == 0:
                                dead.append(u)
                        except Exception:
                            dead.append(u)

                    if (invalid or dead) and self.llm_client is not None:
                        reason_lines = []
                        if invalid:
                            reason_lines.append(f"- invalid_url(not in prefetch): {invalid[0]}")
                        if dead:
                            reason_lines.append(f"- dead_url(http>=400 or unknown): {dead[0]}")
                        conversation.append({
                            "role": "user",
                            "content": (
                                "注意: 返信の出典URLに問題があります（捏造/リンク切れの疑い）。\n"
                                + "\n".join(reason_lines)
                                + "\n\n"
                                "リサーチ結果に含まれるURLだけで、かつ到達可能なURLで回答し直してください。\n"
                                "見つからなければ『見つからない』と明言してください。"
                            ),
                        })
                        llm_result = self.llm_client.chat(conversation)
                        if isinstance(llm_result, LLMError):
                            logger.error("LLM retry (url-guard) failed: %s", llm_result.message)
                            self._slack_send(f"エラー: LLM呼び出しに失敗しました — {llm_result.message}")
                            return
                        self.record_llm_call(
                            provider="openrouter",
                            model=llm_result.model,
                            input_tokens=llm_result.input_tokens,
                            output_tokens=llm_result.output_tokens,
                            task_id=task_id,
                        )
                        actions = parse_response(llm_result.content)
            if actions:
                action_list = ", ".join(a.action_type for a in actions)
                self._trace_event(f"実行アクションを決定: {action_list}")
            else:
                self._trace_event("実行アクションなし（直接応答）")
            conversation.append({"role": "assistant", "content": llm_result.content})

            # Save assistant response to conversation memory (Req 1.5)
            try:
                self.conversation_memory.append(ConversationEntry(
                    timestamp=datetime.now(timezone.utc),
                    role="assistant",
                    content=llm_result.content,
                    task_id=task_id,
                ))
            except Exception:
                logger.warning("Failed to save assistant response to conversation memory", exc_info=True)

            # 6. アクション実行ループ
            self._execute_action_loop(actions, conversation, task_id)

            # Persist a compact interaction log (best-effort)
            try:
                if self.memory_manager is not None:
                    assistant_msgs = [
                        m.get("content", "")
                        for m in conversation
                        if m.get("role") == "assistant"
                    ]
                    response_text = assistant_msgs[-1] if assistant_msgs else ""

                    snapshot_lines = [
                        f"WIP: {len(self.state.wip)}",
                    ]
                    try:
                        pending_consults = self.consultation_store.list_by_status("pending")
                        snapshot_lines.append(f"Pending consults: {len(pending_consults)}")
                    except Exception:
                        pass
                    try:
                        pending_tasks = self.task_queue.list_by_status("pending")
                        snapshot_lines.append(f"Pending tasks: {len(pending_tasks)}")
                    except Exception:
                        pass

                    self.memory_manager.note_interaction(
                        timestamp=now,
                        user_id=user_id,
                        request_text=text,
                        response_text=response_text,
                        snapshot_lines=snapshot_lines,
                    )
                    self.memory_manager.ingest_all_sources()
            except Exception:
                logger.warning("Failed to persist interaction log", exc_info=True)

        except Exception:
            logger.exception("Unexpected error in process_message")
            self._slack_send("エラー: メッセージ処理中に予期しないエラーが発生しました")
        finally:
            self._slack_reply_channel = prev_channel
            self._slack_reply_thread_ts = prev_thread
            self._trace_thread_ts = prev_trace_thread

    def _apply_creator_directive(self, directive: CreatorDirective, *, user_id: str) -> bool:
        """Apply Creator's pause/cancel/resume instruction to task/initiative state.

        Returns True when handled (even if only an error/help message was sent).
        """
        target_task_id = directive.task_id
        consult_id = directive.consult_id

        # If consult_id exists, resolve it (best-effort) and use related_task_id as fallback.
        if consult_id:
            try:
                latest = self.consultation_store.get_latest(consult_id)
                if latest and latest.status != "resolved":
                    self.consultation_store.resolve(consult_id, resolution=directive.raw_text)
                if target_task_id is None and latest and latest.related_task_id:
                    target_task_id = latest.related_task_id
            except Exception:
                logger.warning("Failed to resolve consultation %s", consult_id, exc_info=True)

        # If task_id is still unknown, try to guess by query text.
        if target_task_id is None and directive.query:
            try:
                q = directive.query
                candidates = [
                    t for t in self.task_queue.list_all()
                    if q in (t.description or "") and t.status in ("pending", "running", "paused", "failed", "canceled")
                ]
                candidates.sort(key=lambda t: t.updated_at, reverse=True)
                if len(candidates) == 1:
                    target_task_id = candidates[0].task_id
                elif len(candidates) >= 2:
                    lines = [
                        f"- [{t.task_id}] {t.status} P{t.priority} {t.description[:80]}"
                        for t in candidates[:8]
                    ]
                    self._slack_send(
                        "⚠️ 指示の対象タスクが複数あります。`中止 <task_id>` / `保留 <task_id>` / `再開 <task_id>` のように指定してください。\n"
                        f"候補:\n" + "\n".join(lines)
                    )
                    return True
            except Exception:
                logger.warning("Failed to guess target task for directive", exc_info=True)

        if target_task_id is None:
            self._slack_send(
                "⚠️ 指示の対象（task_id）が特定できませんでした。スレッド内の `task_id:` を含むメッセージに返信するか、"
                "`中止 <task_id>` / `保留 <task_id>` / `再開 <task_id>` を送ってください。"
            )
            return True

        now = datetime.now(timezone.utc)
        reason = f"Creator指示: {directive.raw_text}".strip()

        if directive.kind == "cancel":
            updated = self.task_queue.update_status_tree(
                target_task_id,
                "canceled",
                error=reason,
                result=None,
            )
            self._slack_send(
                f"🛑 中止しました: [{target_task_id}]（対象{len(updated)}件）"
            )
        elif directive.kind == "pause":
            updated = self.task_queue.update_status_tree(
                target_task_id,
                "paused",
                error=reason,
                result=None,
            )
            self._slack_send(
                f"⏸️ 保留しました: [{target_task_id}]（対象{len(updated)}件）"
            )
        else:  # resume
            latest = None
            try:
                latest = self.task_queue._get_latest(target_task_id)
            except Exception:
                latest = None
            if latest is not None and latest.status == "canceled":
                self._slack_send(
                    f"⚠️ 中止済みタスクは再開できません: [{target_task_id}]"
                )
                return True

            updated = self.task_queue.update_status_tree(
                target_task_id,
                "pending",
                error=None,
                result=None,
                only_statuses={"paused"},
            )
            if not updated:
                self._slack_send(
                    f"ℹ️ 保留中タスクが見つかりませんでした: [{target_task_id}]"
                )
            else:
                self._slack_send(
                    f"▶️ 再開しました: [{target_task_id}]（対象{len(updated)}件）"
                )

        # Persist directive outcome into long-term memory (best-effort)
        try:
            mm = getattr(self, "memory_manager", None)
            if mm is not None:
                mm.note_interaction(
                    timestamp=now,
                    user_id=user_id,
                    request_text=f"[directive:{directive.kind}] {directive.raw_text}",
                    response_text=f"applied to task_id={target_task_id}",
                    snapshot_lines=[
                        f"task_id: {target_task_id}",
                        f"consult_id: {consult_id}" if consult_id else "consult_id: n/a",
                    ],
                )
                mm.ingest_all_sources()
        except Exception:
            logger.warning("Failed to persist directive outcome", exc_info=True)

        return True

    def _load_active_initiatives(self) -> list | None:
        """Load active initiatives (in_progress + planned) for context builder."""
        try:
            active = self.initiative_store.list_by_status("in_progress")
            planned = self.initiative_store.list_by_status("planned")
            return active + planned
        except Exception:
            logger.warning("Failed to load active initiatives", exc_info=True)
            return None

    def _load_strategy_direction(self):
        """Load strategy direction for context builder."""
        try:
            return self.strategy_analyzer.analyze()
        except Exception:
            logger.warning("Failed to load strategy direction", exc_info=True)
            return None

    def _execute_action_loop(
        self,
        actions: list[Action],
        conversation: list[dict[str, str]],
        task_id: str,
    ) -> None:
        """アクションを順次実行し、必要に応じてLLMに再問い合わせする."""
        iterations = 0
        memory_action_count = 0
        memory_payload_seen: set[str] = set()
        suppress_ack_reply = False
        work_dir = self.base_dir / "companies" / self.company_id

        while actions and iterations < self._MAX_ACTION_LOOP:
            iterations += 1
            next_actions: list[Action] = []

            for action in actions:
                self._trace_event(f"アクション実行: {action.action_type}")

                if action.action_type == "reply":
                    if suppress_ack_reply and self._looks_like_memory_ack_payload(action.content):
                        logger.info("Skipping ack-like reply in memory guard path (task_id=%s)", task_id)
                        suppress_ack_reply = False
                        continue
                    self._slack_send(action.content)

                elif action.action_type == "control":
                    logger.info("Control action received: %s", action.content[:120])
                    self._trace_event(f"control指示を適用: {self._sanitize_trace_text(action.content)}")
                    for line in action.content.splitlines():
                        cmd = line.strip()
                        if not cmd:
                            continue
                        handled_control, control_reply = self._handle_runtime_control_command(
                            cmd,
                            actor_id="ceo",
                            actor_role="ceo",
                            actor_model=self.llm_client.model if self.llm_client else None,
                        )
                        if handled_control:
                            if control_reply:
                                self._activity_log(
                                    f"CEO control結果: {self._summarize_for_activity_log(control_reply, limit=260)}"
                                )
                            continue
                        try:
                            directive = parse_creator_directive(cmd, thread_context=None)
                        except Exception:
                            directive = None
                        if directive is None:
                            self._slack_send(f"⚠️ control形式が不正です: {cmd}")
                            continue
                        self._apply_creator_directive(directive, user_id="ceo")

                elif action.action_type == "memory":
                    logger.info("Memory action received: %s", action.content[:120])
                    self._trace_event(f"記憶更新を実行: {self._sanitize_trace_text(action.content)}")
                    raw = (action.content or "").strip()
                    if not raw:
                        continue

                    import re

                    first, *rest = raw.splitlines()
                    m = re.match(r"^(curated|daily|pin)\s*[:：]?\s*(.*)$", first.strip(), re.IGNORECASE)
                    if m:
                        op = m.group(1).lower()
                        head = (m.group(2) or "").strip()
                        tail = "\n".join(rest).strip() if rest else ""
                        payload = (head + ("\n" + tail if tail else "")).strip()
                    else:
                        op = "daily"
                        payload = raw

                    payload_key = f"{op}:{' '.join(payload.split()).lower()[:600]}"
                    if payload_key in memory_payload_seen:
                        logger.info("Skipping duplicate memory action in task loop (task_id=%s)", task_id)
                        conversation.append({"role": "user", "content": "メモリ保存スキップ: 重複内容"})
                        suppress_ack_reply = True
                        continue

                    if memory_action_count >= self._MAX_MEMORY_ACTIONS_PER_TASK:
                        logger.warning(
                            "Memory action guard activated (task_id=%s, count=%d)",
                            task_id,
                            memory_action_count,
                        )
                        conversation.append({"role": "user", "content": "メモリ保存スキップ: guard(memory_limit)"})
                        suppress_ack_reply = True
                        continue

                    if self._looks_like_memory_ack_payload(payload):
                        logger.info("Skipping ack-like memory payload (task_id=%s)", task_id)
                        conversation.append({"role": "user", "content": "メモリ保存スキップ: ack_loop_guard"})
                        suppress_ack_reply = True
                        continue

                    def _split_title_body(text: str) -> tuple[str | None, str]:
                        s = (text or "").strip()
                        if not s:
                            return None, ""
                        lines = s.splitlines()
                        if (
                            len(lines) >= 2
                            and lines[0].strip()
                            and len(lines[0].strip()) <= 80
                            and not lines[0].lstrip().startswith(("-", "*", "#"))
                        ):
                            title = lines[0].strip()
                            body = "\n".join(lines[1:]).strip()
                            return (title if body else None), (body or title)
                        return None, s

                    try:
                        if op == "pin":
                            doc_id = None
                            if self.memory_manager is not None:
                                doc_id = self.memory_manager.pin(payload)
                                self.memory_manager.ingest_all_sources()
                            result_text = f"メモリ保存: pin OK ({doc_id or 'no-index'})"
                        elif op == "curated":
                            title, body = _split_title_body(payload)
                            self.memory_vault.append(body, title=title, author="ceo")
                            if self.memory_manager is not None:
                                self.memory_manager.ingest_all_sources()
                            result_text = "メモリ保存: curated OK"
                        else:
                            title, body = _split_title_body(payload)
                            self.memory_vault.append_daily(body, title=title, author="ceo")
                            if self.memory_manager is not None:
                                self.memory_manager.ingest_all_sources()
                            result_text = "メモリ保存: daily OK"

                        memory_payload_seen.add(payload_key)
                        memory_action_count += 1

                        try:
                            self.policy_memory.ingest_text(
                                payload,
                                source=f"memory_{op}",
                                user_id="ceo",
                                task_id=task_id,
                            )
                        except Exception:
                            logger.warning("Failed to ingest policy memory from memory action", exc_info=True)

                        try:
                            self.adaptive_memory.ingest_text(
                                payload,
                                source=f"memory_{op}",
                                user_id="ceo",
                                task_id=task_id,
                            )
                        except Exception:
                            logger.warning("Failed to ingest adaptive memory from memory action", exc_info=True)

                        try:
                            self.procedure_store.ingest_text(
                                payload,
                                source=f"memory_{op}",
                                user_id="ceo",
                                task_id=task_id,
                            )
                        except Exception:
                            logger.warning("Failed to ingest procedure SoT from memory action", exc_info=True)
                    except Exception as exc:
                        logger.warning("Memory action failed: %s", exc, exc_info=True)
                        result_text = f"メモリ保存エラー: {exc}"

                    conversation.append({"role": "user", "content": result_text})

                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )

                    follow_actions = parse_response(llm_result.content)
                    if self._is_ack_only_memory_followup(follow_actions):
                        logger.info("Detected ack-only memory follow-up; stopping recursion (task_id=%s)", task_id)
                        next_actions = []
                        break

                    conversation.append({"role": "assistant", "content": llm_result.content})
                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = follow_actions
                    break

                elif action.action_type == "commitment":
                    logger.info("Commitment action received: %s", action.content[:120])
                    raw = (action.content or "").strip()
                    if not raw:
                        continue

                    import re
                    from datetime import date

                    first, *rest = raw.splitlines()
                    first = first.strip()

                    result_text = ""
                    try:
                        m_close = re.match(
                            r"^(close|done|cancel|canceled)\s+([0-9a-f]{8})(?:\s*[:：]\s*(.*))?$",
                            first,
                            re.IGNORECASE,
                        )
                        if m_close:
                            cmd = m_close.group(1).lower()
                            cid = m_close.group(2)
                            note = (m_close.group(3) or "").strip()
                            if not note and rest:
                                note = "\n".join(rest).strip()
                            status = "done" if cmd in ("close", "done") else "canceled"
                            updated = self.commitment_store.close(cid, note=note, status=status)
                            if self.memory_manager is not None:
                                self.memory_manager.ingest_all_sources()
                            result_text = f"commitment {status}: {updated.commitment_id}"
                        else:
                            m_add = re.match(r"^(add|open)\s*[:：]?\s*(.*)$", first, re.IGNORECASE)
                            if m_add:
                                head = (m_add.group(2) or "").strip()
                                payload = (head + ("\n" + "\n".join(rest) if rest else "")).strip()
                            else:
                                payload = raw

                            # Extract due=YYYY-MM-DD (optional)
                            due_date = None
                            m_due = re.search(r"\bdue\s*=\s*(\d{4}-\d{2}-\d{2})\b", payload)
                            if m_due:
                                try:
                                    due_date = date.fromisoformat(m_due.group(1))
                                except Exception:
                                    due_date = None
                                payload = re.sub(r"\bdue\s*=\s*\d{4}-\d{2}-\d{2}\b", "", payload).strip()

                            def _split_title_body(text: str) -> tuple[str, str]:
                                s = (text or "").strip()
                                if not s:
                                    return "", ""
                                lines = s.splitlines()
                                if (
                                    len(lines) >= 2
                                    and lines[0].strip()
                                    and len(lines[0].strip()) <= 80
                                    and not lines[0].lstrip().startswith(("-", "*", "#"))
                                ):
                                    title = lines[0].strip()
                                    body = "\n".join(lines[1:]).strip()
                                    return title, (body or title)
                                return "", s

                            title, body = _split_title_body(payload)
                            entry, created = self.commitment_store.ensure_open(
                                body,
                                title=title,
                                owner="ceo",
                                due_date=due_date,
                                related_task_id=task_id,
                            )
                            if self.memory_manager is not None:
                                self.memory_manager.ingest_all_sources()
                            verb = "created" if created else "exists"
                            result_text = f"commitment {verb}: {entry.commitment_id}"
                    except Exception as exc:
                        logger.warning("Commitment action failed: %s", exc, exc_info=True)
                        result_text = f"commitment error: {exc}"

                    conversation.append({"role": "user", "content": result_text})

                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )
                    conversation.append({"role": "assistant", "content": llm_result.content})
                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = parse_response(llm_result.content)
                    break

                elif action.action_type == "done":
                    content = (action.content or "").strip()
                    self._slack_send(content or "完了しました。")

                elif action.action_type == "shell_command":
                    logger.info("Executing shell: %s", action.content)
                    self._trace_event(f"シェル実行中: {self._sanitize_trace_text(action.content)}")
                    self._activity_log("CEOツール利用: shell")
                    shell_result = execute_shell(
                        command=action.content,
                        cwd=work_dir,
                    )

                    # Shell execution ledger tracking is disabled in minimal mode.

                    # Build follow-up message with shell result
                    result_text = self._format_shell_result(shell_result)
                    conversation.append({"role": "user", "content": result_text})

                    # Re-query LLM with the shell result
                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )
                    conversation.append({"role": "assistant", "content": llm_result.content})

                    # Save assistant follow-up to conversation memory
                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = parse_response(llm_result.content)
                    break  # Process new actions in next iteration

                elif action.action_type == "consult":
                    logger.info("Consultation requested: %s", action.content[:120])
                    self._trace_event(f"consult判定中: {self._sanitize_trace_text(action.content)}")
                    self._activity_log("CEO判断: consultを検討")
                    consult_text = action.content.strip()
                    assessment = assess_creator_consultation(
                        consult_text,
                        constitution=self.state.constitution,
                    )

                    if not assessment.is_major:
                        logger.info("Treating consultation as minor (reason=%s); proceeding autonomously", assessment.reason)
                        autonomy_note = (
                            "（自律方針）以下は重大な意思決定ではないためCreatorには相談しません。\n"
                            "あなた（CEO AI）が最も安全・低コスト・可逆な選択を仮決定して作業を継続してください。\n"
                            f"- 相談内容: {consult_text}\n"
                            "\n"
                            "制約:\n"
                            "- 課金/契約/アカウント作成/広告出稿/ドメイン購入など「お金が動く」行為はしない\n"
                            "- 会社の目的/ビジョン/憲法の変更はしない（必要なら重大事項として別途<consult>）\n"
                            "- 外部公開は機密/炎上/規約リスクがない範囲で小さく。迷う場合は公開しない\n"
                            "\n"
                            "この方針に従い、以降は<consult>を使わず進めてください。"
                        )
                        conversation.append({"role": "user", "content": autonomy_note})

                        if self.llm_client is None:
                            break

                        llm_result = self.llm_client.chat(conversation)
                        if isinstance(llm_result, LLMError):
                            logger.error("Follow-up LLM call failed: %s", llm_result.message)
                            self._slack_send(
                                f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                            )
                            return

                        self.record_llm_call(
                            provider="openrouter",
                            model=llm_result.model,
                            input_tokens=llm_result.input_tokens,
                            output_tokens=llm_result.output_tokens,
                            task_id=task_id,
                        )
                        conversation.append({"role": "assistant", "content": llm_result.content})

                        try:
                            self.conversation_memory.append(ConversationEntry(
                                timestamp=datetime.now(timezone.utc),
                                role="assistant",
                                content=llm_result.content,
                                task_id=task_id,
                            ))
                        except Exception:
                            logger.warning(
                                "Failed to save assistant follow-up to conversation memory",
                                exc_info=True,
                            )

                        next_actions = parse_response(llm_result.content)
                        break

                    try:
                        entry, created = self.consultation_store.ensure_pending(
                            consult_text,
                            related_task_id=task_id,
                        )
                        if not created:
                            logger.info(
                                "Consultation already pending (consult_id=%s, task_id=%s)",
                                entry.consultation_id,
                                task_id,
                            )
                            return

                        message = (
                            f"🤝 相談 [consult_id: {entry.consultation_id}]\n\n"
                            f"{consult_text}\n\n"
                            f"（解決メモを残す場合: `resolve {entry.consultation_id}: ...`）"
                        )
                    except Exception:
                        logger.warning("Failed to record consultation", exc_info=True)
                        message = f"🤝 相談\n\n{consult_text}"
                    self._slack_send(message)
                    return

                elif action.action_type == "research":
                    logger.info("Executing research: %s", action.content)
                    self._trace_event(f"Web検索を実行: {self._sanitize_trace_text(action.content)}")
                    self._activity_log(f"CEOツール利用: web_search ({self._summarize_for_activity_log(action.content, limit=120)})")
                    search_results = self.web_searcher.search(action.content)

                    # Save each result as a ResearchNote
                    now = datetime.now(timezone.utc)
                    for sr in search_results:
                        note = ResearchNote(
                            query=action.content,
                            source_url=sr.url,
                            title=sr.title,
                            snippet=sr.snippet,
                            summary=sr.snippet,
                            retrieved_at=now,
                        )
                        try:
                            self.research_note_store.save(note)
                        except Exception:
                            logger.warning("Failed to save research note", exc_info=True)

                    # Build summary text
                    if search_results:
                        summary_parts = [f"リサーチ結果 (query={action.content}):"]
                        for i, sr in enumerate(search_results, 1):
                            summary_parts.append(f"{i}. {sr.title}\n   {sr.url}\n   {sr.snippet}")
                        result_text = "\n".join(summary_parts)
                    else:
                        result_text = f"リサーチ結果 (query={action.content}): 検索結果なし"

                    conversation.append({"role": "user", "content": result_text})

                    # Re-query LLM with the results
                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )
                    conversation.append({"role": "assistant", "content": llm_result.content})

                    # Save assistant follow-up to conversation memory
                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = parse_response(llm_result.content)
                    break  # Process new actions in next iteration

                elif action.action_type == "mcp":
                    logger.info("Executing MCP: %s", action.content[:160])
                    self._trace_event(f"MCPを実行: {self._sanitize_trace_text(action.content)}")
                    self._activity_log(f"CEOツール利用: mcp ({self._summarize_for_activity_log(action.content, limit=160)})")
                    result_text = self.mcp_client.run_action(action.content)
                    conversation.append({"role": "user", "content": result_text})

                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )
                    conversation.append({"role": "assistant", "content": llm_result.content})

                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = parse_response(llm_result.content)
                    break  # Process new actions in next iteration

                elif action.action_type == "publish":
                    logger.info("Executing publish: %s", action.content)
                    self._trace_event(f"publish操作を実行: {self._sanitize_trace_text(action.content)}")
                    self._activity_log(f"CEOツール利用: publish ({self._summarize_for_activity_log(action.content, limit=120)})")
                    content = action.content.strip()
                    parts = content.split(":", 2)
                    operation = parts[0] if parts else ""

                    if operation == "create_repo" and len(parts) >= 3:
                        repo_name = parts[1]
                        description = parts[2]
                        pub_result = self.git_publisher.create_repo(repo_name, description)
                        if pub_result.success:
                            try:
                                self.service_registry.register(
                                    name=repo_name,
                                    description=description,
                                    agent_id="manager",
                                )
                            except Exception:
                                logger.warning("Failed to register service", exc_info=True)
                            result_text = (
                                f"公開結果: {pub_result.message}"
                                f" (URL: {pub_result.repo_url})"
                            )
                        else:
                            result_text = f"公開エラー: {pub_result.message}"

                    elif operation == "commit" and len(parts) >= 3:
                        repo_path_str = parts[1]
                        message = parts[2]
                        repo_path = work_dir / repo_path_str
                        pub_result = self.git_publisher.commit_and_push(repo_path, message)
                        if pub_result.success:
                            result_text = f"公開結果: {pub_result.message}"
                        else:
                            result_text = f"公開エラー: {pub_result.message}"

                    elif operation == "self_commit":
                        message = content[len("self_commit:"):].strip()
                        if not message:
                            result_text = "公開エラー: self_commit のメッセージが空です"
                        else:
                            repo_root = Path(os.environ.get("APP_REPO_PATH", "/opt/apps/ai-company"))
                            safe_files = [
                                "src",
                                "tests",
                                "docs",
                                "dashboard",
                                ".github",
                                "Dockerfile",
                                "docker-compose.yml",
                                "pyproject.toml",
                                "README.md",
                                ".env.template",
                                "ai-company.service",
                                "setup-venv.sh",
                                "deploy-to-vps.sh",
                                "install-watchdog.sh",
                                "watchdog.py",
                                "migrate-to-host.sh",
                            ]
                            pub_result = self.git_publisher.commit_and_push(repo_root, message, files=safe_files)
                            if pub_result.success:
                                result_text = f"公開結果: {pub_result.message}"
                            else:
                                result_text = f"公開エラー: {pub_result.message}"

                    else:
                        result_text = f"公開エラー: 不明な操作形式です: {content}"

                    conversation.append({"role": "user", "content": result_text})

                    # Re-query LLM with the results
                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )
                    conversation.append({"role": "assistant", "content": llm_result.content})

                    # Save assistant follow-up to conversation memory
                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = parse_response(llm_result.content)
                    break  # Process new actions in next iteration

                elif action.action_type == "delegate":
                    logger.info("Delegating to sub-agent: %s", action.content[:120])
                    content = action.content.strip()
                    role_part, _, desc = content.partition(":")
                    desc = desc.strip() or content
                    assignment = self._resolve_delegate_assignment(role_part, desc)
                    role = assignment["role"]
                    assignment_kind = assignment["kind"]
                    employee = assignment.get("employee")

                    now = datetime.now(timezone.utc)
                    spent = compute_window_cost(self.state.ledger_events, now)
                    budget_limit = DEFAULT_BUDGET_LIMIT_USD
                    if self.state.constitution and self.state.constitution.budget:
                        budget_limit = self.state.constitution.budget.limit_usd
                    budget_remaining = max(0.0, budget_limit - spent)

                    creator_intent = ""
                    for msg in conversation:
                        if msg.get("role") == "user":
                            creator_intent = (msg.get("content") or "").strip()
                            if creator_intent:
                                break

                    delegation_brief = "\n".join([
                        "【CEO委任ブリーフ】",
                        "- 分業原則: CEOは目的/制約を定義し、実装のHowは社員AIが決める。",
                        f"- 委任先: {assignment_kind}",
                        f"- role: {role}",
                        f"- task_id: {task_id}",
                        f"- 予算残: ${budget_remaining:.2f} (limit=${budget_limit:.2f})",
                        "- 期待: 目的達成に必要な具体手順を自律的に設計・実行し、証跡付きで報告する。",
                        "- エスカレーション: 方針矛盾/高リスク/予算超過見込み時は報告する。",
                        "",
                        "【Creator意図(要約元)】",
                        (creator_intent[:500] or "(なし)"),
                        "",
                        "【依頼本文】",
                        desc,
                    ])
                    model_hint = action.model or (employee.model if employee is not None else "auto")
                    self._trace_event(
                        f"社員AIへ委任: role={role} kind={assignment_kind} model={model_hint}"
                    )
                    self._trace_event("社員AIへの指示:\n```\n" + self._sanitize_trace_text(delegation_brief[:1600]) + "\n```")
                    if employee is not None:
                        self._activity_log(
                            f"CEO→社員AI 委任: employee={employee.name}({employee.employee_id}) "
                            f"role={role} model={model_hint} "
                            f"task={self._summarize_for_activity_log(desc, limit=240)}"
                        )
                    else:
                        self._activity_log(
                            f"CEO→アルバイト 委任: role={role} model={model_hint} "
                            f"task={self._summarize_for_activity_log(desc, limit=240)}"
                        )

                    try:
                        target_name = employee.name if employee is not None else role
                        target_budget = (
                            float(employee.budget_limit_usd)
                            if employee is not None
                            else min(1.0, max(0.2, budget_remaining))
                        )
                        result = self.sub_agent_runner.spawn(
                            name=target_name,
                            role=role,
                            task_description=delegation_brief,
                            budget_limit_usd=target_budget,
                            model=action.model,
                            persistent_employee=employee.model_dump() if employee is not None else None,
                        )
                        result_text = f"サブエージェント結果 (role={role}):\n{result}"
                        if result.lstrip().startswith("⚠️ 社員AI中断報告"):
                            self._slack_send(result)
                        if employee is not None:
                            self._activity_log(
                                f"社員AI→CEO 報告: employee={employee.name}({employee.employee_id}) role={role} "
                                f"result={self._summarize_for_activity_log(result, limit=320)}"
                            )
                        else:
                            self._activity_log(
                                f"アルバイト→CEO 報告: role={role} "
                                f"result={self._summarize_for_activity_log(result, limit=320)}"
                            )
                    except Exception as exc:
                        logger.warning("Sub-agent spawn failed: %s", exc, exc_info=True)
                        result_text = f"サブエージェントエラー (role={role}): {exc}"
                        if employee is not None:
                            self._activity_log(
                                f"社員AIエラー: employee={employee.name}({employee.employee_id}) role={role} "
                                f"error={self._summarize_for_activity_log(str(exc), limit=220)}"
                            )
                        else:
                            self._activity_log(
                                f"アルバイトエラー: role={role} error={self._summarize_for_activity_log(str(exc), limit=220)}"
                            )

                    conversation.append({"role": "user", "content": result_text})

                    # Re-query LLM with the results
                    if self.llm_client is None:
                        break

                    llm_result = self.llm_client.chat(conversation)
                    if isinstance(llm_result, LLMError):
                        logger.error("Follow-up LLM call failed: %s", llm_result.message)
                        self._slack_send(
                            f"エラー: LLM再問い合わせに失敗しました — {llm_result.message}",
                        )
                        return

                    self.record_llm_call(
                        provider="openrouter",
                        model=llm_result.model,
                        input_tokens=llm_result.input_tokens,
                        output_tokens=llm_result.output_tokens,
                        task_id=task_id,
                    )
                    conversation.append({"role": "assistant", "content": llm_result.content})

                    # Save assistant follow-up to conversation memory
                    try:
                        self.conversation_memory.append(ConversationEntry(
                            timestamp=datetime.now(timezone.utc),
                            role="assistant",
                            content=llm_result.content,
                            task_id=task_id,
                        ))
                    except Exception:
                        logger.warning(
                            "Failed to save assistant follow-up to conversation memory",
                            exc_info=True,
                        )

                    next_actions = parse_response(llm_result.content)
                    break  # Process new actions in next iteration

                elif action.action_type == "plan":
                    logger.info("Plan action received: %s", action.content[:120])
                    # Extract original user message for parent task description
                    user_msg = ""
                    for msg in conversation:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                            break
                    task_desc = user_msg[:100] if user_msg else "タスク分解"
                    self._handle_plan_action(action, task_description=task_desc)
                    # plan does not trigger a follow-up LLM call; continue to next action

            # If no shell_command triggered a new LLM call, we're done
            if not next_actions:
                break
            actions = next_actions

    def _handle_plan_action(self, action: Action, task_description: str) -> None:
        """<plan>アクションからサブタスクを登録する.

        Args:
            action: action_type="plan" のActionオブジェクト
            task_description: 親タスクの説明に使うCreatorメッセージの要約
        """
        subtasks = parse_plan_content(action.content)

        if not subtasks:
            logger.warning("plan action contained no subtasks")
            self._slack_send("⚠️ タスク分解: サブタスクが見つかりませんでした")
            return

        # 親タスクを登録
        parent = self.task_queue.add(
            description=f"[親] {task_description}",
            priority=1,
            source="creator",
            slack_channel=self._slack_reply_channel,
            slack_thread_ts=self._slack_reply_thread_ts,
        )

        # サブタスクを依存関係付きで登録
        task_id_map: dict[int, str] = {}  # plan内番号 → 実際のtask_id
        for st in subtasks:
            depends_on = [task_id_map[d] for d in st.depends_on_indices if d in task_id_map]
            entry = self.task_queue.add_with_deps(
                description=st.description,
                depends_on=depends_on,
                parent_task_id=parent.task_id,
                priority=1,
                source="creator",
                slack_channel=self._slack_reply_channel,
                slack_thread_ts=self._slack_reply_thread_ts,
            )
            task_id_map[st.index] = entry.task_id

        # Creatorに報告
        self._slack_send(f"📋 {len(subtasks)}件のサブタスクを登録しました。完了時にまとめて報告します。")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_delegate_role(role_part: str, desc: str) -> str:
        import re

        generic = {"role", "agent", "sub-agent", "subagent", "worker", "社員", "employee", "staff", "member"}
        role = (role_part or "").strip()
        if not role:
            role = "worker"

        role_l = role.lower()
        if role_l not in generic:
            return role

        first_token = (desc or "").strip().split()[0] if (desc or "").strip() else ""
        token = first_token.strip(" :：,-_()").lower()
        if token and token not in generic and re.fullmatch(r"[a-z][a-z0-9_-]{1,40}", token):
            return token

        m = re.search(r"\b(researcher|writer|analyst|developer|engineer|web-developer|devops|qa|designer)\b", (desc or "").lower())
        if m:
            return m.group(1)

        return "worker"

    def _resolve_delegate_assignment(self, role_part: str, desc: str) -> dict:
        token = (role_part or "").strip()
        lower = token.lower()

        def _fallback_role(raw_role: str) -> str:
            return self._normalize_delegate_role(raw_role, desc)

        if lower.startswith(("part-time@", "parttime@", "アルバイト@", "baito@")):
            raw = token.split("@", 1)[1].strip() if "@" in token else ""
            return {
                "kind": "part-time",
                "employee": None,
                "role": _fallback_role(raw),
            }

        if lower.startswith(("employee@", "社員@")):
            target = token.split("@", 1)[1].strip() if "@" in token else ""
            employee = self.employee_store.resolve_active(target)
            if employee is not None:
                return {
                    "kind": "employee",
                    "employee": employee,
                    "role": employee.role,
                }
            return {
                "kind": "part-time",
                "employee": None,
                "role": _fallback_role(target),
            }

        by_key = self.employee_store.resolve_active(token)
        if by_key is not None:
            return {
                "kind": "employee",
                "employee": by_key,
                "role": by_key.role,
            }

        role = _fallback_role(token) if token else _fallback_role("worker")
        by_role = self.employee_store.find_active_by_role(role)
        if by_role is not None:
            return {
                "kind": "employee",
                "employee": by_role,
                "role": by_role.role,
            }

        return {
            "kind": "part-time",
            "employee": None,
            "role": role,
        }

    @staticmethod
    def _is_agent_list_request(text: str) -> bool:
        normalized = (text or "").replace(" ", "").replace("　", "").lower()
        if not normalized:
            return False

        has_meta_intent = any(
            k in normalized
            for k in ("ロジック", "実装", "修正", "改善", "整備", "テスト", "検証", "ルーティング", "判定")
        )
        has_example_intent = any(
            k in normalized
            for k in ("要求された時", "要求されたとき", "聞かれた時", "聞かれたとき", "訊かれた時", "訊かれたとき", "尋ねられた時", "尋ねられたとき", "への回答", "に答えられる", "を返せる", "答えるように", "できるように", "ようにしてください")
        )
        if has_meta_intent or has_example_intent:
            return False

        has_agent_word = any(
            k in normalized
            for k in ("社員ai", "社員", "エージェント", "sub-agent", "subagent", "worker")
        )
        has_list_word = any(
            k in normalized
            for k in ("一覧", "リスト", "最近", "直近", "動いて", "稼働", "アクティブ", "教えて", "共有")
        )
        has_procedure_hint = any(k in normalized for k in ("手順", "runbook", "procedure", "sot"))
        return has_agent_word and has_list_word and not has_procedure_hint

    def _build_agent_list_reply(self, request_text: str = "") -> str:
        try:
            all_runtime_agents = [a for a in self.agent_registry._list_all() if a.agent_id != "ceo"]
            employees = self.employee_store.list_all()
        except Exception:
            logger.warning("Failed to load agent list", exc_info=True)
            return "社員AIの一覧取得中にエラーが発生しました。もう一度試してください。"

        normalized = (request_text or "").replace(" ", "").replace("　", "").lower()
        asks_recent = any(k in normalized for k in ("最近", "直近", "動いて", "稼働", "アクティブ", "現在"))

        def _fmt_runtime(agent) -> str:
            ts = agent.updated_at.strftime("%Y-%m-%d %H:%M")
            model = (agent.model or "unknown").strip() or "unknown"
            return (
                f"- {agent.name}（role={agent.role} / status={agent.status} / "
                f"model={model} / updated={ts} UTC）"
            )

        lines: list[str] = []

        if employees:
            active_employees = [e for e in employees if e.status == "active"]
            inactive_employees = [e for e in employees if e.status != "active"]
            lines.append(f"正社員AIは {len(active_employees)} 名（登録合計 {len(employees)} 名）です。")
            for e in sorted(active_employees, key=lambda x: x.updated_at, reverse=True)[:12]:
                ts = e.updated_at.strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"- {e.name}（id={e.employee_id} / role={e.role} / model={e.model} / "
                    f"purpose={self._summarize_for_activity_log(e.purpose, limit=50)} / updated={ts} UTC）"
                )
            if inactive_employees and not asks_recent:
                lines.append("現在停止中の正社員AI:")
                for e in sorted(inactive_employees, key=lambda x: x.updated_at, reverse=True)[:5]:
                    ts = e.updated_at.strftime("%Y-%m-%d %H:%M")
                    lines.append(f"- {e.name}（id={e.employee_id} / role={e.role} / model={e.model} / updated={ts} UTC）")
        else:
            lines.append("正社員AIはまだ登録されていません。")

        runtime_part_time = [a for a in all_runtime_agents if not str(a.agent_id).startswith("emp-")]
        runtime_part_time.sort(key=lambda a: a.updated_at, reverse=True)
        active_part_time = [a for a in runtime_part_time if a.status == "active"]

        if active_part_time:
            lines.append(f"現在稼働中のアルバイトAIは {len(active_part_time)} 名です。")
            for agent in active_part_time[:8]:
                lines.append(_fmt_runtime(agent))
        elif runtime_part_time:
            lines.append("現在アクティブなアルバイトAIはいません。")
            lines.append("直近で動いていたアルバイトAI:")
            for agent in runtime_part_time[:5]:
                lines.append(_fmt_runtime(agent))

        if len(lines) == 0:
            return "現在、社員AIはまだ作成されていません。"
        return "\n".join(lines)

    @staticmethod
    def _is_procedure_library_request(text: str) -> bool:
        normalized = (text or "").replace(" ", "").replace("　", "").lower()
        if not normalized:
            return False
        has_library_word = any(k in normalized for k in ("一覧", "リスト", "library", "ライブラリ", "どんな", "ある"))
        has_target_word = any(
            k in normalized
            for k in ("手順", "runbook", "procedure", "sot", "共有手順", "手順sot")
        )
        return has_library_word and has_target_word

    def _build_procedure_library_reply(self) -> str:
        try:
            docs = self.procedure_store.list_active()
        except Exception:
            logger.warning("Failed to load procedure library", exc_info=True)
            return "手順ドキュメントの確認中にエラーが発生しました。もう一度試してください。"

        if not docs:
            return "確認しました。現在、保存済みの手順ドキュメントはありません。必要なら今回の作業手順を保存します。"

        private_count = sum(1 for d in docs if d.visibility == "private")
        shared_count = sum(1 for d in docs if d.visibility == "shared")
        return (
            f"手順ドキュメントは保存済みです（社内用 {private_count} 件 / 共有 {shared_count} 件）。"
            "必要な作業名を指定してくれれば、該当手順だけ再掲します。"
        )

    @staticmethod
    def _is_max_turns_question(text: str) -> bool:
        normalized = (text or "").replace(" ", "").replace("　", "").lower()
        if not normalized:
            return False
        has_target = any(k in normalized for k in ("最大会話ターン数", "最大ターン数", "max_task_turns", "maxturn"))
        has_question = any(k in normalized for k in ("何", "いくつ", "教えて", "確認", "設定", "値", "なっていますか", "ですか"))
        return has_target and has_question

    @staticmethod
    def _is_time_question(text: str) -> bool:
        normalized = (text or "").replace(" ", "").replace("　", "").lower()
        if not normalized:
            return False
        if any(k in normalized for k in ("time now", "time.now", "現在時刻", "今何時", "いま何時", "いまの時刻", "日時")):
            return True
        return any(k in normalized for k in ("何時", "なんじ", "時刻")) and any(
            q in normalized for q in ("教えて", "確認", "知りたい", "ですか")
        )

    def _handle_runtime_control_command(
        self,
        command: str,
        *,
        actor_id: str,
        actor_role: str,
        actor_model: str | None = None,
    ) -> tuple[bool, str]:
        try:
            handled_employee, employee_reply = self._handle_employee_control_command(
                command,
                actor_id=actor_id,
                actor_model=actor_model,
            )
            if handled_employee:
                return True, employee_reply
            return self.alarm_scheduler.handle_control_command(
                command,
                actor_id=actor_id,
                actor_role=actor_role,
                actor_model=actor_model,
            )
        except Exception as exc:
            logger.warning("Failed to process runtime control command: %s", command, exc_info=True)
            return True, f"⚠️ controlコマンド処理中にエラーが発生しました: {exc}"

    def _handle_employee_control_command(
        self,
        command: str,
        *,
        actor_id: str,
        actor_model: str | None,
    ) -> tuple[bool, str]:
        cmd = (command or "").strip()
        if not cmd:
            return False, ""

        m = re.match(r"^employee\s+resume\s+([A-Za-z0-9_-]+)\s*$", cmd, re.IGNORECASE)
        if m:
            run_id = m.group(1).strip()
            checkpoint = self.sub_agent_runner.get_run_checkpoint(run_id)
            if checkpoint is None:
                return True, f"⚠️ 指定run_idのチェックポイントが見つかりません: {run_id}"

            role = str(checkpoint.get("role") or "worker").strip() or "worker"
            model = str(checkpoint.get("model") or (actor_model or "")).strip() or None
            employee_id = str(checkpoint.get("employee_id") or "").strip()
            employee = self.employee_store.get_by_id(employee_id) if employee_id else None

            if employee is not None and employee.status != "active":
                try:
                    employee = self.employee_store.update_status(employee.employee_id, "active")
                except Exception:
                    logger.warning("Failed to activate employee for resume: %s", employee.employee_id, exc_info=True)

            base_task = str(checkpoint.get("task_description") or "").strip() or "前回中断したタスクを再開する"
            progress = checkpoint.get("progress") or []
            if isinstance(progress, list):
                progress_lines = [f"- {str(item)}" for item in progress[-8:] if str(item).strip()]
            else:
                progress_lines = []
            if not progress_lines:
                progress_lines = ["- 進捗ログなし"]

            pending_hint = str(checkpoint.get("pending_hint") or "").strip() or "未完了作業を洗い出して継続"
            task_description = "\n".join([
                base_task,
                "",
                f"run_id={run_id} の中断点から再開してください。",
                "実施済み:",
                *progress_lines,
                "再開要件:",
                f"- {pending_hint}",
                "- 重複実行を避け、未完了作業から再開する",
                "- 完了時は結果と検証証跡を報告する",
            ])

            target_name = employee.name if employee is not None else role
            target_budget = float(employee.budget_limit_usd) if employee is not None else 1.0
            try:
                result = self.sub_agent_runner.spawn(
                    name=target_name,
                    role=role,
                    task_description=task_description,
                    budget_limit_usd=target_budget,
                    model=model,
                    ignore_wip_limit=True,
                    persistent_employee=employee.model_dump() if employee is not None else None,
                )
            except Exception as exc:
                logger.warning("Failed to resume employee run: %s", run_id, exc_info=True)
                return True, f"⚠️ run_id {run_id} の再開に失敗しました: {exc}"

            return True, f"▶️ run_id {run_id} を再開しました。\n{result}"

        m = re.match(r"^employee\s+list(?:\s+(\d+))?\s*$", cmd, re.IGNORECASE)
        if m:
            limit = int(m.group(1) or 20)
            rows = self.employee_store.list_all()
            if not rows:
                return True, "正社員AIはまだ登録されていません。"
            rows.sort(key=lambda x: x.updated_at, reverse=True)
            lines = [f"正社員AI一覧（{len(rows)}名, 表示上限{limit}）"]
            for e in rows[: max(1, limit)]:
                ts = e.updated_at.strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"- {e.name}（id={e.employee_id} / role={e.role} / status={e.status} / model={e.model} / updated={ts} UTC）"
                )
            return True, "\n".join(lines)

        m = re.match(r"^employee\s+create\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)(?:\s*\|\s*(.+))?$", cmd, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            role = m.group(2).strip()
            purpose = m.group(3).strip()
            opts = (m.group(4) or "").strip()

            model = (actor_model or "").strip() or "openai/gpt-4.1-mini"
            budget = 1.0
            for opt in [x.strip() for x in opts.split(";") if x.strip()]:
                if "=" not in opt:
                    continue
                key, val = opt.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                if key == "model" and val:
                    model = val
                elif key in ("budget", "budget_usd"):
                    try:
                        budget = max(0.05, float(val))
                    except Exception:
                        pass

            entry, created = self.employee_store.ensure_active(
                name=name,
                role=role,
                purpose=purpose,
                model=model,
                budget_limit_usd=budget,
            )
            if created:
                return True, (
                    "✅ 正社員AIを登録しました。\n"
                    f"- name: {entry.name}\n"
                    f"- id: {entry.employee_id}\n"
                    f"- role: {entry.role}\n"
                    f"- model: {entry.model}\n"
                    f"- budget_limit: ${entry.budget_limit_usd:.2f}\n"
                    f"- memory: {self.employee_store.memory_path(entry.employee_id)}"
                )
            return True, (
                f"ℹ️ 既存の正社員AIを再利用します: {entry.name}（id={entry.employee_id} / role={entry.role} / model={entry.model}）"
            )

        m = re.match(r"^employee\s+(activate|deactivate)\s+(.+)$", cmd, re.IGNORECASE)
        if m:
            mode = m.group(1).lower()
            key = m.group(2).strip()
            entry = self.employee_store.resolve_active(key) if mode == "deactivate" else self.employee_store.resolve_active(key)
            if entry is None:
                entry = self.employee_store.get_by_id(key) or self.employee_store.find_by_name(key)
            if entry is None:
                return True, f"⚠️ 指定の正社員AIが見つかりません: {key}"
            updated = self.employee_store.update_status(entry.employee_id, "active" if mode == "activate" else "inactive")
            return True, f"✅ 正社員AIの状態を更新しました: {updated.name}（id={updated.employee_id} / status={updated.status}）"

        m = re.match(r"^employee\s+memory\s+(.+)$", cmd, re.IGNORECASE)
        if m:
            key = m.group(1).strip()
            entry = self.employee_store.get_by_id(key) or self.employee_store.find_by_name(key)
            if entry is None:
                return True, f"⚠️ 指定の正社員AIが見つかりません: {key}"
            text = self.employee_store.read_memory(entry.employee_id, max_chars=2500).strip()
            if not text:
                return True, f"{entry.name} のメモリはまだ空です。"
            return True, f"{entry.name} のメモリ抜粋:\n{text}"

        return False, ""



    @staticmethod
    def _normalize_url(url: str) -> str:
        u = (url or "").strip()
        u = u.rstrip(")]}>.,'、。")
        if u.startswith("http://"):
            u = "https://" + u[len("http://"):]
        if u.endswith("/"):
            u = u[:-1]
        return u

    @staticmethod
    def _extract_http_urls(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"https?://[^\s)\]}>\"']+", text)

    @staticmethod
    def _should_prefetch_web_search(text: str) -> bool:
        # Heuristic: prefetch web search for time-sensitive external questions.
        s = (text or "").strip()
        if not s:
            return False
        lowered = s.lower()

        internal_markers = (
            "vps",
            "traefik",
            "docker",
            "compose",
            "system prompt",
            "システムプロンプト",
            "ロジック",
            "どのファイル",
            "最大会話ターン数",
            "max_task_turns",
            "社員ai",
            "社員 ai",
            "エージェント",
        )
        if any(m in lowered for m in internal_markers):
            return False

        if re.search(r"\b20(2[5-9]|[3-9]\d)\b", lowered):
            return True
        if any(k in s for k in ("最新", "最近", "ニュース", "リリース", "発表", "バージョン")):
            return True
        return False

    def _read_turn_limit_settings(self) -> dict[str, str]:
        def _as_int(env_key: str, fallback: int) -> int:
            raw = (os.environ.get(env_key) or "").strip()
            if not raw:
                return fallback
            try:
                return int(raw)
            except Exception:
                return fallback

        sub_turns = _as_int("SUB_AGENT_MAX_TURNS", 0)
        sub_wall = _as_int("SUB_AGENT_MAX_WALL_SECONDS", 0)
        auto_turns = _as_int("AUTONOMOUS_MAX_TURNS", 100)
        auto_wall = _as_int("AUTONOMOUS_MAX_WALL_SECONDS", 0)

        def _fmt(value: int, suffix: str = "") -> str:
            if value <= 0:
                return "0 (上限なし)"
            return f"{value}{suffix}"

        return {
            "sub_agent_max_turns_text": _fmt(sub_turns),
            "sub_agent_max_wall_text": _fmt(sub_wall, "秒"),
            "autonomous_max_turns_text": _fmt(auto_turns),
            "autonomous_max_wall_text": _fmt(auto_wall, "秒"),
        }

    def _load_slack_context(self) -> None:
        """Load last known Slack context (best-effort) for autonomous reports."""
        try:
            if not self._slack_context_path.exists():
                return
            raw = self._slack_context_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return
            channel = (data.get("channel") or "").strip() or None
            thread_ts = (data.get("thread_ts") or "").strip() or None
            self._slack_last_channel = channel
            self._slack_last_thread_ts = thread_ts
        except Exception:
            logger.warning("Failed to load Slack context", exc_info=True)

    def _persist_slack_context(self) -> None:
        """Persist last known Slack context (best-effort)."""
        try:
            self._slack_context_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "channel": self._slack_last_channel,
                "thread_ts": self._slack_last_thread_ts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._slack_context_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.warning("Failed to persist Slack context", exc_info=True)

    def _remember_slack_context(self, channel: str | None, thread_ts: str | None) -> None:
        """Remember Slack context for later autonomous completion reports."""
        ch = (channel or "").strip() or None
        if not ch:
            return
        self._slack_last_channel = ch
        self._slack_last_thread_ts = (thread_ts or "").strip() or None
        self._persist_slack_context()

    def _bootstrap_trace_thread(self, request_text: str) -> None:
        if not self._trace_enabled or self.slack is None:
            return
        if self._slack_reply_thread_ts:
            self._trace_thread_ts = self._slack_reply_thread_ts
            return
        if not self._slack_reply_channel:
            return

        excerpt = " ".join((request_text or "").split())
        if len(excerpt) > 120:
            excerpt = excerpt[:120] + "…"
        header = f"🧭 処理ログ開始\n質問: {excerpt or '(empty)'}"
        ts = self.slack.send_message(header, channel=self._slack_reply_channel)
        if ts:
            self._trace_thread_ts = ts
            self._slack_reply_thread_ts = ts

    def _trace_event(self, message: str) -> None:
        if not self._trace_enabled or self.slack is None:
            return
        channel = self._slack_reply_channel
        thread_ts = self._trace_thread_ts or self._slack_reply_thread_ts
        if not channel or not thread_ts:
            return
        safe = self._sanitize_trace_text(message)
        if not safe:
            return
        self.slack.send_message(f"🧭 {safe}", channel=channel, thread_ts=thread_ts)

    def _sanitize_trace_text(self, text: str) -> str:
        import re

        s = (text or "").strip()
        if not s:
            return ""

        for key in ("OPENROUTER_API_KEY", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
            secret = os.environ.get(key)
            if secret:
                s = s.replace(secret, "***")

        s = re.sub(r"(?i)\b(api[_-]?key|token|password|passwd|secret)\s*[:=]\s*[^\s]+", r"\1=***", s)
        if len(s) > 1800:
            s = s[:1800] + "…"
        return s

    @staticmethod
    def _summarize_for_activity_log(text: str, *, limit: int = 300) -> str:
        s = " ".join((text or "").split())
        if len(s) > limit:
            return s[:limit] + "…"
        return s

    def _activity_log(self, message: str) -> None:
        if not self._activity_log_enabled or self.slack is None:
            return
        channel = self._activity_log_channel
        if not channel:
            return
        safe = self._sanitize_trace_text(message)
        if not safe:
            return
        self.slack.send_message(f"📌 {safe}", channel=channel)

    def _slack_send(
        self,
        text: str,
        *,
        channel: str | None = None,
        thread_ts: str | None = None,
    ) -> str | None:
        """Send a message via Slack if the bot is configured."""
        if self.slack is not None:
            target_channel = channel or self._slack_reply_channel or self._slack_default_channel or self._slack_last_channel
            if target_channel and target_channel != self._activity_log_channel:
                self._activity_log(f"CEO→Creator: {self._summarize_for_activity_log(text, limit=700)}")
            return self.slack.send_message(
                text,
                channel=target_channel,
                thread_ts=thread_ts or self._slack_reply_thread_ts,
            )
        logger.warning("Slack not configured, message not sent: %s", text[:100])
        return None

    @staticmethod
    def _format_shell_result(result: ShellResult) -> str:
        """Format a ShellResult for inclusion in the LLM conversation."""
        parts = [f"コマンド実行結果 (return_code={result.return_code}):"]
        if result.timed_out:
            parts.append("⚠️ タイムアウトしました")
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        return "\n".join(parts)


    @staticmethod
    def _looks_like_memory_ack_payload(payload: str) -> bool:
        """Detect likely memory-ack loop payloads from the model itself."""
        text = (payload or "").strip().lower()
        if not text:
            return False
        loop_markers = (
            "curated ok",
            "daily ok",
            "pin ok",
            "メモリ保存指示",
            "保存指示",
            "承認しました",
            "再承認",
            "継続いたします",
            "継続します",
        )
        return any(marker in text for marker in loop_markers)


    def _is_ack_only_memory_followup(self, actions: list[Action]) -> bool:
        """Return True when follow-up consists only of ack-like memory/reply actions."""
        if not actions:
            return False

        saw_any = False
        for action in actions:
            if action.action_type == "memory":
                saw_any = True
                if not self._looks_like_memory_ack_payload(action.content):
                    return False
                continue
            if action.action_type == "reply":
                saw_any = True
                if not self._looks_like_memory_ack_payload(action.content):
                    return False
                continue
            return False
        return saw_any
