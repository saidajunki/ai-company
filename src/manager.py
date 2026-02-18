"""Manager – orchestration layer tying all components together.

Provides:
- init_company_directory: Create the full directory structure and default constitution
- Manager class: Synchronous logic layer for event-driven operations

Requirements: 7.1, 7.2, 7.6
"""

from __future__ import annotations

import logging
import os
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
from task_queue import TaskQueue
from vision_loader import DEFAULT_VISION, VisionLoader

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
        self.web_searcher = WebSearcher()
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

        return action, description

    # ------------------------------------------------------------------
    # Budget check (Req 5.3, 5.4)
    # ------------------------------------------------------------------

    def check_budget(self) -> bool:
        """Budget management is disabled in minimal mode."""
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
        """LLM cost/ledger tracking is disabled in minimal mode."""
        now = datetime.now(timezone.utc)
        return LedgerEvent(
            timestamp=now,
            event_type="llm_call",
            agent_id=agent_id,
            task_id=task_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unit_price_usd_per_1k_input_tokens=0.0,
            unit_price_usd_per_1k_output_tokens=0.0,
            price_retrieved_at=now,
            estimated_cost_usd=0.0,
        )

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
        self._slack_reply_channel = slack_channel
        self._slack_reply_thread_ts = slack_thread_ts
        try:
            stripped = (text or "").strip()

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

            if self._is_agent_list_request(stripped):
                self._slack_send(self._build_agent_list_reply())
                return

            if self._is_procedure_library_request(stripped):
                self._slack_send(self._build_procedure_library_reply())
                return
            recalled_procedure = self.procedure_store.find_best_for_request(stripped)
            if recalled_procedure is not None:
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
            )

            conversation: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]

            # 3. LLM呼び出し
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
                if action.action_type == "reply":
                    if suppress_ack_reply and self._looks_like_memory_ack_payload(action.content):
                        logger.info("Skipping ack-like reply in memory guard path (task_id=%s)", task_id)
                        suppress_ack_reply = False
                        continue
                    self._slack_send(action.content)

                elif action.action_type == "control":
                    logger.info("Control action received: %s", action.content[:120])
                    for line in action.content.splitlines():
                        cmd = line.strip()
                        if not cmd:
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
                    self._slack_send(f"完了: {action.content}")

                elif action.action_type == "shell_command":
                    logger.info("Executing shell: %s", action.content)
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

                elif action.action_type == "publish":
                    logger.info("Executing publish: %s", action.content)
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
                            pub_result = self.git_publisher.commit_and_push(repo_root, message)
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
                    role, _, desc = content.partition(":")
                    role = role.strip() or "worker"
                    desc = desc.strip() or content

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

                    try:
                        result = self.sub_agent_runner.spawn(
                            name=role,
                            role=role,
                            task_description=delegation_brief,
                            model=action.model,
                        )
                        result_text = f"サブエージェント結果 (role={role}):\n{result}"
                    except Exception as exc:
                        logger.warning("Sub-agent spawn failed: %s", exc, exc_info=True)
                        result_text = f"サブエージェントエラー (role={role}): {exc}"

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
        parent = self.task_queue.add(description=f"[親] {task_description}", priority=1, source="creator")

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
            )
            task_id_map[st.index] = entry.task_id

        # Creatorに報告
        self._slack_send(f"📋 タスク分解完了 ({len(subtasks)}件のサブタスク)")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            for k in ("要求された時", "聞かれた時", "訊かれた時", "への回答", "に答えられる", "を返せる")
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

    def _build_agent_list_reply(self) -> str:
        try:
            all_agents = [a for a in self.agent_registry._list_all() if a.agent_id != "ceo"]
        except Exception:
            logger.warning("Failed to load agent list", exc_info=True)
            return "社員AIの一覧取得中にエラーが発生しました。もう一度試してください。"

        if not all_agents:
            return "現在、社員AIはまだ作成されていません。必要なら役割を指定して作成します。"

        all_agents.sort(key=lambda a: a.updated_at, reverse=True)
        active_agents = [a for a in all_agents if a.status == "active"]

        lines: list[str] = []
        if active_agents:
            lines.append(f"最近動いている社員AIは {len(active_agents)} 名です。")
            for agent in active_agents[:8]:
                ts = agent.updated_at.strftime("%Y-%m-%d %H:%M")
                lines.append(f"- {agent.name}（{agent.role}） model={agent.model} / 更新: {ts} UTC")
        else:
            lines.append("現在アクティブな社員AIはいません。")
            lines.append("直近で動いていた社員AI:")
            for agent in all_agents[:5]:
                ts = agent.updated_at.strftime("%Y-%m-%d %H:%M")
                lines.append(f"- {agent.name}（{agent.role} / {agent.status}） 更新: {ts} UTC")

        lines.append("必要なら、この中から担当を指定して次タスクを振り分けます。")
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

    def _slack_send(
        self,
        text: str,
        *,
        channel: str | None = None,
        thread_ts: str | None = None,
    ) -> None:
        """Send a message via Slack if the bot is configured."""
        if self.slack is not None:
            self.slack.send_message(
                text,
                channel=channel or self._slack_reply_channel,
                thread_ts=thread_ts or self._slack_reply_thread_ts,
            )
        else:
            logger.warning("Slack not configured, message not sent: %s", text[:100])

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
        has_memory = False
        for action in actions:
            if action.action_type == "memory":
                has_memory = True
                if not self._looks_like_memory_ack_payload(action.content):
                    return False
                continue
            if action.action_type == "reply":
                if not self._looks_like_memory_ack_payload(action.content):
                    return False
                continue
            return False
        return has_memory
