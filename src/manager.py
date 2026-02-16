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
from typing import Literal

from agent_registry import AgentRegistry
from autonomous_loop import AutonomousLoop
from constitution_store import constitution_save
from context_builder import build_system_prompt
from conversation_memory import ConversationMemory
from cost_aggregator import compute_window_cost, is_budget_exceeded
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
)
from pricing import (
    get_pricing_with_fallback,
    load_pricing_cache,
    pricing_cache_path,
)
from recovery import determine_recovery_action, RecoveryAction
from report_formatter import CostSummary, ReportData, format_report
from response_parser import Action, parse_response
from service_registry import ServiceRegistry
from shell_executor import ShellResult, execute_shell
from sub_agent_runner import SubAgentRunner
from task_queue import TaskQueue
from vision_loader import VisionLoader

logger = logging.getLogger(__name__)


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

        # Set externally after construction
        self.llm_client: LLMClient | None = None
        self.slack: "SlackBot | None" = None  # noqa: F821 — forward ref

        # Conversation memory (Req 1.1, 1.5)
        self.conversation_memory = ConversationMemory(base_dir, company_id)

        # Autonomous growth components
        self.vision_loader = VisionLoader(base_dir, company_id)
        self.task_queue = TaskQueue(base_dir, company_id)
        self.agent_registry = AgentRegistry(base_dir, company_id)
        self.service_registry = ServiceRegistry(base_dir, company_id)
        self.sub_agent_runner = SubAgentRunner(self)
        self.autonomous_loop = AutonomousLoop(self)

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

        data = ReportData(
            timestamp=now,
            company_id=self.company_id,
            wip=list(self.state.wip),
            cost=cost_summary,
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
    # Message processing — Think → Act → Report (Req 3.1–3.4, 4.1–4.6)
    # ------------------------------------------------------------------

    _MAX_WIP = 3
    _MAX_ACTION_LOOP = 10

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

    def process_message(self, text: str, user_id: str) -> None:
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

        try:
            # 1. 予算チェック
            if self.check_budget():
                logger.warning("Budget exceeded, rejecting message")
                self._slack_send("予算上限に達したため処理できません")
                return

            if self.llm_client is None:
                logger.error("LLM client not configured")
                self._slack_send("エラー: LLMクライアントが設定されていません")
                return

            # 2. コンテキスト構築
            now = datetime.now(timezone.utc)

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

            spent = compute_window_cost(self.state.ledger_events, now)
            limit = DEFAULT_BUDGET_LIMIT_USD
            if self.state.constitution and self.state.constitution.budget:
                limit = self.state.constitution.budget.limit_usd

            recent_decisions = self.state.decision_log[-5:]

            # Load recent conversation history (Req 1.2)
            try:
                conversation_history = self.conversation_memory.recent()
            except Exception:
                logger.warning("Failed to load conversation history", exc_info=True)
                conversation_history = None

            # Load vision text (Req 2.1)
            try:
                vision_text = self.vision_loader.load()
            except Exception:
                logger.warning("Failed to load vision", exc_info=True)
                vision_text = None

            system_prompt = build_system_prompt(
                constitution=self.state.constitution,
                wip=self.state.wip,
                recent_decisions=recent_decisions,
                budget_spent=spent,
                budget_limit=limit,
                conversation_history=conversation_history,
                vision_text=vision_text,
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

        except Exception:
            logger.exception("Unexpected error in process_message")
            self._slack_send("エラー: メッセージ処理中に予期しないエラーが発生しました")

    def _execute_action_loop(
        self,
        actions: list[Action],
        conversation: list[dict[str, str]],
        task_id: str,
    ) -> None:
        """アクションを順次実行し、必要に応じてLLMに再問い合わせする."""
        iterations = 0
        work_dir = self.base_dir / "companies" / self.company_id

        while actions and iterations < self._MAX_ACTION_LOOP:
            iterations += 1
            next_actions: list[Action] = []

            for action in actions:
                if action.action_type == "reply":
                    self._slack_send(action.content)

                elif action.action_type == "done":
                    self._slack_send(f"完了: {action.content}")

                elif action.action_type == "shell_command":
                    logger.info("Executing shell: %s", action.content)
                    shell_result = execute_shell(
                        command=action.content,
                        cwd=work_dir,
                    )

                    # Record shell_exec event in ledger
                    now = datetime.now(timezone.utc)
                    shell_event = LedgerEvent(
                        timestamp=now,
                        event_type="shell_exec",
                        agent_id="manager",
                        task_id=task_id,
                        estimated_cost_usd=0,
                        metadata={
                            "command": shell_result.command,
                            "return_code": shell_result.return_code,
                            "duration_seconds": shell_result.duration_seconds,
                        },
                    )
                    append_ledger_event(self.base_dir, self.company_id, shell_event)
                    self.state.ledger_events.append(shell_event)

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

                    next_actions = parse_response(llm_result.content)
                    break  # Process new actions in next iteration

            # If no shell_command triggered a new LLM call, we're done
            if not next_actions:
                break
            actions = next_actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slack_send(self, text: str) -> None:
        """Send a message via Slack if the bot is configured."""
        if self.slack is not None:
            self.slack.send_message(text)
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
