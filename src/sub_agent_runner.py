"""Sub-Agent Runner — サブエージェントの生成と実行を担当する.

CEO_AIが特定タスク用にスポーンする軽量エージェントを管理する。
同一プロセス内で同期的にLLM会話ループを実行し、結果を返す。

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from cost_aggregator import compute_window_cost
from llm_client import LLMClient, LLMError
from response_parser import parse_response
from shell_executor import execute_shell

if TYPE_CHECKING:
    from datetime import datetime
    from manager import Manager

logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 10
DEFAULT_WIP_LIMIT = 3


class SubAgentRunner:
    """サブエージェントの生成と実行を担当する."""

    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def spawn(
        self,
        name: str,
        role: str,
        task_description: str,
        budget_limit_usd: float = 1.0,
        model: str | None = None,
    ) -> str:
        """サブエージェントをスポーンし、タスクを実行する.

        Args:
            name: エージェント名
            role: エージェントの役割
            task_description: タスクの説明
            budget_limit_usd: 予算上限（USD）
            model: 使用するLLMモデル名。未指定/空文字列の場合はCEO_AIのモデルにフォールバック。

        Returns:
            結果文字列（完了メッセージまたはエラー）
        """
        # WIP limit check: count active agents excluding CEO
        active_agents = self.manager.agent_registry.list_active()
        non_ceo_active = [a for a in active_agents if a.agent_id != "ceo"]
        if len(non_ceo_active) >= DEFAULT_WIP_LIMIT:
            msg = f"WIP制限({DEFAULT_WIP_LIMIT})に達しているためスポーンできません"
            logger.warning(msg)
            return msg

        # Determine effective model (empty string treated as None)
        effective_model = model or (
            self.manager.llm_client.model if self.manager.llm_client else "unknown"
        )

        # Generate agent_id
        agent_id = f"sub-{uuid4().hex[:6]}"

        # Create independent LLMClient for this sub-agent
        sub_client = self._create_llm_client(effective_model)

        # Register in AgentRegistry with effective model name
        self.manager.agent_registry.register(
            agent_id=agent_id,
            name=name,
            role=role,
            model=effective_model,
            budget_limit_usd=budget_limit_usd,
        )

        # Build prompt and run conversation
        system_prompt = self._build_sub_agent_prompt(role, task_description)
        try:
            result = self._run_conversation(
                agent_id=agent_id,
                system_prompt=system_prompt,
                task_description=task_description,
                budget_limit_usd=budget_limit_usd,
                llm_client=sub_client,
            )
        except Exception as exc:
            logger.exception("Sub-agent %s failed: %s", agent_id, exc)
            self.manager.agent_registry.update_status(agent_id, "inactive")
            result = f"エラー: {exc}"

        # Mark agent as inactive after completion
        try:
            self.manager.agent_registry.update_status(agent_id, "inactive")
        except Exception:
            logger.warning("Failed to deactivate agent %s", agent_id, exc_info=True)

        return result

    def _create_llm_client(self, model: str) -> LLMClient | None:
        """指定モデルで独立LLMClientを生成する.

        Args:
            model: 使用するモデル名

        Returns:
            新しいLLMClientインスタンス。manager.llm_clientがNoneの場合はNone。
        """
        if self.manager.llm_client is None:
            return None
        return LLMClient(
            api_key=self.manager.llm_client.api_key,
            model=model,
            timeout=self.manager.llm_client.timeout,
        )

    def _build_sub_agent_prompt(self, role: str, task_description: str) -> str:
        """サブエージェント用のシステムプロンプトを構築する."""
        return (
            f"あなたはAI会社のサブエージェントです。\n"
            f"役割: {role}\n"
            f"タスク: {task_description}\n\n"
            f"以下のタグを使って応答してください:\n"
            f"<shell>実行するシェルコマンド</shell>\n"
            f"<reply>報告テキスト</reply>\n"
            f"<done>完了メッセージ</done>\n\n"
            f"タスクが完了したら必ず<done>タグで結果を報告してください。"
        )

    def _run_conversation(
        self,
        agent_id: str,
        system_prompt: str,
        task_description: str,
        budget_limit_usd: float,
        llm_client: LLMClient | None = None,
    ) -> str:
        """LLM会話ループを実行し、結果を返す."""
        if llm_client is None:
            return "エラー: LLMクライアントが設定されていません"

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]

        work_dir = self.manager.base_dir / "companies" / self.manager.company_id

        for turn in range(MAX_CONVERSATION_TURNS):
            # Budget check
            if self._is_budget_exceeded(agent_id, budget_limit_usd):
                logger.warning("Budget exceeded for agent %s", agent_id)
                self.manager.agent_registry.update_status(agent_id, "inactive")
                return f"予算上限(${budget_limit_usd})に達したため停止しました"

            # LLM call
            llm_result = llm_client.chat(conversation)

            if isinstance(llm_result, LLMError):
                logger.error("LLM call failed for agent %s: %s", agent_id, llm_result.message)
                return f"LLMエラー: {llm_result.message}"

            # Record cost
            self.manager.record_llm_call(
                provider="openrouter",
                model=llm_result.model,
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                agent_id=agent_id,
            )

            conversation.append({"role": "assistant", "content": llm_result.content})

            # Parse response
            actions = parse_response(llm_result.content)

            done_result = None
            needs_followup = False

            for action in actions:
                if action.action_type == "done":
                    done_result = action.content
                elif action.action_type == "shell_command":
                    shell_result = execute_shell(command=action.content, cwd=work_dir)
                    result_text = (
                        f"コマンド実行結果 (return_code={shell_result.return_code}):\n"
                    )
                    if shell_result.stdout:
                        result_text += f"stdout:\n{shell_result.stdout}\n"
                    if shell_result.stderr:
                        result_text += f"stderr:\n{shell_result.stderr}\n"
                    conversation.append({"role": "user", "content": result_text})
                    needs_followup = True
                    break  # Process shell result in next turn

            if done_result is not None:
                return done_result

            if not needs_followup:
                # No shell command and no done → treat last reply as result
                reply_actions = [a for a in actions if a.action_type == "reply"]
                if reply_actions:
                    return reply_actions[-1].content
                return llm_result.content

        return "最大会話ターン数に達したため停止しました"

    def _is_budget_exceeded(self, agent_id: str, budget_limit_usd: float) -> bool:
        """指定エージェントのコストが予算上限を超えているか確認する."""
        agent_events = [
            e for e in self.manager.state.ledger_events
            if e.agent_id == agent_id and e.estimated_cost_usd is not None
        ]
        total = sum(e.estimated_cost_usd for e in agent_events)
        return total >= budget_limit_usd
