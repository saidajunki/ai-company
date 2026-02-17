"""Sub-Agent Runner — サブエージェントの生成と実行を担当する.

CEO_AIが特定タスク用にスポーンする軽量エージェントを管理する。
同一プロセス内で同期的にLLM会話ループを実行し、結果を返す。

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from cost_aggregator import compute_window_cost
from llm_client import LLMClient, LLMError
from model_catalog import select_model_for_role
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
        # 1. Explicit model= from delegate tag takes priority
        # 2. Role-based auto-selection from model_catalog
        # 3. Fallback to CEO model
        ceo_model = self.manager.llm_client.model if self.manager.llm_client else "unknown"
        if model:
            effective_model = model
        else:
            effective_model = select_model_for_role(role, fallback_model=ceo_model)

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
        purpose = None
        try:
            if self.manager.state.constitution is not None:
                purpose = self.manager.state.constitution.purpose
        except Exception:
            purpose = None

        vision_text = None
        try:
            vision_text = self.manager.vision_loader.load()
        except Exception:
            vision_text = None

        rolling_summary_text = None
        recalled_memories = None
        try:
            mm = getattr(self.manager, "memory_manager", None)
            if mm is not None:
                mm.ingest_all_sources()
                rolling_summary_text = mm.summary_for_prompt()
                recalled_memories = mm.recall_for_prompt(task_description, limit=6)
        except Exception:
            logger.warning("Failed to build memory context for sub-agent", exc_info=True)

        system_prompt = self._build_sub_agent_prompt(
            role,
            task_description,
            purpose=purpose,
            vision_text=vision_text,
            rolling_summary_text=rolling_summary_text,
            recalled_memories=recalled_memories,
        )
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

        # Persist sub-agent outcome into long-term summary/journal (best-effort)
        try:
            mm = getattr(self.manager, "memory_manager", None)
            if mm is not None:
                mm.note_interaction(
                    timestamp=datetime.now(timezone.utc),
                    user_id=agent_id,
                    request_text=f"[sub-agent:{role}] {task_description}",
                    response_text=result,
                    snapshot_lines=[f"sub-agent: {name}", f"model: {effective_model}"],
                )
        except Exception:
            logger.warning("Failed to persist sub-agent outcome", exc_info=True)

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

    def _build_sub_agent_prompt(
        self,
        role: str,
        task_description: str,
        *,
        purpose: str | None = None,
        vision_text: str | None = None,
        rolling_summary_text: str | None = None,
        recalled_memories: list[str] | None = None,
    ) -> str:
        """サブエージェント用のシステムプロンプトを構築する."""
        company_root = self.manager.base_dir / "companies" / self.manager.company_id
        purpose_text = (purpose or "").strip() or "未設定"
        vision = (vision_text or "").strip() or "未設定"

        lines: list[str] = [
            "あなたはAI会社のサブエージェントです。",
            f"役割: {role}",
            f"タスク: {task_description}",
            "",
            "## 会社の目的",
            purpose_text,
            "",
            "## ビジョン",
            vision,
            "",
            "## 社内ナレッジ参照先",
            f"- {company_root / 'constitution.yaml'}",
            f"- {company_root / 'vision.md'}",
            f"- {company_root / 'state' / 'rolling_summary.md'}",
            f"- {company_root / 'state' / 'memory.sqlite3'}",
            f"- {company_root / 'journal'}",
            "必要に応じて<shell>でファイル参照して構いません。",
        ]

        if rolling_summary_text and rolling_summary_text.strip():
            lines.extend(["", rolling_summary_text.strip()])

        if recalled_memories is not None:
            lines.append("")
            lines.append("## 長期記憶（リコール）")
            if recalled_memories:
                lines.extend(recalled_memories[:10])
            else:
                lines.append("リコールなし")

        lines.extend([
            "",
            "以下のタグを使って応答してください:",
            "<shell>実行するシェルコマンド</shell>",
            "<research>Web検索クエリ</research>",
            "<reply>報告テキスト</reply>",
            "<done>完了メッセージ</done>",
            "",
            "タスクが完了したら必ず<done>タグで結果を報告してください。",
        ])

        return "\n".join(lines)

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
                elif action.action_type == "research":
                    query = action.content.strip()
                    if not query:
                        continue
                    try:
                        search_results = self.manager.web_searcher.search(query)
                    except Exception:
                        logger.warning(
                            "Sub-agent research failed (agent_id=%s, query=%s)",
                            agent_id,
                            query,
                            exc_info=True,
                        )
                        search_results = []

                    now = datetime.now(timezone.utc)
                    for sr in search_results:
                        try:
                            self.manager.research_note_store.save(ResearchNote(
                                query=query,
                                source_url=sr.url,
                                title=sr.title,
                                snippet=sr.snippet,
                                summary=sr.snippet,
                                retrieved_at=now,
                            ))
                        except Exception:
                            logger.warning(
                                "Failed to save research note from sub-agent",
                                exc_info=True,
                            )

                    if search_results:
                        parts = [f"リサーチ結果 (query={query}):"]
                        for i, sr in enumerate(search_results, 1):
                            parts.append(f"{i}. {sr.title}\n   {sr.url}\n   {sr.snippet}")
                        result_text = "\n".join(parts)
                    else:
                        result_text = f"リサーチ結果 (query={query}): 検索結果なし"

                    conversation.append({"role": "user", "content": result_text})
                    needs_followup = True
                    break

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
