"""Autonomous Loop — メインループ内で自律タスク実行を制御する.

tick() はハートビートサイクルから呼ばれ、WIP/予算チェック後に
pendingタスクを1つ選択して実行する。pendingがなければLLMに提案を依頼する。

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from llm_client import LLMError
from models import TaskEntry
from response_parser import parse_response
from shell_executor import execute_shell

if TYPE_CHECKING:
    from manager import Manager

logger = logging.getLogger(__name__)

DEFAULT_WIP_LIMIT = 3
MAX_TASK_TURNS = 50


class AutonomousLoop:
    """メインループ内で自律タスク実行を制御する."""

    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def tick(self) -> None:
        """1サイクル分の自律実行を行う.

        1. WIPに空きがあるか確認
        2. 予算に余裕があるか確認
        3. pendingタスクを選択（なければLLMに提案を依頼）
        4. タスクを実行
        5. 結果を報告
        """
        try:
            # 1. WIP check
            running = self.manager.task_queue.list_by_status("running")
            wip_limit = self._get_wip_limit()
            if len(running) >= wip_limit:
                logger.info("WIP full (%d/%d), skipping tick", len(running), wip_limit)
                return

            # 2. Budget check
            if self.manager.check_budget():
                logger.info("Budget exceeded, skipping tick")
                return

            # 3. Pick task
            task = self._pick_task()
            if task is None:
                # No pending tasks — ask LLM to propose new ones
                proposed = self._propose_tasks()
                if not proposed:
                    logger.info("No tasks proposed, skipping tick")
                    return
                task = self._pick_task()
                if task is None:
                    return

            # 4. Execute task
            self._execute_task(task)

        except Exception:
            logger.exception("Error in autonomous loop tick")

    def _pick_task(self) -> TaskEntry | None:
        """次に実行するタスクを選択する."""
        return self.manager.task_queue.next_pending()

    def _propose_tasks(self) -> list[TaskEntry]:
        """LLMに新しいタスクの提案を依頼する."""
        if self.manager.llm_client is None:
            logger.warning("LLM client not configured, cannot propose tasks")
            return []

        try:
            vision_text = self.manager.vision_loader.load()
        except Exception:
            logger.warning("Failed to load vision", exc_info=True)
            vision_text = ""

        # Creator score policy (optional)
        purpose = ""
        policy_text = ""
        try:
            if self.manager.state.constitution:
                purpose = self.manager.state.constitution.purpose
                pol = getattr(self.manager.state.constitution, "creator_score_policy", None)
                if pol and getattr(pol, "enabled", False):
                    policy_text = (
                        "評価はCreatorスコア(0-100)を最重要KPIとする。"
                        f"優先は「{pol.priority}」。"
                        "各軸は 面白さ/コスト効率/現実性/進化性（各0-25）。"
                    )
        except Exception:
            pass

        latest_review = ""
        try:
            r = self.manager.creator_review_store.latest()
            if r:
                latest_review = f"直近レビュー: {r.score_total_100}/100 コメント: {r.comment}"
        except Exception:
            pass

        # Gather task history for context
        history_parts: list[str] = []
        pending_ids: list[str] = []
        try:
            completed = self.manager.task_queue.list_by_status("completed")
            completed.sort(key=lambda t: t.updated_at, reverse=True)
            for t in completed[:5]:
                result_short = (t.result or "")[:100]
                history_parts.append(f"  完了: {t.description} → {result_short}")
        except Exception:
            pass
        try:
            failed = self.manager.task_queue.list_by_status("failed")
            failed.sort(key=lambda t: t.updated_at, reverse=True)
            for t in failed[:3]:
                error_short = (t.error or "不明")[:100]
                history_parts.append(f"  失敗: {t.description} — {error_short}")
        except Exception:
            pass
        try:
            pending = self.manager.task_queue.list_by_status("pending")
            for t in pending:
                pending_ids.append(f"  [{t.task_id}] {t.description}")
        except Exception:
            pass

        history_text = "\n".join(history_parts) if history_parts else "なし"
        pending_text = "\n".join(pending_ids) if pending_ids else "なし"

        prompt = (
            "あなたはAI会社の社長AIです。\n"
            f"目的: {purpose}\n"
            f"{policy_text}\n"
            f"{latest_review}\n"
            f"ビジョン:\n{vision_text}\n\n"
            f"最近のタスク履歴:\n{history_text}\n\n"
            f"既存のpendingタスク:\n{pending_text}\n\n"
            "現在pendingのタスクがありません。\n"
            "会社のビジョンと評価方針に基づいて、次に取り組むべき施策（タスク）を1〜3個提案してください。\n"
            "各施策は1行で簡潔に。可能なら「最初の一手」と「想定スコア(面白さ/コスト効率/現実性/進化性)」を添えてください。\n"
            "既存タスクに依存する場合は depends_on:task_id1,task_id2 を末尾に追加してください。\n"
            "フォーマット:\n"
            "- 施策1の説明 | 最初の一手: ... | 想定: 面白さa/25 コスト効率b/25 現実性c/25 進化性d/25\n"
            "- 施策2の説明 | depends_on:task_id1,task_id2\n"
        )

        messages = [
            {"role": "system", "content": "タスク提案アシスタント"},
            {"role": "user", "content": prompt},
        ]

        try:
            result = self.manager.llm_client.chat(messages)
        except Exception:
            logger.exception("LLM call failed during task proposal")
            return []

        if isinstance(result, LLMError):
            logger.error("LLM error during task proposal: %s", result.message)
            return []

        # Record LLM cost
        try:
            self.manager.record_llm_call(
                provider="openrouter",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                task_id="propose-tasks",
            )
        except Exception:
            logger.warning("Failed to record LLM call", exc_info=True)

        # Parse response to extract task descriptions
        tasks: list[TaskEntry] = []
        for line in result.content.splitlines():
            line = line.strip()
            # Match lines starting with "- " or "* " or numbered "1. "
            match = re.match(r"^[-*]\s+(.+)$|^\d+\.\s+(.+)$", line)
            if match:
                desc = (match.group(1) or match.group(2)).strip()
                # If the model included metadata (e.g. " | 最初の一手: ..."), keep only the core description.
                desc = desc.split("|", 1)[0].strip()
                if not desc:
                    continue

                # Parse depends_on:id1,id2 from the full line
                deps: list[str] = []
                dep_match = re.search(r"depends_on:\s*([\w,]+)", line)
                if dep_match:
                    deps = [d.strip() for d in dep_match.group(1).split(",") if d.strip()]

                try:
                    if deps:
                        entry = self.manager.task_queue.add_with_deps(desc, depends_on=deps)
                    else:
                        entry = self.manager.task_queue.add(desc)
                    tasks.append(entry)
                except Exception:
                    logger.warning("Failed to add proposed task: %s", desc, exc_info=True)

        logger.info("Proposed %d new tasks", len(tasks))
        return tasks

    def _execute_task(self, task: TaskEntry) -> None:
        """タスクをLLMに渡して実行する."""
        if self.manager.llm_client is None:
            logger.warning("LLM client not configured, cannot execute task")
            return

        # Update status to running
        try:
            self.manager.task_queue.update_status(task.task_id, "running")
        except Exception:
            logger.exception("Failed to update task status to running")
            return

        work_dir = self.manager.base_dir / "companies" / self.manager.company_id

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "あなたはAI会社の社長AIです。タスクを実行してください。\n"
                    "シェルコマンドが必要な場合は<shell>コマンド</shell>で指示してください。\n"
                    "Creatorに相談が必要な場合は<consult>相談内容</consult>で送ってください。\n"
                    "完了したら<done>結果の要約</done>で報告してください。"
                ),
            },
            {"role": "user", "content": f"タスク: {task.description}"},
        ]

        try:
            had_shell = False
            for _turn in range(MAX_TASK_TURNS):
                # Budget check each turn
                if self.manager.check_budget():
                    self.manager.task_queue.update_status(
                        task.task_id, "failed", error="予算超過"
                    )
                    self._report(f"タスク中断(予算超過): {task.description}")
                    return

                result = self.manager.llm_client.chat(messages)

                if isinstance(result, LLMError):
                    self.manager.task_queue.update_status(
                        task.task_id, "failed", error=result.message
                    )
                    self._report(f"タスク失敗(LLMエラー): {task.description}")
                    return

                # Record cost
                try:
                    self.manager.record_llm_call(
                        provider="openrouter",
                        model=result.model,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        task_id=task.task_id,
                    )
                except Exception:
                    logger.warning("Failed to record LLM call", exc_info=True)

                messages.append({"role": "assistant", "content": result.content})
                actions = parse_response(result.content)

                done_result = None
                needs_followup = False

                for action in actions:
                    if action.action_type == "done":
                        done_result = action.content
                    elif action.action_type == "reply":
                        self._report(action.content)
                    elif action.action_type == "consult":
                        try:
                            entry = self.manager.consultation_store.add(
                                action.content.strip(),
                                related_task_id=task.task_id,
                            )
                            self._report(
                                f"🤝 相談 [consult_id: {entry.consultation_id}]\n\n{action.content.strip()}"
                            )
                        except Exception:
                            self._report(f"🤝 相談\n\n{action.content.strip()}")
                        self.manager.task_queue.update_status(
                            task.task_id, "failed", error="相談待ち"
                        )
                        return
                    elif action.action_type == "shell_command":
                        had_shell = True
                        shell_result = execute_shell(command=action.content, cwd=work_dir)
                        result_text = (
                            f"コマンド実行結果 (return_code={shell_result.return_code}):\n"
                        )
                        if shell_result.stdout:
                            result_text += f"stdout:\n{shell_result.stdout}\n"
                        if shell_result.stderr:
                            result_text += f"stderr:\n{shell_result.stderr}\n"
                        messages.append({"role": "user", "content": result_text})
                        needs_followup = True
                        break
                    elif action.action_type == "delegate":
                        content = action.content.strip()
                        role, _, desc = content.partition(":")
                        role = role.strip() or "worker"
                        desc = desc.strip() or content
                        try:
                            sub_result = self.manager.sub_agent_runner.spawn(
                                name=role,
                                role=role,
                                task_description=desc,
                            )
                            result_text = f"サブエージェント結果 (role={role}):\n{sub_result}"
                        except Exception as exc:
                            logger.warning("Sub-agent spawn failed: %s", exc, exc_info=True)
                            result_text = f"サブエージェントエラー (role={role}): {exc}"
                        messages.append({"role": "user", "content": result_text})
                        needs_followup = True
                        break

                if done_result is not None:
                    # Quality gate: verify output if shell commands were used
                    q_score, q_notes = None, None
                    enable_quality_gate = os.environ.get("TASK_QUALITY_GATE", "0") == "1"
                    if had_shell and enable_quality_gate:
                        try:
                            q_score, q_notes = self._verify_task_output(task, messages)
                        except Exception:
                            logger.warning("Quality verification failed", exc_info=True)

                    if q_score is not None and q_score < 0.5:
                        self.manager.task_queue.update_status(
                            task.task_id, "failed",
                            error=f"品質不足 (score={q_score:.2f}): {q_notes}",
                            quality_score=q_score,
                            quality_notes=q_notes,
                        )
                        self._report(
                            f"タスク品質不足: {task.description}\n"
                            f"スコア: {q_score:.2f} — {q_notes}"
                        )
                    else:
                        self.manager.task_queue.update_status(
                            task.task_id, "completed", result=done_result,
                            quality_score=q_score,
                            quality_notes=q_notes,
                        )
                        self._report(f"タスク完了: {task.description}\n結果: {done_result}")
                    return

                if not needs_followup:
                    # No shell and no done — treat as completed
                    self.manager.task_queue.update_status(
                        task.task_id, "completed", result=result.content
                    )
                    self._report(f"タスク完了: {task.description}")
                    return

            # Max turns reached
            self.manager.task_queue.update_status(
                task.task_id, "failed", error="最大ターン数到達"
            )
            self._report(f"タスク中断(最大ターン数): {task.description}")

        except Exception as exc:
            logger.exception("Error executing task %s", task.task_id)
            try:
                self.manager.task_queue.update_status(
                    task.task_id, "failed", error=str(exc)
                )
            except Exception:
                logger.warning("Failed to update task status to failed", exc_info=True)
            self._report(f"タスク失敗(エラー): {task.description}")

    def _get_wip_limit(self) -> int:
        """WIP制限を取得する."""
        try:
            constitution = self.manager.state.constitution
            if constitution and constitution.work_principles:
                return constitution.work_principles.wip_limit
        except Exception:
            pass
        return DEFAULT_WIP_LIMIT
    def _verify_task_output(
        self, task: TaskEntry, conversation: list[dict[str, str]]
    ) -> tuple[float, str]:
        """LLMにタスク成果物の品質を評価させる.

        Returns (score, notes). On LLM failure returns (1.0, "verification skipped").
        """
        if self.manager.llm_client is None:
            return 1.0, "verification skipped: no LLM client"

        # Build a compact summary of the conversation for review
        summary_parts: list[str] = []
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "…"
            summary_parts.append(f"[{role}] {content}")
        summary = "\n".join(summary_parts[-10:])  # last 10 messages

        review_prompt = (
            "以下のタスク実行ログを確認し、品質を評価してください。\n"
            f"タスク: {task.description}\n\n"
            f"実行ログ:\n{summary}\n\n"
            "以下のフォーマットで回答してください:\n"
            "score: 0.0〜1.0の数値（1.0が最高品質）\n"
            "notes: 評価コメント（1行）\n"
        )

        messages = [
            {"role": "system", "content": "タスク品質評価アシスタント。簡潔に評価してください。"},
            {"role": "user", "content": review_prompt},
        ]

        try:
            result = self.manager.llm_client.chat(messages)
        except Exception:
            logger.warning("Quality verification LLM call failed", exc_info=True)
            return 1.0, "verification skipped: LLM call exception"

        if isinstance(result, LLMError):
            logger.warning("Quality verification LLM error: %s", result.message)
            return 1.0, "verification skipped: LLM error"

        # Record cost
        try:
            self.manager.record_llm_call(
                provider="openrouter",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                task_id=task.task_id,
            )
        except Exception:
            logger.warning("Failed to record quality verification LLM call", exc_info=True)

        # Parse score and notes from response
        import re as _re
        score = 1.0
        notes = result.content.strip()

        score_match = _re.search(r"score:\s*([\d.]+)", result.content, _re.IGNORECASE)
        if score_match:
            try:
                parsed = float(score_match.group(1))
                if 0.0 <= parsed <= 1.0:
                    score = parsed
            except ValueError:
                pass

        notes_match = _re.search(r"notes:\s*(.+)", result.content, _re.IGNORECASE)
        if notes_match:
            notes = notes_match.group(1).strip()

        return score, notes

    def _report(self, message: str) -> None:
        """Slackに結果を報告する."""
        try:
            self.manager._slack_send(message)
        except Exception:
            logger.warning("Failed to send report: %s", message, exc_info=True)
