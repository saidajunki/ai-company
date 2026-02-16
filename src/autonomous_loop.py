"""Autonomous Loop — メインループ内で自律タスク実行を制御する.

tick() はハートビートサイクルから呼ばれ、WIP/予算チェック後に
pendingタスクを1つ選択して実行する。pendingがなければLLMに提案を依頼する。

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import logging
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

        prompt = (
            "あなたはAI会社の社長AIです。\n"
            f"ビジョン:\n{vision_text}\n\n"
            "現在pendingのタスクがありません。\n"
            "会社のビジョンに基づいて、次に取り組むべきタスクを1〜3個提案してください。\n"
            "各タスクは1行で簡潔に記述してください。\n"
            "フォーマット:\n"
            "- タスク1の説明\n"
            "- タスク2の説明\n"
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
                if desc:
                    try:
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
                    "完了したら<done>結果の要約</done>で報告してください。"
                ),
            },
            {"role": "user", "content": f"タスク: {task.description}"},
        ]

        try:
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
                    elif action.action_type == "shell_command":
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

                if done_result is not None:
                    self.manager.task_queue.update_status(
                        task.task_id, "completed", result=done_result
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

    def _report(self, message: str) -> None:
        """Slackに結果を報告する."""
        try:
            self.manager._slack_send(message)
        except Exception:
            logger.warning("Failed to send report: %s", message, exc_info=True)
