"""Autonomous Loop — メインループ内で自律タスク実行を制御する.

tick() はハートビートサイクルから呼ばれ、WIP/予算チェック後に
pendingタスクを1つ選択して実行する。pendingがなければLLMに提案を依頼する。

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from artifact_verifier import ArtifactVerifier
from context_builder import TaskHistoryContext, _build_task_history_section
from llm_client import LLMError
from models import TaskEntry
from priority_classifier import PriorityClassifier
from response_parser import parse_response
from shell_command_tracker import ShellCommandTracker
from shell_executor import execute_shell

if TYPE_CHECKING:
    from manager import Manager

logger = logging.getLogger(__name__)

DEFAULT_WIP_LIMIT = 3
MAX_TASK_TURNS = 50
# runningタスクがこの秒数以上updated_atから経過したらstuckとみなす
STUCK_TASK_TIMEOUT_SECONDS = 1800  # 30分


class AutonomousLoop:
    """メインループ内で自律タスク実行を制御する."""

    def __init__(self, manager: Manager) -> None:
        self.manager = manager

    def tick(self) -> None:
        """1サイクル分の自律実行を行う.

        1. stuckしたrunningタスクを検出してfailedにする
        2. WIPに空きがあるか確認
        3. 予算に余裕があるか確認
        4. pendingタスクを選択（なければLLMに提案を依頼）
        5. タスクを実行
        6. 結果を報告
        """
        try:
            # 0a. Reap stuck running tasks
            self._reap_stuck_tasks()

            # 0b. Retry failed tasks
            self._retry_failed_tasks()

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
        """LLMに新しいタスクの提案を依頼する.

        まず InitiativePlanner による計画を試み、失敗または空の場合は
        既存のLLMベースのタスク提案にフォールバックする。
        """
        # --- Initiative-based planning (preferred) ---
        initiative_planner = getattr(self.manager, "initiative_planner", None)
        if initiative_planner is not None:
            try:
                initiatives = initiative_planner.plan()
                if initiatives:
                    tasks: list[TaskEntry] = []
                    for ini in initiatives:
                        for tid in ini.task_ids:
                            task = self.manager.task_queue._get_latest(tid)
                            if task:
                                tasks.append(task)
                    if tasks:
                        logger.info(
                            "Initiative planner proposed %d tasks from %d initiatives",
                            len(tasks),
                            len(initiatives),
                        )
                        return tasks
                    logger.info("Initiative planner returned initiatives but no tasks found in queue")
            except Exception:
                logger.warning(
                    "Initiative planner failed, falling back to LLM proposal",
                    exc_info=True,
                )

        # --- Fallback: LLM-based task proposal ---
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

        # Long-term memory context (best-effort)
        rolling_summary_text = None
        recalled_memories = None
        try:
            mm = getattr(self.manager, "memory_manager", None)
            if mm is not None:
                mm.ingest_all_sources()
                rolling_summary_text = mm.summary_for_prompt()
                recalled_memories = mm.recall_for_prompt(
                    f"タスク提案 {purpose}\n{vision_text}\n{latest_review}",
                    limit=6,
                )
        except Exception:
            logger.warning("Failed to build memory context for task proposal", exc_info=True)

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
        if rolling_summary_text:
            prompt += "\n\n" + rolling_summary_text
        if recalled_memories is not None:
            prompt += "\n\n## 長期記憶（リコール）\n"
            prompt += "\n".join(recalled_memories) if recalled_memories else "リコールなし"

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
        tasks = []
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
                    priority = PriorityClassifier.classify(desc, "autonomous")
                    if deps:
                        entry = self.manager.task_queue.add_with_deps(desc, depends_on=deps, priority=priority, source="autonomous")
                    else:
                        entry = self.manager.task_queue.add(desc, priority=priority, source="autonomous")
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

        # Build task history context (Requirements 5.1, 5.2)
        try:
            task_history = TaskHistoryContext(
                completed=self.manager.task_queue.list_by_status("completed")[-10:],
                failed=self.manager.task_queue.list_by_status("failed")[-5:],
                running=self.manager.task_queue.list_by_status("running"),
            )
            task_history_text = _build_task_history_section(task_history)
        except Exception:
            logger.warning("Failed to build task history context", exc_info=True)
            task_history_text = ""

        # Long-term memory context (best-effort)
        rolling_summary_text = None
        recalled_memories = None
        try:
            mm = getattr(self.manager, "memory_manager", None)
            if mm is not None:
                mm.ingest_all_sources()
                rolling_summary_text = mm.summary_for_prompt()
                recalled_memories = mm.recall_for_prompt(task.description, limit=6)
        except Exception:
            logger.warning("Failed to build memory context for task execution", exc_info=True)

        system_content = (
            "あなたはAI会社の社長AIです。タスクを実行してください。\n"
            "シェルコマンドが必要な場合は<shell>コマンド</shell>で指示してください。\n"
            "Creatorに相談が必要な場合は<consult>相談内容</consult>で送ってください。\n"
            "社員エージェントに委任する場合は<delegate>role:タスク説明 model=モデル名</delegate>で指示してください（model=は省略可）。\n"
            "完了したら<done>結果の要約</done>で報告してください。"
        )
        if task_history_text:
            system_content += "\n\n" + task_history_text
        if rolling_summary_text:
            system_content += "\n\n" + rolling_summary_text
        if recalled_memories is not None:
            system_content += "\n\n## 長期記憶（リコール）\n"
            system_content += "\n".join(recalled_memories) if recalled_memories else "リコールなし"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"タスク: {task.description}"},
        ]

        try:
            shell_tracker = ShellCommandTracker()
            for _turn in range(MAX_TASK_TURNS):
                # Budget check each turn
                if self.manager.check_budget():
                    self.manager.task_queue.update_status(
                        task.task_id, "failed", error="予算超過"
                    )
                    self._check_parent_completion(task)
                    self._report(f"タスク中断(予算超過): {task.description}")
                    return

                result = self.manager.llm_client.chat(messages)

                if isinstance(result, LLMError):
                    self.manager.task_queue.update_status(
                        task.task_id, "failed", error=result.message
                    )
                    self._check_parent_completion(task)
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
                        consult_text = action.content.strip()
                        try:
                            entry, created = self.manager.consultation_store.ensure_pending(
                                consult_text,
                                related_task_id=task.task_id,
                            )
                            if created:
                                self._report(
                                    f"🤝 相談 [consult_id: {entry.consultation_id}]\n\n{consult_text}"
                                )
                            else:
                                logger.info(
                                    "Consultation already pending (consult_id=%s, task_id=%s)",
                                    entry.consultation_id,
                                    task.task_id,
                                )
                        except Exception:
                            self._report(f"🤝 相談\n\n{consult_text}")
                        self.manager.task_queue.update_status(
                            task.task_id, "failed", error="相談待ち"
                        )
                        try:
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                mm.note_interaction(
                                    timestamp=datetime.now(timezone.utc),
                                    user_id="autonomous_loop",
                                    request_text=f"[task:{task.task_id}] {task.description}",
                                    response_text="FAILED: 相談待ち",
                                    snapshot_lines=[f"consult: {consult_text[:120]}"],
                                )
                                mm.ingest_all_sources()
                        except Exception:
                            logger.warning("Failed to persist task outcome", exc_info=True)
                        self._check_parent_completion(task)
                        return
                    elif action.action_type == "shell_command":
                        shell_result = execute_shell(command=action.content, cwd=work_dir)
                        shell_tracker.record(action.content, shell_result.return_code)
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
                                model=action.model,
                            )
                            result_text = f"サブエージェント結果 (role={role}):\n{sub_result}"
                        except Exception as exc:
                            logger.warning("Sub-agent spawn failed: %s", exc, exc_info=True)
                            result_text = f"サブエージェントエラー (role={role}): {exc}"
                        messages.append({"role": "user", "content": result_text})
                        needs_followup = True
                        break

                if done_result is not None:
                    # Step 1: Shell command all-failed check (Req 2.2, 2.3)
                    if shell_tracker.had_any_commands() and shell_tracker.all_failed():
                        failed_cmds = shell_tracker.failed_commands()
                        error_msg = "全シェルコマンドが失敗: " + "; ".join(
                            f"{r.command} (rc={r.return_code})" for r in failed_cmds
                        )
                        self.manager.task_queue.update_status(
                            task.task_id, "failed", error=error_msg
                        )
                        try:
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                mm.note_interaction(
                                    timestamp=datetime.now(timezone.utc),
                                    user_id="autonomous_loop",
                                    request_text=f"[task:{task.task_id}] {task.description}",
                                    response_text=f"FAILED: {error_msg}",
                                    snapshot_lines=["reason: all_shell_failed"],
                                )
                                mm.ingest_all_sources()
                        except Exception:
                            logger.warning("Failed to persist task outcome", exc_info=True)
                        self._check_parent_completion(task)
                        self._report(f"タスク失敗(全コマンド失敗): {task.description}\n{error_msg}")
                        return

                    # Step 2: Artifact verification (Req 3.1, 3.2, 3.3)
                    artifact_verifier = ArtifactVerifier(work_dir)
                    all_text = done_result + "\n" + "\n".join(
                        m.get("content", "") for m in messages
                    )
                    artifact_paths = artifact_verifier.extract_file_paths(all_text)
                    if artifact_paths:
                        artifact_result = artifact_verifier.verify(artifact_paths)
                        if not artifact_result.all_exist:
                            error_msg = "成果物が見つかりません: " + ", ".join(artifact_result.missing)
                            self.manager.task_queue.update_status(
                                task.task_id, "failed", error=error_msg
                            )
                            try:
                                mm = getattr(self.manager, "memory_manager", None)
                                if mm is not None:
                                    mm.note_interaction(
                                        timestamp=datetime.now(timezone.utc),
                                        user_id="autonomous_loop",
                                        request_text=f"[task:{task.task_id}] {task.description}",
                                        response_text=f"FAILED: {error_msg}",
                                        snapshot_lines=["reason: artifact_missing"],
                                    )
                                    mm.ingest_all_sources()
                            except Exception:
                                logger.warning("Failed to persist task outcome", exc_info=True)
                            self._check_parent_completion(task)
                            self._report(f"タスク失敗(成果物欠損): {task.description}\n{error_msg}")
                            return

                    # Step 3: Quality Gate - always active (Req 1.1, 5.1, 5.2)
                    q_score, q_notes = None, None
                    if shell_tracker.had_any_commands():
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
                        try:
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                mm.note_interaction(
                                    timestamp=datetime.now(timezone.utc),
                                    user_id="autonomous_loop",
                                    request_text=f"[task:{task.task_id}] {task.description}",
                                    response_text=f"FAILED(quality): {q_notes}",
                                    snapshot_lines=[f"quality_score: {q_score:.2f}"],
                                )
                                mm.ingest_all_sources()
                        except Exception:
                            logger.warning("Failed to persist task outcome", exc_info=True)
                        self._check_parent_completion(task)
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
                        try:
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                mm.note_interaction(
                                    timestamp=datetime.now(timezone.utc),
                                    user_id="autonomous_loop",
                                    request_text=f"[task:{task.task_id}] {task.description}",
                                    response_text=done_result,
                                    snapshot_lines=[
                                        f"quality_score: {q_score:.2f}" if q_score is not None else "quality_score: n/a",
                                    ],
                                )
                                mm.ingest_all_sources()
                        except Exception:
                            logger.warning("Failed to persist task outcome", exc_info=True)
                        self._report(f"タスク完了: {task.description}\n結果: {done_result}")
                        self._check_initiative_completion(task.task_id)
                        self._check_parent_completion(task)
                    return

                if not needs_followup:
                    # No shell and no done — treat as completed
                    self.manager.task_queue.update_status(
                        task.task_id, "completed", result=result.content
                    )
                    try:
                        mm = getattr(self.manager, "memory_manager", None)
                        if mm is not None:
                            mm.note_interaction(
                                timestamp=datetime.now(timezone.utc),
                                user_id="autonomous_loop",
                                request_text=f"[task:{task.task_id}] {task.description}",
                                response_text=result.content,
                                snapshot_lines=["done_tag: none"],
                            )
                            mm.ingest_all_sources()
                    except Exception:
                        logger.warning("Failed to persist task outcome", exc_info=True)
                    self._report(f"タスク完了: {task.description}")
                    self._check_initiative_completion(task.task_id)
                    self._check_parent_completion(task)
                    return

            # Max turns reached
            self.manager.task_queue.update_status(
                task.task_id, "failed", error="最大ターン数到達"
            )
            try:
                mm = getattr(self.manager, "memory_manager", None)
                if mm is not None:
                    mm.note_interaction(
                        timestamp=datetime.now(timezone.utc),
                        user_id="autonomous_loop",
                        request_text=f"[task:{task.task_id}] {task.description}",
                        response_text="FAILED: 最大ターン数到達",
                        snapshot_lines=["reason: max_turns"],
                    )
                    mm.ingest_all_sources()
            except Exception:
                logger.warning("Failed to persist task outcome", exc_info=True)
            self._check_parent_completion(task)
            self._report(f"タスク中断(最大ターン数): {task.description}")

        except Exception as exc:
            logger.exception("Error executing task %s", task.task_id)
            try:
                self.manager.task_queue.update_status(
                    task.task_id, "failed", error=str(exc)
                )
            except Exception:
                logger.warning("Failed to update task status to failed", exc_info=True)
            self._check_parent_completion(task)
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
    ) -> tuple[float | None, str]:
        """LLMにタスク成果物の品質を評価させる.

        Returns (score, notes). On LLM failure returns (None, "verification skipped: ...").
        """
        if self.manager.llm_client is None:
            return None, "verification skipped: no LLM client"

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
            return None, "verification skipped: LLM call exception"

        if isinstance(result, LLMError):
            logger.warning("Quality verification LLM error: %s", result.message)
            return None, "verification skipped: LLM error"

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

    def _reap_stuck_tasks(self) -> None:
        """updated_atから一定時間経過したrunningタスクをfailedにする."""
        now = datetime.now(timezone.utc)
        running = self.manager.task_queue.list_by_status("running")
        for task in running:
            updated = task.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            elapsed = (now - updated).total_seconds()
            if elapsed >= STUCK_TASK_TIMEOUT_SECONDS:
                logger.warning(
                    "Reaping stuck task %s (running for %ds): %s",
                    task.task_id,
                    int(elapsed),
                    task.description[:80],
                )
                try:
                    self.manager.task_queue.update_status(
                        task.task_id,
                        "failed",
                        error=f"タイムアウト({int(elapsed)}秒間進捗なし)",
                    )
                    self._check_parent_completion(task)
                    self._report(
                        f"⏰ タスクタイムアウト: {task.description[:60]}\n"
                        f"({int(elapsed)}秒間進捗なし → failed)"
                    )
                except Exception:
                    logger.exception("Failed to reap stuck task %s", task.task_id)

    def _retry_failed_tasks(self) -> None:
        """リトライ可能な失敗タスクをpendingに戻す."""
        failed = self.manager.task_queue.list_by_status("failed")
        for task in sorted(failed, key=lambda t: t.priority):
            # エスカレーション済みタスクはスキップ
            if task.error and task.error.startswith("[escalated]"):
                continue
            if task.retry_count < task.max_retries:
                logger.info(
                    "Retrying task %s (retry %d/%d, error: %s)",
                    task.task_id,
                    task.retry_count + 1,
                    task.max_retries,
                    task.error,
                )
                self.manager.task_queue.update_status_for_retry(
                    task.task_id, retry_count=task.retry_count + 1
                )
            else:
                self._escalate_to_creator(task)

    def _escalate_to_creator(self, task: TaskEntry) -> None:
        """max_retries到達タスクをCreatorにエスカレーションする."""
        content = (
            f"タスク '{task.description}' が{task.max_retries}回リトライ後も失敗しました。\n"
            f"最終エラー: {task.error or '不明'}\n"
            f"task_id: {task.task_id}"
        )
        try:
            entry, created = self.manager.consultation_store.ensure_pending(
                content,
                related_task_id=task.task_id,
            )
            if created:
                self._report(
                    f"🚨 エスカレーション [consult_id: {entry.consultation_id}]\n\n{content}"
                )
            else:
                logger.info(
                    "Escalation already pending (consult_id=%s, task_id=%s)",
                    entry.consultation_id,
                    task.task_id,
                )
        except Exception:
            logger.warning("Failed to escalate task %s", task.task_id, exc_info=True)
            self._report(f"🚨 エスカレーション\n\n{content}")

        # エスカレーション済みマーカーを付けて再処理を防止
        try:
            self.manager.task_queue.update_status(
                task.task_id, "failed",
                error=f"[escalated] {task.error or '不明'}",
            )
        except Exception:
            logger.warning("Failed to mark task as escalated: %s", task.task_id, exc_info=True)

    def _check_initiative_completion(self, task_id: str) -> None:
        """タスク完了時にイニシアチブの全タスク完了を検知し、振り返りを生成する."""
        initiative_store = getattr(self.manager, "initiative_store", None)
        initiative_planner = getattr(self.manager, "initiative_planner", None)
        if initiative_store is None:
            return

        try:
            # Check all active initiatives (planned or in_progress)
            for status in ("planned", "in_progress"):
                for initiative in initiative_store.list_by_status(status):
                    if task_id not in initiative.task_ids:
                        continue

                    # Check if ALL tasks in this initiative are completed
                    all_completed = True
                    for tid in initiative.task_ids:
                        task_entry = self.manager.task_queue._get_latest(tid)
                        if task_entry is None or task_entry.status != "completed":
                            all_completed = False
                            break

                    if not all_completed:
                        continue

                    # All tasks completed — mark initiative as completed
                    initiative_store.update_status(initiative.initiative_id, "completed")
                    logger.info(
                        "Initiative completed: %s (%s)",
                        initiative.title,
                        initiative.initiative_id,
                    )

                    # Generate retrospective
                    if initiative_planner is not None:
                        try:
                            retro = initiative_planner.generate_retrospective(
                                initiative.initiative_id,
                            )
                            if retro:
                                self._report(
                                    f"🎉 イニシアチブ完了: {initiative.title}\n振り返り: {retro}"
                                )
                            else:
                                self._report(f"🎉 イニシアチブ完了: {initiative.title}")
                        except Exception:
                            logger.exception(
                                "Failed to generate retrospective for %s",
                                initiative.initiative_id,
                            )
                            self._report(f"🎉 イニシアチブ完了: {initiative.title}")
                    else:
                        self._report(f"🎉 イニシアチブ完了: {initiative.title}")
        except Exception:
            logger.exception("Error checking initiative completion for task %s", task_id)
    def _check_parent_completion(self, task: TaskEntry) -> None:
        """サブタスク完了/失敗時に親タスクの状態を更新する."""
        if task.parent_task_id is None:
            return
        try:
            siblings = self.manager.task_queue.list_by_parent(task.parent_task_id)
            # 全サブタスク完了なら親をcompletedに
            if all(s.status == "completed" for s in siblings):
                self.manager.task_queue.update_status(
                    task.parent_task_id, "completed", result="全サブタスク完了"
                )
                return
            # 永久失敗サブタスク（retry_count >= max_retries）があれば親をfailedに
            if any(
                s.status == "failed" and s.retry_count >= s.max_retries
                for s in siblings
            ):
                self.manager.task_queue.update_status(
                    task.parent_task_id, "failed", error="サブタスク永久失敗"
                )
        except Exception:
            logger.exception(
                "Error checking parent completion for task %s", task.task_id
            )
