"""Sub-Agent Runner — サブエージェントの生成と実行を担当する.

CEO_AIが特定タスク用にスポーンする軽量エージェントを管理する。
同一プロセス内で同期的にLLM会話ループを実行し、結果を返す。

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from cost_aggregator import compute_window_cost
from llm_client import LLMClient, LLMError
from model_catalog import select_model_for_role
from models import ResearchNote
from response_parser import parse_response
from shell_executor import execute_shell

if TYPE_CHECKING:
    from datetime import datetime
    from manager import Manager

logger = logging.getLogger(__name__)

def _read_int_env(key: str, default: int) -> int:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


MAX_CONVERSATION_TURNS = _read_int_env("SUB_AGENT_MAX_TURNS", 0)
MAX_CONVERSATION_WALL_SECONDS = _read_int_env("SUB_AGENT_MAX_WALL_SECONDS", 0)
DEFAULT_WIP_LIMIT = 3
_PROGRESS_ITEMS_LIMIT = 20



_JP_SURNAMES = ['佐藤','鈴木','高橋','田中','渡辺','伊藤','山本','中村','小林','加藤']
_JP_GIVEN_NAMES = ['陽菜','結衣','葵','凛','咲良','悠斗','蓮','颯太','陽向','悠真']


def generate_japanese_employee_name(seed: str, existing_names: set[str] | None = None) -> str:
    existing = existing_names or set()
    suffix = seed.split('-', 1)[-1]
    try:
        n = int(suffix, 16)
    except Exception:
        n = sum(ord(c) for c in seed)

    base = f"{_JP_SURNAMES[n % len(_JP_SURNAMES)]} {_JP_GIVEN_NAMES[(n // len(_JP_SURNAMES)) % len(_JP_GIVEN_NAMES)]}"
    if base not in existing:
        return base

    for i in range(2, 100):
        cand = f"{base}（{i}）"
        if cand not in existing:
            return cand

    return base
@dataclass(frozen=True)
class RoleObjective:
    """Role-specific objective function and boundaries."""

    objective: str
    success_metrics: list[str]
    owned_decisions: list[str]
    escalation_conditions: list[str]


def _resolve_role_objective(role: str) -> RoleObjective:
    """Return objective profile for a sub-agent role."""
    normalized = (role or "").strip().lower()

    if any(k in normalized for k in ("infra", "sre", "ops", "運用", "インフラ")):
        return RoleObjective(
            objective="可用性・安全性・再現性を優先して運用基盤を安定化する",
            success_metrics=[
                "障害復旧時間を短縮する",
                "設定変更に証跡とロールバック手順を残す",
                "運用手順を再実行可能な形で文書化する",
            ],
            owned_decisions=[
                "サービス構成・設定値・起動順序の最適化",
                "監視/ログ/復旧コマンドの具体化",
                "デプロイ/再起動/切り戻し手順の設計",
            ],
            escalation_conditions=[
                "予算増額や有料契約が必要",
                "会社方針や公開範囲に抵触する可能性",
                "データ損失リスクが高い不可逆操作",
            ],
        )

    if any(k in normalized for k in ("dev", "coder", "engineer", "開発", "実装", "web")):
        return RoleObjective(
            objective="CEOが定めた目的・期限・予算内で、実装品質と速度を最大化する",
            success_metrics=[
                "要求仕様に一致した動作を実現する",
                "最小変更で安全にリリース可能にする",
                "検証可能な成果と再現コマンドを残す",
            ],
            owned_decisions=[
                "技術選定・実装方法・詳細設計",
                "デバッグ方針・テスト方針・リファクタリング範囲",
                "実装手順の分解と実行順",
            ],
            escalation_conditions=[
                "要件が矛盾して実装方針を確定できない",
                "予算/期限を超過しそう",
                "セキュリティ・法務リスクが高い",
            ],
        )

    if any(k in normalized for k in ("finance", "budget", "予算", "会計", "財務")):
        return RoleObjective(
            objective="コスト効率を最大化しつつ、事業継続性を守る",
            success_metrics=[
                "予算超過リスクを早期検知する",
                "支出判断の根拠を定量で提示する",
                "代替案の費用対効果を比較提示する",
            ],
            owned_decisions=[
                "費用試算・コスト配分・削減案の設計",
                "モデル/サービスの費用最適化",
                "予算執行の優先順位提案",
            ],
            escalation_conditions=[
                "予算上限の変更が必要",
                "新規課金・契約を伴う",
                "会社方針に関わる投資判断",
            ],
        )

    if any(k in normalized for k in ("research", "analyst", "調査", "分析", "戦略")):
        return RoleObjective(
            objective="意思決定に使える情報の精度と鮮度を最大化する",
            success_metrics=[
                "一次情報に基づく根拠を提示する",
                "不確実性と前提条件を明示する",
                "意思決定可能な比較案に整理する",
            ],
            owned_decisions=[
                "調査範囲・検索手順・要約粒度",
                "情報の信頼性評価と取捨選択",
                "提案オプションの整理方法",
            ],
            escalation_conditions=[
                "外部発信や対外約束に直結する結論",
                "方針矛盾があり優先順位を決められない",
                "追加予算なしでは検証不能",
            ],
        )

    return RoleObjective(
        objective="担当タスクを最短で前進させ、CEOの意思決定負荷を減らす",
        success_metrics=[
            "作業を中断させない具体的進捗を出す",
            "結果と根拠を簡潔に共有する",
            "再利用可能な知見を残す",
        ],
        owned_decisions=[
            "タスク実行順と具体手段",
            "必要な調査/実装/検証の設計",
            "ログ・証跡の取り方",
        ],
        escalation_conditions=[
            "予算/方針/期限の制約を満たせない",
            "不可逆で高リスクな操作が必要",
            "要件の優先順位が不明確",
        ],
    )


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
        ignore_wip_limit: bool = False,
        persistent_employee: dict | None = None,
    ) -> str:
        """サブエージェントをスポーンし、タスクを実行する.

        Args:
            name: エージェント名
            role: エージェントの役割
            task_description: タスクの説明
            budget_limit_usd: 予算上限（USD）
            model: 使用するLLMモデル名。未指定/空文字列の場合はCEO_AIのモデルにフォールバック。
            ignore_wip_limit: True の場合、WIP制限を無視してスポーンする。
            persistent_employee: 正社員AIプロファイル（employee_id/name/role/model/purpose）。

        Returns:
            結果文字列（完了メッセージまたはエラー）
        """
        # WIP limit check: count active agents excluding CEO
        if not ignore_wip_limit:
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
        employee_profile = persistent_employee or {}
        is_persistent_employee = bool(employee_profile)

        if is_persistent_employee:
            agent_id = str(employee_profile.get("employee_id") or f"emp-{uuid4().hex[:6]}")
            display_name = str(employee_profile.get("name") or name).strip() or name
            role = str(employee_profile.get("role") or role).strip() or role
            if not model:
                model = str(employee_profile.get("model") or "").strip() or None
            try:
                if budget_limit_usd <= 0:
                    budget_limit_usd = float(employee_profile.get("budget_limit_usd") or 1.0)
            except Exception:
                budget_limit_usd = max(0.05, budget_limit_usd)
        else:
            # Generate agent_id
            agent_id = f"sub-{uuid4().hex[:6]}"
            # Assign a unique Japanese employee name for clear logs.
            try:
                existing_names = {a.name for a in self.manager.agent_registry._list_all() if getattr(a, 'name', None)}
            except Exception:
                existing_names = set()
            display_name = generate_japanese_employee_name(agent_id, existing_names)

        if model:
            effective_model = model
        else:
            effective_model = select_model_for_role(role, fallback_model=ceo_model, task_description=task_description)

        # Create independent LLMClient for this sub-agent
        sub_client = self._create_llm_client(effective_model)
        self._log_activity(
            f"社員AI起動: name={display_name} role={role} model={effective_model} "
            f"task={self._summarize(task_description, limit=220)}"
        )

        # Register in AgentRegistry with effective model name
        self.manager.agent_registry.register(
            agent_id=agent_id,
            name=display_name,
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
        employee_memory_text = None
        employee_purpose = None
        try:
            mm = getattr(self.manager, "memory_manager", None)
            if mm is not None:
                mm.ingest_all_sources()
                rolling_summary_text = mm.summary_for_prompt()
                recalled_memories = mm.recall_for_prompt(task_description, limit=6)
        except Exception:
            logger.warning("Failed to build memory context for sub-agent", exc_info=True)

        if is_persistent_employee:
            try:
                employee_id = str(employee_profile.get("employee_id") or agent_id)
                employee_memory_text = self.manager.employee_store.read_memory(employee_id, max_chars=3200)
                employee_purpose = str(employee_profile.get("purpose") or "").strip() or None
            except Exception:
                logger.warning("Failed to load employee-specific memory", exc_info=True)

        system_prompt = self._build_sub_agent_prompt(
            role,
            task_description,
            purpose=purpose,
            vision_text=vision_text,
            rolling_summary_text=rolling_summary_text,
            recalled_memories=recalled_memories,
            employee_name=display_name if is_persistent_employee else None,
            employee_id=str(employee_profile.get("employee_id") or agent_id) if is_persistent_employee else None,
            employee_purpose=employee_purpose,
            employee_memory_text=employee_memory_text,
            employment_type="employee" if is_persistent_employee else "part-time",
        )
        try:
            result = self._run_conversation(
                agent_id=agent_id,
                agent_name=display_name,
                role=role,
                system_prompt=system_prompt,
                task_description=task_description,
                budget_limit_usd=budget_limit_usd,
                llm_client=sub_client,
            )
        except Exception as exc:
            logger.exception("Sub-agent %s failed: %s", agent_id, exc)
            self.manager.agent_registry.update_status(agent_id, "inactive")
            result = f"エラー: {exc}"
            self._log_activity(f"社員AI異常終了: name={display_name} role={role} error={self._summarize(str(exc), limit=200)}")

        # Mark agent as inactive after completion
        try:
            self.manager.agent_registry.update_status(agent_id, "inactive")
        except Exception:
            logger.warning("Failed to deactivate agent %s", agent_id, exc_info=True)

        self._log_activity(
            f"社員AI完了: name={display_name} role={role} result={self._summarize(result, limit=260)}"
        )

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

        if is_persistent_employee:
            try:
                employee_id = str(employee_profile.get("employee_id") or agent_id)
                self.manager.employee_store.append_memory(
                    employee_id,
                    title="委任タスク実行",
                    content=(
                        f"task:\n{task_description.strip()}\n\n"
                        f"result:\n{(result or '').strip()}"
                    ),
                    source="delegation",
                )
            except Exception:
                logger.warning("Failed to append employee memory", exc_info=True)

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
        employee_name: str | None = None,
        employee_id: str | None = None,
        employee_purpose: str | None = None,
        employee_memory_text: str | None = None,
        employment_type: str = "part-time",
    ) -> str:
        """サブエージェント用のシステムプロンプトを構築する."""
        company_root = self.manager.base_dir / "companies" / self.manager.company_id
        purpose_text = (purpose or "").strip() or "未設定"
        vision = (vision_text or "").strip() or "未設定"
        objective = _resolve_role_objective(role)
        now_utc = datetime.now(timezone.utc)
        now_jst = now_utc.astimezone(timezone(timedelta(hours=9)))

        lines: list[str] = [
            "あなたはAI会社のサブエージェントです。",
            f"役割: {role}",
            f"雇用区分: {'正社員AI' if employment_type == 'employee' else 'アルバイトAI'}",
            f"タスク: {task_description}",
            "",
            "## この役割の目的関数",
            f"- {objective.objective}",
            "成功指標:",
            *[f"- {metric}" for metric in objective.success_metrics],
            "",
            "あなたが主体的に決める範囲:",
            *[f"- {item}" for item in objective.owned_decisions],
            "",
            "Creator/CEOに確認すべき条件:",
            *[f"- {item}" for item in objective.escalation_conditions],
            "",
            "CEOは目的・予算・期限・価値観を定義し、あなたは実装/運用のHowを決める担当です。",
            "専門実務の詳細をCEOに戻さず、まず自分で具体案を作って実行してください。",
            "軽微な環境整備（依存関係のインストール/設定修正/再起動など）は独断で続行してよい（例: wp が PHP 不足で動かないなら PHP をインストールして再検証する）。",
            "同じコマンド列を繰り返し使うと判断したら、/opt/apps/ai-company/tools/ai new <name> で共通ツール化するか、手順SoTとして保存して再利用する。",
            "許可取りはしない。確認が必要なのは「会社方針/予算/外部課金・契約/不可逆・高リスク操作」に関わる場合だけ。",
            "長時間タスクは進捗を分割して進める。中断時は run_id・実施済み・残作業 を必ず報告し、再開可能な状態で止める。",
            "",
            "## 組織ルール",
            "- 正社員AIは継続担当として一貫性を維持する。重要な進捗・判断は自分のメモリに残す。",
            "- アルバイトAIは短期スポット担当。必要最小限の実装/調査を迅速に完了する。",
            "- 雇用区分を勝手に変更しない。",
            "",
            "## 現在時刻",
            f"- UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- JST: {now_jst.strftime('%Y-%m-%d %H:%M:%S')}",
            "- 現在時刻の再確認が必要なら `<control>time now</control>` を使う。",
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
            f"- {company_root / 'knowledge' / 'MEMORY.md'}",
            f"- {company_root / 'state' / 'rolling_summary.md'}",
            f"- {company_root / 'state' / 'memory.sqlite3'}",
            f"- {company_root / 'state' / 'commitments.ndjson'}",
            f"- {company_root / 'protocols' / 'mcp_servers.yaml'}",
            f"- {company_root / 'state' / 'alarms.json'}",
            f"- {company_root / 'protocols' / 'newsroom_sources.yaml'}",
            f"- {company_root / 'state' / 'newsroom_state.json'}",
            f"- {company_root / 'journal'}",
            "- /opt/apps/ai-company/tools/ai",
            "- /opt/apps/ai-company/tools/README.md",
            "必要に応じて<shell>でファイル参照して構いません。",
        ]

        if employment_type == "employee":
            lines.extend([
                "",
                "## あなたの社員プロファイル",
                f"- name: {employee_name or '(unknown)'}",
                f"- employee_id: {employee_id or '(unknown)'}",
                f"- purpose: {employee_purpose or '(未設定)'}",
            ])
            if employee_memory_text and employee_memory_text.strip():
                lines.extend([
                    "",
                    "## あなた専用の記憶（抜粋）",
                    employee_memory_text.strip(),
                ])

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
            "<mcp>MCPツール呼び出し（社内共通ツール）</mcp>",
            "<memory>社内メモリに残す内容</memory>",
            "<control>alarm/time制御コマンド</control>",
            "<reply>報告テキスト</reply>",
            "<done>完了メッセージ</done>",
            "",
            "control例:",
            "- alarm add once 2026-02-19T12:00:00+09:00 | self | 10分後に進捗を再確認する",
            "- alarm add cron 0 * * * * | role:web-developer;budget=0.5 | サイト死活確認して報告",
            "- alarm list",
            "- alarm cancel <alarm_id>",
            "- time now",
            "",
            "タスクが完了したら必ず<done>タグで結果を報告してください。",
        ])

        return "\n".join(lines)

    def _run_conversation(
        self,
        agent_id: str,
        agent_name: str,
        role: str,
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
        run_id = f"run-{uuid4().hex[:10]}"
        started_at = datetime.now(timezone.utc)
        max_turns = _read_int_env("SUB_AGENT_MAX_TURNS", MAX_CONVERSATION_TURNS)
        max_wall_seconds = _read_int_env("SUB_AGENT_MAX_WALL_SECONDS", MAX_CONVERSATION_WALL_SECONDS)
        progress_items: list[str] = []
        memory_payload_seen: set[str] = set()

        def _elapsed_seconds() -> int:
            return int((datetime.now(timezone.utc) - started_at).total_seconds())

        def _remember(note: str) -> None:
            note_text = (note or "").strip()
            if not note_text:
                return
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            progress_items.append(f"{stamp} {note_text}")
            if len(progress_items) > _PROGRESS_ITEMS_LIMIT:
                del progress_items[:-_PROGRESS_ITEMS_LIMIT]

        def _save_checkpoint(
            *,
            status: str,
            reason: str,
            turn: int,
            pending_hint: str = "",
            result_text: str = "",
        ) -> None:
            self._persist_run_checkpoint(
                run_id=run_id,
                agent_id=agent_id,
                agent_name=agent_name,
                role=role,
                model=llm_client.model,
                task_description=task_description,
                turn=turn,
                elapsed_seconds=_elapsed_seconds(),
                status=status,
                reason=reason,
                pending_hint=pending_hint,
                progress_items=progress_items,
                result_text=result_text,
            )

        def _interrupt(
            *,
            reason: str,
            turn: int,
            pending_hint: str,
        ) -> str:
            _save_checkpoint(
                status="interrupted",
                reason=reason,
                turn=turn,
                pending_hint=pending_hint,
            )
            report = self._build_interruption_report(
                run_id=run_id,
                agent_name=agent_name,
                role=role,
                reason=reason,
                turn=turn,
                elapsed_seconds=_elapsed_seconds(),
                progress_items=progress_items,
                pending_hint=pending_hint,
            )
            self._log_activity(
                f"社員AI中断: name={agent_name} role={role} run_id={run_id} "
                f"reason={self._summarize(reason, limit=120)}"
            )
            return report

        resume_run_id = self._extract_resume_run_id(task_description)
        if resume_run_id:
            checkpoint = self.get_run_checkpoint(resume_run_id)
            resume_context = self._build_resume_context(checkpoint) if checkpoint else ""
            if resume_context:
                conversation.append({"role": "user", "content": resume_context})
                _remember(f"再開コンテキストを適用(run_id={resume_run_id})")

        _save_checkpoint(status="running", reason="started", turn=0)

        turn = 0
        while True:
            turn += 1

            if max_turns > 0 and turn > max_turns:
                return _interrupt(
                    reason=f"会話ターン上限({max_turns})に到達",
                    turn=turn - 1,
                    pending_hint="作業を2〜5ステップ単位に分割し、employee resumeで継続してください。",
                )

            if max_wall_seconds > 0 and _elapsed_seconds() >= max_wall_seconds:
                return _interrupt(
                    reason=f"実行時間上限({max_wall_seconds}秒)に到達",
                    turn=turn - 1,
                    pending_hint="同じrun_idで再開し、残作業を継続してください。",
                )

            if self._is_budget_exceeded(agent_id, budget_limit_usd):
                logger.warning("Budget exceeded for agent %s", agent_id)
                self.manager.agent_registry.update_status(agent_id, "inactive")
                _remember(f"予算上限(${budget_limit_usd})到達")
                return _interrupt(
                    reason=f"予算上限(${budget_limit_usd})に到達",
                    turn=turn - 1,
                    pending_hint="予算見直し、またはタスクを細分化して再開してください。",
                )

            llm_result = llm_client.chat(conversation)

            if isinstance(llm_result, LLMError):
                logger.error("LLM call failed for agent %s: %s", agent_id, llm_result.message)
                _save_checkpoint(
                    status="failed",
                    reason=f"LLMエラー: {llm_result.message}",
                    turn=turn,
                    result_text=f"LLMエラー: {llm_result.message}",
                )
                return f"LLMエラー: {llm_result.message}"

            self.manager.record_llm_call(
                provider="openrouter",
                model=llm_result.model,
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                agent_id=agent_id,
            )

            conversation.append({"role": "assistant", "content": llm_result.content})

            actions = parse_response(llm_result.content)
            if actions:
                kinds = ", ".join(a.action_type for a in actions)
                self._log_activity(f"社員AI処理: name={agent_name} role={role} actions={kinds}")

            if self._is_ack_only_memory_followup(actions):
                _remember("loop guard: ack-only followup")
                reply_actions = [a for a in actions if a.action_type == "reply" and (a.content or "").strip()]
                guard_text = reply_actions[-1].content.strip() if reply_actions else "loop guard: ack-only followup"
                self._log_activity(
                    f"社員AIループ抑止: name={agent_name} role={role} reason=ack_only_memory_followup"
                )
                _save_checkpoint(
                    status="completed",
                    reason="loop_guard_ack_only_followup",
                    turn=turn,
                    result_text=guard_text,
                )
                return guard_text

            done_result = None
            needs_followup = False

            for action in actions:
                if action.action_type == "done":
                    done_result = action.content
                    _remember(f"done候補: {self._summarize(done_result, limit=80)}")
                elif action.action_type == "shell_command":
                    command_summary = self._summarize(action.content, limit=100)
                    _remember(f"shell実行: {command_summary}")
                    self._log_activity(f"社員AIツール利用: name={agent_name} tool=shell")
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
                    break
                elif action.action_type == "research":
                    query = action.content.strip()
                    if not query:
                        continue
                    _remember(f"web検索: {self._summarize(query, limit=90)}")
                    self._log_activity(f"社員AIツール利用: name={agent_name} tool=web_search")
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
                elif action.action_type == "mcp":
                    self._log_activity(f"社員AIツール利用: name={agent_name} tool=mcp")
                    payload = (action.content or "").strip()
                    if not payload:
                        continue
                    _remember(f"mcp呼び出し: {self._summarize(payload, limit=90)}")
                    try:
                        result_text = self.manager.mcp_client.run_action(payload)
                    except Exception as exc:
                        logger.warning("Sub-agent MCP call failed: %s", exc, exc_info=True)
                        result_text = f"MCP呼び出しエラー: {exc}"

                    conversation.append({"role": "user", "content": result_text})
                    needs_followup = True
                    break
                elif action.action_type == "memory":
                    self._log_activity(f"社員AIツール利用: name={agent_name} tool=memory")
                    raw = (action.content or "").strip()
                    if not raw:
                        continue
                    _remember(f"memory保存: {self._summarize(raw, limit=90)}")

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
                        _remember("loop guard: duplicate memory payload")
                        conversation.append({"role": "user", "content": "メモリ保存スキップ: duplicate payload(loop_guard)"})
                        self._log_activity(
                            f"社員AIループ抑止: name={agent_name} role={role} reason=duplicate_memory_payload"
                        )
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
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                doc_id = mm.pin(payload)
                                mm.ingest_all_sources()
                            result_text = f"メモリ保存: pin OK ({doc_id or 'no-index'})"
                        elif op == "curated":
                            title, body = _split_title_body(payload)
                            self.manager.memory_vault.append(body, title=title, author=agent_id)
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                mm.ingest_all_sources()
                            result_text = "メモリ保存: curated OK"
                        else:
                            title, body = _split_title_body(payload)
                            self.manager.memory_vault.append_daily(body, title=title, author=agent_id)
                            mm = getattr(self.manager, "memory_manager", None)
                            if mm is not None:
                                mm.ingest_all_sources()
                            result_text = "メモリ保存: daily OK"
                        memory_payload_seen.add(payload_key)
                    except Exception as exc:
                        logger.warning("Sub-agent memory action failed: %s", exc, exc_info=True)
                        result_text = f"メモリ保存エラー: {exc}"

                    conversation.append({"role": "user", "content": result_text})
                    needs_followup = True
                    break
                elif action.action_type == "control":
                    self._log_activity(f"社員AIツール利用: name={agent_name} tool=control")
                    result_parts: list[str] = []
                    handled_any = False
                    for raw_cmd in (action.content or "").splitlines():
                        cmd = raw_cmd.strip()
                        if not cmd:
                            continue
                        _remember(f"control実行: {self._summarize(cmd, limit=90)}")
                        handled, result_text = self.manager._handle_runtime_control_command(
                            cmd,
                            actor_id=agent_id,
                            actor_role=role,
                            actor_model=llm_client.model if llm_client is not None else None,
                        )
                        if handled:
                            handled_any = True
                            if result_text:
                                result_parts.append(result_text)
                    if not handled_any:
                        result_parts.append(f"control未対応: {action.content}")
                    result_text = "\n".join(result_parts).strip()
                    conversation.append({"role": "user", "content": result_text})
                    needs_followup = True
                    break
                elif action.action_type == "reply":
                    reply_text = (action.content or "").strip()
                    if reply_text:
                        _remember(f"reply: {self._summarize(reply_text, limit=90)}")

            if done_result is not None:
                self._log_activity(
                    f"社員AI→CEO 完了報告: name={agent_name} role={role} "
                    f"done={self._summarize(done_result, limit=220)}"
                )

                reply_actions = [a for a in actions if a.action_type == "reply"]
                if reply_actions:
                    reply_text = (reply_actions[-1].content or "").strip()
                    done_text = (done_result or "").strip()

                    merged = reply_text or done_text
                    if reply_text and done_text and done_text not in reply_text:
                        merged = (reply_text + "\n\n" + done_text).strip()

                    _save_checkpoint(
                        status="completed",
                        reason="done",
                        turn=turn,
                        result_text=merged,
                    )
                    return merged

                _save_checkpoint(
                    status="completed",
                    reason="done",
                    turn=turn,
                    result_text=done_result,
                )
                return done_result

            if not needs_followup:
                reply_actions = [a for a in actions if a.action_type == "reply"]
                if reply_actions:
                    text = reply_actions[-1].content
                    self._log_activity(
                        f"社員AI→CEO reply: name={agent_name} role={role} "
                        f"text={self._summarize(text, limit=220)}"
                    )
                    _save_checkpoint(
                        status="completed",
                        reason="reply_only",
                        turn=turn,
                        result_text=text,
                    )
                    return text

                self._log_activity(
                    f"社員AI→CEO 応答: name={agent_name} role={role} "
                    f"text={self._summarize(llm_result.content, limit=220)}"
                )
                _save_checkpoint(
                    status="completed",
                    reason="assistant_only",
                    turn=turn,
                    result_text=llm_result.content,
                )
                return llm_result.content

            _save_checkpoint(status="running", reason="followup", turn=turn)

    @staticmethod
    def _looks_like_memory_ack_payload(payload: str) -> bool:
        text = (payload or "").strip().lower()
        if not text:
            return False
        loop_markers = (
            "curated ok",
            "daily ok",
            "pin ok",
            "メモリ保存",
            "保存しました",
            "保存済み",
            "loop_guard",
            "duplicate payload",
        )
        return any(marker in text for marker in loop_markers)

    @classmethod
    def _is_ack_only_memory_followup(cls, actions: list) -> bool:
        if not actions:
            return False

        saw_any = False
        for action in actions:
            if action.action_type in ("memory", "reply"):
                saw_any = True
                if not cls._looks_like_memory_ack_payload(action.content):
                    return False
                continue
            return False
        return saw_any

    @staticmethod
    def _extract_resume_run_id(text: str) -> str | None:
        raw = (text or "").strip()
        if not raw:
            return None
        m = re.search(r"(?:run_id|run)\s*[:=]\s*([A-Za-z0-9_-]{4,64})", raw, re.IGNORECASE)
        if not m:
            return None
        return m.group(1)

    def _run_checkpoint_path(self) -> Path:
        return self.manager.base_dir / "companies" / self.manager.company_id / "state" / "sub_agent_runs.ndjson"

    def _persist_run_checkpoint(
        self,
        *,
        run_id: str,
        agent_id: str,
        agent_name: str,
        role: str,
        model: str,
        task_description: str,
        turn: int,
        elapsed_seconds: int,
        status: str,
        reason: str,
        pending_hint: str,
        progress_items: list[str],
        result_text: str,
    ) -> None:
        path = self._run_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "employee_id": agent_id if str(agent_id).startswith("emp-") else "",
            "role": role,
            "model": model,
            "status": status,
            "reason": reason,
            "task_description": task_description,
            "turn": turn,
            "elapsed_seconds": elapsed_seconds,
            "pending_hint": pending_hint,
            "progress": progress_items[-_PROGRESS_ITEMS_LIMIT:],
            "result_text": self._summarize(result_text, limit=400),
        }
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def get_run_checkpoint(self, run_id: str) -> dict | None:
        key = (run_id or '').strip()
        if not key:
            return None
        path = self._run_checkpoint_path()
        if not path.exists():
            return None

        latest: dict | None = None
        try:
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if str(item.get('run_id') or '') == key:
                        latest = item
        except Exception:
            logger.warning('Failed to read sub-agent checkpoints', exc_info=True)
            return None
        return latest

    def _build_resume_context(self, checkpoint: dict | None) -> str:
        if not checkpoint:
            return ''
        run_id = str(checkpoint.get('run_id') or '').strip()
        if not run_id:
            return ''
        progress = checkpoint.get('progress') or []
        if isinstance(progress, list):
            progress_lines = [f"- {str(x)}" for x in progress[-8:] if str(x).strip()]
        else:
            progress_lines = []
        if not progress_lines:
            progress_lines = ['- 進捗ログなし']
        pending_hint = str(checkpoint.get('pending_hint') or '').strip() or '残作業を洗い出して継続してください。'

        return '\n'.join([
            '前回中断した作業のチェックポイントを読み込みました。',
            f'run_id: {run_id}',
            '実施済み:',
            *progress_lines,
            '継続方針:',
            f'- {pending_hint}',
            'この情報を踏まえて、重複実行を避けて次の未完了作業から再開してください。',
        ])

    def _build_interruption_report(
        self,
        *,
        run_id: str,
        agent_name: str,
        role: str,
        reason: str,
        turn: int,
        elapsed_seconds: int,
        progress_items: list[str],
        pending_hint: str,
    ) -> str:
        progress_lines = [f"- {item}" for item in progress_items[-8:]]
        if not progress_lines:
            progress_lines = ['- 進捗ログなし']

        return '\n'.join([
            '⚠️ 社員AI中断報告',
            f'run_id: {run_id}',
            f'agent: {agent_name} (role={role})',
            f'理由: {reason}',
            f'進捗: {turn}ターン / {elapsed_seconds}秒',
            '実施済み:',
            *progress_lines,
            '再開方針:',
            f'- {pending_hint}',
            f'- 再開コマンド例: employee resume {run_id}',
        ])

    @staticmethod
    def _summarize(text: str, *, limit: int = 240) -> str:
        s = " ".join((text or "").split())
        if len(s) > limit:
            return s[:limit] + "…"
        return s

    def _log_activity(self, message: str) -> None:
        try:
            fn = getattr(self.manager, "_activity_log", None)
            if callable(fn):
                fn(message)
        except Exception:
            logger.warning("Failed to emit sub-agent activity log", exc_info=True)

    def _is_budget_exceeded(self, agent_id: str, budget_limit_usd: float) -> bool:
        """指定エージェントのコストが予算上限を超えているか確認する."""
        agent_events = [
            e for e in self.manager.state.ledger_events
            if e.agent_id == agent_id and e.estimated_cost_usd is not None
        ]
        total = sum(e.estimated_cost_usd for e in agent_events)
        return total >= budget_limit_usd
