"""Initiative planner for autonomous business initiative generation.

Plans 1-3 business initiatives based on vision, constitution, strategy
direction, and past performance. Saves to InitiativeStore and adds
first-step tasks to TaskQueue.

Requirements: 1.1, 1.2, 1.3, 1.4, 3.4
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from initiative_store import InitiativeStore
from models import InitiativeEntry, InitiativeScores
from strategy_analyzer import StrategyAnalyzer

if TYPE_CHECKING:
    from manager import Manager

logger = logging.getLogger(__name__)

# Cost-efficiency threshold: scores at or below this (out of 25) are
# considered high-cost and trigger the "consulting" status when the
# estimated cost would exceed 50% of the budget limit.
_HIGH_COST_THRESHOLD = 5


class InitiativePlanner:
    """ビジョン・憲法・戦略方針・過去実績に基づいてイニシアチブを計画する."""

    COOLDOWN_SECONDS = 30 * 60  # 30分
    MAX_INITIATIVES_PER_CYCLE = 1

    def __init__(
        self,
        manager: Manager,
        initiative_store: InitiativeStore,
        strategy_analyzer: StrategyAnalyzer,
    ) -> None:
        self._manager = manager
        self._initiative_store = initiative_store
        self._strategy_analyzer = strategy_analyzer
        self._last_planned_at: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self) -> list[InitiativeEntry]:
        """ビジョン・憲法・戦略方針・過去実績に基づいてイニシアチブを計画する.

        クールダウン期間中、またはアクティブなイニシアチブが存在する場合は
        空リストを返す。1サイクルあたり最大1件のイニシアチブを生成する。
        """
        # クールダウンチェック
        if self._last_planned_at is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_planned_at).total_seconds()
            if elapsed < self.COOLDOWN_SECONDS:
                logger.info("Initiative cooldown active (%d/%d sec)", elapsed, self.COOLDOWN_SECONDS)
                return []

        # アクティブイニシアチブチェック
        active = self._initiative_store.list_by_status("planned") + \
                 self._initiative_store.list_by_status("in_progress")
        if active:
            logger.info("Active initiatives exist (%d), skipping planning", len(active))
            return []

        try:
            result = self._plan_impl()
            if result:
                self._last_planned_at = datetime.now(timezone.utc)
            return result
        except Exception:
            logger.exception("Initiative planning failed")
            return []

    def generate_retrospective(self, initiative_id: str) -> str:
        """完了したイニシアチブの振り返りを生成する."""
        entry = self._initiative_store.get(initiative_id)
        if entry is None:
            logger.warning("Initiative not found for retrospective: %s", initiative_id)
            return ""

        prompt = self._build_retrospective_prompt(entry)
        messages = [{"role": "user", "content": prompt}]

        llm_client = self._manager.llm_client
        if llm_client is None:
            logger.error("LLM client not configured")
            return "振り返り生成失敗: LLMクライアント未設定"

        result = llm_client.chat(messages)

        # Record LLM call
        from llm_client import LLMResponse

        if isinstance(result, LLMResponse):
            self._manager.record_llm_call(
                provider="openrouter",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                task_id=f"retrospective-{initiative_id}",
            )
            retrospective_text = result.content.strip()
        else:
            logger.warning("LLM error during retrospective: %s", result.message)
            retrospective_text = "振り返り生成失敗"

        # Update initiative with retrospective
        self._initiative_store.update_status(
            initiative_id,
            status="completed",
            retrospective=retrospective_text,
        )

        return retrospective_text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _plan_impl(self) -> list[InitiativeEntry]:
        """Core planning logic."""
        llm_client = self._manager.llm_client
        if llm_client is None:
            logger.error("LLM client not configured")
            return []

        # Gather context
        vision = self._manager.vision_loader.load()
        constitution = self._manager.state.constitution
        strategy = self._strategy_analyzer.analyze()
        recent_initiatives = self._initiative_store.recent(limit=5)

        # Build prompt
        prompt = self._build_plan_prompt(
            vision, constitution, strategy, recent_initiatives,
        )
        messages = [{"role": "user", "content": prompt}]

        # Call LLM
        result = llm_client.chat(messages)

        from llm_client import LLMResponse

        if not isinstance(result, LLMResponse):
            logger.warning("LLM error during planning: %s", result.message)
            return []

        # Record LLM call
        self._manager.record_llm_call(
            provider="openrouter",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            task_id="initiative-planning",
        )

        # Parse response
        initiatives = self._parse_initiatives(result.content)
        if not initiatives:
            logger.warning("No initiatives parsed from LLM response")
            return []

        # Limit to MAX_INITIATIVES_PER_CYCLE (1件)
        initiatives = initiatives[:self.MAX_INITIATIVES_PER_CYCLE]

        # Budget check and save
        budget_limit = 10.0
        if constitution and constitution.budget:
            budget_limit = constitution.budget.limit_usd

        entries: list[InitiativeEntry] = []
        for init_data in initiatives:
            entry = init_data

            # Check if high-cost: cost_efficiency <= threshold suggests high cost
            if (
                entry.estimated_scores is not None
                and entry.estimated_scores.cost_efficiency <= _HIGH_COST_THRESHOLD
            ):
                entry = entry.model_copy(update={"status": "consulting"})

            # Save to store
            self._initiative_store.save(entry)

            # Add first step as task (only if not consulting)
            if entry.first_step and entry.status != "consulting":
                task_entry = self._manager.task_queue.add(entry.first_step)
                # Update initiative with the task_id
                updated = entry.model_copy(
                    update={"task_ids": [task_entry.task_id]},
                )
                self._initiative_store.save(updated)
                entries.append(updated)
            else:
                entries.append(entry)

        return entries

    def _build_plan_prompt(
        self,
        vision: str,
        constitution,
        strategy,
        recent_initiatives: list[InitiativeEntry],
    ) -> str:
        """LLMに渡すイニシアチブ計画プロンプトを構築する."""
        parts: list[str] = []

        parts.append("あなたはAI会社の社長AIです。以下の情報に基づいて、新しいビジネスイニシアチブを1〜3件提案してください。")
        parts.append("")

        # Vision
        parts.append("## ビジョン")
        parts.append(vision)
        parts.append("")

        # Constitution
        if constitution:
            parts.append("## 憲法（抜粋）")
            parts.append(f"- 目的: {constitution.purpose}")
            if constitution.budget:
                parts.append(f"- 予算上限: ${constitution.budget.limit_usd}/時間")
            if constitution.creator_score_policy:
                parts.append(f"- スコア方針: {constitution.creator_score_policy.priority}")
            parts.append("")

        # Strategy direction
        if strategy and strategy.summary:
            parts.append("## 現在の戦略方針")
            parts.append(strategy.summary)
            if strategy.weak_axes:
                parts.append(f"- 改善が必要な軸: {', '.join(strategy.weak_axes)}")
            parts.append("")

        # Recent initiatives
        if recent_initiatives:
            parts.append("## 過去のイニシアチブ（直近）")
            for init in recent_initiatives[-3:]:
                status_ja = {
                    "planned": "計画済み",
                    "in_progress": "進行中",
                    "completed": "完了",
                    "abandoned": "中止",
                    "consulting": "相談待ち",
                }.get(init.status, init.status)
                parts.append(f"- [{status_ja}] {init.title}: {init.description}")
            parts.append("")

        # Output format
        parts.append("## 出力形式")
        parts.append("以下のJSON配列形式で回答してください。1〜3件のイニシアチブを提案してください。")
        parts.append("```json")
        parts.append('[')
        parts.append('  {')
        parts.append('    "title": "イニシアチブのタイトル",')
        parts.append('    "description": "イニシアチブの説明",')
        parts.append('    "estimated_cost_usd": 3.0,')
        parts.append('    "scores": {')
        parts.append('      "interestingness": 20,')
        parts.append('      "cost_efficiency": 15,')
        parts.append('      "realism": 18,')
        parts.append('      "evolvability": 22')
        parts.append('    },')
        parts.append('    "first_step": "最初に実行すべきタスクの説明"')
        parts.append('  }')
        parts.append(']')
        parts.append("```")
        parts.append("")
        parts.append("注意:")
        parts.append("- 各スコアは0〜25の整数")
        parts.append("- first_stepは具体的で実行可能なタスク")
        parts.append("- シェルで完結しやすい活動を優先")

        return "\n".join(parts)

    def _build_retrospective_prompt(self, entry: InitiativeEntry) -> str:
        """振り返り生成プロンプトを構築する."""
        parts: list[str] = []
        parts.append("以下のイニシアチブの振り返りを生成してください。")
        parts.append("")
        parts.append(f"## イニシアチブ: {entry.title}")
        parts.append(f"説明: {entry.description}")
        parts.append(f"ステータス: {entry.status}")
        if entry.estimated_scores:
            parts.append(f"想定スコア: 面白さ{entry.estimated_scores.interestingness}/25, "
                         f"コスト効率{entry.estimated_scores.cost_efficiency}/25, "
                         f"現実性{entry.estimated_scores.realism}/25, "
                         f"進化性{entry.estimated_scores.evolvability}/25")
        parts.append("")
        parts.append("以下の観点で振り返りを記述してください:")
        parts.append("1. 成果の要約")
        parts.append("2. 学び（うまくいったこと、改善点）")
        parts.append("3. 次への示唆（今後のイニシアチブへの提案）")
        return "\n".join(parts)

    def _parse_initiatives(self, content: str) -> list[InitiativeEntry]:
        """LLMレスポンスからイニシアチブをパースする."""
        now = datetime.now(timezone.utc)

        # Try to extract JSON array from the response
        json_data = self._extract_json_array(content)
        if json_data is None:
            logger.warning("Failed to extract JSON from LLM response")
            return []

        entries: list[InitiativeEntry] = []
        for item in json_data:
            try:
                scores_data = item.get("scores", {})
                scores = InitiativeScores(
                    interestingness=self._clamp_score(scores_data.get("interestingness", 15)),
                    cost_efficiency=self._clamp_score(scores_data.get("cost_efficiency", 15)),
                    realism=self._clamp_score(scores_data.get("realism", 15)),
                    evolvability=self._clamp_score(scores_data.get("evolvability", 15)),
                )

                title = str(item.get("title", "")).strip()
                description = str(item.get("description", "")).strip()
                first_step = str(item.get("first_step", "")).strip()

                if not title or not description or not first_step:
                    logger.warning("Skipping initiative with missing fields: %s", item)
                    continue

                entry = InitiativeEntry(
                    initiative_id=uuid4().hex[:8],
                    title=title,
                    description=description,
                    status="planned",
                    estimated_scores=scores,
                    first_step=first_step,
                    created_at=now,
                    updated_at=now,
                )
                entries.append(entry)
            except Exception:
                logger.warning("Failed to parse initiative item: %s", item, exc_info=True)
                continue

        # Limit to 3 initiatives max
        return entries[:3]

    @staticmethod
    def _extract_json_array(content: str) -> list[dict] | None:
        """LLMレスポンスからJSON配列を抽出する."""
        # Try to find JSON in code blocks first
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
        if code_block:
            text = code_block.group(1).strip()
        else:
            text = content.strip()

        # Try parsing as JSON array
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _clamp_score(value: int | float | None) -> int:
        """スコアを0-25の範囲にクランプする."""
        if value is None:
            return 15
        return max(0, min(25, int(value)))
