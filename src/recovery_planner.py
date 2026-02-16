"""Recovery planner for autonomous idle-state handling.

Determines whether the AI company can autonomously plan new initiatives
based on Creator score history, or needs to consult the Creator for direction.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from initiative_planner import InitiativePlanner
    from manager import Manager

logger = logging.getLogger(__name__)

_AUTONOMOUS_SCORE_THRESHOLD = 40


class RecoveryPlanner:
    """アイドル状態からの復帰を判断・実行する."""

    def __init__(self, manager: Manager, initiative_planner: InitiativePlanner) -> None:
        self._manager = manager
        self._initiative_planner = initiative_planner

    def should_plan_autonomously(self) -> bool:
        """Creatorスコア履歴に基づいて自律計画可能か判断する.

        条件: スコアが1件以上存在し、直近スコアが40点以上
        """
        latest = self._manager.creator_review_store.latest()
        if latest is None:
            return False
        return latest.score_total_100 >= _AUTONOMOUS_SCORE_THRESHOLD

    def handle_idle(self) -> str:
        """アイドル状態を処理する.

        自律計画可能なら Initiative_Planner を呼び、
        そうでなければ Creator に相談を送信する。
        戻り値は実行したアクションの説明。
        """
        if self.should_plan_autonomously():
            return self._plan_autonomously()
        return self._consult_creator()

    def _plan_autonomously(self) -> str:
        """InitiativePlannerを呼び出して自律的にイニシアチブを計画する."""
        try:
            entries = self._initiative_planner.plan()
            titles = [e.title for e in entries] if entries else []
            self._manager._slack_send("自律的にイニシアチブを計画しました")
            if titles:
                return f"自律的にイニシアチブを計画しました: {', '.join(titles)}"
            return "自律的にイニシアチブを計画しました（計画結果なし）"
        except Exception:
            logger.exception("Autonomous planning failed in RecoveryPlanner")
            return self._consult_creator()

    def _consult_creator(self) -> str:
        """Creatorに方向性の相談を送信する."""
        msg = "タスクがなく、自律計画の条件を満たしていません。Creatorの方向性をお待ちしています。"
        self._manager._slack_send(msg)
        return "Creatorに相談を送信しました"
