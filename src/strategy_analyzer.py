"""Strategy analyzer for Creator score trends and weak axis detection.

Analyzes recent CreatorReview scores and initiative history to produce
a StrategyDirection with rule-based summary (no LLM required).

Requirements: 2.1, 2.2, 2.4
"""

from __future__ import annotations

from creator_review_store import CreatorReviewStore
from initiative_store import InitiativeStore
from models import CreatorReview, StrategyDirection

# Mapping from model field names to Japanese axis names
_AXIS_MAP: dict[str, str] = {
    "score_interestingness_25": "面白さ",
    "score_cost_efficiency_25": "コスト効率",
    "score_realism_25": "現実性",
    "score_evolvability_25": "進化性",
}

_DEFAULT_SUMMARY = "まずは面白さ重視で小さく試す"


class StrategyAnalyzer:
    """Creatorスコアとイニシアチブ実績を分析し、戦略方針を導出する."""

    def __init__(
        self,
        creator_review_store: CreatorReviewStore,
        initiative_store: InitiativeStore,
    ) -> None:
        self._creator_review_store = creator_review_store
        self._initiative_store = initiative_store

    def analyze(self) -> StrategyDirection:
        """直近のCreatorスコアとイニシアチブ実績を分析し、戦略方針を返す."""
        reviews = self._creator_review_store.recent(limit=5)

        if not reviews:
            return StrategyDirection(summary=_DEFAULT_SUMMARY)

        score_trends = self.compute_score_trends(reviews)
        weak_axes = self.detect_weak_axes(reviews)

        strengthen: list[str] = []
        avoid: list[str] = []
        pivot_suggestions: list[str] = []

        # Identify strong axes (top scorers) and weak axes
        if score_trends:
            max_score = max(score_trends.values())
            min_score = min(score_trends.values())

            for axis, score in score_trends.items():
                if score == max_score and max_score > 0:
                    strengthen.append(axis)
                if score == min_score and min_score < max_score:
                    avoid.append(axis)

        for axis in weak_axes:
            pivot_suggestions.append(f"{axis}の改善が必要（連続低スコア）")

        summary = self._build_summary(strengthen, avoid, weak_axes, score_trends)

        return StrategyDirection(
            strengthen=strengthen,
            avoid=avoid,
            pivot_suggestions=pivot_suggestions,
            weak_axes=weak_axes,
            score_trends=score_trends,
            summary=summary,
        )

    def compute_score_trends(
        self, reviews: list[CreatorReview]
    ) -> dict[str, float]:
        """各軸のスコアトレンド（直近5件の平均）を算出する."""
        recent = reviews[-5:] if len(reviews) > 5 else reviews

        if not recent:
            return {}

        trends: dict[str, float] = {}
        for field, ja_name in _AXIS_MAP.items():
            scores = [
                getattr(r, field)
                for r in recent
                if getattr(r, field) is not None
            ]
            if scores:
                trends[ja_name] = sum(scores) / len(scores)

        return trends

    def detect_weak_axes(
        self, reviews: list[CreatorReview]
    ) -> list[str]:
        """連続2回以上10点以下の軸を検出する."""
        if len(reviews) < 2:
            return []

        weak: list[str] = []

        for field, ja_name in _AXIS_MAP.items():
            # Check for consecutive scores <= 10
            consecutive_low = 0
            found_weak = False
            for review in reviews:
                score = getattr(review, field)
                if score is not None and score <= 10:
                    consecutive_low += 1
                    if consecutive_low >= 2:
                        found_weak = True
                        break
                else:
                    consecutive_low = 0

            if found_weak:
                weak.append(ja_name)

        return weak

    def _build_summary(
        self,
        strengthen: list[str],
        avoid: list[str],
        weak_axes: list[str],
        score_trends: dict[str, float],
    ) -> str:
        """ルールベースで戦略方針サマリーを生成する."""
        parts: list[str] = []

        if strengthen:
            parts.append(f"強化方向: {', '.join(strengthen)}")

        if avoid:
            parts.append(f"注意方向: {', '.join(avoid)}")

        if weak_axes:
            parts.append(f"改善必要: {', '.join(weak_axes)}（連続低スコア）")

        if not parts:
            return _DEFAULT_SUMMARY

        return "。".join(parts)
