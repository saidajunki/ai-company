"""Tests for StrategyAnalyzer.

Requirements: 2.1, 2.2, 2.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from creator_review_store import CreatorReviewStore
from initiative_store import InitiativeStore
from models import CreatorReview, StrategyDirection
from strategy_analyzer import StrategyAnalyzer

NOW = datetime.now(timezone.utc)


def _make_review(
    interestingness: int | None = 20,
    cost_efficiency: int | None = 15,
    realism: int | None = 18,
    evolvability: int | None = 22,
    total: int | None = None,
) -> CreatorReview:
    scores = {
        "interestingness": interestingness,
        "cost_efficiency": cost_efficiency,
        "realism": realism,
        "evolvability": evolvability,
    }
    if total is None:
        total = sum(v for v in scores.values() if v is not None)
    return CreatorReview(
        timestamp=NOW,
        score_total_100=total,
        score_interestingness_25=interestingness,
        score_cost_efficiency_25=cost_efficiency,
        score_realism_25=realism,
        score_evolvability_25=evolvability,
    )


def _make_analyzer(tmp_path: Path, reviews: list[CreatorReview] | None = None) -> StrategyAnalyzer:
    crs = CreatorReviewStore(tmp_path, "test-co")
    ins = InitiativeStore(tmp_path, "test-co")
    if reviews:
        for r in reviews:
            crs.save(r)
    return StrategyAnalyzer(crs, ins)


class TestComputeScoreTrends:
    def test_single_review(self, tmp_path):
        """1件のレビューから各軸の平均を算出."""
        review = _make_review(20, 15, 18, 22)
        analyzer = _make_analyzer(tmp_path)
        trends = analyzer.compute_score_trends([review])
        assert trends["面白さ"] == 20.0
        assert trends["コスト効率"] == 15.0
        assert trends["現実性"] == 18.0
        assert trends["進化性"] == 22.0

    def test_multiple_reviews_average(self, tmp_path):
        """複数レビューの算術平均."""
        reviews = [
            _make_review(20, 10, 15, 25),
            _make_review(10, 20, 25, 15),
        ]
        analyzer = _make_analyzer(tmp_path)
        trends = analyzer.compute_score_trends(reviews)
        assert trends["面白さ"] == 15.0
        assert trends["コスト効率"] == 15.0
        assert trends["現実性"] == 20.0
        assert trends["進化性"] == 20.0

    def test_uses_only_last_5(self, tmp_path):
        """6件以上の場合、直近5件のみ使用."""
        reviews = [
            _make_review(0, 0, 0, 0, total=0),  # oldest, should be excluded
            _make_review(10, 10, 10, 10),
            _make_review(10, 10, 10, 10),
            _make_review(10, 10, 10, 10),
            _make_review(10, 10, 10, 10),
            _make_review(10, 10, 10, 10),
        ]
        analyzer = _make_analyzer(tmp_path)
        trends = analyzer.compute_score_trends(reviews)
        assert trends["面白さ"] == 10.0

    def test_empty_reviews(self, tmp_path):
        """空リストの場合は空辞書."""
        analyzer = _make_analyzer(tmp_path)
        assert analyzer.compute_score_trends([]) == {}

    def test_none_scores_excluded(self, tmp_path):
        """Noneスコアは平均計算から除外."""
        reviews = [
            _make_review(20, None, 10, 15, total=45),
            _make_review(10, None, 20, 25, total=55),
        ]
        analyzer = _make_analyzer(tmp_path)
        trends = analyzer.compute_score_trends(reviews)
        assert trends["面白さ"] == 15.0
        assert "コスト効率" not in trends
        assert trends["現実性"] == 15.0
        assert trends["進化性"] == 20.0


class TestDetectWeakAxes:
    def test_consecutive_low_scores(self, tmp_path):
        """連続2回10点以下の軸を検出."""
        reviews = [
            _make_review(5, 20, 20, 20),
            _make_review(8, 20, 20, 20),
        ]
        analyzer = _make_analyzer(tmp_path)
        weak = analyzer.detect_weak_axes(reviews)
        assert "面白さ" in weak
        assert "コスト効率" not in weak

    def test_non_consecutive_not_detected(self, tmp_path):
        """非連続の低スコアは検出しない."""
        reviews = [
            _make_review(5, 20, 20, 20),
            _make_review(20, 20, 20, 20),  # breaks the streak
            _make_review(5, 20, 20, 20),
        ]
        analyzer = _make_analyzer(tmp_path)
        weak = analyzer.detect_weak_axes(reviews)
        assert "面白さ" not in weak

    def test_exactly_10_is_weak(self, tmp_path):
        """10点ちょうどは「10点以下」に含まれる."""
        reviews = [
            _make_review(10, 20, 20, 20),
            _make_review(10, 20, 20, 20),
        ]
        analyzer = _make_analyzer(tmp_path)
        weak = analyzer.detect_weak_axes(reviews)
        assert "面白さ" in weak

    def test_11_is_not_weak(self, tmp_path):
        """11点は「10点以下」に含まれない."""
        reviews = [
            _make_review(11, 20, 20, 20),
            _make_review(11, 20, 20, 20),
        ]
        analyzer = _make_analyzer(tmp_path)
        weak = analyzer.detect_weak_axes(reviews)
        assert "面白さ" not in weak

    def test_single_review_no_weak(self, tmp_path):
        """1件のみでは連続2回にならない."""
        reviews = [_make_review(5, 5, 5, 5)]
        analyzer = _make_analyzer(tmp_path)
        assert analyzer.detect_weak_axes(reviews) == []

    def test_multiple_weak_axes(self, tmp_path):
        """複数軸が同時に弱い場合."""
        reviews = [
            _make_review(5, 3, 20, 20),
            _make_review(8, 7, 20, 20),
        ]
        analyzer = _make_analyzer(tmp_path)
        weak = analyzer.detect_weak_axes(reviews)
        assert "面白さ" in weak
        assert "コスト効率" in weak

    def test_empty_reviews(self, tmp_path):
        """空リストの場合は空リスト."""
        analyzer = _make_analyzer(tmp_path)
        assert analyzer.detect_weak_axes([]) == []

    def test_none_scores_break_streak(self, tmp_path):
        """Noneスコアは連続をリセットする."""
        reviews = [
            _make_review(5, 20, 20, 20),
            _make_review(None, 20, 20, 20, total=60),
            _make_review(5, 20, 20, 20),
        ]
        analyzer = _make_analyzer(tmp_path)
        weak = analyzer.detect_weak_axes(reviews)
        assert "面白さ" not in weak


class TestAnalyze:
    def test_empty_store_returns_default(self, tmp_path):
        """レビューなしの場合、デフォルト戦略を返す."""
        analyzer = _make_analyzer(tmp_path)
        result = analyzer.analyze()
        assert result.summary == "まずは面白さ重視で小さく試す"
        assert result.score_trends == {}
        assert result.weak_axes == []

    def test_with_reviews(self, tmp_path):
        """レビューありの場合、トレンドと弱軸を含む."""
        reviews = [
            _make_review(5, 20, 15, 18),
            _make_review(8, 22, 18, 20),
        ]
        analyzer = _make_analyzer(tmp_path, reviews)
        result = analyzer.analyze()
        assert result.score_trends
        assert "面白さ" in result.weak_axes
        assert result.summary != ""

    def test_returns_strategy_direction_type(self, tmp_path):
        """戻り値がStrategyDirection型."""
        analyzer = _make_analyzer(tmp_path, [_make_review()])
        result = analyzer.analyze()
        assert isinstance(result, StrategyDirection)

    def test_strengthen_contains_highest_axis(self, tmp_path):
        """最高スコア軸がstrengthenに含まれる."""
        reviews = [_make_review(25, 5, 5, 5)]
        analyzer = _make_analyzer(tmp_path, reviews)
        result = analyzer.analyze()
        assert "面白さ" in result.strengthen

    def test_weak_axes_generate_pivot_suggestions(self, tmp_path):
        """弱軸がある場合、ピボット提案が生成される."""
        reviews = [
            _make_review(5, 20, 20, 20),
            _make_review(8, 22, 18, 20),
        ]
        analyzer = _make_analyzer(tmp_path, reviews)
        result = analyzer.analyze()
        assert any("面白さ" in s for s in result.pivot_suggestions)

    def test_summary_includes_weak_axes(self, tmp_path):
        """サマリーに弱軸情報が含まれる."""
        reviews = [
            _make_review(5, 20, 20, 20),
            _make_review(3, 22, 18, 20),
        ]
        analyzer = _make_analyzer(tmp_path, reviews)
        result = analyzer.analyze()
        assert "面白さ" in result.summary
        assert "改善必要" in result.summary
