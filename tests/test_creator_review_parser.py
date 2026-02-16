"""Unit tests for creator_review_parser."""

from __future__ import annotations

from creator_review_parser import parse_creator_review


class TestParseCreatorReview:
    def test_parses_axes_and_computes_total(self):
        text = (
            "面白さ: 20/25, コスト効率: 10/25, 現実性: 15/25, 進化性: 18/25\n"
            "コメント: 良い方向。継続。\n"
            "指示: 継続"
        )
        review = parse_creator_review(text, user_id="U123")
        assert review is not None
        assert review.score_interestingness_25 == 20
        assert review.score_cost_efficiency_25 == 10
        assert review.score_realism_25 == 15
        assert review.score_evolvability_25 == 18
        assert review.score_total_100 == 63
        assert "良い方向" in review.comment
        assert review.user_id == "U123"

    def test_parses_total_only(self):
        text = "総合: 80/100\nコメント: 方向性は良い"
        review = parse_creator_review(text)
        assert review is not None
        assert review.score_total_100 == 80
        assert review.score_interestingness_25 is None

    def test_parses_fullwidth_digits(self):
        text = (
            "面白さ：２５／２５\n"
            "コスト効率：２０／２５\n"
            "現実性：１５／２５\n"
            "進化性：１０／２５\n"
            "コメント：OK"
        )
        review = parse_creator_review(text)
        assert review is not None
        assert review.score_total_100 == 70

    def test_returns_none_for_non_review_text(self):
        review = parse_creator_review("こんにちは")
        assert review is None

