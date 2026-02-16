"""Unit tests for approval_text_classifier."""

from __future__ import annotations

from approval_text_classifier import classify_approval_text


class TestClassifyApprovalTextHeuristics:
    def test_approve_keywords(self):
        assert classify_approval_text("いいよ、進めて") == "approved"

    def test_reject_keywords(self):
        assert classify_approval_text("それはダメ。見送りで") == "rejected"

    def test_empty_is_unknown(self):
        assert classify_approval_text("") == "unknown"

    def test_conflicting_is_unknown(self):
        assert classify_approval_text("OKだけどやめて") == "unknown"

