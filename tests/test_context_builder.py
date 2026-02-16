"""Tests for context_builder — conversation history section.

Requirements: 1.2, 1.4
"""

from __future__ import annotations

from datetime import datetime

from context_builder import build_system_prompt
from models import ConversationEntry


def _make_entry(role: str, content: str, ts: datetime | None = None) -> ConversationEntry:
    return ConversationEntry(
        timestamp=ts or datetime(2025, 1, 15, 10, 0, 0),
        role=role,
        content=content,
    )


def _build_prompt(**kwargs) -> str:
    """Helper with sensible defaults so tests only specify what they care about."""
    defaults = dict(
        constitution=None,
        wip=[],
        recent_decisions=[],
        budget_spent=0.0,
        budget_limit=10.0,
    )
    defaults.update(kwargs)
    return build_system_prompt(**defaults)


class TestVisionSection:
    """Tests for the ビジョン・事業方針 section added by vision_text param."""

    def test_no_vision_shows_placeholder(self):
        prompt = _build_prompt(vision_text=None)
        assert "## ビジョン・事業方針" in prompt
        assert "ビジョン未設定" in prompt

    def test_empty_string_shows_placeholder(self):
        prompt = _build_prompt(vision_text="")
        assert "ビジョン未設定" in prompt

    def test_vision_text_appears(self):
        prompt = _build_prompt(vision_text="シェルで完結しやすい活動に寄せる")
        assert "## ビジョン・事業方針" in prompt
        assert "シェルで完結しやすい活動に寄せる" in prompt
        assert "ビジョン未設定" not in prompt

    def test_vision_section_after_constitution(self):
        """Vision section should appear after constitution section."""
        prompt = _build_prompt(vision_text="テストビジョン")
        constitution_pos = prompt.index("## 会社憲法")
        vision_pos = prompt.index("## ビジョン・事業方針")
        wip_pos = prompt.index("## 現在のWIP")
        assert constitution_pos < vision_pos < wip_pos

    def test_existing_callers_unaffected(self):
        """Omitting vision_text still works (backward compat)."""
        prompt = build_system_prompt(
            constitution=None,
            wip=[],
            recent_decisions=[],
            budget_spent=0.0,
            budget_limit=10.0,
        )
        assert "## ビジョン・事業方針" in prompt
        assert "ビジョン未設定" in prompt


class TestConversationHistorySection:
    """Tests for the 会話履歴 section added by conversation_history param."""

    def test_no_history_shows_placeholder(self):
        prompt = _build_prompt(conversation_history=None)
        assert "## 会話履歴" in prompt
        assert "会話履歴なし" in prompt

    def test_empty_list_shows_placeholder(self):
        prompt = _build_prompt(conversation_history=[])
        assert "会話履歴なし" in prompt

    def test_single_entry_appears(self):
        entry = _make_entry("user", "こんにちは", datetime(2025, 1, 15, 10, 30, 0))
        prompt = _build_prompt(conversation_history=[entry])
        assert "[user]" in prompt
        assert "2025-01-15 10:30:00" in prompt
        assert "こんにちは" in prompt

    def test_multiple_entries_all_appear(self):
        entries = [
            _make_entry("user", "質問です", datetime(2025, 1, 15, 10, 0, 0)),
            _make_entry("assistant", "回答です", datetime(2025, 1, 15, 10, 1, 0)),
            _make_entry("system", "システムメッセージ", datetime(2025, 1, 15, 10, 2, 0)),
        ]
        prompt = _build_prompt(conversation_history=entries)
        assert "[user]" in prompt
        assert "[assistant]" in prompt
        assert "[system]" in prompt
        assert "質問です" in prompt
        assert "回答です" in prompt
        assert "システムメッセージ" in prompt

    def test_entry_format_structured(self):
        """Each entry shows role, timestamp, and content in structured format (Req 1.4)."""
        entry = _make_entry("assistant", "応答テスト", datetime(2025, 6, 1, 14, 30, 45))
        prompt = _build_prompt(conversation_history=[entry])
        assert "- [assistant] 2025-06-01 14:30:45: 応答テスト" in prompt

    def test_existing_callers_unaffected(self):
        """Omitting conversation_history still works (backward compat)."""
        prompt = build_system_prompt(
            constitution=None,
            wip=[],
            recent_decisions=[],
            budget_spent=0.0,
            budget_limit=10.0,
        )
        assert "## 会話履歴" in prompt
        assert "会話履歴なし" in prompt
