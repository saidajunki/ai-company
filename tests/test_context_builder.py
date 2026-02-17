"""Tests for context_builder — conversation history section.

Requirements: 1.2, 1.4
"""

from __future__ import annotations

from datetime import datetime

from context_builder import build_system_prompt
from models import ConversationEntry, ResearchNote


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


def _make_research_note(
    summary: str = "テスト要約",
    source_url: str = "https://example.com/article",
    retrieved_at: datetime | None = None,
    **kwargs,
) -> ResearchNote:
    return ResearchNote(
        query="テストクエリ",
        source_url=source_url,
        title="テスト記事",
        snippet="テストスニペット",
        summary=summary,
        retrieved_at=retrieved_at or datetime(2025, 7, 1, 12, 0, 0),
        **kwargs,
    )


class TestResearchNotesSection:
    """Tests for the リサーチノート section (Requirements 3.1, 3.2, 3.3)."""

    def test_no_notes_shows_placeholder(self):
        """Req 3.3: リサーチノートなし when no notes."""
        prompt = _build_prompt(research_notes=None)
        assert "## リサーチノート" in prompt
        assert "リサーチノートなし" in prompt

    def test_empty_list_shows_placeholder(self):
        """Req 3.3: リサーチノートなし when empty list."""
        prompt = _build_prompt(research_notes=[])
        assert "リサーチノートなし" in prompt

    def test_single_note_appears(self):
        """Req 3.2: retrieved_at, source_url, summary are displayed."""
        note = _make_research_note(
            summary="AI市場の動向分析",
            source_url="https://example.com/ai-trends",
            retrieved_at=datetime(2025, 7, 1, 14, 30, 0),
        )
        prompt = _build_prompt(research_notes=[note])
        assert "## リサーチノート" in prompt
        assert "2025-07-01 14:30:00" in prompt
        assert "https://example.com/ai-trends" in prompt
        assert "AI市場の動向分析" in prompt
        assert "リサーチノートなし" not in prompt

    def test_multiple_notes_all_appear(self):
        """Req 3.1: multiple notes are included."""
        notes = [
            _make_research_note(summary=f"要約{i}", source_url=f"https://example.com/{i}")
            for i in range(3)
        ]
        prompt = _build_prompt(research_notes=notes)
        for i in range(3):
            assert f"要約{i}" in prompt
            assert f"https://example.com/{i}" in prompt

    def test_max_10_notes(self):
        """Req 3.1: at most 10 notes are displayed."""
        notes = [
            _make_research_note(summary=f"要約{i}", source_url=f"https://example.com/{i}")
            for i in range(15)
        ]
        prompt = _build_prompt(research_notes=notes)
        for i in range(10):
            assert f"要約{i}" in prompt
        for i in range(10, 15):
            assert f"要約{i}" not in prompt

    def test_section_position_before_conversation(self):
        """Research section appears between budget and conversation."""
        note = _make_research_note()
        prompt = _build_prompt(research_notes=[note])
        budget_pos = prompt.index("## 予算状況")
        research_pos = prompt.index("## リサーチノート")
        conversation_pos = prompt.index("## 会話履歴")
        assert budget_pos < research_pos < conversation_pos

    def test_existing_callers_unaffected(self):
        """Omitting research_notes still works (backward compat)."""
        prompt = build_system_prompt(
            constitution=None,
            wip=[],
            recent_decisions=[],
            budget_spent=0.0,
            budget_limit=10.0,
        )
        assert "## リサーチノート" in prompt
        assert "リサーチノートなし" in prompt


# --- Initiative & Strategy section tests (Req 5.1, 5.2, 5.3) ---

from models import InitiativeEntry, StrategyDirection


def _make_initiative(
    title: str = "テストイニシアチブ",
    description: str = "テスト説明",
    status: str = "in_progress",
    initiative_id: str = "ini-001",
) -> InitiativeEntry:
    return InitiativeEntry(
        initiative_id=initiative_id,
        title=title,
        description=description,
        status=status,
        created_at=datetime(2025, 7, 1, 12, 0, 0),
        updated_at=datetime(2025, 7, 1, 12, 0, 0),
    )


class TestInitiativeSection:
    """Tests for the イニシアチブ section (Requirements 5.1, 5.3)."""

    def test_no_initiatives_shows_placeholder(self):
        """Req 5.3: アクティブなイニシアチブなし when None."""
        prompt = _build_prompt(active_initiatives=None)
        assert "## イニシアチブ" in prompt
        assert "アクティブなイニシアチブなし" in prompt

    def test_empty_list_shows_placeholder(self):
        """Req 5.3: アクティブなイニシアチブなし when empty list."""
        prompt = _build_prompt(active_initiatives=[])
        assert "アクティブなイニシアチブなし" in prompt

    def test_single_initiative_appears(self):
        """Req 5.1: active initiative title and description are displayed."""
        ini = _make_initiative(title="OSSツール公開", description="GitHub公開計画")
        prompt = _build_prompt(active_initiatives=[ini])
        assert "## イニシアチブ" in prompt
        assert "OSSツール公開" in prompt
        assert "GitHub公開計画" in prompt
        assert "アクティブなイニシアチブなし" not in prompt

    def test_multiple_initiatives_all_appear(self):
        """Req 5.1: multiple initiatives are listed."""
        initiatives = [
            _make_initiative(title=f"施策{i}", initiative_id=f"ini-{i}")
            for i in range(3)
        ]
        prompt = _build_prompt(active_initiatives=initiatives)
        for i in range(3):
            assert f"施策{i}" in prompt

    def test_initiative_shows_status(self):
        """Req 5.1: status is included in the output."""
        ini = _make_initiative(status="planned")
        prompt = _build_prompt(active_initiatives=[ini])
        assert "[planned]" in prompt

    def test_backward_compat_without_initiatives(self):
        """Omitting active_initiatives still works."""
        prompt = build_system_prompt(
            constitution=None,
            wip=[],
            recent_decisions=[],
            budget_spent=0.0,
            budget_limit=10.0,
        )
        assert "## イニシアチブ" in prompt
        assert "アクティブなイニシアチブなし" in prompt


class TestStrategySection:
    """Tests for the 戦略方針 section (Requirement 5.2)."""

    def test_no_strategy_shows_placeholder(self):
        prompt = _build_prompt(strategy_direction=None)
        assert "## 戦略方針" in prompt
        assert "戦略方針未設定" in prompt

    def test_empty_summary_shows_placeholder(self):
        sd = StrategyDirection(summary="")
        prompt = _build_prompt(strategy_direction=sd)
        assert "戦略方針未設定" in prompt

    def test_strategy_summary_appears(self):
        """Req 5.2: strategy summary is included in the prompt."""
        sd = StrategyDirection(summary="面白さ重視で小さく試す方針を継続")
        prompt = _build_prompt(strategy_direction=sd)
        assert "## 戦略方針" in prompt
        assert "面白さ重視で小さく試す方針を継続" in prompt
        assert "戦略方針未設定" not in prompt

    def test_backward_compat_without_strategy(self):
        """Omitting strategy_direction still works."""
        prompt = build_system_prompt(
            constitution=None,
            wip=[],
            recent_decisions=[],
            budget_spent=0.0,
            budget_limit=10.0,
        )
        assert "## 戦略方針" in prompt
        assert "戦略方針未設定" in prompt

    def test_section_order(self):
        """Initiative and strategy sections appear after budget."""
        ini = _make_initiative()
        sd = StrategyDirection(summary="テスト方針")
        prompt = _build_prompt(active_initiatives=[ini], strategy_direction=sd)
        budget_pos = prompt.index("## 予算状況")
        initiative_pos = prompt.index("## イニシアチブ")
        strategy_pos = prompt.index("## 戦略方針")
        research_pos = prompt.index("## リサーチノート")
        assert budget_pos < initiative_pos < strategy_pos < research_pos


class TestModelCatalogSection:
    """Tests for the 利用可能なモデル section added by model_catalog_text param."""

    def test_no_catalog_omits_section(self):
        prompt = _build_prompt(model_catalog_text=None)
        assert "## 利用可能なモデル" not in prompt

    def test_empty_string_omits_section(self):
        prompt = _build_prompt(model_catalog_text="")
        assert "## 利用可能なモデル" not in prompt

    def test_catalog_text_appears(self):
        catalog = "- google/gemini-2.5-flash [fast, cheap]"
        prompt = _build_prompt(model_catalog_text=catalog)
        assert "## 利用可能なモデル" in prompt
        assert catalog in prompt

    def test_catalog_includes_guidance(self):
        prompt = _build_prompt(model_catalog_text="モデル一覧テスト")
        assert "社員エージェントに委任する際、タスクに適したモデルを選択できます。" in prompt
        assert "コーディング・分析: coding/analysisカテゴリのモデル" in prompt
        assert "簡単なタスク・情報収集: fast/cheapカテゴリのモデル" in prompt
        assert "汎用タスク: generalカテゴリのモデル" in prompt

    def test_catalog_section_before_format(self):
        prompt = _build_prompt(model_catalog_text="テストカタログ")
        catalog_pos = prompt.index("## 利用可能なモデル")
        format_pos = prompt.index("## 応答フォーマット")
        assert catalog_pos < format_pos

    def test_backward_compat_without_catalog(self):
        prompt = build_system_prompt(
            constitution=None,
            wip=[],
            recent_decisions=[],
            budget_spent=0.0,
            budget_limit=10.0,
        )
        assert "## 応答フォーマット" in prompt
        assert "## 利用可能なモデル" not in prompt


class TestDelegateFormatSection:
    """Tests for delegate tag format in 応答フォーマット section."""

    def test_delegate_tag_in_format(self):
        prompt = _build_prompt()
        assert "<delegate>" in prompt
        assert "</delegate>" in prompt

    def test_delegate_model_format_explained(self):
        prompt = _build_prompt()
        assert "role:タスク説明 model=モデル名" in prompt

    def test_delegate_model_optional_note(self):
        prompt = _build_prompt()
        assert "model=は省略可能" in prompt


class TestLongTermMemorySections:
    def test_memory_sections_present(self):
        prompt = _build_prompt(
            rolling_summary="## 永続メモリ（要約）\n- pinned note",
            recalled_memories=["- [2025-01-01 00:00:00] (conversation) memory hit"],
        )
        assert "## 永続メモリ（要約）" in prompt
        assert "pinned note" in prompt
        assert "## 長期記憶（リコール）" in prompt
        assert "memory hit" in prompt
