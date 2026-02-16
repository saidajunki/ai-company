"""Tests for InitiativePlanner.

Requirements: 1.1, 1.2, 1.3, 1.4, 3.4
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from initiative_planner import InitiativePlanner, _HIGH_COST_THRESHOLD
from initiative_store import InitiativeStore
from llm_client import LLMResponse, LLMError
from models import (
    ConstitutionModel,
    InitiativeEntry,
    InitiativeScores,
    StrategyDirection,
)
from strategy_analyzer import StrategyAnalyzer
from task_queue import TaskQueue
from vision_loader import VisionLoader

NOW = datetime.now(timezone.utc)


def _make_llm_response(initiatives: list[dict]) -> LLMResponse:
    """Create a mock LLM response with JSON initiative data."""
    content = "```json\n" + json.dumps(initiatives, ensure_ascii=False) + "\n```"
    return LLMResponse(
        content=content,
        input_tokens=100,
        output_tokens=200,
        model="test-model",
        finish_reason="stop",
    )


def _default_initiative_data(
    title: str = "OSSツール公開",
    description: str = "小さなCLIツールを作成して公開する",
    first_step: str = "READMEのドラフトを作成する",
    interestingness: int = 20,
    cost_efficiency: int = 18,
    realism: int = 22,
    evolvability: int = 15,
) -> dict:
    return {
        "title": title,
        "description": description,
        "estimated_cost_usd": 2.0,
        "scores": {
            "interestingness": interestingness,
            "cost_efficiency": cost_efficiency,
            "realism": realism,
            "evolvability": evolvability,
        },
        "first_step": first_step,
    }


def _make_manager(tmp_path: Path) -> MagicMock:
    """Create a mock Manager with required attributes."""
    manager = MagicMock()
    manager.base_dir = tmp_path
    manager.company_id = "test-co"
    manager.vision_loader = VisionLoader(tmp_path, "test-co")
    manager.task_queue = TaskQueue(tmp_path, "test-co")
    manager.state = SimpleNamespace(
        constitution=ConstitutionModel(),
    )
    manager.llm_client = MagicMock()
    manager.record_llm_call = MagicMock()
    return manager


def _make_planner(tmp_path: Path, manager: MagicMock | None = None) -> tuple[InitiativePlanner, MagicMock]:
    """Create an InitiativePlanner with mocked dependencies."""
    if manager is None:
        manager = _make_manager(tmp_path)
    store = InitiativeStore(tmp_path, "test-co")
    from creator_review_store import CreatorReviewStore
    crs = CreatorReviewStore(tmp_path, "test-co")
    analyzer = StrategyAnalyzer(crs, store)
    planner = InitiativePlanner(manager, store, analyzer)
    return planner, manager


class TestPlan:
    """plan() メソッドのテスト."""

    def test_generates_one_initiative(self, tmp_path):
        """1件のイニシアチブを生成できる."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert len(result) == 1
        assert result[0].title == "OSSツール公開"

    def test_caps_at_one_per_cycle(self, tmp_path):
        """LLMが複数件返しても1件に制限する (MAX_INITIATIVES_PER_CYCLE=1)."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data(title=f"施策{i}") for i in range(5)]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert len(result) == 1

    def test_required_fields_present(self, tmp_path):
        """各イニシアチブに必須フィールドが含まれる."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert len(result) == 1
        entry = result[0]
        assert entry.title
        assert entry.description
        assert entry.first_step
        assert entry.estimated_scores is not None
        assert entry.estimated_scores.interestingness >= 0
        assert entry.estimated_scores.cost_efficiency >= 0
        assert entry.estimated_scores.realism >= 0
        assert entry.estimated_scores.evolvability >= 0

    def test_saves_to_initiative_store(self, tmp_path):
        """イニシアチブがInitiativeStoreに保存される."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        store = InitiativeStore(tmp_path, "test-co")
        stored = store.get(result[0].initiative_id)
        assert stored is not None
        assert stored.title == "OSSツール公開"

    def test_adds_first_step_to_task_queue(self, tmp_path):
        """最初の一手がタスクキューに追加される."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        tasks = manager.task_queue.list_all()
        assert len(tasks) == 1
        assert tasks[0].description == "READMEのドラフトを作成する"

    def test_high_cost_sets_consulting_status(self, tmp_path):
        """cost_efficiencyが閾値以下の場合、consultingステータスになる."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data(cost_efficiency=3)]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert len(result) == 1
        assert result[0].status == "consulting"

    def test_consulting_does_not_add_task(self, tmp_path):
        """consultingステータスのイニシアチブはタスクキューに追加しない."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data(cost_efficiency=3)]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        planner.plan()

        tasks = manager.task_queue.list_all()
        assert len(tasks) == 0

    def test_cost_efficiency_at_threshold_is_consulting(self, tmp_path):
        """cost_efficiencyがちょうど閾値の場合もconsulting."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data(cost_efficiency=_HIGH_COST_THRESHOLD)]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert result[0].status == "consulting"

    def test_cost_efficiency_above_threshold_is_planned(self, tmp_path):
        """cost_efficiencyが閾値超の場合はplanned."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data(cost_efficiency=_HIGH_COST_THRESHOLD + 1)]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert result[0].status == "planned"

    def test_records_llm_call(self, tmp_path):
        """LLM呼び出しが記録される."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        planner.plan()

        manager.record_llm_call.assert_called_once()

    def test_llm_error_returns_empty(self, tmp_path):
        """LLMエラー時は空リストを返す."""
        planner, manager = _make_planner(tmp_path)
        manager.llm_client.chat.return_value = LLMError(
            error_type="api_error",
            message="Server error",
            status_code=500,
        )

        result = planner.plan()

        assert result == []

    def test_no_llm_client_returns_empty(self, tmp_path):
        """LLMクライアント未設定時は空リストを返す."""
        planner, manager = _make_planner(tmp_path)
        manager.llm_client = None

        result = planner.plan()

        assert result == []

    def test_skips_items_with_missing_title(self, tmp_path):
        """タイトルが空のアイテムはスキップする."""
        planner, manager = _make_planner(tmp_path)
        data = [
            _default_initiative_data(title=""),
            _default_initiative_data(title="有効な施策"),
        ]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert len(result) == 1
        assert result[0].title == "有効な施策"

    def test_first_initiative_selected_when_multiple_returned(self, tmp_path):
        """LLMが複数返しても最初の1件のみ選択される."""
        planner, manager = _make_planner(tmp_path)
        data = [
            _default_initiative_data(title="高コスト施策", cost_efficiency=2),
            _default_initiative_data(title="通常施策", cost_efficiency=20),
        ]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()

        assert len(result) == 1
        assert result[0].title == "高コスト施策"
        assert result[0].status == "consulting"


class TestPlanRateLimiting:
    """plan() のレート制限テスト (Requirements: 4.1, 4.2, 4.3, 4.4, 4.5)."""

    def test_cooldown_blocks_second_call(self, tmp_path):
        """クールダウン期間中は空リストを返す (Req 4.1, 4.2)."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        # First call succeeds
        result1 = planner.plan()
        assert len(result1) == 1

        # Second call within cooldown returns empty
        # (the first call saved a "planned" initiative AND set _last_planned_at)
        result2 = planner.plan()
        assert result2 == []

    def test_cooldown_expires_allows_planning(self, tmp_path):
        """クールダウン期間経過後は計画可能 (Req 4.1, 4.5)."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        # Simulate past planning
        from datetime import timedelta
        planner._last_planned_at = datetime.now(timezone.utc) - timedelta(minutes=31)

        # Should be allowed (cooldown expired), but active initiative check may block
        # So we need no active initiatives in the store
        result = planner.plan()
        assert len(result) == 1

    def test_cooldown_not_expired_returns_empty(self, tmp_path):
        """クールダウン未経過時は空リストを返す (Req 4.2)."""
        planner, manager = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        from datetime import timedelta
        planner._last_planned_at = datetime.now(timezone.utc) - timedelta(minutes=10)

        result = planner.plan()
        assert result == []
        # LLM should NOT have been called
        manager.llm_client.chat.assert_not_called()

    def test_active_planned_initiative_blocks_planning(self, tmp_path):
        """planned ステータスのイニシアチブが存在する場合はスキップ (Req 4.4)."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        # Pre-save a planned initiative
        existing = InitiativeEntry(
            initiative_id="existing01",
            title="既存施策",
            description="進行中の施策",
            status="planned",
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(existing)

        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()
        assert result == []
        manager.llm_client.chat.assert_not_called()

    def test_active_in_progress_initiative_blocks_planning(self, tmp_path):
        """in_progress ステータスのイニシアチブが存在する場合はスキップ (Req 4.4)."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        existing = InitiativeEntry(
            initiative_id="existing02",
            title="進行中施策",
            description="進行中の施策",
            status="in_progress",
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(existing)

        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()
        assert result == []

    def test_completed_initiative_does_not_block(self, tmp_path):
        """completed ステータスのイニシアチブは計画をブロックしない."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        existing = InitiativeEntry(
            initiative_id="done01",
            title="完了施策",
            description="完了した施策",
            status="completed",
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(existing)

        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()
        assert len(result) == 1

    def test_consulting_initiative_does_not_block(self, tmp_path):
        """consulting ステータスのイニシアチブは計画をブロックしない."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        existing = InitiativeEntry(
            initiative_id="consult01",
            title="相談待ち施策",
            description="相談待ちの施策",
            status="consulting",
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(existing)

        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()
        assert len(result) == 1

    def test_max_one_initiative_per_cycle(self, tmp_path):
        """1サイクルあたり最大1件のイニシアチブ (Req 4.3)."""
        planner, manager = _make_planner(tmp_path)
        data = [
            _default_initiative_data(title="施策A"),
            _default_initiative_data(title="施策B"),
            _default_initiative_data(title="施策C"),
        ]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        result = planner.plan()
        assert len(result) == 1
        assert result[0].title == "施策A"

    def test_last_planned_at_set_on_success(self, tmp_path):
        """成功時に _last_planned_at が設定される (Req 4.5)."""
        planner, manager = _make_planner(tmp_path)
        assert planner._last_planned_at is None

        data = [_default_initiative_data()]
        manager.llm_client.chat.return_value = _make_llm_response(data)

        planner.plan()
        assert planner._last_planned_at is not None

    def test_last_planned_at_not_set_on_empty_result(self, tmp_path):
        """結果が空の場合は _last_planned_at が設定されない."""
        planner, manager = _make_planner(tmp_path)
        manager.llm_client = None

        planner.plan()
        assert planner._last_planned_at is None

    def test_cooldown_constant_is_30_minutes(self):
        """COOLDOWN_SECONDS が30分 (1800秒) であること."""
        assert InitiativePlanner.COOLDOWN_SECONDS == 1800

    def test_max_initiatives_constant_is_one(self):
        """MAX_INITIATIVES_PER_CYCLE が1であること."""
        assert InitiativePlanner.MAX_INITIATIVES_PER_CYCLE == 1


class TestGenerateRetrospective:
    """generate_retrospective() メソッドのテスト."""

    def test_generates_retrospective(self, tmp_path):
        """振り返りテキストを生成する."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        entry = InitiativeEntry(
            initiative_id="retro01",
            title="テスト施策",
            description="テスト用の施策",
            status="completed",
            estimated_scores=InitiativeScores(
                interestingness=20, cost_efficiency=15, realism=18, evolvability=22,
            ),
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(entry)

        manager.llm_client.chat.return_value = LLMResponse(
            content="成果: テストが成功しました。学び: 自動化は重要。次への示唆: CI/CDを強化。",
            input_tokens=50,
            output_tokens=100,
            model="test-model",
            finish_reason="stop",
        )

        result = planner.generate_retrospective("retro01")

        assert "テストが成功" in result
        # Verify it was saved
        updated = store.get("retro01")
        assert updated is not None
        assert updated.retrospective is not None
        assert "テストが成功" in updated.retrospective

    def test_nonexistent_initiative_returns_empty(self, tmp_path):
        """存在しないイニシアチブIDの場合は空文字を返す."""
        planner, manager = _make_planner(tmp_path)

        result = planner.generate_retrospective("nonexistent")

        assert result == ""

    def test_llm_error_returns_failure_text(self, tmp_path):
        """LLMエラー時は失敗テキストを返す."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        entry = InitiativeEntry(
            initiative_id="retro02",
            title="テスト施策",
            description="テスト用",
            status="completed",
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(entry)

        manager.llm_client.chat.return_value = LLMError(
            error_type="api_error",
            message="Server error",
        )

        result = planner.generate_retrospective("retro02")

        assert "失敗" in result

    def test_no_llm_client_returns_failure(self, tmp_path):
        """LLMクライアント未設定時は失敗テキストを返す."""
        planner, manager = _make_planner(tmp_path)
        store = InitiativeStore(tmp_path, "test-co")

        entry = InitiativeEntry(
            initiative_id="retro03",
            title="テスト施策",
            description="テスト用",
            status="completed",
            created_at=NOW,
            updated_at=NOW,
        )
        store.save(entry)
        manager.llm_client = None

        result = planner.generate_retrospective("retro03")

        assert "失敗" in result


class TestParseInitiatives:
    """_parse_initiatives() のテスト."""

    def test_parses_json_in_code_block(self, tmp_path):
        """コードブロック内のJSONをパースする."""
        planner, _ = _make_planner(tmp_path)
        content = '```json\n[{"title":"A","description":"B","first_step":"C","scores":{"interestingness":20,"cost_efficiency":15,"realism":18,"evolvability":22}}]\n```'

        result = planner._parse_initiatives(content)

        assert len(result) == 1
        assert result[0].title == "A"

    def test_parses_raw_json_array(self, tmp_path):
        """コードブロックなしのJSON配列をパースする."""
        planner, _ = _make_planner(tmp_path)
        data = [_default_initiative_data()]
        content = json.dumps(data, ensure_ascii=False)

        result = planner._parse_initiatives(content)

        assert len(result) == 1

    def test_invalid_json_returns_empty(self, tmp_path):
        """不正なJSONの場合は空リストを返す."""
        planner, _ = _make_planner(tmp_path)

        result = planner._parse_initiatives("これはJSONではありません")

        assert result == []

    def test_clamps_scores_to_valid_range(self, tmp_path):
        """スコアが0-25の範囲にクランプされる."""
        planner, _ = _make_planner(tmp_path)
        data = [_default_initiative_data(interestingness=30, cost_efficiency=-5)]
        content = json.dumps(data, ensure_ascii=False)

        result = planner._parse_initiatives(content)

        assert result[0].estimated_scores.interestingness == 25
        assert result[0].estimated_scores.cost_efficiency == 0


class TestExtractJsonArray:
    """_extract_json_array() のテスト."""

    def test_extracts_from_code_block(self):
        content = '```json\n[{"a": 1}]\n```'
        result = InitiativePlanner._extract_json_array(content)
        assert result == [{"a": 1}]

    def test_extracts_raw_array(self):
        content = '[{"a": 1}]'
        result = InitiativePlanner._extract_json_array(content)
        assert result == [{"a": 1}]

    def test_wraps_single_object(self):
        content = '{"a": 1}'
        result = InitiativePlanner._extract_json_array(content)
        assert result == [{"a": 1}]

    def test_returns_none_for_invalid(self):
        result = InitiativePlanner._extract_json_array("not json")
        assert result is None

    def test_finds_array_in_text(self):
        content = 'Here are the results:\n[{"a": 1}]\nEnd.'
        result = InitiativePlanner._extract_json_array(content)
        assert result == [{"a": 1}]


class TestClampScore:
    def test_normal_value(self):
        assert InitiativePlanner._clamp_score(15) == 15

    def test_above_max(self):
        assert InitiativePlanner._clamp_score(30) == 25

    def test_below_min(self):
        assert InitiativePlanner._clamp_score(-5) == 0

    def test_none_returns_default(self):
        assert InitiativePlanner._clamp_score(None) == 15
