"""Unit tests for data models (Task 1.2)."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from models import (
    ConstitutionModel,
    DecisionLogEntry,
    HeartbeatState,
    LedgerEvent,
    ModelPricing,
    PricingCache,
)

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# --- ConstitutionModel ---

class TestConstitutionModel:
    def test_default_creation(self):
        c = ConstitutionModel()
        assert c.version == 1
        assert c.purpose == "研究開発中心のAI組織"
        assert c.budget.limit_usd == 10
        assert c.budget.window_minutes == 60
        assert c.work_principles.wip_limit == 3
        assert c.disclosure_policy.default == "public"

    def test_custom_values(self):
        c = ConstitutionModel(version=2, purpose="テスト組織")
        assert c.version == 2
        assert c.purpose == "テスト組織"


# --- LedgerEvent ---

class TestLedgerEvent:
    def test_valid_llm_call(self):
        e = LedgerEvent(
            timestamp=NOW,
            event_type="llm_call",
            agent_id="manager",
            task_id="task-1",
            provider="openrouter",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            unit_price_usd_per_1k_input_tokens=0.03,
            unit_price_usd_per_1k_output_tokens=0.06,
            price_retrieved_at=NOW,
            estimated_cost_usd=0.006,
        )
        assert e.event_type == "llm_call"
        assert e.estimated_cost_usd == 0.006

    def test_valid_shell_exec(self):
        e = LedgerEvent(
            timestamp=NOW,
            event_type="shell_exec",
            agent_id="manager",
            task_id="task-2",
        )
        assert e.event_type == "shell_exec"
        assert e.provider is None

    def test_llm_call_missing_required_fields(self):
        """llm_call without LLM-specific fields should fail validation."""
        with pytest.raises(ValidationError, match="llm_call event requires fields"):
            LedgerEvent(
                timestamp=NOW,
                event_type="llm_call",
                agent_id="manager",
                task_id="task-1",
            )

    def test_llm_call_partial_fields(self):
        """llm_call with only some LLM fields should fail."""
        with pytest.raises(ValidationError, match="llm_call event requires fields"):
            LedgerEvent(
                timestamp=NOW,
                event_type="llm_call",
                agent_id="manager",
                task_id="task-1",
                provider="openrouter",
                model="gpt-4",
                # missing tokens, prices, etc.
            )

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            LedgerEvent(
                timestamp=NOW,
                event_type="unknown",
                agent_id="manager",
                task_id="task-1",
            )

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            LedgerEvent(
                timestamp=NOW,
                event_type="shell_exec",
                agent_id="manager",
                task_id="task-1",
                input_tokens=-1,
            )

    def test_valid_api_call(self):
        e = LedgerEvent(
            timestamp=NOW,
            event_type="api_call",
            agent_id="manager",
            task_id="task-3",
            api_call_count=5,
            api_unit_price_usd=0.0,
            estimated_cost_usd=0.0,
        )
        assert e.api_call_count == 5

    def test_metadata_dict(self):
        e = LedgerEvent(
            timestamp=NOW,
            event_type="decision",
            agent_id="manager",
            task_id="task-4",
            metadata={"key": "value"},
        )
        assert e.metadata == {"key": "value"}



# --- DecisionLogEntry ---

class TestDecisionLogEntry:
    def test_valid_entry(self):
        e = DecisionLogEntry(
            date=date(2025, 1, 15),
            decision="OpenRouterを使用する",
            why="コスト効率が良い",
            scope="LLM呼び出し全般",
            revisit="月次レビュー",
        )
        assert e.decision == "OpenRouterを使用する"
        assert e.status is None

    def test_with_status(self):
        e = DecisionLogEntry(
            date=date(2025, 1, 15),
            decision="憲法変更: 予算上限を$20に",
            why="実験規模拡大",
            scope="budget.limit_usd",
            revisit="2週間後",
            status="proposed",
            request_id="abc-123",
            related_constitution_field="budget.limit_usd",
        )
        assert e.status == "proposed"
        assert e.request_id == "abc-123"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            DecisionLogEntry(
                date=date(2025, 1, 15),
                decision="テスト",
                # missing why, scope, revisit
            )

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            DecisionLogEntry(
                date=date(2025, 1, 15),
                decision="テスト",
                why="テスト",
                scope="テスト",
                revisit="テスト",
                status="invalid_status",
            )


# --- HeartbeatState ---

class TestHeartbeatState:
    def test_valid_heartbeat(self):
        h = HeartbeatState(
            updated_at=NOW,
            manager_pid=1234,
            status="running",
            current_wip=["task-1", "task-2"],
        )
        assert h.status == "running"
        assert len(h.current_wip) == 2

    def test_empty_wip(self):
        h = HeartbeatState(
            updated_at=NOW,
            manager_pid=1234,
            status="idle",
        )
        assert h.current_wip == []

    def test_wip_limit_of_3(self):
        """WIP は最大3件まで (Req 3.1 WIP制限)."""
        with pytest.raises(ValidationError):
            HeartbeatState(
                updated_at=NOW,
                manager_pid=1234,
                status="running",
                current_wip=["t1", "t2", "t3", "t4"],
            )

    def test_wip_exactly_3(self):
        h = HeartbeatState(
            updated_at=NOW,
            manager_pid=1234,
            status="running",
            current_wip=["t1", "t2", "t3"],
        )
        assert len(h.current_wip) == 3

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            HeartbeatState(
                updated_at=NOW,
                manager_pid=1234,
                status="crashed",
            )

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            HeartbeatState(updated_at=NOW)


# --- PricingCache ---

class TestPricingCache:
    def test_valid_cache(self):
        p = PricingCache(
            retrieved_at=NOW,
            models={
                "gpt-4": ModelPricing(
                    input_price_per_1k=0.03,
                    output_price_per_1k=0.06,
                    retrieved_at=NOW,
                ),
            },
        )
        assert "gpt-4" in p.models
        assert p.models["gpt-4"].input_price_per_1k == 0.03

    def test_empty_models(self):
        p = PricingCache(retrieved_at=NOW)
        assert p.models == {}

    def test_multiple_models(self):
        p = PricingCache(
            retrieved_at=NOW,
            models={
                "gpt-4": ModelPricing(
                    input_price_per_1k=0.03,
                    output_price_per_1k=0.06,
                    retrieved_at=NOW,
                ),
                "claude-3": ModelPricing(
                    input_price_per_1k=0.015,
                    output_price_per_1k=0.075,
                    retrieved_at=NOW,
                ),
            },
        )
        assert len(p.models) == 2


# --- New models for autonomous growth ---

from models import AgentEntry, ConversationEntry, ServiceEntry, TaskEntry


# --- ConversationEntry ---

class TestConversationEntry:
    def test_valid_user_message(self):
        e = ConversationEntry(
            timestamp=NOW,
            role="user",
            content="こんにちは",
            user_id="U123",
        )
        assert e.role == "user"
        assert e.content == "こんにちは"
        assert e.user_id == "U123"
        assert e.task_id is None

    def test_valid_assistant_message(self):
        e = ConversationEntry(
            timestamp=NOW,
            role="assistant",
            content="了解しました",
        )
        assert e.role == "assistant"
        assert e.user_id is None

    def test_valid_system_message(self):
        e = ConversationEntry(
            timestamp=NOW,
            role="system",
            content="System prompt",
            task_id="task-1",
        )
        assert e.role == "system"
        assert e.task_id == "task-1"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            ConversationEntry(
                timestamp=NOW,
                role="unknown",
                content="test",
            )

    def test_missing_content(self):
        with pytest.raises(ValidationError):
            ConversationEntry(timestamp=NOW, role="user")

    def test_json_round_trip(self):
        e = ConversationEntry(
            timestamp=NOW,
            role="user",
            content="テスト",
            user_id="U1",
            task_id="T1",
        )
        restored = ConversationEntry.model_validate_json(e.model_dump_json())
        assert restored == e


# --- TaskEntry ---

class TestTaskEntry:
    def test_valid_task(self):
        t = TaskEntry(
            task_id="t-1",
            description="テストタスク",
            priority=1,
            status="pending",
            created_at=NOW,
            updated_at=NOW,
        )
        assert t.task_id == "t-1"
        assert t.priority == 1
        assert t.agent_id == "ceo"

    def test_default_priority(self):
        t = TaskEntry(
            task_id="t-2",
            description="デフォルト優先度",
            status="pending",
            created_at=NOW,
            updated_at=NOW,
        )
        assert t.priority == 3

    def test_priority_bounds(self):
        with pytest.raises(ValidationError):
            TaskEntry(
                task_id="t-3",
                description="invalid",
                priority=0,
                status="pending",
                created_at=NOW,
                updated_at=NOW,
            )
        with pytest.raises(ValidationError):
            TaskEntry(
                task_id="t-4",
                description="invalid",
                priority=6,
                status="pending",
                created_at=NOW,
                updated_at=NOW,
            )

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            TaskEntry(
                task_id="t-5",
                description="bad status",
                status="cancelled",
                created_at=NOW,
                updated_at=NOW,
            )

    def test_with_result_and_error(self):
        t = TaskEntry(
            task_id="t-6",
            description="完了タスク",
            status="completed",
            created_at=NOW,
            updated_at=NOW,
            result="成功",
            error=None,
            agent_id="sub-1",
        )
        assert t.result == "成功"
        assert t.agent_id == "sub-1"

    def test_json_round_trip(self):
        t = TaskEntry(
            task_id="t-7",
            description="往復テスト",
            priority=2,
            status="running",
            created_at=NOW,
            updated_at=NOW,
            result="partial",
            agent_id="sub-2",
        )
        restored = TaskEntry.model_validate_json(t.model_dump_json())
        assert restored == t


# --- AgentEntry ---

class TestAgentEntry:
    def test_valid_agent(self):
        a = AgentEntry(
            agent_id="ceo",
            name="CEO AI",
            role="ceo",
            model="gpt-4",
            budget_limit_usd=10.0,
            created_at=NOW,
            updated_at=NOW,
        )
        assert a.agent_id == "ceo"
        assert a.status == "active"

    def test_inactive_agent(self):
        a = AgentEntry(
            agent_id="sub-1",
            name="Worker",
            role="researcher",
            model="gpt-3.5",
            budget_limit_usd=1.0,
            status="inactive",
            created_at=NOW,
            updated_at=NOW,
        )
        assert a.status == "inactive"

    def test_negative_budget_rejected(self):
        with pytest.raises(ValidationError):
            AgentEntry(
                agent_id="bad",
                name="Bad",
                role="test",
                model="gpt-4",
                budget_limit_usd=-1.0,
                created_at=NOW,
                updated_at=NOW,
            )

    def test_zero_budget_allowed(self):
        a = AgentEntry(
            agent_id="free",
            name="Free",
            role="test",
            model="gpt-4",
            budget_limit_usd=0.0,
            created_at=NOW,
            updated_at=NOW,
        )
        assert a.budget_limit_usd == 0.0

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            AgentEntry(
                agent_id="x",
                name="X",
                role="test",
                model="gpt-4",
                budget_limit_usd=1.0,
                status="suspended",
                created_at=NOW,
                updated_at=NOW,
            )

    def test_json_round_trip(self):
        a = AgentEntry(
            agent_id="ceo",
            name="CEO AI",
            role="ceo",
            model="gpt-4",
            budget_limit_usd=10.0,
            created_at=NOW,
            updated_at=NOW,
        )
        restored = AgentEntry.model_validate_json(a.model_dump_json())
        assert restored == a


# --- ServiceEntry ---

class TestServiceEntry:
    def test_valid_service(self):
        s = ServiceEntry(
            name="my-oss",
            description="OSSプロジェクト",
            created_at=NOW,
            updated_at=NOW,
            agent_id="ceo",
        )
        assert s.name == "my-oss"
        assert s.status == "active"

    def test_archived_service(self):
        s = ServiceEntry(
            name="old-tool",
            description="古いツール",
            status="archived",
            created_at=NOW,
            updated_at=NOW,
            agent_id="sub-1",
        )
        assert s.status == "archived"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            ServiceEntry(
                name="bad",
                description="bad",
                status="deleted",
                created_at=NOW,
                updated_at=NOW,
                agent_id="ceo",
            )

    def test_missing_agent_id(self):
        with pytest.raises(ValidationError):
            ServiceEntry(
                name="no-agent",
                description="missing agent",
                created_at=NOW,
                updated_at=NOW,
            )

    def test_json_round_trip(self):
        s = ServiceEntry(
            name="test-svc",
            description="テストサービス",
            created_at=NOW,
            updated_at=NOW,
            agent_id="ceo",
        )
        restored = ServiceEntry.model_validate_json(s.model_dump_json())
        assert restored == s


# --- ResearchNote (Req 2.1, 2.2) ---

from models import ResearchNote


class TestResearchNote:
    def test_valid_note_with_published_at(self):
        n = ResearchNote(
            query="AI company trends",
            source_url="https://example.com/article",
            title="AI Trends 2025",
            snippet="AI companies are growing...",
            summary="AI業界のトレンドまとめ",
            published_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
            retrieved_at=NOW,
        )
        assert n.query == "AI company trends"
        assert n.source_url == "https://example.com/article"
        assert n.published_at is not None

    def test_valid_note_without_published_at(self):
        """published_at が不明な場合は None を許容する (Req 2.2)."""
        n = ResearchNote(
            query="test query",
            source_url="https://example.com",
            title="Test",
            snippet="snippet",
            summary="summary",
            retrieved_at=NOW,
        )
        assert n.published_at is None

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ResearchNote(
                query="test",
                source_url="https://example.com",
                # missing title, snippet, summary, retrieved_at
            )

    def test_json_round_trip(self):
        n = ResearchNote(
            query="テストクエリ",
            source_url="https://example.com/jp",
            title="テスト記事",
            snippet="スニペット",
            summary="要約テスト",
            published_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
            retrieved_at=NOW,
        )
        restored = ResearchNote.model_validate_json(n.model_dump_json())
        assert restored == n

    def test_json_round_trip_none_published_at(self):
        n = ResearchNote(
            query="query",
            source_url="https://example.com",
            title="title",
            snippet="snippet",
            summary="summary",
            retrieved_at=NOW,
        )
        restored = ResearchNote.model_validate_json(n.model_dump_json())
        assert restored == n
        assert restored.published_at is None
