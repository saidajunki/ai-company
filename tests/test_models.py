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
