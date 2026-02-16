"""Unit tests for AgentRegistry.

Tests: register + get round-trip, update_status, list_active filtering,
ensure_ceo idempotency, empty registry behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from agent_registry import AgentRegistry


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    return AgentRegistry(tmp_path, "test-co")


class TestRegisterAndGet:
    def test_register_returns_agent_entry(self, registry: AgentRegistry) -> None:
        agent = registry.register("a1", "Agent One", "worker", "gpt-4o", 5.0)
        assert agent.agent_id == "a1"
        assert agent.name == "Agent One"
        assert agent.role == "worker"
        assert agent.model == "gpt-4o"
        assert agent.budget_limit_usd == 5.0
        assert agent.status == "active"

    def test_get_round_trip(self, registry: AgentRegistry) -> None:
        registry.register("a1", "Agent One", "worker", "gpt-4o", 5.0)
        fetched = registry.get("a1")
        assert fetched is not None
        assert fetched.agent_id == "a1"
        assert fetched.name == "Agent One"
        assert fetched.role == "worker"
        assert fetched.model == "gpt-4o"
        assert fetched.budget_limit_usd == 5.0

    def test_get_nonexistent_returns_none(self, registry: AgentRegistry) -> None:
        assert registry.get("no-such-id") is None

    def test_register_multiple_agents(self, registry: AgentRegistry) -> None:
        registry.register("a1", "One", "worker", "gpt-4o", 1.0)
        registry.register("a2", "Two", "researcher", "claude", 2.0)
        assert registry.get("a1") is not None
        assert registry.get("a2") is not None


class TestUpdateStatus:
    def test_active_to_inactive(self, registry: AgentRegistry) -> None:
        registry.register("a1", "Agent", "worker", "gpt-4o", 5.0)
        registry.update_status("a1", "inactive")
        agent = registry.get("a1")
        assert agent is not None
        assert agent.status == "inactive"

    def test_inactive_to_active(self, registry: AgentRegistry) -> None:
        registry.register("a1", "Agent", "worker", "gpt-4o", 5.0)
        registry.update_status("a1", "inactive")
        registry.update_status("a1", "active")
        agent = registry.get("a1")
        assert agent is not None
        assert agent.status == "active"

    def test_update_nonexistent_raises(self, registry: AgentRegistry) -> None:
        with pytest.raises(ValueError, match="Agent not found"):
            registry.update_status("no-such-id", "inactive")

    def test_updated_at_changes(self, registry: AgentRegistry) -> None:
        agent = registry.register("a1", "Agent", "worker", "gpt-4o", 5.0)
        original_updated = agent.updated_at
        registry.update_status("a1", "inactive")
        updated = registry.get("a1")
        assert updated is not None
        assert updated.updated_at >= original_updated

    def test_created_at_preserved(self, registry: AgentRegistry) -> None:
        agent = registry.register("a1", "Agent", "worker", "gpt-4o", 5.0)
        registry.update_status("a1", "inactive")
        updated = registry.get("a1")
        assert updated is not None
        assert updated.created_at == agent.created_at

    def test_other_fields_preserved(self, registry: AgentRegistry) -> None:
        registry.register("a1", "Agent", "worker", "gpt-4o", 5.0)
        registry.update_status("a1", "inactive")
        agent = registry.get("a1")
        assert agent is not None
        assert agent.name == "Agent"
        assert agent.role == "worker"
        assert agent.model == "gpt-4o"
        assert agent.budget_limit_usd == 5.0


class TestListActive:
    def test_all_active(self, registry: AgentRegistry) -> None:
        registry.register("a1", "One", "worker", "gpt-4o", 1.0)
        registry.register("a2", "Two", "researcher", "claude", 2.0)
        active = registry.list_active()
        assert len(active) == 2

    def test_filters_inactive(self, registry: AgentRegistry) -> None:
        registry.register("a1", "One", "worker", "gpt-4o", 1.0)
        registry.register("a2", "Two", "researcher", "claude", 2.0)
        registry.update_status("a1", "inactive")
        active = registry.list_active()
        assert len(active) == 1
        assert active[0].agent_id == "a2"

    def test_empty_registry(self, registry: AgentRegistry) -> None:
        assert registry.list_active() == []

    def test_all_inactive(self, registry: AgentRegistry) -> None:
        registry.register("a1", "One", "worker", "gpt-4o", 1.0)
        registry.update_status("a1", "inactive")
        assert registry.list_active() == []


class TestEnsureCeo:
    def test_creates_ceo_when_missing(self, registry: AgentRegistry) -> None:
        ceo = registry.ensure_ceo("gpt-4o")
        assert ceo.agent_id == "ceo"
        assert ceo.name == "CEO AI"
        assert ceo.role == "ceo"
        assert ceo.model == "gpt-4o"
        assert ceo.budget_limit_usd == 10.0
        assert ceo.status == "active"

    def test_idempotent(self, registry: AgentRegistry) -> None:
        ceo1 = registry.ensure_ceo("gpt-4o")
        ceo2 = registry.ensure_ceo("gpt-4o")
        assert ceo1.agent_id == ceo2.agent_id
        assert ceo1.created_at == ceo2.created_at

    def test_does_not_duplicate(self, registry: AgentRegistry) -> None:
        registry.ensure_ceo("gpt-4o")
        registry.ensure_ceo("gpt-4o")
        active = registry.list_active()
        ceo_entries = [a for a in active if a.agent_id == "ceo"]
        assert len(ceo_entries) == 1


class TestEmptyRegistry:
    def test_get_returns_none(self, registry: AgentRegistry) -> None:
        assert registry.get("anything") is None

    def test_list_active_empty(self, registry: AgentRegistry) -> None:
        assert registry.list_active() == []
