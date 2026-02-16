"""Unit tests for ServiceRegistry.

Tests: register + get round-trip, deduplication (same name updates),
list_all, empty registry behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from service_registry import ServiceRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ServiceRegistry:
    return ServiceRegistry(tmp_path, "test-co")


class TestRegisterAndGet:
    def test_register_returns_service_entry(self, registry: ServiceRegistry) -> None:
        svc = registry.register("my-tool", "A useful tool", "ceo")
        assert svc.name == "my-tool"
        assert svc.description == "A useful tool"
        assert svc.agent_id == "ceo"
        assert svc.status == "active"

    def test_get_round_trip(self, registry: ServiceRegistry) -> None:
        registry.register("my-tool", "A useful tool", "ceo")
        fetched = registry.get("my-tool")
        assert fetched is not None
        assert fetched.name == "my-tool"
        assert fetched.description == "A useful tool"
        assert fetched.agent_id == "ceo"
        assert fetched.status == "active"

    def test_get_nonexistent_returns_none(self, registry: ServiceRegistry) -> None:
        assert registry.get("no-such-service") is None

    def test_register_multiple_services(self, registry: ServiceRegistry) -> None:
        registry.register("svc-a", "Service A", "ceo")
        registry.register("svc-b", "Service B", "agent-1")
        assert registry.get("svc-a") is not None
        assert registry.get("svc-b") is not None


class TestDeduplication:
    def test_same_name_updates_description(self, registry: ServiceRegistry) -> None:
        registry.register("my-tool", "Version 1", "ceo")
        registry.register("my-tool", "Version 2", "ceo")
        fetched = registry.get("my-tool")
        assert fetched is not None
        assert fetched.description == "Version 2"

    def test_same_name_updates_agent_id(self, registry: ServiceRegistry) -> None:
        registry.register("my-tool", "A tool", "ceo")
        registry.register("my-tool", "A tool", "agent-1")
        fetched = registry.get("my-tool")
        assert fetched is not None
        assert fetched.agent_id == "agent-1"

    def test_same_name_appears_once_in_list_all(self, registry: ServiceRegistry) -> None:
        registry.register("my-tool", "V1", "ceo")
        registry.register("my-tool", "V2", "ceo")
        all_services = registry.list_all()
        names = [s.name for s in all_services]
        assert names.count("my-tool") == 1

    def test_created_at_preserved_on_update(self, registry: ServiceRegistry) -> None:
        original = registry.register("my-tool", "V1", "ceo")
        registry.register("my-tool", "V2", "ceo")
        updated = registry.get("my-tool")
        assert updated is not None
        assert updated.created_at == original.created_at

    def test_updated_at_changes_on_update(self, registry: ServiceRegistry) -> None:
        original = registry.register("my-tool", "V1", "ceo")
        updated = registry.register("my-tool", "V2", "ceo")
        assert updated.updated_at >= original.updated_at


class TestListAll:
    def test_multiple_services(self, registry: ServiceRegistry) -> None:
        registry.register("svc-a", "A", "ceo")
        registry.register("svc-b", "B", "agent-1")
        all_services = registry.list_all()
        assert len(all_services) == 2
        names = {s.name for s in all_services}
        assert names == {"svc-a", "svc-b"}

    def test_empty_registry(self, registry: ServiceRegistry) -> None:
        assert registry.list_all() == []


class TestEmptyRegistry:
    def test_get_returns_none(self, registry: ServiceRegistry) -> None:
        assert registry.get("anything") is None

    def test_list_all_empty(self, registry: ServiceRegistry) -> None:
        assert registry.list_all() == []
