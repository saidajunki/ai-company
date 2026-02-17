"""Unit tests for SubAgentRunner (Task 9.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_registry import AgentRegistry
from llm_client import LLMClient, LLMError, LLMResponse
from manager import Manager, init_company_directory
from models import LedgerEvent
from sub_agent_runner import SubAgentRunner, DEFAULT_WIP_LIMIT, MAX_CONVERSATION_TURNS


CID = "test-co"


def _make_mock_llm() -> MagicMock:
    """Create a standard mocked LLM client."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.model = "test-model"
    mock_llm.api_key = "test-api-key"
    mock_llm.timeout = 30.0
    mock_llm.chat.return_value = LLMResponse(
        content="<done>タスク完了</done>",
        input_tokens=100,
        output_tokens=50,
        model="test-model",
        finish_reason="stop",
    )
    return mock_llm


def _make_manager(tmp_path: Path) -> Manager:
    """Create a Manager with mocked LLM client and agent_registry."""
    init_company_directory(tmp_path, CID)
    mgr = Manager(tmp_path, CID)

    # Set up agent_registry
    mgr.agent_registry = AgentRegistry(tmp_path, CID)

    # Set up mocked LLM client
    mgr.llm_client = _make_mock_llm()
    mgr.slack = MagicMock()
    return mgr


# ---------------------------------------------------------------------------
# _build_sub_agent_prompt
# ---------------------------------------------------------------------------

class TestBuildSubAgentPrompt:
    def test_prompt_contains_role(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)
        prompt = runner._build_sub_agent_prompt("リサーチャー", "市場調査を行う")
        assert "リサーチャー" in prompt

    def test_prompt_contains_task_description(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)
        prompt = runner._build_sub_agent_prompt("開発者", "APIを実装する")
        assert "APIを実装する" in prompt

    def test_prompt_contains_both_role_and_task(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)
        prompt = runner._build_sub_agent_prompt("テスター", "テストを書く")
        assert "テスター" in prompt
        assert "テストを書く" in prompt

    def test_prompt_contains_response_format(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)
        prompt = runner._build_sub_agent_prompt("dev", "task")
        assert "<done>" in prompt
        assert "<shell>" in prompt


# ---------------------------------------------------------------------------
# WIP limit enforcement
# ---------------------------------------------------------------------------

class TestWipLimit:
    def test_spawn_rejected_when_wip_full(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)

        # Register 3 active non-CEO agents
        for i in range(DEFAULT_WIP_LIMIT):
            mgr.agent_registry.register(
                agent_id=f"sub-{i:06d}",
                name=f"Agent {i}",
                role="worker",
                model="test-model",
                budget_limit_usd=1.0,
            )

        result = runner.spawn("New Agent", "worker", "some task")
        assert "WIP制限" in result

    def test_spawn_allowed_when_under_limit(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        # Register 2 active non-CEO agents (under limit of 3)
        for i in range(2):
            mgr.agent_registry.register(
                agent_id=f"sub-{i:06d}",
                name=f"Agent {i}",
                role="worker",
                model="test-model",
                budget_limit_usd=1.0,
            )

        result = runner.spawn("New Agent", "worker", "some task")
        assert "WIP制限" not in result

    def test_ceo_not_counted_in_wip(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        # Register CEO + 2 non-CEO agents
        mgr.agent_registry.ensure_ceo("test-model")
        for i in range(2):
            mgr.agent_registry.register(
                agent_id=f"sub-{i:06d}",
                name=f"Agent {i}",
                role="worker",
                model="test-model",
                budget_limit_usd=1.0,
            )

        result = runner.spawn("New Agent", "worker", "some task")
        assert "WIP制限" not in result


# ---------------------------------------------------------------------------
# Basic spawn flow
# ---------------------------------------------------------------------------

class TestSpawnFlow:
    def test_spawn_returns_done_content(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        result = runner.spawn("Worker", "developer", "コードを書く")
        assert result == "タスク完了"

    def test_spawn_includes_internal_knowledge_context(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く")

        first_messages = mock_sub.chat.call_args_list[0][0][0]
        system_prompt = first_messages[0]["content"]
        assert "## 社内ナレッジ参照先" in system_prompt
        assert "vision.md" in system_prompt
        assert "## 永続メモリ（要約）" in system_prompt
        assert "## 長期記憶（リコール）" in system_prompt

    def test_spawn_registers_agent(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く")

        # Should have at least one sub-agent registered
        all_agents = mgr.agent_registry._list_all()
        sub_agents = [a for a in all_agents if a.agent_id.startswith("sub-")]
        assert len(sub_agents) >= 1

    def test_spawn_records_llm_cost(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く")

        # At least one ledger event with sub-agent's agent_id
        sub_events = [
            e for e in mgr.state.ledger_events
            if e.agent_id.startswith("sub-")
        ]
        assert len(sub_events) >= 1

    def test_spawn_deactivates_agent_after_completion(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く")

        all_agents = mgr.agent_registry._list_all()
        sub_agents = [a for a in all_agents if a.agent_id.startswith("sub-")]
        assert len(sub_agents) >= 1
        # After completion, agent should be inactive
        assert sub_agents[0].status == "inactive"

    def test_spawn_with_llm_error(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        mock_sub.chat.return_value = LLMError(
            error_type="api_error",
            message="API failure",
        )
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        result = runner.spawn("Worker", "developer", "コードを書く")
        assert "LLMエラー" in result

    def test_spawn_without_llm_client(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.llm_client = None
        runner = SubAgentRunner(mgr)

        result = runner.spawn("Worker", "developer", "コードを書く")
        assert "LLMクライアント" in result


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    def test_budget_exceeded_stops_agent(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)

        call_count = 0
        mock_sub = _make_mock_llm()

        def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            # First call succeeds with shell command to trigger follow-up
            if call_count == 1:
                return LLMResponse(
                    content="<shell>echo hello</shell>",
                    input_tokens=100,
                    output_tokens=50,
                    model="test-model",
                    finish_reason="stop",
                )
            return LLMResponse(
                content="<done>完了</done>",
                input_tokens=100,
                output_tokens=50,
                model="test-model",
                finish_reason="stop",
            )

        mock_sub.chat.side_effect = mock_chat
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        # Pre-populate ledger with costs that will exceed budget on first check
        # We need to know the agent_id, so we patch uuid4
        with patch("sub_agent_runner.uuid4") as mock_uuid:
            mock_uuid.return_value = MagicMock(hex="abcdef1234567890")
            agent_id = "sub-abcdef"

            # Add existing cost events for this agent
            now = datetime.now(timezone.utc)
            for _ in range(5):
                event = LedgerEvent(
                    timestamp=now,
                    event_type="llm_call",
                    agent_id=agent_id,
                    task_id="test",
                    provider="openrouter",
                    model="test-model",
                    input_tokens=100,
                    output_tokens=50,
                    unit_price_usd_per_1k_input_tokens=0.01,
                    unit_price_usd_per_1k_output_tokens=0.03,
                    price_retrieved_at=now,
                    estimated_cost_usd=0.5,
                )
                mgr.state.ledger_events.append(event)

            result = runner.spawn("Worker", "developer", "コードを書く", budget_limit_usd=1.0)
            assert "予算上限" in result


# ---------------------------------------------------------------------------
# Shell command execution
# ---------------------------------------------------------------------------

class TestShellExecution:
    def test_shell_command_triggers_followup(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)

        call_count = 0
        mock_sub = _make_mock_llm()

        def mock_chat(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="<shell>echo hello</shell>",
                    input_tokens=100,
                    output_tokens=50,
                    model="test-model",
                    finish_reason="stop",
                )
            return LLMResponse(
                content="<done>シェル実行完了</done>",
                input_tokens=100,
                output_tokens=50,
                model="test-model",
                finish_reason="stop",
            )

        mock_sub.chat.side_effect = mock_chat
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        result = runner.spawn("Worker", "developer", "echo実行")
        assert result == "シェル実行完了"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Max conversation turns
# ---------------------------------------------------------------------------

class TestMaxTurns:
    def test_stops_at_max_turns(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        runner = SubAgentRunner(mgr)

        mock_sub = _make_mock_llm()
        # Always return shell commands to keep looping
        mock_sub.chat.return_value = LLMResponse(
            content="<shell>echo loop</shell>",
            input_tokens=10,
            output_tokens=10,
            model="test-model",
            finish_reason="stop",
        )
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        result = runner.spawn("Worker", "developer", "無限ループ")
        assert "最大会話ターン数" in result
        assert mock_sub.chat.call_count == MAX_CONVERSATION_TURNS


# ---------------------------------------------------------------------------
# Model selection (Task 2.1)
# ---------------------------------------------------------------------------

class TestModelSelection:
    """Tests for spawn() model parameter and _create_llm_client()."""

    def test_spawn_with_explicit_model_registers_that_model(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く", model="google/gemini-2.5-flash")

        all_agents = mgr.agent_registry._list_all()
        sub_agents = [a for a in all_agents if a.agent_id.startswith("sub-")]
        assert len(sub_agents) == 1
        assert sub_agents[0].model == "google/gemini-2.5-flash"

    def test_spawn_without_model_falls_back_to_ceo_model(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く")

        all_agents = mgr.agent_registry._list_all()
        sub_agents = [a for a in all_agents if a.agent_id.startswith("sub-")]
        assert len(sub_agents) == 1
        assert sub_agents[0].model == "test-model"

    def test_spawn_with_empty_string_model_falls_back_to_ceo(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mock_sub = _make_mock_llm()
        runner = SubAgentRunner(mgr)
        runner._create_llm_client = MagicMock(return_value=mock_sub)

        runner.spawn("Worker", "developer", "コードを書く", model="")

        all_agents = mgr.agent_registry._list_all()
        sub_agents = [a for a in all_agents if a.agent_id.startswith("sub-")]
        assert len(sub_agents) == 1
        assert sub_agents[0].model == "test-model"

    def test_spawn_without_llm_client_uses_unknown_model(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.llm_client = None
        runner = SubAgentRunner(mgr)

        runner.spawn("Worker", "developer", "コードを書く")

        all_agents = mgr.agent_registry._list_all()
        sub_agents = [a for a in all_agents if a.agent_id.startswith("sub-")]
        assert len(sub_agents) == 1
        assert sub_agents[0].model == "unknown"

    def test_create_llm_client_returns_independent_instance(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        # Use a real LLMClient for the manager so _create_llm_client works
        mgr.llm_client = LLMClient(api_key="test-key", model="ceo-model", timeout=60.0)
        runner = SubAgentRunner(mgr)

        sub_client = runner._create_llm_client("sub-model")

        assert sub_client is not None
        assert isinstance(sub_client, LLMClient)
        assert sub_client.model == "sub-model"
        assert sub_client.api_key == "test-key"
        assert sub_client.timeout == 60.0
        assert sub_client is not mgr.llm_client

    def test_create_llm_client_returns_none_when_no_manager_client(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        mgr.llm_client = None
        runner = SubAgentRunner(mgr)

        result = runner._create_llm_client("some-model")

        assert result is None
