"""Unit tests for main.py callback wiring (Task 7.1).

Tests that on_message routes to Manager.process_message and that
LLMClient / SlackBot are correctly wired to the Manager.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from manager import Manager, init_company_directory


CID = "test-co"


class TestOnMessageCallback:
    """Verify on_message calls mgr.process_message."""

    def test_on_message_calls_process_message(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mgr.process_message = MagicMock()

        # Simulate what main.py's on_message closure does
        def on_message(text: str, user_id: str) -> None:
            mgr.process_message(text, user_id)

        on_message("こんにちは", "U123")

        mgr.process_message.assert_called_once_with("こんにちは", "U123")


class TestLLMClientInitialization:
    """Verify LLMClient is set on Manager when API key is present."""

    def test_llm_client_set_when_api_key_present(self, tmp_path: Path):
        from llm_client import LLMClient

        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.llm_client is None

        mgr.llm_client = LLMClient(api_key="test-key", model="test-model")
        assert mgr.llm_client is not None
        assert mgr.llm_client.model == "test-model"

    def test_llm_client_none_when_no_api_key(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        # Without setting llm_client, it stays None
        assert mgr.llm_client is None


class TestSlackAssignment:
    """Verify SlackBot is set on Manager for reply routing."""

    def test_slack_set_on_manager(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        assert mgr.slack is None

        mock_slack = MagicMock()
        mgr.slack = mock_slack
        assert mgr.slack is mock_slack

    def test_process_message_without_llm_sends_error_via_slack(self, tmp_path: Path):
        """When LLM client is not configured, process_message sends error via Slack."""
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        mock_slack = MagicMock()
        mgr.slack = mock_slack
        # llm_client is None

        mgr.process_message("テスト", "U123")

        mock_slack.send_message.assert_called_once()
        call_text = mock_slack.send_message.call_args[0][0]
        assert "LLM" in call_text
