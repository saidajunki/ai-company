from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from manager import Manager, init_company_directory


def _make_manager(tmp_path: Path) -> Manager:
    init_company_directory(tmp_path, "test-co")
    mgr = Manager(tmp_path, "test-co")
    return mgr


def test_fetches_thread_context_when_missing(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    mock_slack = MagicMock()
    mock_slack.fetch_thread_context.return_value = "channel=C1 thread_ts=111.222\n- [user:U1] ..."
    mgr.slack = mock_slack

    mgr.process_message(
        "今何時ですか",
        user_id="U1",
        slack_channel="C1",
        slack_thread_ts="111.222",
        slack_thread_context=None,
    )

    mock_slack.fetch_thread_context.assert_called_once_with(
        channel="C1",
        thread_ts="111.222",
        exclude_ts=None,
    )


def test_skips_fallback_when_thread_context_provided(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    mock_slack = MagicMock()
    mgr.slack = mock_slack

    mgr.process_message(
        "今何時ですか",
        user_id="U1",
        slack_channel="C1",
        slack_thread_ts="111.222",
        slack_thread_context="channel=C1 thread_ts=111.222\n- [user:U1] already",
    )

    mock_slack.fetch_thread_context.assert_not_called()
