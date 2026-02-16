"""Unit tests for watchdog.py (standalone host-side script).

Since watchdog.py lives at project root (not in src/), we adjust sys.path.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# watchdog.py is at project root, not in src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import watchdog  # noqa: E402


# ---------------------------------------------------------------------------
# check_heartbeat
# ---------------------------------------------------------------------------

class TestCheckHeartbeat:
    def test_fresh_heartbeat_returns_false(self, tmp_path: Path):
        """A recent timestamp → not stale."""
        hb_file = tmp_path / "heartbeat.json"
        now = datetime.now(timezone.utc)
        hb_file.write_text(json.dumps({"updated_at": now.isoformat()}))
        assert watchdog.check_heartbeat(str(hb_file)) is False

    def test_stale_heartbeat_returns_true(self, tmp_path: Path):
        """Timestamp older than 20 minutes → stale."""
        hb_file = tmp_path / "heartbeat.json"
        old = datetime.now(timezone.utc) - timedelta(minutes=25)
        hb_file.write_text(json.dumps({"updated_at": old.isoformat()}))
        assert watchdog.check_heartbeat(str(hb_file)) is True

    def test_exactly_20_minutes_is_stale(self, tmp_path: Path):
        """Exactly 20 minutes elapsed → stale (>= threshold)."""
        hb_file = tmp_path / "heartbeat.json"
        old = datetime.now(timezone.utc) - timedelta(minutes=20)
        hb_file.write_text(json.dumps({"updated_at": old.isoformat()}))
        assert watchdog.check_heartbeat(str(hb_file)) is True

    def test_missing_file_returns_true(self, tmp_path: Path):
        """File doesn't exist → stale."""
        assert watchdog.check_heartbeat(str(tmp_path / "nope.json")) is True

    def test_invalid_json_returns_true(self, tmp_path: Path):
        """Malformed JSON → stale."""
        hb_file = tmp_path / "heartbeat.json"
        hb_file.write_text("not json at all{{{")
        assert watchdog.check_heartbeat(str(hb_file)) is True

    def test_missing_updated_at_key_returns_true(self, tmp_path: Path):
        """JSON without updated_at → stale."""
        hb_file = tmp_path / "heartbeat.json"
        hb_file.write_text(json.dumps({"status": "running"}))
        assert watchdog.check_heartbeat(str(hb_file)) is True

    def test_invalid_timestamp_returns_true(self, tmp_path: Path):
        """Non-parseable timestamp → stale."""
        hb_file = tmp_path / "heartbeat.json"
        hb_file.write_text(json.dumps({"updated_at": "not-a-date"}))
        assert watchdog.check_heartbeat(str(hb_file)) is True

    def test_custom_threshold(self, tmp_path: Path):
        """Custom threshold_minutes is respected."""
        hb_file = tmp_path / "heartbeat.json"
        ts = datetime.now(timezone.utc) - timedelta(minutes=6)
        hb_file.write_text(json.dumps({"updated_at": ts.isoformat()}))
        # 5 min threshold, 6 min elapsed → stale
        assert watchdog.check_heartbeat(str(hb_file), threshold_minutes=5) is True
        # 10 min threshold, 6 min elapsed → fresh
        assert watchdog.check_heartbeat(str(hb_file), threshold_minutes=10) is False

    def test_naive_timestamp_treated_as_utc(self, tmp_path: Path):
        """Naive datetime (no tz) is treated as UTC."""
        hb_file = tmp_path / "heartbeat.json"
        # Write a naive ISO timestamp (no +00:00)
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        hb_file.write_text(json.dumps({"updated_at": now_naive.isoformat()}))
        assert watchdog.check_heartbeat(str(hb_file)) is False


# ---------------------------------------------------------------------------
# restart_container
# ---------------------------------------------------------------------------

class TestRestartContainer:
    @patch("watchdog.subprocess.run")
    def test_success(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0, stdout="ai-company\n", stderr="")
        success, output = watchdog.restart_container("ai-company")
        assert success is True
        assert "ai-company" in output
        mock_run.assert_called_once_with(
            ["docker", "restart", "ai-company"],
            capture_output=True,
            text=True,
            timeout=120,
        )

    @patch("watchdog.subprocess.run")
    def test_failure(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: no such container")
        success, output = watchdog.restart_container("bad-container")
        assert success is False
        assert "no such container" in output

    @patch("watchdog.subprocess.run", side_effect=Exception("docker not found"))
    def test_exception(self, mock_run: MagicMock):
        success, output = watchdog.restart_container()
        assert success is False
        assert "docker not found" in output


# ---------------------------------------------------------------------------
# notify_slack
# ---------------------------------------------------------------------------

class TestNotifySlack:
    @patch("watchdog.urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = watchdog.notify_slack("https://hooks.slack.com/test", "hello")
        assert result is True

    @patch("watchdog.urllib.request.urlopen", side_effect=Exception("network error"))
    def test_failure(self, mock_urlopen: MagicMock):
        result = watchdog.notify_slack("https://hooks.slack.com/test", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    @patch.dict("os.environ", {
        "HEARTBEAT_PATH": "/tmp/hb.json",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
        "CONTAINER_NAME": "my-container",
    })
    @patch("watchdog.notify_slack", return_value=True)
    @patch("watchdog.restart_container", return_value=(True, "my-container"))
    @patch("watchdog.request_manager_restart", return_value=(False, "no-docker"))
    @patch("watchdog.heartbeat_is_stale", return_value=(True, 21.0))
    @patch("watchdog.read_heartbeat", return_value={"updated_at": "2020-01-01T00:00:00+00:00"})
    def test_stale_triggers_restart_and_notify(
        self,
        mock_read: MagicMock,
        mock_is_stale: MagicMock,
        mock_request_restart: MagicMock,
        mock_restart: MagicMock,
        mock_notify: MagicMock,
    ):
        watchdog.main()
        mock_read.assert_called_once()
        mock_is_stale.assert_called_once()
        mock_request_restart.assert_called_once()
        mock_restart.assert_called_once_with("my-container")
        assert mock_notify.call_count == 2
        # Verify at least one message mentions the container name
        msgs = [c.args[1] for c in mock_notify.call_args_list]
        assert any("my-container" in m for m in msgs)

    @patch.dict("os.environ", {
        "HEARTBEAT_PATH": "/tmp/hb.json",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
    })
    @patch("watchdog.notify_slack")
    @patch("watchdog.restart_container")
    @patch("watchdog.request_manager_restart")
    @patch("watchdog.heartbeat_is_stale", return_value=(False, 1.0))
    @patch("watchdog.read_heartbeat", return_value={"updated_at": "2020-01-01T00:00:00+00:00"})
    def test_fresh_does_nothing(
        self,
        mock_read: MagicMock,
        mock_is_stale: MagicMock,
        mock_request_restart: MagicMock,
        mock_restart: MagicMock,
        mock_notify: MagicMock,
    ):
        watchdog.main()
        mock_read.assert_called_once()
        mock_is_stale.assert_called_once()
        mock_request_restart.assert_not_called()
        mock_restart.assert_not_called()
        mock_notify.assert_not_called()

    @patch.dict("os.environ", {"HEARTBEAT_PATH": "", "SLACK_WEBHOOK_URL": ""}, clear=False)
    @patch("watchdog.read_heartbeat", return_value={"updated_at": "2020-01-01T00:00:00+00:00"})
    @patch("watchdog.heartbeat_is_stale", return_value=(False, 0.1))
    @patch("watchdog.request_manager_restart")
    @patch("watchdog.restart_container")
    @patch("watchdog.notify_slack")
    def test_missing_heartbeat_path_still_checks_and_exits(
        self,
        mock_notify: MagicMock,
        mock_restart: MagicMock,
        mock_request_restart: MagicMock,
        mock_is_stale: MagicMock,
        mock_read: MagicMock,
    ):
        watchdog.main()
        mock_read.assert_called_once()
        mock_is_stale.assert_called_once()
        mock_request_restart.assert_not_called()
        mock_restart.assert_not_called()
        mock_notify.assert_not_called()

    @patch.dict("os.environ", {
        "HEARTBEAT_PATH": "/tmp/hb.json",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
        "CONTAINER_NAME": "my-container",
    })
    @patch("watchdog.notify_slack", return_value=True)
    @patch("watchdog.restart_container", return_value=(False, "permission denied"))
    @patch("watchdog.request_manager_restart", return_value=(False, "no-docker"))
    @patch("watchdog.heartbeat_is_stale", return_value=(True, 21.0))
    @patch("watchdog.read_heartbeat", return_value={"updated_at": "2020-01-01T00:00:00+00:00"})
    def test_restart_failure_notifies_with_error(
        self,
        mock_read: MagicMock,
        mock_is_stale: MagicMock,
        mock_request_restart: MagicMock,
        mock_restart: MagicMock,
        mock_notify: MagicMock,
    ):
        watchdog.main()
        assert mock_notify.call_count == 2
        msg = mock_notify.call_args_list[-1].args[1]
        assert "失敗" in msg
        assert "permission denied" in msg

    @patch.dict("os.environ", {
        "HEARTBEAT_PATH": "/tmp/hb.json",
        "SLACK_WEBHOOK_URL": "",
    })
    @patch("watchdog.notify_slack")
    @patch("watchdog.restart_container", return_value=(True, "ok"))
    @patch("watchdog.request_manager_restart", return_value=(False, "no-docker"))
    @patch("watchdog.heartbeat_is_stale", return_value=(True, 21.0))
    @patch("watchdog.read_heartbeat", return_value={"updated_at": "2020-01-01T00:00:00+00:00"})
    def test_no_webhook_skips_slack(
        self,
        mock_read: MagicMock,
        mock_is_stale: MagicMock,
        mock_request_restart: MagicMock,
        mock_restart: MagicMock,
        mock_notify: MagicMock,
    ):
        watchdog.main()
        mock_restart.assert_called_once()
        mock_notify.assert_not_called()
