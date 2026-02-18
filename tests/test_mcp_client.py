"""Unit tests for MCPClient."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mcp_client import MCPClient


def _write_mcp_config(tmp_path: Path) -> Path:
    path = tmp_path / "companies" / "co" / "protocols" / "mcp_servers.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "servers:",
            "  vps-monitor:",
            "    api_base: \"https://example.com/mcp\"",
            "    token_env: \"TEST_MCP_TOKEN\"",
            "    desc: \"test\"",
            "",
        ]),
        encoding="utf-8",
    )
    return path


class TestParseAction:
    def test_shorthand_is_tools_call(self):
        req = MCPClient.parse_action("list_containers")
        assert req is not None
        assert req.method == "tools/call"
        assert req.name == "list_containers"
        assert req.arguments == {}

    def test_yaml_tools_list(self):
        req = MCPClient.parse_action("server: vps-monitor\nmethod: tools/list")
        assert req is not None
        assert req.method == "tools/list"
        assert req.server == "vps-monitor"


class TestRunAction:
    def test_tools_list_formats_names(self, tmp_path: Path, monkeypatch):
        _write_mcp_config(tmp_path)
        monkeypatch.setenv("TEST_MCP_TOKEN", "secret")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": "list_containers"}, {"name": "get_container_logs"}]},
        }

        with patch("mcp_client.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            client = MCPClient(tmp_path, "co")
            out = client.run_action("server: vps-monitor\nmethod: tools/list")
            assert "list_containers" in out
            assert "get_container_logs" in out

    def test_tools_call_extracts_text_content(self, tmp_path: Path):
        _write_mcp_config(tmp_path)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "OK"}]},
        }

        with patch("mcp_client.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
            client = MCPClient(tmp_path, "co")
            out = client.run_action("server: vps-monitor\nname: list_containers\narguments: {}")
            assert "MCP結果" in out
            assert "OK" in out

