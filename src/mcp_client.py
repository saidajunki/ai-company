"""MCP Client — JSON-RPC over HTTP tool bridge.

This module provides a minimal MCP client for calling MCP servers over HTTP
(`tools/list` and `tools/call`) so the CEO AI and sub-agents can use MCP as a
shared internal tool.

Config file (YAML, company SoT):
  companies/<company_id>/protocols/mcp_servers.yaml

Example:
  servers:
    vps-monitor:
      api_base: "https://mcp.app.babl.tech"
      token_env: "AI_COMPANY_MCP_VPS_MONITOR_TOKEN"
      desc: "VPS監視MCPサーバ"

LLM action payload inside <mcp> ... </mcp> (JSON or YAML):
  # tools/list
  server: vps-monitor
  method: tools/list

  # tools/call
  server: vps-monitor
  method: tools/call
  name: list_containers
  arguments: {}

Shorthand:
  list_containers
    -> method=tools/call, arguments={}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    api_base: str
    token_env: str | None = None
    desc: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class MCPActionRequest:
    server: str | None
    method: str
    name: str | None = None
    arguments: dict[str, Any] | None = None


class MCPClient:
    """Simple MCP client (HTTP JSON-RPC)."""

    def __init__(
        self,
        base_dir: Path,
        company_id: str,
        *,
        config_path: Path | None = None,
        default_server_env: str = "AI_COMPANY_MCP_DEFAULT_SERVER",
    ) -> None:
        self.base_dir = base_dir
        self.company_id = company_id
        self.config_path = config_path or (
            base_dir / "companies" / company_id / "protocols" / "mcp_servers.yaml"
        )
        self.default_server_env = default_server_env

    # -----------------------------
    # Config
    # -----------------------------

    def list_servers(self) -> list[MCPServerSpec]:
        data = self._load_config()
        servers = data.get("servers")
        if not isinstance(servers, dict):
            return []

        specs: list[MCPServerSpec] = []
        for name, cfg in servers.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(cfg, dict):
                continue
            api_base = str(cfg.get("api_base") or cfg.get("url") or "").strip()
            if not api_base:
                continue
            token_env = str(cfg.get("token_env") or "").strip() or None
            desc = str(cfg.get("desc") or cfg.get("description") or "").strip() or None
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
            try:
                if cfg.get("timeout_seconds") is not None:
                    timeout_seconds = int(cfg.get("timeout_seconds"))
            except Exception:
                timeout_seconds = DEFAULT_TIMEOUT_SECONDS
            specs.append(MCPServerSpec(
                name=name.strip(),
                api_base=api_base.rstrip("/"),
                token_env=token_env,
                desc=desc,
                timeout_seconds=timeout_seconds,
            ))
        return specs

    def format_servers_for_prompt(self, *, limit: int = 8) -> str:
        servers = self.list_servers()
        if not servers:
            return "（未設定）"
        lines: list[str] = []
        for s in servers[:limit]:
            desc = f" — {s.desc}" if s.desc else ""
            lines.append(f"- {s.name}: {s.api_base}{desc}")
        if len(servers) > limit:
            lines.append(f"- ... ({len(servers) - limit} more)")
        return "\n".join(lines)

    def _load_config(self) -> dict[str, Any]:
        try:
            if not self.config_path.exists():
                return {}
            raw = self.config_path.read_text(encoding="utf-8")
            payload = yaml.safe_load(raw) if raw.strip() else None
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.warning("Failed to load MCP config: %s", self.config_path, exc_info=True)
            return {}

    def _resolve_default_server(self, servers: list[MCPServerSpec]) -> str | None:
        env = (os.environ.get(self.default_server_env) or "").strip()
        if env:
            return env
        return servers[0].name if servers else None

    def _get_server(self, name: str | None) -> MCPServerSpec | None:
        servers = self.list_servers()
        if not servers:
            return None
        requested = (name or "").strip()
        if not requested:
            requested = self._resolve_default_server(servers) or ""
        for s in servers:
            if s.name == requested:
                return s
        return None

    # -----------------------------
    # Action parsing
    # -----------------------------

    @staticmethod
    def parse_action(text: str) -> MCPActionRequest | None:
        raw = (text or "").strip()
        if not raw:
            return None

        data: Any = None
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
            except Exception:
                data = None

        if data is None:
            try:
                data = yaml.safe_load(raw)
            except Exception:
                data = None

        if not isinstance(data, dict):
            # Shorthand: treat as tools/call name
            return MCPActionRequest(
                server=None,
                method="tools/call",
                name=raw,
                arguments={},
            )

        server = (data.get("server") or data.get("srv") or None)
        method = (data.get("method") or data.get("m") or "").strip()
        name = (data.get("name") or data.get("tool") or None)
        arguments = data.get("arguments") if "arguments" in data else data.get("args")

        # Default method based on fields
        if not method:
            method = "tools/call" if name else "tools/list"

        # Normalize args
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        return MCPActionRequest(
            server=str(server).strip() if server else None,
            method=str(method).strip(),
            name=str(name).strip() if name else None,
            arguments=arguments,
        )

    # -----------------------------
    # Execution
    # -----------------------------

    def run_action(self, payload_text: str) -> str:
        req = self.parse_action(payload_text)
        if req is None:
            return "MCP結果: 入力が空です"

        server = self._get_server(req.server)
        if server is None:
            available = self.format_servers_for_prompt()
            return (
                "MCP結果: サーバ設定が見つかりません。\n"
                f"- 設定ファイル: {self.config_path}\n"
                f"- 利用可能サーバ:\n{available}"
            )

        method = req.method.strip()
        params: dict[str, Any] | None = None
        label = f"server={server.name} method={method}"

        if method == "tools/list":
            params = None
        elif method == "tools/call":
            tool_name = (req.name or "").strip()
            if not tool_name:
                return f"MCP結果 ({label}): tools/call には name が必要です"
            params = {"name": tool_name, "arguments": req.arguments or {}}
            label = f"server={server.name} tool={tool_name}"
        else:
            return f"MCP結果 ({label}): 未対応メソッドです（対応: tools/list, tools/call）"

        ok, data, err = self._request(server, method, params=params)
        if not ok:
            return f"MCP結果 ({label}): エラー\n{err or 'unknown_error'}"

        # Format result
        if method == "tools/list":
            tools = (
                (data or {}).get("tools")
                if isinstance(data, dict)
                else None
            )
            if not isinstance(tools, list):
                return f"MCP結果 ({label}): tools/list 応答形式が不正です"
            names = []
            for t in tools:
                if isinstance(t, dict) and t.get("name"):
                    names.append(str(t["name"]))
            lines = "\n".join(f"- {n}" for n in names[:60]) if names else "（ツールなし）"
            return f"MCP結果 ({label}):\n{lines}"

        # tools/call
        # MCP servers usually return: result: { content: [{type:'text', text:'...'}], isError?: bool }
        content = None
        if isinstance(data, dict):
            content = data.get("content")
        texts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text") or ""))
        out = "\n".join(t for t in texts if t) if texts else json.dumps(data, ensure_ascii=False)[:6000]
        if len(out) > 6000:
            out = out[:6000] + "…"
        return f"MCP結果 ({label}):\n{out}"

    def _request(
        self,
        server: MCPServerSpec,
        method: str,
        *,
        params: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        token = None
        if server.token_env:
            token = (os.environ.get(server.token_env) or "").strip() or None
        # Backward compat / simple default
        if token is None:
            token = (os.environ.get("MCP_TOKEN") or "").strip() or None

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1,
        }
        if params is not None:
            payload["params"] = params

        try:
            with httpx.Client(timeout=server.timeout_seconds) as client:
                resp = client.post(server.api_base, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
        except Exception as exc:
            return False, None, f"http_error: {exc}"

        if not isinstance(body, dict):
            return False, None, "invalid_response: non-object json"
        if body.get("error"):
            return False, None, json.dumps(body.get("error"), ensure_ascii=False)
        result = body.get("result")
        if isinstance(result, dict):
            return True, result, None
        # Some servers might return non-dict result; keep as wrapped
        return True, {"value": result}, None

