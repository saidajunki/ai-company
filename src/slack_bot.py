"""Slack Bot integration for AI Company Manager.

Uses Slack Bolt (Socket Mode) to:
- Send 10-minute reports to the governance channel
- Send approval requests and handle ✅/❌ reactions
- Receive messages from Creator

Req 3.2: Reports to channel C0AF21AFC14
Req 4.1-4.7: Approval protocol via reactions
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

log = logging.getLogger("ai-company.slack")

GOVERNANCE_CHANNEL = os.environ.get("SLACK_CHANNEL_ID", "C0AF21AFC14")
CREATOR_USER_ID = os.environ.get("CREATOR_SLACK_USER_ID", "").strip() or None


class SlackBot:
    """Manages Slack connectivity via Socket Mode."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        *,
        on_reaction: Callable[[str, str, str], None] | None = None,
        on_message: Callable[[str, str], None] | None = None,
        on_approval_text: Callable[[str, str, str], None] | None = None,
        approval_store_path: str | Path | None = None,
        creator_user_id: str | None = CREATOR_USER_ID,
    ) -> None:
        """
        Args:
            bot_token: xoxb-... Bot User OAuth Token
            app_token: xapp-... App-Level Token (Socket Mode)
            on_reaction: callback(request_id, reaction, user_id) for ✅/❌
            on_message: callback(text, user_id) for incoming messages
        """
        self.app = App(token=bot_token)
        self.handler = SocketModeHandler(self.app, app_token)
        self._on_reaction = on_reaction
        self._on_message = on_message
        self._on_approval_text = on_approval_text
        self._creator_user_id = creator_user_id
        self._thread: threading.Thread | None = None
        # Track message ts -> request_id for reaction mapping
        self._approval_messages: dict[str, str] = {}
        self._approval_store_path = Path(approval_store_path) if approval_store_path else None
        self._load_approval_mapping()

        self._register_handlers()

    def _load_approval_mapping(self) -> None:
        """Load approval message mapping from disk (best-effort)."""
        if self._approval_store_path is None:
            return
        try:
            if not self._approval_store_path.exists():
                return
            import json

            raw = self._approval_store_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self._approval_messages = {str(k): str(v) for k, v in data.items()}
        except Exception:
            log.exception("Failed to load approval mapping from %s", self._approval_store_path)

    def _persist_approval_mapping(self) -> None:
        """Persist approval message mapping to disk (best-effort)."""
        if self._approval_store_path is None:
            return
        try:
            import json

            self._approval_store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._approval_store_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._approval_messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._approval_store_path)
        except Exception:
            log.exception("Failed to persist approval mapping to %s", self._approval_store_path)

    def _register_handlers(self) -> None:
        @self.app.event("reaction_added")
        def handle_reaction(event: dict, say) -> None:
            reaction = event.get("reaction", "")
            if reaction not in ("white_check_mark", "x"):
                return

            item = event.get("item", {})
            msg_ts = item.get("ts", "")
            user_id = event.get("user", "")

            request_id = self._approval_messages.get(msg_ts)
            if not request_id:
                return

            if self._creator_user_id and user_id != self._creator_user_id:
                log.info("Ignoring reaction from non-creator user: %s", user_id)
                return

            mapped = "approved" if reaction == "white_check_mark" else "rejected"
            log.info(
                "Reaction %s on request %s by %s",
                mapped, request_id, user_id,
            )

            if self._on_reaction:
                self._on_reaction(request_id, mapped, user_id)

        @self.app.event("app_mention")
        def handle_app_mention(event: dict, say) -> None:
            text = event.get("text", "")
            user_id = event.get("user", "")
            if not text or not user_id:
                return
            # Strip the mention tag (e.g. "<@U12345> hello" -> "hello")
            import re
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if not text:
                return

            log.info("Mention from %s: %s", user_id, text[:80])
            if self._on_message:
                self._on_message(text, user_id)

        @self.app.event("message")
        def handle_message(event: dict, say) -> None:
            text = event.get("text", "")
            user_id = event.get("user", "")
            if not text or not user_id:
                return
            # Ignore bot messages
            if event.get("bot_id"):
                return

            # Approval via thread reply / request_id mention
            request_id = None
            thread_ts = event.get("thread_ts") or ""
            if thread_ts:
                request_id = self._approval_messages.get(thread_ts)

            if request_id is None:
                # Fallback: parse request_id from message body
                import re

                m = re.search(r"request_id\s*[:：]\s*([0-9a-fA-F-]{8,})", text)
                if m:
                    request_id = m.group(1)

            if request_id and self._on_approval_text:
                if self._creator_user_id and user_id != self._creator_user_id:
                    log.info("Ignoring approval text from non-creator user: %s", user_id)
                    return
                log.info("Approval reply detected for %s by %s", request_id, user_id)
                self._on_approval_text(request_id, text, user_id)
                return

            log.info("Message from %s: %s", user_id, text[:80])
            if self._on_message:
                self._on_message(text, user_id)

    def start(self) -> None:
        """Start Socket Mode connection in a background thread."""
        self._thread = threading.Thread(
            target=self.handler.start,
            daemon=True,
            name="slack-socket-mode",
        )
        self._thread.start()
        log.info("Slack Socket Mode started (channel: %s)", GOVERNANCE_CHANNEL)

    def stop(self) -> None:
        """Disconnect from Slack."""
        try:
            self.handler.close()
        except Exception:
            log.exception("Error closing Slack handler")

    def send_report(self, report_text: str) -> str | None:
        """Post a 10-minute report to the governance channel.

        Returns the message timestamp (ts) or None on failure.
        """
        try:
            result = self.app.client.chat_postMessage(
                channel=GOVERNANCE_CHANNEL,
                text=report_text,
                mrkdwn=True,
            )
            ts = result.get("ts")
            log.info("Report posted (ts=%s)", ts)
            return ts
        except Exception:
            log.exception("Failed to send report")
            return None

    def send_approval_request(
        self, message: str, request_id: str,
    ) -> str | None:
        """Post an approval request and track its message ts.

        Returns the message timestamp (ts) or None on failure.
        """
        try:
            result = self.app.client.chat_postMessage(
                channel=GOVERNANCE_CHANNEL,
                text=message,
                mrkdwn=True,
            )
            ts = result.get("ts")
            if ts:
                self._approval_messages[ts] = request_id
                self._persist_approval_mapping()
                log.info(
                    "Approval request posted (ts=%s, request_id=%s)",
                    ts, request_id,
                )
            return ts
        except Exception:
            log.exception("Failed to send approval request")
            return None

    def send_message(self, text: str) -> str | None:
        """Post a generic message to the governance channel."""
        try:
            result = self.app.client.chat_postMessage(
                channel=GOVERNANCE_CHANNEL,
                text=text,
            )
            return result.get("ts")
        except Exception:
            log.exception("Failed to send message")
            return None
