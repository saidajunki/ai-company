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
from typing import Callable

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

log = logging.getLogger("ai-company.slack")

GOVERNANCE_CHANNEL = os.environ.get("SLACK_CHANNEL_ID", "C0AF21AFC14")


class SlackBot:
    """Manages Slack connectivity via Socket Mode."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        *,
        on_reaction: Callable[[str, str, str], None] | None = None,
        on_message: Callable[[str, str], None] | None = None,
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
        self._thread: threading.Thread | None = None
        # Track message ts -> request_id for reaction mapping
        self._approval_messages: dict[str, str] = {}

        self._register_handlers()

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

            mapped = "approved" if reaction == "white_check_mark" else "rejected"
            log.info(
                "Reaction %s on request %s by %s",
                mapped, request_id, user_id,
            )

            if self._on_reaction:
                self._on_reaction(request_id, mapped, user_id)

        @self.app.event("message")
        def handle_message(event: dict, say) -> None:
            text = event.get("text", "")
            user_id = event.get("user", "")
            if not text or not user_id:
                return
            # Ignore bot messages
            if event.get("bot_id"):
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
