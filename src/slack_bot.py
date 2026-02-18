"""Slack Bot integration for AI Company Manager.

Uses Slack Bolt (Socket Mode) to:
- Send 10-minute reports to the governance channel
- Send approval requests and handle ✅/❌ reactions
- Receive messages from Creator

Req 3.2: Reports to channel C0AF21AFC14
Req 4.1-4.7: Approval protocol via reactions
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from event_deduper import EventDeduper

log = logging.getLogger("ai-company.slack")

GOVERNANCE_CHANNEL = os.environ.get("SLACK_CHANNEL_ID", "C0AF21AFC14")
CREATOR_USER_ID = os.environ.get("CREATOR_SLACK_USER_ID", "").strip() or None
_DEFAULT_DEDUP_TTL_SECONDS = int(os.environ.get("SLACK_EVENT_DEDUP_TTL_SECONDS", "900"))  # 15m


class SlackBot:
    """Manages Slack connectivity via Socket Mode."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        *,
        on_reaction: Callable[[str, str, str], None] | None = None,
        on_message: Callable[..., None] | None = None,
        on_approval_text: Callable[..., None] | None = None,
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
        # Best-effort de-dup for Slack message events (Socket Mode can deliver both message + app_mention)
        self._event_deduper = EventDeduper(ttl_seconds=_DEFAULT_DEDUP_TTL_SECONDS)

        self._register_handlers()

    def _should_process_message(self, *, channel: str, ts: str) -> bool:
        """Return True if this (channel, ts) event should be processed (dedup within TTL)."""
        if not channel or not ts:
            return True
        return self._event_deduper.should_process(f"{channel}:{ts}")

    def _call_on_approval_text(
        self,
        *,
        request_id: str,
        text: str,
        user_id: str,
        channel: str | None,
        thread_ts: str | None,
        thread_context: str | None,
    ) -> None:
        if not self._on_approval_text:
            return
        try:
            argc = len(inspect.signature(self._on_approval_text).parameters)
        except Exception:
            argc = 3

        # Backward compat: old callbacks are (request_id, text, user_id)
        if argc <= 3:
            self._on_approval_text(request_id, text, user_id)
            return

        # New callbacks: (request_id, text, user_id, channel, thread_ts, thread_context)
        self._on_approval_text(request_id, text, user_id, channel, thread_ts, thread_context)

    def _call_on_message(
        self,
        *,
        text: str,
        user_id: str,
        channel: str | None,
        thread_ts: str | None,
        thread_context: str | None,
    ) -> None:
        if not self._on_message:
            return
        try:
            argc = len(inspect.signature(self._on_message).parameters)
        except Exception:
            argc = 2

        # Backward compat: old callbacks are (text, user_id)
        if argc <= 2:
            self._on_message(text, user_id)
            return

        # New callbacks: (text, user_id, channel, thread_ts, thread_context)
        self._on_message(text, user_id, channel, thread_ts, thread_context)

    def _format_thread_context(
        self,
        *,
        channel: str,
        thread_ts: str,
        exclude_ts: str | None = None,
        max_chars: int = 12_000,
    ) -> str | None:
        """Fetch and format full thread transcript (best-effort)."""
        try:
            messages: list[dict] = []
            cursor: str | None = None
            while True:
                kwargs = {"channel": channel, "ts": thread_ts, "limit": 200}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = self.app.client.conversations_replies(**kwargs)
                batch = resp.get("messages") or []
                if isinstance(batch, list):
                    messages.extend(batch)
                meta = resp.get("response_metadata") or {}
                cursor = (meta.get("next_cursor") or "").strip() or None
                if not cursor:
                    break

            lines: list[str] = []
            for m in messages:
                text = (m.get("text") or "").strip()
                if not text:
                    continue
                if exclude_ts and (m.get("ts") == exclude_ts):
                    continue
                subtype = (m.get("subtype") or "").strip()
                if subtype in ("message_changed", "message_deleted"):
                    continue

                ts_raw = m.get("ts") or ""
                try:
                    ts_f = float(ts_raw)
                    dt = datetime.fromtimestamp(ts_f, tz=timezone.utc)
                    ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = str(ts_raw)

                is_bot = bool(m.get("bot_id")) or subtype == "bot_message"
                user = m.get("user") or ("bot" if is_bot else "unknown")
                who = "creator" if (self._creator_user_id and user == self._creator_user_id) else user
                role = "assistant" if is_bot else "user"

                if len(text) > 2000:
                    text = text[:2000] + "…"
                lines.append(f"- [{role}:{who}] {ts}: {text}")

            if not lines:
                return None

            header = f"channel={channel} thread_ts={thread_ts}"
            content = "\n".join([header, *lines])
            if len(content) <= max_chars:
                return content

            head = "\n".join([header, *lines[:30]])
            tail = "\n".join(lines[-30:])
            trimmed = f"{head}\n...\n{tail}"
            return trimmed[:max_chars] + "…"
        except Exception:
            log.exception(
                "Failed to fetch thread context (channel=%s thread_ts=%s)",
                channel,
                thread_ts,
            )
            return None


    def fetch_thread_context(
        self,
        *,
        channel: str,
        thread_ts: str,
        exclude_ts: str | None = None,
    ) -> str | None:
        """Public helper to fetch thread context for Manager fallback usage."""
        context = self._format_thread_context(
            channel=channel,
            thread_ts=thread_ts,
            exclude_ts=exclude_ts,
        )
        if context and context.strip():
            log.info(
                "Thread context loaded (channel=%s thread_ts=%s chars=%d)",
                channel,
                thread_ts,
                len(context),
            )
        else:
            log.info(
                "Thread context unavailable (channel=%s thread_ts=%s)",
                channel,
                thread_ts,
            )
        return context

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
            channel = event.get("channel", "") or ""
            thread_ts = event.get("thread_ts") or ""
            ts = event.get("ts") or ""
            if not text or not user_id:
                return
            if channel and ts and not self._should_process_message(channel=channel, ts=ts):
                log.info("Skipping duplicate app_mention event (channel=%s ts=%s)", channel, ts)
                return
            # Strip the mention tag (e.g. "<@U12345> hello" -> "hello")
            import re
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if not text:
                return

            log.info("Mention from %s: %s", user_id, text[:80])
            thread_context = None
            if channel and thread_ts:
                thread_context = self.fetch_thread_context(
                    channel=channel,
                    thread_ts=thread_ts,
                    exclude_ts=event.get("ts") or None,
                )
            self._call_on_message(
                text=text,
                user_id=user_id,
                channel=channel or None,
                thread_ts=thread_ts or None,
                thread_context=thread_context,
            )

        @self.app.event("message")
        def handle_message(event: dict, say) -> None:
            text = event.get("text", "")
            user_id = event.get("user", "")
            channel = event.get("channel", "") or ""
            ts = event.get("ts") or ""
            if not text or not user_id:
                return
            # Ignore bot messages
            if event.get("bot_id"):
                return
            # Ignore non-standard message subtypes (edited/deleted/etc)
            if event.get("subtype"):
                return
            if channel and ts and not self._should_process_message(channel=channel, ts=ts):
                log.info("Skipping duplicate message event (channel=%s ts=%s)", channel, ts)
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
                thread_context = None
                if channel and thread_ts:
                    thread_context = self.fetch_thread_context(
                        channel=channel,
                        thread_ts=thread_ts,
                        exclude_ts=event.get("ts") or None,
                    )
                self._call_on_approval_text(
                    request_id=request_id,
                    text=text,
                    user_id=user_id,
                    channel=channel or None,
                    thread_ts=thread_ts or None,
                    thread_context=thread_context,
                )
                return

            log.info("Message from %s: %s", user_id, text[:80])
            thread_context = None
            if channel and thread_ts:
                thread_context = self.fetch_thread_context(
                    channel=channel,
                    thread_ts=thread_ts,
                    exclude_ts=event.get("ts") or None,
                )
            self._call_on_message(
                text=text,
                user_id=user_id,
                channel=channel or None,
                thread_ts=thread_ts or None,
                thread_context=thread_context,
            )

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

    def send_message(
        self,
        text: str,
        *,
        channel: str | None = None,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post a generic message to a channel, optionally as a thread reply."""
        try:
            kwargs: dict[str, object] = {
                "channel": channel or GOVERNANCE_CHANNEL,
                "text": text,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            result = self.app.client.chat_postMessage(**kwargs)
            return result.get("ts")
        except Exception:
            log.exception("Failed to send message")
            return None
