"""AI Company Manager – long-running entry point.

Starts the Manager process, initializes the company directory,
connects to Slack via Socket Mode, and runs an event loop that
periodically updates the heartbeat and sends 10-minute reports.

This is the container's main process (PID 1 inside Docker).
Req 7.1: Single long-running process.
Req 7.6: Heartbeat updated periodically.
Req 3.2: Reports sent to Slack.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from amendment import approve_amendment, reject_amendment
from approval_text_classifier import classify_approval_text
from llm_client import LLMClient
from manager import Manager, init_company_directory
from manager_state import constitution_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ai-company")

# Configuration via environment variables
COMPANY_ID = os.environ.get("COMPANY_ID", "alpha")
BASE_DIR = Path(os.environ.get("BASE_DIR", "/opt/apps/ai-company/data"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "180"))  # 3 min
REPORT_INTERVAL = 0  # hard-disabled in minimal mode
DAILY_BRIEF_INTERVAL = int(os.environ.get("DAILY_BRIEF_INTERVAL", "0"))  # disabled by default

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
CREATOR_SLACK_USER_ID = os.environ.get("CREATOR_SLACK_USER_ID", "").strip() or None
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

_shutdown = False


def _handle_signal(signum: int, _frame) -> None:
    global _shutdown
    log.info("Signal %s received, shutting down gracefully...", signum)
    _shutdown = True


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info("=== AI Company Manager starting ===")
    log.info("Company: %s | Base: %s | Heartbeat interval: %ds", COMPANY_ID, BASE_DIR, HEARTBEAT_INTERVAL)
    log.info("LLM Model: %s", OPENROUTER_MODEL)

    # Initialize directory structure and default constitution
    init_company_directory(BASE_DIR, COMPANY_ID)
    log.info("Company directory initialized")

    # Create Manager and run startup sequence
    mgr = Manager(BASE_DIR, COMPANY_ID)
    action, description = mgr.startup()
    log.info("Startup complete – recovery action: %s (%s)", action, description)

    # Initialize LLM client if API key is configured (Req 1.6)
    if OPENROUTER_API_KEY:
        mgr.llm_client = LLMClient(api_key=OPENROUTER_API_KEY, model=OPENROUTER_MODEL)
        log.info("LLM client initialized (model: %s)", OPENROUTER_MODEL)
        # Update CEO agent model (may have been registered as "unknown" during startup)
        try:
            ceo = mgr.agent_registry.get("ceo")
            if ceo and ceo.model != OPENROUTER_MODEL:
                mgr.agent_registry.update_status("ceo", ceo.status)
                # Re-register with correct model
                from datetime import datetime, timezone
                from models import AgentEntry
                from ndjson_store import ndjson_append
                updated_ceo = ceo.model_copy(update={
                    "model": OPENROUTER_MODEL,
                    "updated_at": datetime.now(timezone.utc),
                })
                ndjson_append(mgr.agent_registry._path, updated_ceo)
                log.info("CEO agent model updated: %s → %s", ceo.model, OPENROUTER_MODEL)
        except Exception:
            log.warning("Failed to update CEO agent model", exc_info=True)
    else:
        log.warning("OPENROUTER_API_KEY not set – LLM features disabled")

    # Pricing/ledger based budget control is disabled in minimal mode.

    # --- Slack integration ---
    slack = None
    if SLACK_BOT_TOKEN and SLACK_APP_TOKEN:
        from slack_bot import SlackBot

        def on_reaction(request_id: str, result: str, user_id: str) -> None:
            log.info("Approval %s for %s by %s", result, request_id, user_id)

            # Find the proposal in decision_log by request_id
            # Check if already processed (approved/rejected entry exists)
            proposal = None
            already_processed = False
            for entry in mgr.state.decision_log:
                if entry.request_id == request_id:
                    if entry.status in ("approved", "rejected"):
                        already_processed = True
                        break
                    if entry.status == "proposed":
                        proposal = entry

            if already_processed:
                log.warning("Request %s already processed", request_id)
                slack.send_message(
                    f"⚠️ 承認リクエスト `{request_id}` は既に処理済みです"
                )
                return

            if proposal is None:
                log.warning("No pending proposal found for request_id=%s", request_id)
                slack.send_message(
                    f"⚠️ 承認リクエスト `{request_id}` に対応する提案が見つかりません"
                )
                return

            const_path = constitution_path(BASE_DIR, COMPANY_ID)

            try:
                if result == "approved":
                    updated = approve_amendment(
                        constitution=mgr.state.constitution,
                        proposal=proposal,
                        constitution_path=const_path,
                        base_dir=BASE_DIR,
                        company_id=COMPANY_ID,
                    )
                    mgr.state.constitution = updated
                    # Reload decision_log to include the new approved entry
                    from manager_state import restore_state
                    mgr.state.decision_log = restore_state(BASE_DIR, COMPANY_ID).decision_log
                    slack.send_message(
                        f"✅ 憲法変更が承認されました (v{updated.version})\n"
                        f"変更内容: {proposal.decision}"
                    )
                    log.info("Amendment approved: %s (v%d)", proposal.decision, updated.version)

                elif result == "rejected":
                    reject_amendment(
                        proposal=proposal,
                        base_dir=BASE_DIR,
                        company_id=COMPANY_ID,
                    )
                    # Reload decision_log to include the new rejected entry
                    from manager_state import restore_state
                    mgr.state.decision_log = restore_state(BASE_DIR, COMPANY_ID).decision_log
                    slack.send_message(
                        f"❌ 憲法変更が却下されました\n"
                        f"変更内容: {proposal.decision}"
                    )
                    log.info("Amendment rejected: %s", proposal.decision)

                else:
                    log.warning("Unknown reaction result: %s", result)

            except Exception:
                log.exception("Error processing reaction for %s", request_id)
                slack.send_message(
                    f"⚠️ 承認処理中にエラーが発生しました (request_id: {request_id})"
                )

        def on_approval_text(
            request_id: str,
            text: str,
            user_id: str,
            channel: str | None = None,
            thread_ts: str | None = None,
            thread_context: str | None = None,
        ) -> None:
            """Handle free-form approval replies (thread reply preferred)."""
            log.info("Approval text for %s by %s: %s", request_id, user_id, text[:120])

            def _reply(message: str) -> None:
                slack.send_message(message, channel=channel, thread_ts=thread_ts)

            # Find pending proposal in decision_log by request_id
            proposal = None
            already_processed = False
            for entry in mgr.state.decision_log:
                if entry.request_id == request_id:
                    if entry.status in ("approved", "rejected"):
                        already_processed = True
                        break
                    if entry.status == "proposed":
                        proposal = entry

            if already_processed:
                _reply(f"⚠️ 承認リクエスト `{request_id}` は既に処理済みです")
                return

            if proposal is None:
                _reply(f"⚠️ 承認リクエスト `{request_id}` に対応する提案が見つかりません")
                return

            request_summary = (
                f"request_id={request_id}\n"
                f"decision={proposal.decision}\n"
                f"why={proposal.why}\n"
                f"scope={proposal.scope}\n"
            )
            if thread_context and thread_context.strip():
                request_summary += f"\nslack_thread_context:\n{thread_context.strip()[:6000]}\n"
            decision = classify_approval_text(
                text,
                request_summary=request_summary,
                llm_client=mgr.llm_client,
            )

            if decision == "unknown":
                _reply(
                    f"⚠️ 判定できませんでした: `{request_id}`\n"
                    f"スレッドで「OK/NG」「進めて/やめて」など意図が分かる形で返信してください。"
                )
                return

            const_path = constitution_path(BASE_DIR, COMPANY_ID)

            try:
                if decision == "approved":
                    updated = approve_amendment(
                        constitution=mgr.state.constitution,
                        proposal=proposal,
                        constitution_path=const_path,
                        base_dir=BASE_DIR,
                        company_id=COMPANY_ID,
                    )
                    mgr.state.constitution = updated
                    from manager_state import restore_state
                    mgr.state.decision_log = restore_state(BASE_DIR, COMPANY_ID).decision_log
                    _reply(
                        f"✅ 憲法変更が承認されました (v{updated.version})\n"
                        f"変更内容: {proposal.decision}"
                    )
                    log.info("Amendment approved via text: %s (v%d)", proposal.decision, updated.version)

                elif decision == "rejected":
                    reject_amendment(
                        proposal=proposal,
                        base_dir=BASE_DIR,
                        company_id=COMPANY_ID,
                    )
                    from manager_state import restore_state
                    mgr.state.decision_log = restore_state(BASE_DIR, COMPANY_ID).decision_log
                    _reply(
                        f"❌ 憲法変更が却下されました\n"
                        f"変更内容: {proposal.decision}"
                    )
                    log.info("Amendment rejected via text: %s", proposal.decision)

            except Exception:
                log.exception("Error processing approval text for %s", request_id)
                _reply(f"⚠️ 承認処理中にエラーが発生しました (request_id: {request_id})")

        def on_message(
            text: str,
            user_id: str,
            channel: str | None = None,
            thread_ts: str | None = None,
            thread_context: str | None = None,
        ) -> None:
            log.info(
                "Creator message: %s (from %s, channel=%s, thread_ts=%s)",
                text[:100],
                user_id,
                channel,
                thread_ts,
            )
            mgr.process_message(
                text,
                user_id,
                slack_channel=channel,
                slack_thread_ts=thread_ts,
                slack_thread_context=thread_context,
            )

        approval_store_path = BASE_DIR / "companies" / COMPANY_ID / "state" / "slack_approval_mapping.json"

        slack = SlackBot(
            bot_token=SLACK_BOT_TOKEN,
            app_token=SLACK_APP_TOKEN,
            on_reaction=on_reaction,
            on_message=on_message,
            on_approval_text=on_approval_text,
            approval_store_path=approval_store_path,
            creator_user_id=CREATOR_SLACK_USER_ID,
        )
        slack.start()
        mgr.slack = slack
        log.info("Slack Bot connected (Socket Mode)")

        # Send startup notification
        slack.send_message("🟢 AI Company Manager 起動完了")
    else:
        log.warning("SLACK_BOT_TOKEN or SLACK_APP_TOKEN not set – running without Slack")

    # --- Main loop ---
    last_report_at = time.monotonic()
    last_daily_brief_at = time.monotonic()
    log.info("Entering main loop (heartbeat every %ds, report every %ds)...", HEARTBEAT_INTERVAL, REPORT_INTERVAL)

    while not _shutdown:
        try:
            # Update heartbeat
            from heartbeat import update_heartbeat
            status = "running" if mgr.state.wip else "idle"
            mgr.state.heartbeat = update_heartbeat(
                BASE_DIR, COMPANY_ID,
                status=status,
                current_wip=mgr.state.wip,
                pid=os.getpid(),
            )

            # Send periodic report (optional)
            elapsed = time.monotonic() - last_report_at
            if REPORT_INTERVAL > 0 and elapsed >= REPORT_INTERVAL and slack:
                try:
                    report = mgr.generate_report()
                    ts = slack.send_report(report)
                    if ts:
                        log.info("10-min report sent (ts=%s)", ts)
                    last_report_at = time.monotonic()
                except Exception:
                    log.exception("Failed to send 10-min report")

            # Send daily brief (optional; KPI loop)
            if DAILY_BRIEF_INTERVAL > 0 and slack:
                daily_elapsed = time.monotonic() - last_daily_brief_at
                if daily_elapsed >= DAILY_BRIEF_INTERVAL:
                    try:
                        brief = mgr.generate_daily_brief()
                        ts = slack.send_report(brief)
                        if ts:
                            log.info("Daily brief sent (ts=%s)", ts)
                        last_daily_brief_at = time.monotonic()
                    except Exception:
                        log.exception("Failed to send daily brief")

        except Exception:
            log.exception("Error in main loop iteration")

        # Sleep in small increments so we can respond to signals quickly
        for _ in range(HEARTBEAT_INTERVAL):
            if _shutdown:
                break
            time.sleep(1)

    # Graceful shutdown
    if slack:
        slack.send_message("🔴 AI Company Manager シャットダウン中...")
        slack.stop()

    log.info("=== AI Company Manager stopped ===")


if __name__ == "__main__":
    main()
