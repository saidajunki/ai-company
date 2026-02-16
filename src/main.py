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

from manager import Manager, init_company_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ai-company")

# Configuration via environment variables
COMPANY_ID = os.environ.get("COMPANY_ID", "alpha")
BASE_DIR = Path(os.environ.get("BASE_DIR", "/app/data"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "180"))  # 3 min
REPORT_INTERVAL = int(os.environ.get("REPORT_INTERVAL", "600"))  # 10 min

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")

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

    # Initialize directory structure and default constitution
    init_company_directory(BASE_DIR, COMPANY_ID)
    log.info("Company directory initialized")

    # Create Manager and run startup sequence
    mgr = Manager(BASE_DIR, COMPANY_ID)
    action, description = mgr.startup()
    log.info("Startup complete – recovery action: %s (%s)", action, description)

    # --- Slack integration ---
    slack = None
    if SLACK_BOT_TOKEN and SLACK_APP_TOKEN:
        from slack_bot import SlackBot

        def on_reaction(request_id: str, result: str, user_id: str) -> None:
            log.info("Approval %s for %s by %s", result, request_id, user_id)
            # TODO: route to amendment/approval logic

        def on_message(text: str, user_id: str) -> None:
            log.info("Creator message: %s (from %s)", text[:100], user_id)
            # TODO: route to task processing

        slack = SlackBot(
            bot_token=SLACK_BOT_TOKEN,
            app_token=SLACK_APP_TOKEN,
            on_reaction=on_reaction,
            on_message=on_message,
        )
        slack.start()
        log.info("Slack Bot connected (Socket Mode)")

        # Send startup notification
        slack.send_message(
            f"🟢 AI Company Manager 起動完了\n"
            f"会社: {COMPANY_ID} | 復旧アクション: {action.value} ({description})"
        )
    else:
        log.warning("SLACK_BOT_TOKEN or SLACK_APP_TOKEN not set – running without Slack")

    # --- Main loop ---
    last_report_at = time.monotonic()
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

            # Check budget
            if mgr.check_budget():
                log.warning("Budget exceeded – pausing LLM/API calls")

            # Send 10-minute report (Req 3.2)
            elapsed = time.monotonic() - last_report_at
            if elapsed >= REPORT_INTERVAL and slack:
                try:
                    report = mgr.generate_report()
                    ts = slack.send_report(report)
                    if ts:
                        log.info("10-min report sent (ts=%s)", ts)
                    last_report_at = time.monotonic()
                except Exception:
                    log.exception("Failed to send 10-min report")

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
