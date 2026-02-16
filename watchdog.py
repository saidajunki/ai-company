#!/usr/bin/env python3
"""Watchdog: ホスト側スタンドアロン heartbeat 監視スクリプト.

コンテナ外で動作し、heartbeat.json を監視して stale なら
docker restart → Slack 通知を行う。

外部依存なし（stdlib のみ: json, os, subprocess, urllib.request, datetime, logging）。

Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.request
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def check_heartbeat(heartbeat_path: str, threshold_minutes: int = 20) -> bool:
    """heartbeat.json を読み、stale かどうかを返す.

    Returns:
        True if stale (再起動が必要), False if fresh.
    """
    try:
        with open(heartbeat_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("Heartbeat file not found: %s", heartbeat_path)
        return True
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read heartbeat file: %s", exc)
        return True

    try:
        updated_at_str = data["updated_at"]
        updated_at = datetime.fromisoformat(updated_at_str)
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Invalid heartbeat data: %s", exc)
        return True

    now = datetime.now(timezone.utc)
    # Ensure updated_at is timezone-aware for comparison
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    elapsed = (now - updated_at).total_seconds()
    is_stale = elapsed >= threshold_minutes * 60
    if is_stale:
        logger.info(
            "Heartbeat stale: %.1f minutes elapsed (threshold=%d)",
            elapsed / 60,
            threshold_minutes,
        )
    else:
        logger.info(
            "Heartbeat fresh: %.1f minutes elapsed (threshold=%d)",
            elapsed / 60,
            threshold_minutes,
        )
    return is_stale


def restart_container(container_name: str = "ai-company") -> tuple[bool, str]:
    """docker restart を実行する.

    Returns:
        (success, output) — success=True なら stdout, False なら stderr.
    """
    try:
        result = subprocess.run(
            ["docker", "restart", container_name],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Container '%s' restarted successfully.", container_name)
            return True, result.stdout.strip()
        else:
            logger.error(
                "docker restart failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        msg = f"docker restart timed out for '{container_name}'"
        logger.error(msg)
        return False, msg
    except Exception as exc:
        msg = f"docker restart error: {exc}"
        logger.error(msg)
        return False, msg


def notify_slack(webhook_url: str, message: str) -> bool:
    """Slack Webhook に通知を送信する.

    Returns:
        True on success, False on failure.
    """
    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            logger.info("Slack notification sent (status=%d).", resp.status)
            return True
    except Exception as exc:
        logger.error("Slack notification failed: %s", exc)
        return False


def main() -> None:
    """エントリポイント: 環境変数 → heartbeat チェック → restart → Slack 通知."""
    heartbeat_path = os.environ.get("HEARTBEAT_PATH", "")
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    container_name = os.environ.get("CONTAINER_NAME", "ai-company")

    if not heartbeat_path:
        logger.error("HEARTBEAT_PATH is not set.")
        return
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL is not set; Slack notifications disabled.")

    stale = check_heartbeat(heartbeat_path)

    if not stale:
        logger.info("Heartbeat OK — nothing to do.")
        return

    # Heartbeat is stale → restart container
    success, output = restart_container(container_name)

    if success:
        msg = f"🔄 Watchdog: コンテナ `{container_name}` を再起動しました。"
    else:
        msg = f"⚠️ Watchdog: コンテナ `{container_name}` の再起動に失敗しました。\n```{output}```"

    if webhook_url:
        notify_slack(webhook_url, msg)


if __name__ == "__main__":
    main()
