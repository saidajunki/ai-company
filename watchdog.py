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
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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


def _docker_exec(container_name: str, cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, *cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "docker exec timed out"
    except Exception as exc:
        return 1, "", str(exc)


def read_heartbeat(
    *,
    heartbeat_path: Optional[str],
    container_name: str,
    company_id: str,
    base_dir_in_container: str = "/app/data",
) -> Optional[Dict[str, Any]]:
    """Heartbeat JSON を読む（ホストパス優先、なければ docker exec で読む）。"""
    if heartbeat_path:
        try:
            with open(heartbeat_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Heartbeat file not found: %s", heartbeat_path)
        except Exception as exc:
            logger.error("Failed to read heartbeat file %s: %s", heartbeat_path, exc)

    heartbeat_in_container = f"{base_dir_in_container}/companies/{company_id}/state/heartbeat.json"
    rc, out, err = _docker_exec(container_name, ["cat", heartbeat_in_container], timeout=10)
    if rc != 0:
        logger.warning("Failed to read heartbeat via docker exec (rc=%d): %s", rc, err.strip())
        return None
    try:
        return json.loads(out)
    except Exception as exc:
        logger.error("Failed to parse heartbeat JSON from container: %s", exc)
        return None


def heartbeat_is_stale(
    data: Optional[Dict[str, Any]], threshold_minutes: int = 20
) -> Tuple[bool, float]:
    """Returns (is_stale, elapsed_minutes)."""
    if not data:
        return True, float("inf")
    try:
        updated_at_str = data["updated_at"]
        updated_at = datetime.fromisoformat(updated_at_str)
    except Exception:
        return True, float("inf")

    now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    elapsed = (now - updated_at).total_seconds() / 60.0
    return elapsed >= threshold_minutes, elapsed


def request_manager_restart(
    container_name: str,
    company_id: str,
    *,
    base_dir_in_container: str = "/app/data",
) -> Tuple[bool, str]:
    """(A) 会社コンテナ内の manager 再起動を要求する（supervisor の flag）。"""
    flag_path = f"{base_dir_in_container}/companies/{company_id}/state/restart_manager.flag"
    cmd = ["sh", "-lc", f"mkdir -p $(dirname {flag_path}) && date > {flag_path}"]
    rc, out, err = _docker_exec(container_name, cmd, timeout=10)
    if rc == 0:
        return True, out.strip()
    return False, err.strip() or out.strip()


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
    heartbeat_path = os.environ.get("HEARTBEAT_PATH", "").strip() or None
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    container_name = os.environ.get("CONTAINER_NAME", "ai-company")
    company_id = os.environ.get("COMPANY_ID", "alpha")
    threshold_minutes = int(os.environ.get("THRESHOLD_MINUTES", "20"))
    base_dir_in_container = os.environ.get("BASE_DIR_IN_CONTAINER", "/app/data")

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL is not set; Slack notifications disabled.")

    data = read_heartbeat(
        heartbeat_path=heartbeat_path,
        container_name=container_name,
        company_id=company_id,
        base_dir_in_container=base_dir_in_container,
    )
    stale, elapsed = heartbeat_is_stale(data, threshold_minutes=threshold_minutes)

    if not stale:
        logger.info("Heartbeat OK — nothing to do.")
        return

    # Heartbeat is stale → notify, then (A) restart manager process, then (B) restart container if needed
    msg = (
        f"⚠️ Watchdog: heartbeat が stale です（{elapsed:.1f}分、閾値={threshold_minutes}分）。\n"
        f"(A) manager 再起動を試みます: `{container_name}` / company_id={company_id}"
    )
    if webhook_url:
        notify_slack(webhook_url, msg)

    ok, out = request_manager_restart(
        container_name,
        company_id,
        base_dir_in_container=base_dir_in_container,
    )
    if ok:
        logger.info("Requested manager restart via flag.")
        time.sleep(30)
        data2 = read_heartbeat(
            heartbeat_path=heartbeat_path,
            container_name=container_name,
            company_id=company_id,
            base_dir_in_container=base_dir_in_container,
        )
        stale2, elapsed2 = heartbeat_is_stale(data2, threshold_minutes=threshold_minutes)
        if not stale2:
            msg2 = f"✅ Watchdog: manager 再起動で復帰しました（経過={elapsed2:.1f}分）。"
            if webhook_url:
                notify_slack(webhook_url, msg2)
            return
        logger.warning("Manager restart did not recover heartbeat (elapsed=%.1f).", elapsed2)
    else:
        logger.warning("Manager restart request failed: %s", out)

    success, output = restart_container(container_name)

    if success:
        msg3 = f"🔄 Watchdog: コンテナ `{container_name}` を再起動しました。"
    else:
        msg3 = f"⚠️ Watchdog: コンテナ `{container_name}` の再起動に失敗しました。\n```{output}```"

    if webhook_url:
        notify_slack(webhook_url, msg3)


if __name__ == "__main__":
    main()
