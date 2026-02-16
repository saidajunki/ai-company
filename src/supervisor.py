"""Process supervisor for the AI Company container.

Runs the main manager process as a child and supports in-place restarts
without restarting the whole container.

Watchdog can request a restart by creating (touching) the flag file:
  companies/<company_id>/state/restart_manager.flag
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [supervisor] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ai-company.supervisor")


def _restart_flag_path() -> Path:
    base_dir = Path(os.environ.get("BASE_DIR", "/app/data"))
    company_id = os.environ.get("COMPANY_ID", "alpha")
    return base_dir / "companies" / company_id / "state" / "restart_manager.flag"


def _start_child() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log.info("Starting child: %s -m main", sys.executable)
    return subprocess.Popen([sys.executable, "-m", "main"], env=env)


def _terminate_child(proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    log.info("Terminating child (pid=%s)...", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        log.warning("Child did not exit in %.1fs, killing...", timeout_s)
        proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            log.error("Child still did not exit after kill (pid=%s)", proc.pid)


def main() -> None:
    shutdown = False

    def _handle_signal(signum: int, _frame) -> None:
        nonlocal shutdown
        log.info("Signal %s received, shutting down supervisor...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    restart_flag = _restart_flag_path()
    last_flag_mtime: float | None = None

    child = _start_child()
    backoff_s = 1.0

    while True:
        if shutdown:
            _terminate_child(child)
            break

        # Restart request flag (host watchdog can touch this file)
        try:
            if restart_flag.exists():
                mtime = restart_flag.stat().st_mtime
                if last_flag_mtime is None or mtime > last_flag_mtime:
                    last_flag_mtime = mtime
                    log.warning("Restart flag detected: %s", restart_flag)
                    _terminate_child(child)
                    child = _start_child()
                    backoff_s = 1.0
                    try:
                        restart_flag.unlink()
                    except Exception:
                        log.warning("Failed to remove restart flag: %s", restart_flag, exc_info=True)
        except Exception:
            log.warning("Failed to check restart flag: %s", restart_flag, exc_info=True)

        # Child exit monitoring
        rc = child.poll()
        if rc is not None:
            log.error("Child exited (rc=%s). Restarting in %.1fs...", rc, backoff_s)
            time.sleep(backoff_s)
            child = _start_child()
            backoff_s = min(backoff_s * 2, 30.0)

        time.sleep(1.0)


if __name__ == "__main__":
    main()

