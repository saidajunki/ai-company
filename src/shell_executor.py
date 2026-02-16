"""Shell Executor — コンテナ内でシェルコマンドを安全に実行する.

subprocess.run を使用してコマンドを実行し、stdout/stderr/return_code をキャプチャする。
タイムアウト時はプロセスを強制終了し、出力は最大文字数に切り詰める。

Requirements: 6.1, 6.2, 6.3, 6.5, 6.6
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShellResult:
    """シェルコマンド実行結果."""

    command: str
    stdout: str          # 最大10,000文字に切り詰め
    stderr: str          # 最大10,000文字に切り詰め
    return_code: int
    timed_out: bool
    duration_seconds: float


def truncate_output(text: str, max_chars: int) -> str:
    """出力テキストを最大文字数に切り詰める.

    元のテキストが max_chars 以下の場合はそのまま返す。
    超過する場合は先頭 max_chars 文字に切り詰める。
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def execute_shell(
    command: str,
    cwd: Path | None = None,
    timeout: int = 60,
    max_output_chars: int = 10_000,
) -> ShellResult:
    """シェルコマンドを実行し、結果をキャプチャする.

    Args:
        command: 実行するシェルコマンド
        cwd: 作業ディレクトリ（会社専用領域）
        timeout: タイムアウト秒数（デフォルト60秒）
        max_output_chars: stdout/stderrの最大文字数（デフォルト10,000）

    Returns:
        ShellResult: コマンド実行結果
    """
    start = time.monotonic()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        duration = time.monotonic() - start

        return ShellResult(
            command=command,
            stdout=truncate_output(result.stdout, max_output_chars),
            stderr=truncate_output(result.stderr, max_output_chars),
            return_code=result.returncode,
            timed_out=False,
            duration_seconds=round(duration, 3),
        )

    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        return ShellResult(
            command=command,
            stdout=truncate_output(stdout, max_output_chars),
            stderr=truncate_output(stderr, max_output_chars),
            return_code=-1,
            timed_out=True,
            duration_seconds=round(duration, 3),
        )
