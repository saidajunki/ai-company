"""Shell command result tracking for false-completion prevention.

タスク実行中のシェルコマンドの return_code を記録・集計し、
全コマンド失敗時の虚偽完了を防止する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShellCommandRecord:
    """シェルコマンド実行記録."""

    command: str
    return_code: int


class ShellCommandTracker:
    """タスク実行中のシェルコマンド結果を追跡する."""

    def __init__(self) -> None:
        self._results: list[ShellCommandRecord] = []

    def record(self, command: str, return_code: int) -> None:
        """シェルコマンドの結果を記録する."""
        self._results.append(ShellCommandRecord(command=command, return_code=return_code))

    def had_any_commands(self) -> bool:
        """シェルコマンドが1つ以上実行されたか."""
        return len(self._results) > 0

    def all_failed(self) -> bool:
        """全てのシェルコマンドが失敗したか (return_code != 0).

        コマンドが記録されていない場合は False を返す。
        """
        if not self._results:
            return False
        return all(r.return_code != 0 for r in self._results)

    def has_any_success(self) -> bool:
        """少なくとも1つ成功したコマンドがあるか."""
        return any(r.return_code == 0 for r in self._results)

    def failed_commands(self) -> list[ShellCommandRecord]:
        """失敗したコマンドのリストを返す."""
        return [r for r in self._results if r.return_code != 0]
