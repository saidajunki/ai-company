"""Git Publisher — GitHub CLI経由でリポジトリ作成・コミット・プッシュを行う.

ShellExecutor を使い gh / git コマンドを実行する。
GITHUB_TOKEN 環境変数が未設定の場合はエラーを返す。
全操作の結果は PublishResult で返し、例外は送出しない。

Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from shell_executor import execute_shell

logger = logging.getLogger(__name__)


@dataclass
class PublishResult:
    """公開操作の結果."""

    success: bool
    message: str
    repo_url: str | None = None


class GitPublisher:
    """GitHub CLI / git コマンドで成果物を公開する."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def _check_token(self) -> PublishResult | None:
        """GITHUB_TOKEN 環境変数を確認する.

        Returns:
            トークン未設定の場合は失敗の PublishResult、設定済みなら None。
        """
        if not os.environ.get("GITHUB_TOKEN"):
            return PublishResult(
                success=False,
                message="GITHUB_TOKEN 環境変数が未設定です",
            )
        return None

    def create_repo(self, name: str, description: str) -> PublishResult:
        """パブリック GitHub リポジトリを作成する.

        Args:
            name: リポジトリ名
            description: リポジトリの説明

        Returns:
            PublishResult: 作成結果
        """
        token_err = self._check_token()
        if token_err is not None:
            return token_err

        command = (
            f'gh repo create "{name}" '
            f'--public '
            f'--description "{description}" '
            f'--confirm'
        )

        try:
            result = execute_shell(command=command, cwd=self.work_dir)
        except Exception:
            logger.exception("create_repo で予期しないエラー")
            return PublishResult(success=False, message="リポジトリ作成中に予期しないエラーが発生しました")

        if result.timed_out:
            return PublishResult(success=False, message="リポジトリ作成がタイムアウトしました")

        if result.return_code != 0:
            err = result.stderr.strip() or result.stdout.strip()
            return PublishResult(
                success=False,
                message=f"リポジトリ作成に失敗しました: {err}",
            )

        # stdout からリポジトリURLを抽出
        repo_url = result.stdout.strip().split("\n")[-1].strip()
        if not repo_url.startswith("http"):
            repo_url = f"https://github.com/{name}"

        return PublishResult(
            success=True,
            message=f"リポジトリ {name} を作成しました",
            repo_url=repo_url,
        )

    def commit_and_push(
        self,
        repo_path: Path,
        message: str,
        files: list[str] | None = None,
    ) -> PublishResult:
        """ファイルを git add / commit / push する.

        Args:
            repo_path: git リポジトリのパス
            message: コミットメッセージ
            files: add するファイルリスト（None なら全ファイル）

        Returns:
            PublishResult: コミット・プッシュ結果
        """
        token_err = self._check_token()
        if token_err is not None:
            return token_err

        # git add
        if files:
            add_targets = " ".join(f'"{f}"' for f in files)
            add_cmd = f"git add {add_targets}"
        else:
            add_cmd = "git add -A"

        result = execute_shell(command=add_cmd, cwd=repo_path)
        if result.return_code != 0:
            err = result.stderr.strip() or result.stdout.strip()
            return PublishResult(success=False, message=f"git add に失敗しました: {err}")

        # git commit
        commit_cmd = f'git commit -m "{message}"'
        result = execute_shell(command=commit_cmd, cwd=repo_path)
        if result.return_code != 0:
            err = result.stderr.strip() or result.stdout.strip()
            return PublishResult(success=False, message=f"git commit に失敗しました: {err}")

        # git push
        push_cmd = "git push origin HEAD"
        result = execute_shell(command=push_cmd, cwd=repo_path)
        if result.return_code != 0:
            err = result.stderr.strip() or result.stdout.strip()
            return PublishResult(success=False, message=f"git push に失敗しました: {err}")

        return PublishResult(
            success=True,
            message=f"コミットしてプッシュしました: {message}",
        )
