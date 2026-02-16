"""Tests for git_publisher — GitPublisher のユニットテスト.

Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from shell_executor import ShellResult
from git_publisher import GitPublisher, PublishResult


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _shell_ok(stdout: str = "") -> ShellResult:
    return ShellResult(
        command="...",
        stdout=stdout,
        stderr="",
        return_code=0,
        timed_out=False,
        duration_seconds=0.1,
    )


def _shell_error(stderr: str = "error", return_code: int = 1) -> ShellResult:
    return ShellResult(
        command="...",
        stdout="",
        stderr=stderr,
        return_code=return_code,
        timed_out=False,
        duration_seconds=0.1,
    )


def _shell_timeout() -> ShellResult:
    return ShellResult(
        command="...",
        stdout="",
        stderr="",
        return_code=-1,
        timed_out=True,
        duration_seconds=60.0,
    )


# ---------------------------------------------------------------------------
# create_repo 成功
# ---------------------------------------------------------------------------

class TestCreateRepoSuccess:

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_creates_repo_successfully(self, mock_shell):
        mock_shell.return_value = _shell_ok("https://github.com/org/my-repo")

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.create_repo("org/my-repo", "A test repo")

        assert result.success is True
        assert "my-repo" in result.message
        assert result.repo_url == "https://github.com/org/my-repo"

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_command_contains_name_and_description(self, mock_shell):
        mock_shell.return_value = _shell_ok("https://github.com/org/my-repo")

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        gp.create_repo("org/my-repo", "My description")

        cmd = mock_shell.call_args.kwargs.get("command") or mock_shell.call_args[0][0]
        assert "org/my-repo" in cmd
        assert "My description" in cmd

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_uses_gh_repo_create(self, mock_shell):
        mock_shell.return_value = _shell_ok("https://github.com/org/my-repo")

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        gp.create_repo("org/my-repo", "desc")

        cmd = mock_shell.call_args.kwargs.get("command") or mock_shell.call_args[0][0]
        assert "gh repo create" in cmd
        assert "--public" in cmd


# ---------------------------------------------------------------------------
# create_repo 失敗
# ---------------------------------------------------------------------------

class TestCreateRepoFailure:

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_shell_error_returns_failure(self, mock_shell):
        mock_shell.return_value = _shell_error(stderr="repository already exists")

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.create_repo("org/my-repo", "desc")

        assert result.success is False
        assert "repository already exists" in result.message

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_timeout_returns_failure(self, mock_shell):
        mock_shell.return_value = _shell_timeout()

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.create_repo("org/my-repo", "desc")

        assert result.success is False
        assert "タイムアウト" in result.message

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_exception_returns_failure(self, mock_shell):
        mock_shell.side_effect = RuntimeError("boom")

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.create_repo("org/my-repo", "desc")

        assert result.success is False


# ---------------------------------------------------------------------------
# GITHUB_TOKEN 未設定
# ---------------------------------------------------------------------------

class TestGithubTokenMissing:

    @patch.dict("os.environ", {}, clear=True)
    def test_create_repo_without_token(self):
        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.create_repo("org/my-repo", "desc")

        assert result.success is False
        assert "GITHUB_TOKEN" in result.message

    @patch.dict("os.environ", {}, clear=True)
    def test_commit_and_push_without_token(self):
        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.commit_and_push(Path("/tmp/repo"), "msg")

        assert result.success is False
        assert "GITHUB_TOKEN" in result.message

    @patch.dict("os.environ", {"GITHUB_TOKEN": ""})
    def test_empty_token_treated_as_missing(self):
        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.create_repo("org/my-repo", "desc")

        assert result.success is False
        assert "GITHUB_TOKEN" in result.message


# ---------------------------------------------------------------------------
# commit_and_push 成功
# ---------------------------------------------------------------------------

class TestCommitAndPushSuccess:

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_commit_and_push_all_files(self, mock_shell):
        mock_shell.return_value = _shell_ok()

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.commit_and_push(Path("/tmp/repo"), "initial commit")

        assert result.success is True
        assert mock_shell.call_count == 3  # add, commit, push

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_commit_message_in_command(self, mock_shell):
        mock_shell.return_value = _shell_ok()

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        gp.commit_and_push(Path("/tmp/repo"), "my commit msg")

        # commit is the second call
        commit_call = mock_shell.call_args_list[1]
        cmd = commit_call.kwargs.get("command") or commit_call[0][0]
        assert "my commit msg" in cmd

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_specific_files_added(self, mock_shell):
        mock_shell.return_value = _shell_ok()

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        gp.commit_and_push(Path("/tmp/repo"), "msg", files=["a.py", "b.py"])

        add_call = mock_shell.call_args_list[0]
        cmd = add_call.kwargs.get("command") or add_call[0][0]
        assert "a.py" in cmd
        assert "b.py" in cmd

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_no_files_uses_add_all(self, mock_shell):
        mock_shell.return_value = _shell_ok()

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        gp.commit_and_push(Path("/tmp/repo"), "msg", files=None)

        add_call = mock_shell.call_args_list[0]
        cmd = add_call.kwargs.get("command") or add_call[0][0]
        assert "git add -A" in cmd


# ---------------------------------------------------------------------------
# commit_and_push 失敗
# ---------------------------------------------------------------------------

class TestCommitAndPushFailure:

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_add_failure(self, mock_shell):
        mock_shell.return_value = _shell_error(stderr="pathspec not found")

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.commit_and_push(Path("/tmp/repo"), "msg")

        assert result.success is False
        assert "git add" in result.message

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_commit_failure(self, mock_shell):
        mock_shell.side_effect = [_shell_ok(), _shell_error(stderr="nothing to commit")]

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.commit_and_push(Path("/tmp/repo"), "msg")

        assert result.success is False
        assert "git commit" in result.message

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("git_publisher.execute_shell")
    def test_push_failure(self, mock_shell):
        mock_shell.side_effect = [_shell_ok(), _shell_ok(), _shell_error(stderr="rejected")]

        gp = GitPublisher(work_dir=Path("/tmp/work"))
        result = gp.commit_and_push(Path("/tmp/repo"), "msg")

        assert result.success is False
        assert "git push" in result.message
