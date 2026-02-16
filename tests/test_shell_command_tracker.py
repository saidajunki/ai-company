"""Tests for ShellCommandTracker."""

from src.shell_command_tracker import ShellCommandRecord, ShellCommandTracker


class TestShellCommandRecord:
    def test_fields(self):
        rec = ShellCommandRecord(command="echo hi", return_code=0)
        assert rec.command == "echo hi"
        assert rec.return_code == 0


class TestShellCommandTracker:
    def test_empty_tracker(self):
        t = ShellCommandTracker()
        assert not t.had_any_commands()
        assert not t.all_failed()
        assert not t.has_any_success()
        assert t.failed_commands() == []

    def test_record_single_success(self):
        t = ShellCommandTracker()
        t.record("echo ok", 0)
        assert t.had_any_commands()
        assert not t.all_failed()
        assert t.has_any_success()
        assert t.failed_commands() == []

    def test_record_single_failure(self):
        t = ShellCommandTracker()
        t.record("false", 1)
        assert t.had_any_commands()
        assert t.all_failed()
        assert not t.has_any_success()
        assert len(t.failed_commands()) == 1
        assert t.failed_commands()[0].command == "false"
        assert t.failed_commands()[0].return_code == 1

    def test_mixed_results(self):
        t = ShellCommandTracker()
        t.record("cmd1", 1)
        t.record("cmd2", 0)
        t.record("cmd3", 2)
        assert t.had_any_commands()
        assert not t.all_failed()
        assert t.has_any_success()
        failed = t.failed_commands()
        assert len(failed) == 2
        assert failed[0].command == "cmd1"
        assert failed[1].command == "cmd3"

    def test_all_failed_multiple(self):
        t = ShellCommandTracker()
        t.record("a", 1)
        t.record("b", 127)
        t.record("c", 255)
        assert t.all_failed()
        assert not t.has_any_success()
        assert len(t.failed_commands()) == 3

    def test_all_success(self):
        t = ShellCommandTracker()
        t.record("x", 0)
        t.record("y", 0)
        assert not t.all_failed()
        assert t.has_any_success()
        assert t.failed_commands() == []

    def test_preserves_order(self):
        t = ShellCommandTracker()
        t.record("first", 0)
        t.record("second", 1)
        t.record("third", 0)
        failed = t.failed_commands()
        assert len(failed) == 1
        assert failed[0].command == "second"
