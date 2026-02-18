from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from manager import Manager, init_company_directory


def _make_manager(tmp_path: Path) -> Manager:
    init_company_directory(tmp_path, "test-co")
    mgr = Manager(tmp_path, "test-co")
    return mgr


def test_employee_resume_command_uses_checkpoint(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    checkpoint = {
        "run_id": "run-abc123",
        "role": "developer",
        "model": "openai/gpt-4.1",
        "task_description": "WordPress検証タスク",
        "progress": ["12:00 shell: wp --info", "12:01 shell: php -v"],
        "pending_hint": "PHP導入後にwp再検証",
    }
    mgr.sub_agent_runner.get_run_checkpoint = MagicMock(return_value=checkpoint)
    mgr.sub_agent_runner.spawn = MagicMock(return_value="再開完了")

    handled, reply = mgr._handle_runtime_control_command(
        "employee resume run-abc123",
        actor_id="creator",
        actor_role="ceo",
        actor_model="openai/gpt-4.1-mini",
    )

    assert handled is True
    assert "run_id run-abc123" in reply
    assert "再開完了" in reply
    mgr.sub_agent_runner.spawn.assert_called_once()


def test_employee_resume_command_not_found(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)

    mgr.sub_agent_runner.get_run_checkpoint = MagicMock(return_value=None)

    handled, reply = mgr._handle_runtime_control_command(
        "employee resume run-missing",
        actor_id="creator",
        actor_role="ceo",
        actor_model="openai/gpt-4.1-mini",
    )

    assert handled is True
    assert "見つかりません" in reply
