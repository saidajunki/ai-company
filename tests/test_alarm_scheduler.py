from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alarm_scheduler import AlarmScheduler
from employee_store import EmployeeStore


class _DummySubAgentRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def spawn(
        self,
        *,
        name: str,
        role: str,
        task_description: str,
        budget_limit_usd: float,
        model: str | None = None,
        ignore_wip_limit: bool = False,
        persistent_employee: dict | None = None,
    ) -> str:
        self.calls.append(
            {
                "name": name,
                "role": role,
                "task_description": task_description,
                "budget_limit_usd": budget_limit_usd,
                "model": model,
                "ignore_wip_limit": ignore_wip_limit,
                "persistent_employee": persistent_employee,
            }
        )
        return "ok"


class _DummyAgentRegistry:
    def get(self, _agent_id: str):
        return None


class _DummyManager:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.activity_logs: list[str] = []
        self.sub_agent_runner = _DummySubAgentRunner()
        self.agent_registry = _DummyAgentRegistry()
        self.employee_store = None
        self._slack_default_channel = None
        self._slack_last_channel = None

    def process_message(
        self,
        text: str,
        user_id: str,
        slack_channel: str | None = None,
        slack_thread_ts: str | None = None,
        slack_thread_context: str | None = None,
    ) -> None:
        _ = user_id, slack_channel, slack_thread_ts, slack_thread_context
        self.messages.append(text)

    def _activity_log(self, text: str) -> None:
        self.activity_logs.append(text)


def _new_scheduler(tmp_path: Path) -> AlarmScheduler:
    return AlarmScheduler(tmp_path, "alpha")


def test_alarm_add_once_control_command(tmp_path: Path) -> None:
    scheduler = _new_scheduler(tmp_path)
    handled, reply = scheduler.handle_control_command(
        "alarm add once +10m | ceo | 10分後に進捗を確認する",
        actor_id="ceo",
        actor_role="ceo",
        actor_model="openai/gpt-4.1-mini",
    )
    assert handled is True
    assert "アラーム登録完了" in reply

    state = scheduler._load_state()
    alarms = state["alarms"]
    assert len(alarms) == 1
    assert alarms[0]["schedule_type"] == "once"
    assert alarms[0]["target"] == "ceo"


def test_alarm_add_cron_with_role_options(tmp_path: Path) -> None:
    scheduler = _new_scheduler(tmp_path)
    handled, reply = scheduler.handle_control_command(
        "alarm add cron 0 * * * * | role:web-developer;budget=0.5;model=openai/gpt-4.1 | 毎時の死活確認",
        actor_id="ceo",
        actor_role="ceo",
        actor_model="openai/gpt-4.1-mini",
    )
    assert handled is True
    assert "cron" in reply

    listed_handled, listed = scheduler.handle_control_command(
        "alarm list",
        actor_id="ceo",
        actor_role="ceo",
        actor_model="openai/gpt-4.1-mini",
    )
    assert listed_handled is True
    assert "role:web-developer" in listed
    assert "0 * * * *" in listed


def test_alarm_add_once_self_target_for_subagent(tmp_path: Path) -> None:
    scheduler = _new_scheduler(tmp_path)
    handled, _reply = scheduler.handle_control_command(
        "alarm add once +5m | self | 進捗再確認",
        actor_id="sub-001",
        actor_role="web-developer",
        actor_model="openai/gpt-4.1",
    )
    assert handled is True
    state = scheduler._load_state()
    assert state["alarms"][0]["target"] == "role:web-developer"


def test_alarm_cancel(tmp_path: Path) -> None:
    scheduler = _new_scheduler(tmp_path)
    entry = scheduler.add_once(
        run_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        prompt="テスト",
        owner_agent="ceo",
        target="ceo",
    )
    ok = scheduler.cancel(entry["alarm_id"])
    assert ok is True
    state = scheduler._load_state()
    assert state["alarms"][0]["status"] == "canceled"


def test_alarm_tick_executes_due_entries(tmp_path: Path) -> None:
    scheduler = _new_scheduler(tmp_path)
    ceo_entry = scheduler.add_once(
        run_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        prompt="CEOタスク",
        owner_agent="ceo",
        target="ceo",
    )
    worker_entry = scheduler.add_once(
        run_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        prompt="社員タスク",
        owner_agent="ceo",
        target="role:web-developer",
        budget_limit_usd=0.7,
    )

    state = scheduler._load_state()
    for entry in state["alarms"]:
        if entry["alarm_id"] in (ceo_entry["alarm_id"], worker_entry["alarm_id"]):
            entry["next_run_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    scheduler._save_state(state)

    manager = _DummyManager()
    executed = scheduler.tick(manager)
    assert executed == 2
    assert manager.messages == ["CEOタスク"]
    assert len(manager.sub_agent_runner.calls) == 1
    assert manager.sub_agent_runner.calls[0]["role"] == "web-developer"
    assert manager.sub_agent_runner.calls[0]["ignore_wip_limit"] is True


def test_alarm_tick_targets_persistent_employee(tmp_path: Path) -> None:
    scheduler = _new_scheduler(tmp_path)
    employee_store = EmployeeStore(tmp_path, "alpha")
    employee = employee_store.create(
        name="山田 太郎",
        role="web-developer",
        purpose="継続開発担当",
        model="openai/gpt-4.1",
        budget_limit_usd=0.9,
    )
    entry = scheduler.add_once(
        run_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        prompt="継続タスクを実行",
        owner_agent="ceo",
        target=f"employee:{employee.name}",
        budget_limit_usd=1.0,
    )

    state = scheduler._load_state()
    for alarm in state["alarms"]:
        if alarm["alarm_id"] == entry["alarm_id"]:
            alarm["next_run_at"] = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    scheduler._save_state(state)

    manager = _DummyManager()
    manager.employee_store = employee_store
    executed = scheduler.tick(manager)
    assert executed == 1
    assert len(manager.sub_agent_runner.calls) == 1
    call = manager.sub_agent_runner.calls[0]
    assert call["name"] == employee.name
    assert call["role"] == employee.role
    assert call["persistent_employee"] is not None
