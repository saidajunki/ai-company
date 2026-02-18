from __future__ import annotations

from pathlib import Path

from employee_store import EmployeeStore


def test_employee_create_and_resolve(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path, "alpha")
    entry = store.create(
        name="佐藤 花子",
        role="web-developer",
        purpose="Webサイトの継続開発",
        model="openai/gpt-4.1",
        budget_limit_usd=0.8,
    )
    assert entry.employee_id.startswith("emp-")
    assert store.get_by_id(entry.employee_id) is not None
    assert store.find_active_by_name("佐藤 花子") is not None
    assert store.find_active_by_role("web-developer") is not None


def test_employee_ensure_active_deduplicates_by_name(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path, "alpha")
    first, created1 = store.ensure_active(
        name="鈴木 太郎",
        role="accountant",
        purpose="会計管理",
        model="openai/gpt-4.1-mini",
        budget_limit_usd=0.5,
    )
    second, created2 = store.ensure_active(
        name="鈴木 太郎",
        role="accountant",
        purpose="会計管理",
        model="openai/gpt-4.1-mini",
        budget_limit_usd=0.5,
    )
    assert created1 is True
    assert created2 is False
    assert first.employee_id == second.employee_id


def test_employee_memory_append_and_read(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path, "alpha")
    entry = store.create(
        name="高橋 葵",
        role="marketing",
        purpose="広報運用",
        model="openai/gpt-4.1-mini",
        budget_limit_usd=0.4,
    )
    store.append_memory(
        entry.employee_id,
        title="運用メモ",
        content="SNS運用のKPIを更新した。",
        source="test",
    )
    text = store.read_memory(entry.employee_id, max_chars=2000)
    assert "運用メモ" in text
    assert "KPI" in text
