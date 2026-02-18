"""Persistent employee store (正社員AIレジストリ + 個別メモリ)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import EmployeeEntry
from ndjson_store import ndjson_append, ndjson_read

DEFAULT_EMPLOYEE_BUDGET_USD = 1.0


class EmployeeStore:
    """永続社員AIを管理する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        company_root = base_dir / "companies" / company_id
        self._path = company_root / "state" / "employees.ndjson"
        self._memory_root = company_root / "knowledge" / "employees"

    def create(
        self,
        *,
        name: str,
        role: str,
        purpose: str,
        model: str,
        budget_limit_usd: float = DEFAULT_EMPLOYEE_BUDGET_USD,
    ) -> EmployeeEntry:
        now = datetime.now(timezone.utc)
        entry = EmployeeEntry(
            employee_id=f"emp-{uuid4().hex[:6]}",
            name=name.strip(),
            role=role.strip(),
            purpose=purpose.strip(),
            model=(model or "").strip() or "openai/gpt-4.1-mini",
            budget_limit_usd=max(0.05, float(budget_limit_usd)),
            status="active",
            created_at=now,
            updated_at=now,
        )
        ndjson_append(self._path, entry)
        self._ensure_memory_file(entry.employee_id, entry)
        self.append_memory(
            entry.employee_id,
            title="採用",
            content=(
                f"名前: {entry.name}\n"
                f"役割: {entry.role}\n"
                f"目的: {entry.purpose}\n"
                f"モデル: {entry.model}\n"
                f"予算上限: ${entry.budget_limit_usd:.2f}"
            ),
            source="employee_create",
        )
        return entry

    def ensure_active(
        self,
        *,
        name: str,
        role: str,
        purpose: str,
        model: str,
        budget_limit_usd: float = DEFAULT_EMPLOYEE_BUDGET_USD,
    ) -> tuple[EmployeeEntry, bool]:
        existing = self.find_active_by_name(name)
        if existing is not None:
            return existing, False
        return (
            self.create(
                name=name,
                role=role,
                purpose=purpose,
                model=model,
                budget_limit_usd=budget_limit_usd,
            ),
            True,
        )

    def update_status(self, employee_id: str, status: str) -> EmployeeEntry:
        current = self.get_by_id(employee_id)
        if current is None:
            raise ValueError(f"employee not found: {employee_id}")
        if status not in ("active", "inactive"):
            raise ValueError(f"invalid employee status: {status}")
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        ndjson_append(self._path, updated)
        return updated

    def get_by_id(self, employee_id: str) -> EmployeeEntry | None:
        for entry in self._list_latest():
            if entry.employee_id == employee_id:
                return entry
        return None

    def find_by_name(self, name: str) -> EmployeeEntry | None:
        normalized = _normalize_name(name)
        if not normalized:
            return None
        for entry in self._list_latest():
            if _normalize_name(entry.name) == normalized:
                return entry
        return None

    def find_active_by_name(self, name: str) -> EmployeeEntry | None:
        e = self.find_by_name(name)
        if e is None or e.status != "active":
            return None
        return e

    def find_active_by_role(self, role: str) -> EmployeeEntry | None:
        normalized = (role or "").strip().lower()
        if not normalized:
            return None
        candidates = [e for e in self.list_active() if (e.role or "").strip().lower() == normalized]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.updated_at, reverse=True)
        return candidates[0]

    def resolve_active(self, key: str) -> EmployeeEntry | None:
        k = (key or "").strip()
        if not k:
            return None
        if k.startswith("emp-"):
            e = self.get_by_id(k)
            return e if e is not None and e.status == "active" else None
        e = self.find_active_by_name(k)
        return e

    def list_active(self) -> list[EmployeeEntry]:
        return [e for e in self._list_latest() if e.status == "active"]

    def list_all(self) -> list[EmployeeEntry]:
        return self._list_latest()

    def format_roster(self, *, limit: int = 30) -> str:
        entries = self.list_active()
        if not entries:
            return "（正社員AIなし）"
        entries.sort(key=lambda x: x.updated_at, reverse=True)
        lines = [f"- {e.name}（id={e.employee_id} / role={e.role} / model={e.model}）" for e in entries[:limit]]
        return "\n".join(lines)

    def append_memory(self, employee_id: str, *, title: str, content: str, source: str) -> None:
        path = self.memory_path(employee_id)
        self._memory_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        block = [
            "",
            f"## {now} | {title.strip() or 'メモ'}",
            f"- source: {source}",
            "",
            content.strip(),
            "",
        ]
        if not path.exists():
            path.write_text(f"# Employee Memory: {employee_id}\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(block))

    def read_memory(self, employee_id: str, *, max_chars: int = 3500) -> str:
        path = self.memory_path(employee_id)
        if not path.exists():
            return ""
        raw = path.read_text(encoding="utf-8")
        if len(raw) <= max_chars:
            return raw
        return raw[-max_chars:]

    def memory_path(self, employee_id: str) -> Path:
        return self._memory_root / employee_id / "MEMORY.md"

    def _ensure_memory_file(self, employee_id: str, entry: EmployeeEntry | None = None) -> None:
        path = self.memory_path(employee_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        lines = [f"# Employee Memory: {employee_id}"]
        if entry is not None:
            lines.extend(
                [
                    "",
                    f"- name: {entry.name}",
                    f"- role: {entry.role}",
                    f"- purpose: {entry.purpose}",
                    f"- model: {entry.model}",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _list_latest(self) -> list[EmployeeEntry]:
        entries = ndjson_read(self._path, EmployeeEntry)
        latest: dict[str, EmployeeEntry] = {}
        for entry in entries:
            latest[entry.employee_id] = entry
        return list(latest.values())


def _normalize_name(name: str) -> str:
    return " ".join((name or "").replace("　", " ").split()).lower()
