"""Alarm scheduler for recurring/one-shot autonomous actions."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
DEFAULT_BUDGET_LIMIT_USD = 1.0
MAX_HISTORY = 400


@dataclass
class _CronPattern:
    expr: str
    minutes: set[int]
    hours: set[int]
    days: set[int]
    months: set[int]
    weekdays: set[int]  # 0=Sun .. 6=Sat
    dom_any: bool
    dow_any: bool


class AlarmScheduler:
    """Persistent alarm scheduler (once + cron) for CEO/Sub-Agents."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._state_path = base_dir / "companies" / company_id / "state" / "alarms.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._state_path.exists():
            self._save_state({"alarms": [], "history": []})

    def now_text(self, now: datetime | None = None) -> str:
        t = (now or _utc_now()).astimezone(timezone.utc)
        jst = t.astimezone(JST)
        return (
            f"現在時刻は UTC {t.strftime('%Y-%m-%d %H:%M:%S')} / "
            f"JST {jst.strftime('%Y-%m-%d %H:%M:%S')} です。"
        )

    def handle_control_command(
        self,
        command: str,
        *,
        actor_id: str,
        actor_role: str,
        actor_model: str | None = None,
    ) -> tuple[bool, str]:
        cmd = (command or "").strip()
        if not cmd:
            return False, ""

        low = cmd.lower()
        if low in ("time now", "time.now", "now", "時刻", "現在時刻"):
            return True, self.now_text()

        if not low.startswith("alarm "):
            return False, ""

        self.ensure_initialized()

        m = re.match(r"^alarm\s+list(?:\s+(\d+))?\s*$", cmd, re.IGNORECASE)
        if m:
            limit = int(m.group(1) or 20)
            return True, self._render_alarm_list(limit=limit)

        m = re.match(r"^alarm\s+cancel\s+([0-9a-f]{8})\s*$", cmd, re.IGNORECASE)
        if m:
            alarm_id = m.group(1)
            canceled = self.cancel(alarm_id)
            if canceled:
                return True, f"⏰ アラームを停止しました: {alarm_id}"
            return True, f"⚠️ 指定のアラームが見つかりません: {alarm_id}"

        m = re.match(r"^alarm\s+add\s+once\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$", cmd, re.IGNORECASE)
        if m:
            when_text, target_spec, prompt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            run_at = _parse_datetime_like(when_text)
            target, model, budget, max_runs = self._parse_target_spec(
                target_spec,
                actor_role=actor_role,
                actor_model=actor_model,
            )
            entry = self.add_once(
                run_at=run_at,
                prompt=prompt,
                owner_agent=actor_id,
                target=target,
                model=model,
                budget_limit_usd=budget,
                max_runs=max_runs,
            )
            return True, (
                f"⏰ アラーム登録完了: {entry['alarm_id']} (once)\n"
                f"- 実行時刻: {_fmt_dt(entry['next_run_at'])}\n"
                f"- 実行先: {entry['target']}\n"
                f"- 内容: {entry['prompt']}"
            )

        m = re.match(r"^alarm\s+add\s+cron\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$", cmd, re.IGNORECASE)
        if m:
            expr, target_spec, prompt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            target, model, budget, max_runs = self._parse_target_spec(
                target_spec,
                actor_role=actor_role,
                actor_model=actor_model,
            )
            entry = self.add_cron(
                cron_expr=expr,
                prompt=prompt,
                owner_agent=actor_id,
                target=target,
                model=model,
                budget_limit_usd=budget,
                max_runs=max_runs,
            )
            return True, (
                f"⏰ アラーム登録完了: {entry['alarm_id']} (cron)\n"
                f"- cron: `{entry['cron_expr']}`\n"
                f"- 次回実行: {_fmt_dt(entry['next_run_at'])}\n"
                f"- 実行先: {entry['target']}\n"
                f"- 内容: {entry['prompt']}"
            )

        return True, (
            "⚠️ alarmコマンド形式が不正です。使用例:\n"
            "- `alarm add once 2026-02-19T12:00:00+09:00 | ceo | 状況報告を作成する`\n"
            "- `alarm add cron 0 * * * * | role:web-developer;budget=0.5 | 監視結果を確認して報告する`\n"
            "- `alarm list`\n"
            "- `alarm cancel <alarm_id>`\n"
            "- `time now`"
        )

    def add_once(
        self,
        *,
        run_at: datetime,
        prompt: str,
        owner_agent: str,
        target: str,
        model: str | None = None,
        budget_limit_usd: float = DEFAULT_BUDGET_LIMIT_USD,
        max_runs: int | None = None,
    ) -> dict[str, Any]:
        self.ensure_initialized()
        now = _utc_now()
        run_at_utc = run_at.astimezone(timezone.utc)
        if run_at_utc <= now:
            run_at_utc = now + timedelta(seconds=5)

        entry = {
            "alarm_id": uuid4().hex[:8],
            "status": "active",
            "schedule_type": "once",
            "cron_expr": None,
            "target": target,
            "prompt": prompt.strip(),
            "model": (model or "").strip() or None,
            "budget_limit_usd": float(max(0.05, budget_limit_usd)),
            "max_runs": int(max_runs) if max_runs is not None and max_runs > 0 else None,
            "run_count": 0,
            "failure_count": 0,
            "owner_agent": owner_agent,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_run_at": None,
            "next_run_at": run_at_utc.isoformat(),
            "last_error": None,
        }
        state = self._load_state()
        alarms = state.get("alarms")
        if not isinstance(alarms, list):
            alarms = []
        alarms.append(entry)
        state["alarms"] = alarms
        self._save_state(state)
        return entry

    def add_cron(
        self,
        *,
        cron_expr: str,
        prompt: str,
        owner_agent: str,
        target: str,
        model: str | None = None,
        budget_limit_usd: float = DEFAULT_BUDGET_LIMIT_USD,
        max_runs: int | None = None,
    ) -> dict[str, Any]:
        self.ensure_initialized()
        now = _utc_now()
        pattern = _parse_cron_pattern(cron_expr)
        next_run = _next_cron_after(pattern, now)

        entry = {
            "alarm_id": uuid4().hex[:8],
            "status": "active",
            "schedule_type": "cron",
            "cron_expr": pattern.expr,
            "target": target,
            "prompt": prompt.strip(),
            "model": (model or "").strip() or None,
            "budget_limit_usd": float(max(0.05, budget_limit_usd)),
            "max_runs": int(max_runs) if max_runs is not None and max_runs > 0 else None,
            "run_count": 0,
            "failure_count": 0,
            "owner_agent": owner_agent,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_run_at": None,
            "next_run_at": next_run.isoformat(),
            "last_error": None,
        }
        state = self._load_state()
        alarms = state.get("alarms")
        if not isinstance(alarms, list):
            alarms = []
        alarms.append(entry)
        state["alarms"] = alarms
        self._save_state(state)
        return entry

    def cancel(self, alarm_id: str) -> bool:
        self.ensure_initialized()
        state = self._load_state()
        alarms = state.get("alarms")
        if not isinstance(alarms, list):
            return False
        now = _utc_now().isoformat()
        changed = False
        for entry in alarms:
            if str(entry.get("alarm_id")) == alarm_id and str(entry.get("status")) == "active":
                entry["status"] = "canceled"
                entry["updated_at"] = now
                changed = True
        if changed:
            self._save_state(state)
        return changed

    def tick(self, manager: Any) -> int:
        """Execute due alarms."""
        self.ensure_initialized()
        state = self._load_state()
        alarms = state.get("alarms")
        if not isinstance(alarms, list):
            alarms = []
        history = state.get("history")
        if not isinstance(history, list):
            history = []

        now = _utc_now()
        due_entries: list[dict[str, Any]] = []
        for entry in alarms:
            if str(entry.get("status")) != "active":
                continue
            try:
                next_run = _parse_datetime_like(str(entry.get("next_run_at") or ""))
            except Exception:
                entry["status"] = "failed"
                entry["last_error"] = "invalid next_run_at"
                entry["updated_at"] = now.isoformat()
                continue
            if next_run <= now:
                due_entries.append(entry)

        if not due_entries:
            return 0

        due_entries.sort(key=lambda x: str(x.get("next_run_at") or ""))
        executed = 0

        for entry in due_entries:
            alarm_id = str(entry.get("alarm_id") or "unknown")
            target = str(entry.get("target") or "ceo")
            prompt = str(entry.get("prompt") or "").strip()
            if not prompt:
                entry["status"] = "failed"
                entry["last_error"] = "empty prompt"
                entry["updated_at"] = now.isoformat()
                continue

            try:
                self._execute_entry(manager, entry)
                executed += 1
                entry["failure_count"] = 0
                entry["last_error"] = None
                entry["last_run_at"] = now.isoformat()
                entry["run_count"] = int(entry.get("run_count") or 0) + 1
                entry["updated_at"] = now.isoformat()
                history.append(
                    {
                        "alarm_id": alarm_id,
                        "ran_at": now.isoformat(),
                        "target": target,
                        "result": "ok",
                    }
                )
                self._advance_schedule(entry, now)
            except Exception as exc:
                logger.warning("Alarm execution failed: %s", alarm_id, exc_info=True)
                failure_count = int(entry.get("failure_count") or 0) + 1
                entry["failure_count"] = failure_count
                entry["last_error"] = str(exc)
                entry["updated_at"] = now.isoformat()
                history.append(
                    {
                        "alarm_id": alarm_id,
                        "ran_at": now.isoformat(),
                        "target": target,
                        "result": f"error:{exc}",
                    }
                )
                if str(entry.get("schedule_type")) == "once" and failure_count >= 3:
                    entry["status"] = "failed"
                    entry["next_run_at"] = None
                else:
                    backoff_min = min(60, max(3, failure_count * 5))
                    entry["next_run_at"] = (now + timedelta(minutes=backoff_min)).isoformat()

        state["alarms"] = alarms
        state["history"] = history[-MAX_HISTORY:]
        self._save_state(state)
        return executed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_entry(self, manager: Any, entry: dict[str, Any]) -> None:
        alarm_id = str(entry.get("alarm_id") or "unknown")
        target = str(entry.get("target") or "ceo")
        prompt = str(entry.get("prompt") or "")
        model = (entry.get("model") or "").strip() or None
        budget = float(entry.get("budget_limit_usd") or DEFAULT_BUDGET_LIMIT_USD)

        activity = getattr(manager, "_activity_log", None)
        if callable(activity):
            activity(
                f"アラーム実行: alarm_id={alarm_id} target={target} "
                f"schedule={entry.get('schedule_type')} prompt={_short(prompt, 160)}"
            )

        if target == "ceo":
            manager.process_message(
                prompt,
                user_id=f"alarm:{alarm_id}",
                slack_channel=getattr(manager, "_slack_default_channel", None) or getattr(manager, "_slack_last_channel", None),
                slack_thread_ts=None,
                slack_thread_context=None,
            )
            return

        role = ""
        if target.startswith("role:"):
            role = target.split(":", 1)[1].strip()
        elif target.startswith("agent:"):
            agent_id = target.split(":", 1)[1].strip()
            agent = manager.agent_registry.get(agent_id)
            if agent is not None:
                role = (agent.role or "").strip()
                if model is None:
                    model = (agent.model or "").strip() or None
            if not role:
                role = "worker"
        else:
            role = target.strip() or "worker"

        result = manager.sub_agent_runner.spawn(
            name=role,
            role=role,
            task_description=prompt,
            budget_limit_usd=max(0.05, budget),
            model=model,
            ignore_wip_limit=True,
        )

        if callable(activity):
            activity(
                f"アラーム完了: alarm_id={alarm_id} role={role} "
                f"result={_short(str(result), 180)}"
            )

    def _advance_schedule(self, entry: dict[str, Any], now: datetime) -> None:
        schedule_type = str(entry.get("schedule_type") or "")
        max_runs = entry.get("max_runs")
        run_count = int(entry.get("run_count") or 0)
        if max_runs is not None and run_count >= int(max_runs):
            entry["status"] = "done"
            entry["next_run_at"] = None
            return

        if schedule_type == "once":
            entry["status"] = "done"
            entry["next_run_at"] = None
            return

        if schedule_type == "cron":
            expr = str(entry.get("cron_expr") or "").strip()
            pattern = _parse_cron_pattern(expr)
            entry["next_run_at"] = _next_cron_after(pattern, now).isoformat()
            return

        entry["status"] = "failed"
        entry["next_run_at"] = None
        entry["last_error"] = f"unknown schedule_type: {schedule_type}"

    def _render_alarm_list(self, *, limit: int = 20) -> str:
        state = self._load_state()
        alarms = state.get("alarms")
        if not isinstance(alarms, list):
            return "⏰ アラームは未登録です。"

        active = [a for a in alarms if str(a.get("status")) == "active"]
        active.sort(key=lambda x: str(x.get("next_run_at") or ""))
        if not active:
            return "⏰ 現在有効なアラームはありません。"

        lines = [f"⏰ 有効アラーム {len(active)} 件（表示上限 {limit} 件）"]
        for entry in active[: max(1, limit)]:
            aid = str(entry.get("alarm_id") or "unknown")
            stype = str(entry.get("schedule_type") or "unknown")
            target = str(entry.get("target") or "ceo")
            next_run = _fmt_dt(str(entry.get("next_run_at") or ""))
            if stype == "cron":
                cron = str(entry.get("cron_expr") or "")
                lines.append(
                    f"- [{aid}] cron `{cron}` / next={next_run} / target={target} / "
                    f"task={_short(str(entry.get('prompt') or ''), 70)}"
                )
            else:
                lines.append(
                    f"- [{aid}] once {next_run} / target={target} / "
                    f"task={_short(str(entry.get('prompt') or ''), 70)}"
                )
        return "\n".join(lines)

    def _parse_target_spec(
        self,
        text: str,
        *,
        actor_role: str,
        actor_model: str | None,
    ) -> tuple[str, str | None, float, int | None]:
        parts = [p.strip() for p in (text or "").split(";") if p.strip()]
        target_raw = parts[0] if parts else "self"
        target = _normalize_target(target_raw, actor_role=actor_role)

        model = None
        budget = DEFAULT_BUDGET_LIMIT_USD
        max_runs = None
        for p in parts[1:]:
            if "=" not in p:
                continue
            key, val = p.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "model":
                model = val or None
            elif key in ("budget", "budget_usd"):
                try:
                    budget = max(0.05, float(val))
                except Exception:
                    pass
            elif key in ("max_runs", "maxrun"):
                try:
                    max_runs = max(1, int(val))
                except Exception:
                    pass

        if model is None and target.startswith("role:") and target == f"role:{actor_role.strip()}":
            model = (actor_model or "").strip() or None

        return target, model, budget, max_runs

    def _load_state(self) -> dict[str, Any]:
        self.ensure_initialized()
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.warning("Failed to load alarms state", exc_info=True)
        return {"alarms": [], "history": []}

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _normalize_target(target: str, *, actor_role: str) -> str:
    s = (target or "").strip()
    if not s or s.lower() == "self":
        if (actor_role or "").strip().lower() == "ceo":
            return "ceo"
        return f"role:{(actor_role or 'worker').strip()}"

    low = s.lower()
    if low == "ceo":
        return "ceo"
    if low.startswith("role:"):
        return f"role:{s.split(':', 1)[1].strip() or 'worker'}"
    if low.startswith("agent:"):
        return f"agent:{s.split(':', 1)[1].strip()}"

    return f"role:{s}"


def _parse_datetime_like(text: str) -> datetime:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty datetime")

    now = _utc_now()
    m = re.match(r"^\+(\d+)\s*([smhd])$", s, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "s":
            return now + timedelta(seconds=n)
        if unit == "m":
            return now + timedelta(minutes=n)
        if unit == "h":
            return now + timedelta(hours=n)
        return now + timedelta(days=n)

    s2 = s
    if "T" not in s2 and " " in s2:
        s2 = s2.replace(" ", "T", 1)
    if s2.endswith("Z"):
        s2 = s2[:-1] + "+00:00"
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_cron_pattern(expr: str) -> _CronPattern:
    raw = " ".join((expr or "").split())
    fields = raw.split(" ")
    if len(fields) != 5:
        raise ValueError("cron expression must have 5 fields: m h dom mon dow")

    minute_txt, hour_txt, day_txt, month_txt, weekday_txt = fields
    minutes = _parse_cron_field(minute_txt, 0, 59, allow_7=False)
    hours = _parse_cron_field(hour_txt, 0, 23, allow_7=False)
    days = _parse_cron_field(day_txt, 1, 31, allow_7=False)
    months = _parse_cron_field(month_txt, 1, 12, allow_7=False)
    weekdays = _parse_cron_field(weekday_txt, 0, 6, allow_7=True)

    return _CronPattern(
        expr=raw,
        minutes=minutes,
        hours=hours,
        days=days,
        months=months,
        weekdays=weekdays,
        dom_any=(day_txt.strip() == "*"),
        dow_any=(weekday_txt.strip() == "*"),
    )


def _parse_cron_field(token: str, min_value: int, max_value: int, *, allow_7: bool) -> set[int]:
    out: set[int] = set()
    tok = (token or "").strip()
    if not tok:
        raise ValueError("empty cron field")

    for part in tok.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        base = part
        if "/" in part:
            base, step_txt = part.split("/", 1)
            step = int(step_txt.strip())
            if step <= 0:
                raise ValueError(f"invalid cron step: {part}")

        if base == "*":
            start, end = min_value, max_value
        elif "-" in base:
            left, right = base.split("-", 1)
            start, end = int(left.strip()), int(right.strip())
        else:
            start = end = int(base.strip())

        for value in range(start, end + 1, step):
            vv = value
            if allow_7 and vv == 7:
                vv = 0
            if vv < min_value or vv > max_value:
                raise ValueError(f"cron value out of range: {value}")
            out.add(vv)

    if not out:
        raise ValueError("empty cron field set")
    return out


def _next_cron_after(pattern: _CronPattern, after: datetime) -> datetime:
    now = after.astimezone(timezone.utc).replace(second=0, microsecond=0)
    probe = now + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        if _cron_matches(pattern, probe):
            return probe
        probe += timedelta(minutes=1)
    raise ValueError(f"could not find next cron run within 1 year: {pattern.expr}")


def _cron_matches(pattern: _CronPattern, dt: datetime) -> bool:
    t = dt.astimezone(timezone.utc)
    if t.minute not in pattern.minutes:
        return False
    if t.hour not in pattern.hours:
        return False
    if t.month not in pattern.months:
        return False

    day_match = t.day in pattern.days
    cron_weekday = (t.weekday() + 1) % 7  # Python Mon=0..Sun=6 -> Cron Sun=0
    dow_match = cron_weekday in pattern.weekdays

    if pattern.dom_any and pattern.dow_any:
        return True
    if pattern.dom_any:
        return dow_match
    if pattern.dow_any:
        return day_match
    return day_match or dow_match


def _fmt_dt(text: str) -> str:
    try:
        dt = _parse_datetime_like(text).astimezone(timezone.utc)
        jst = dt.astimezone(JST)
        return f"{dt.strftime('%Y-%m-%d %H:%M')} UTC / {jst.strftime('%Y-%m-%d %H:%M')} JST"
    except Exception:
        return text or "n/a"


def _short(text: str, limit: int) -> str:
    s = " ".join((text or "").split())
    if len(s) > limit:
        return s[:limit] + "…"
    return s


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
