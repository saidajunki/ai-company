"""Policy memory store for durable company direction/rules/budget memories.

Goals:
- Keep durable policy memories extracted from conversations and work.
- Preserve timeline (new vs old) with timestamps.
- Detect conflicts and surface consultation prompts.
- Prevent noisy/non-policy lines from polluting long-term memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import PolicyMemoryEntry
from ndjson_store import ndjson_append, ndjson_read


@dataclass
class PolicyConflict:
    new_entry: PolicyMemoryEntry
    conflicts_with: list[PolicyMemoryEntry] = field(default_factory=list)


@dataclass
class PolicyIngestResult:
    created: list[PolicyMemoryEntry] = field(default_factory=list)
    conflicts: list[PolicyConflict] = field(default_factory=list)


_POLICY_KEYWORDS = (
    "方針", "方向性", "目的", "目指", "ビジョン", "ルール", "禁止", "必ず", "予算", "上限",
    "忘れない", "記憶", "相談", "承認", "公開", "非公開", "自動", "停止", "再開",
    "コミット", "push", "再読込", "restart_manager.flag", "システムプロンプト", "ロジック",
)

_DIRECTIVE_HINTS = (
    "する", "しない", "してください", "すべき", "します", "維持", "優先", "禁止",
    "上限", "固定", "必ず", "忘れない", "記憶", "保存", "主体", "方針", "ルール",
)

_QUESTION_HINTS = (
    "?", "？", "ですか", "ますか", "でしょうか", "教えて", "確認", "何ができます", "覚えていますか",
    "答えて", "説明して", "一覧", "どこ",
)

_NEG_TOKENS = ("禁止", "しない", "やらない", "停止", "無効", "不可", "禁止する", "非公開")
_POS_TOKENS = ("実行", "する", "行う", "許可", "有効", "進める", "行っていく", "主体", "公開", "優先")

_ANCHOR_TOKENS = (
    "予算", "方針", "方向性", "公開", "非公開", "自動", "相談", "承認", "コミット", "push",
    "VPS", "社長", "ルール", "禁止", "再読込", "restart_manager.flag", "システムプロンプト",
)

_BUDGET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ドル|usd|\$)", re.IGNORECASE)


class PolicyMemoryStore:
    """Append-only policy memory store with conflict detection + compaction."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self.base_dir = base_dir
        self.company_id = company_id
        self._path = base_dir / "companies" / company_id / "state" / "policy_memory.ndjson"

    @property
    def path(self) -> Path:
        return self._path

    def ensure_initialized(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def list_all(self) -> list[PolicyMemoryEntry]:
        return ndjson_read(self._path, PolicyMemoryEntry)

    def list_active(self) -> list[PolicyMemoryEntry]:
        return [e for e in self.list_all() if e.status == "active"]

    def list_conflicted(self) -> list[PolicyMemoryEntry]:
        return [e for e in self.list_all() if e.status == "conflicted"]

    def seed_defaults(
        self,
        *,
        app_repo_path: str,
        system_prompt_file: str,
        restart_flag_path: str,
    ) -> None:
        repo_root = str(Path(app_repo_path).expanduser().resolve())
        prompt_path = str(Path(system_prompt_file).expanduser().resolve())
        restart_path = str(Path(restart_flag_path).expanduser().resolve())

        defaults = [
            ("rule", f"システムプロンプト実体: {prompt_path}"),
            ("rule", f"主要ロジック: {repo_root}/src/"),
            ("rule", f"再読込フラグ: {restart_path}"),
            ("direction", "このVPS内で事業を行う主体は社長AIである"),
            ("direction", "自己改変は社長AIの能力強化のために行う"),
        ]
        active = self.list_active()
        existing = {_dedupe_key(e.content) for e in active}
        for category, content in defaults:
            if _dedupe_key(content) in existing:
                continue
            entry = PolicyMemoryEntry(
                memory_id=uuid4().hex[:10],
                created_at=datetime.now(timezone.utc),
                category=category,
                content=content,
                source="bootstrap",
                status="active",
                importance=5,
            )
            ndjson_append(self._path, entry)
            existing.add(_dedupe_key(content))

    def compact(self, *, max_entries: int = 2000) -> None:
        """Rebuild file while removing obvious noise/duplicates and re-evaluating conflicts."""
        self.ensure_initialized()
        entries = self.list_all()
        if not entries:
            return

        entries.sort(key=lambda e: e.created_at)
        if max_entries > 0 and len(entries) > max_entries:
            entries = entries[-max_entries:]

        cleaned: list[PolicyMemoryEntry] = []
        active: list[PolicyMemoryEntry] = []
        seen: set[str] = set()

        for entry in entries:
            content = (entry.content or "").strip()
            if not content:
                continue
            if _looks_question(content):
                continue
            if entry.source != "bootstrap" and not _looks_policy_statement(content):
                continue

            normalized = _dedupe_key(content)
            if normalized in seen:
                continue

            item = entry.model_copy(deep=True)
            item.content = content
            item.status = "active"
            item.conflict_with = []

            conflicts = [e for e in active if _is_conflict(item, e)]
            if conflicts:
                item.status = "conflicted"
                item.conflict_with = [e.memory_id for e in conflicts]
            else:
                active.append(item)

            cleaned.append(item)
            seen.add(normalized)

        self._rewrite_entries(cleaned)

    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> PolicyIngestResult:
        self.ensure_initialized()
        result = PolicyIngestResult()
        lines = _extract_candidate_lines(text)
        if not lines:
            return result

        active = self.list_active()
        existing_norm = {_dedupe_key(e.content) for e in active}

        for line in lines:
            norm = _dedupe_key(line)
            if not norm or norm in existing_norm:
                continue

            category = _categorize(line)
            importance = 5 if category in ("direction", "budget", "rule") else 3
            candidate = PolicyMemoryEntry(
                memory_id=uuid4().hex[:10],
                created_at=datetime.now(timezone.utc),
                category=category,
                content=line,
                source=source,
                user_id=user_id,
                task_id=task_id,
                importance=importance,
                status="active",
            )

            conflicts = [e for e in active if _is_conflict(candidate, e)]
            if conflicts:
                candidate.status = "conflicted"
                candidate.conflict_with = [e.memory_id for e in conflicts]
                result.conflicts.append(PolicyConflict(new_entry=candidate, conflicts_with=conflicts))
            else:
                active.append(candidate)

            ndjson_append(self._path, candidate)
            result.created.append(candidate)
            existing_norm.add(norm)

        return result

    def format_active(self, *, limit: int = 20) -> str:
        entries = self.list_active()
        if not entries:
            return "（なし）"
        entries.sort(key=lambda e: e.created_at, reverse=True)
        lines = []
        for e in entries[:limit]:
            ts = e.created_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] [{e.memory_id}] ({e.category}) {e.content}")
        return "\n".join(lines)

    def format_timeline(self, *, limit: int = 30) -> str:
        entries = self.list_all()
        if not entries:
            return "（なし）"
        entries.sort(key=lambda e: e.created_at, reverse=True)
        lines = []
        for e in entries[:limit]:
            ts = e.created_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] [{e.status}] [{e.memory_id}] ({e.category}) {e.content}")
        return "\n".join(lines)

    def format_conflicts(self, *, limit: int = 10) -> str:
        conflicts = self.list_conflicted()
        if not conflicts:
            return "（衝突なし）"
        conflicts.sort(key=lambda e: e.created_at, reverse=True)
        lines = []
        for e in conflicts[:limit]:
            ts = e.created_at.strftime("%Y-%m-%d %H:%M:%S")
            with_ids = ", ".join(e.conflict_with) if e.conflict_with else "unknown"
            lines.append(f"- [{ts}] [{e.memory_id}] ({e.category}) {e.content} / conflicts: {with_ids}")
        return "\n".join(lines)

    def _rewrite_entries(self, entries: list[PolicyMemoryEntry]) -> None:
        lines = [e.model_dump_json() for e in entries]
        content = "\n".join(lines)
        if content:
            content += "\n"
        self._path.write_text(content, encoding="utf-8")


def _extract_candidate_lines(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    lines: list[str] = []
    blocks = re.split(r"[\n]+", raw)
    for block in blocks:
        for part in re.split(r"[。]\s*", block):
            s = part.strip(" -・\t")
            if len(s) < 8:
                continue
            if _looks_question(s):
                continue
            if _looks_policy_statement(s):
                lines.append(s)

    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        norm = _dedupe_key(line)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(line)
    return deduped


def _looks_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(h in t for h in _QUESTION_HINTS)


def _looks_policy_statement(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if not any(k in t for k in _POLICY_KEYWORDS):
        return False
    if any(h in t for h in _DIRECTIVE_HINTS):
        return True
    if any(k in t for k in ("システムプロンプト", "再読込フラグ", "主要ロジック", "ディレクトリ", "ファイル")):
        return True
    return False


def _categorize(text: str) -> str:
    t = text
    if any(k in t for k in ("予算", "上限", "$", "ドル", "60分", "1時間", "時間あたり", "/h", "per hour")):
        return "budget"
    if any(k in t for k in ("方針", "方向性", "ビジョン", "目的", "目指", "主体", "公開", "非公開")):
        return "direction"
    if any(k in t for k in ("ルール", "禁止", "必ず", "しない", "再読込", "commit", "push", "コミット")):
        return "rule"
    if any(k in t for k in ("ディレクトリ", "パス", "ファイル", "場所")):
        return "operation"
    return "fact"


def _dedupe_key(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    if ":" in raw:
        head, tail = raw.split(":", 1)
        label = head.strip()
        body = tail.strip()
        if label in ("再読込フラグ", "システムプロンプト実体", "主要ロジック"):
            path_like = Path(body).expanduser()
            if not path_like.is_absolute():
                path_like = (Path("/opt/apps/ai-company") / path_like).resolve()
            else:
                path_like = path_like.resolve()
            canonical = str(path_like)
            if label == "主要ロジック":
                canonical = canonical.rstrip("/") + "/"
            return f"{label}:{canonical}".lower()

    return re.sub(r"\s+", "", raw).lower()


def _intent_sign(text: str) -> int:
    t = text or ""
    neg = any(k in t for k in _NEG_TOKENS)
    pos = any(k in t for k in _POS_TOKENS)
    if neg and not pos:
        return -1
    if pos and not neg:
        return 1
    return 0


def _anchors(text: str) -> set[str]:
    t = text or ""
    return {a for a in _ANCHOR_TOKENS if a in t}


def _extract_budget_value(text: str) -> float | None:
    t = text or ""
    if not any(k in t for k in ("予算", "上限", "ドル", "$", "usd")):
        return None
    m = _BUDGET_RE.search(t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _publicity_state(text: str) -> int:
    t = text or ""
    if any(k in t for k in ("非公開", "公開しない", "公開は禁止", "公開を禁止", "公開しません")):
        return -1
    if "公開" in t:
        return 1
    return 0


def _is_conflict(new: PolicyMemoryEntry, old: PolicyMemoryEntry) -> bool:
    if old.status != "active":
        return False

    new_public = _publicity_state(new.content)
    old_public = _publicity_state(old.content)
    if new_public != 0 and old_public != 0 and new_public != old_public:
        return True

    new_budget = _extract_budget_value(new.content)
    old_budget = _extract_budget_value(old.content)
    if new_budget is not None and old_budget is not None and abs(new_budget - old_budget) > 1e-9:
        return True

    if new.category != old.category:
        return False

    new_sign = _intent_sign(new.content)
    old_sign = _intent_sign(old.content)
    if new_sign == 0 or old_sign == 0 or new_sign == old_sign:
        return False

    overlap = _anchors(new.content) & _anchors(old.content)
    return len(overlap) > 0
