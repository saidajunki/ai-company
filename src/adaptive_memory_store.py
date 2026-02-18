"""Adaptive long-term memory store.

This store keeps important memories beyond fixed categories (budget/policy/rules),
creates domain directories automatically, and prunes low-value stale memories.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import AdaptiveMemoryEntry
from ndjson_store import ndjson_append, ndjson_read


@dataclass
class AdaptiveIngestResult:
    created: list[AdaptiveMemoryEntry] = field(default_factory=list)


_IMPORTANT_KEYWORDS = (
    "決定", "方針", "ルール", "要件", "仕様", "手順", "運用", "障害", "原因", "対策",
    "予算", "コスト", "収益", "KPI", "ロードマップ", "期限", "締切", "担当", "依存",
    "公開", "非公開", "顧客", "ユーザー", "市場", "URL", "http", "パス", "ディレクトリ",
    "VPS", "Docker", "Traefik", "systemctl", "デプロイ", "再起動", "認証", "トークン",
    "システムプロンプト", "記憶", "メモリ", "index", "sqlite", "vector", "agent", "社員",
)

_QUESTION_HINTS = (
    "?", "？", "ですか", "ますか", "でしょうか", "教えて", "確認", "何が", "どこ", "いつ",
)

_DOMAIN_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("governance", ("方針", "ルール", "憲法", "意思決定", "禁止", "必ず")),
    ("budget-finance", ("予算", "コスト", "収益", "請求", "支出", "利益")),
    ("infra-ops", ("VPS", "Docker", "Traefik", "systemctl", "deploy", "デプロイ", "再起動", "監視")),
    ("product-tech", ("機能", "仕様", "実装", "OSS", "リリース", "API", "モデル", "プロンプト")),
    ("memory-system", ("記憶", "メモリ", "index", "sqlite", "vector", "検索", "要約")),
    ("org-people", ("社員", "採用", "役割", "責任", "委譲", "チーム")),
    ("market-customer", ("顧客", "ユーザー", "市場", "話題", "SNS", "PR")),
    ("security", ("認証", "トークン", "鍵", "脆弱", "攻撃", "権限", "秘密")),
]


class AdaptiveMemoryStore:
    """Store and organize important memories in dynamic domains."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self.base_dir = base_dir
        self.company_id = company_id
        company_root = base_dir / "companies" / company_id
        self._state_path = company_root / "state" / "adaptive_memory.ndjson"
        self._domains_root = company_root / "knowledge" / "domains"
        self._domain_index = self._domains_root / "INDEX.md"

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def domains_root(self) -> Path:
        return self._domains_root

    def ensure_initialized(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._domains_root.mkdir(parents=True, exist_ok=True)
        if not self._state_path.exists():
            self._state_path.write_text("", encoding="utf-8")
        if not self._domain_index.exists():
            self._domain_index.write_text("# Memory Domains\n\n", encoding="utf-8")

    def list_all(self) -> list[AdaptiveMemoryEntry]:
        return ndjson_read(self._state_path, AdaptiveMemoryEntry)

    def list_active(self) -> list[AdaptiveMemoryEntry]:
        return [e for e in self.list_all() if e.status == "active"]

    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> AdaptiveIngestResult:
        self.ensure_initialized()
        result = AdaptiveIngestResult()

        candidates = _extract_candidates(text, source=source)
        if not candidates:
            return result

        existing = {
            _dedupe_key(e.domain, e.content)
            for e in self.list_all()
            if e.status != "pruned"
        }

        for candidate in candidates:
            domain = _infer_domain(candidate)
            dedupe = _dedupe_key(domain, candidate)
            if dedupe in existing:
                continue

            now = datetime.now(timezone.utc)
            importance = _score_importance(candidate, source=source)
            entry = AdaptiveMemoryEntry(
                memory_id=uuid4().hex[:10],
                created_at=now,
                updated_at=now,
                domain=domain,
                status="active",
                content=candidate,
                source=source,
                user_id=user_id,
                task_id=task_id,
                importance=importance,
                tags=_extract_tags(candidate),
            )
            ndjson_append(self._state_path, entry)
            self._append_domain_note(entry)
            result.created.append(entry)
            existing.add(dedupe)

        if result.created:
            self._refresh_domain_index()

        return result

    def compact_and_prune(
        self,
        *,
        low_importance_days: int = 7,
        medium_importance_days: int = 30,
        max_entries: int = 3000,
    ) -> None:
        """Deduplicate and forget stale low-value memories."""
        self.ensure_initialized()
        entries = self.list_all()
        if not entries:
            return

        entries.sort(key=lambda e: e.created_at)
        if max_entries > 0 and len(entries) > max_entries:
            entries = entries[-max_entries:]

        now = datetime.now(timezone.utc)
        seen: set[str] = set()
        compacted: list[AdaptiveMemoryEntry] = []

        for entry in entries:
            content = (entry.content or "").strip()
            if not content:
                continue

            key = _dedupe_key(entry.domain, content)
            if key in seen:
                continue
            seen.add(key)

            item = entry.model_copy(deep=True)
            item.content = content
            item.updated_at = now

            age_days = max(0, (now - item.created_at).days)
            if item.importance <= 2 and age_days >= low_importance_days:
                item.status = "pruned"
            elif item.importance == 3 and age_days >= medium_importance_days:
                item.status = "archived"
            elif item.status == "pruned" and age_days < low_importance_days:
                item.status = "active"

            compacted.append(item)

        self._rewrite(compacted)
        self._refresh_domain_index()

    def format_active(self, *, limit: int = 24) -> str:
        entries = self.list_active()
        if not entries:
            return "（なし）"

        entries.sort(key=lambda e: (e.importance, e.created_at), reverse=True)
        lines: list[str] = []
        for entry in entries[:limit]:
            ts = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] [{entry.domain}] (imp:{entry.importance}) {entry.content}")
        return "\n".join(lines)

    def format_domains(self, *, limit: int = 16) -> str:
        entries = self.list_all()
        if not entries:
            return "（なし）"

        stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"active": 0, "archived": 0, "pruned": 0}
        )
        for entry in entries:
            st = entry.status if entry.status in ("active", "archived", "pruned") else "active"
            stats[entry.domain][st] += 1

        ordered = sorted(
            stats.items(),
            key=lambda kv: (kv[1]["active"], kv[1]["archived"]),
            reverse=True,
        )
        lines: list[str] = []
        for domain, counts in ordered[:limit]:
            lines.append(
                f"- {domain}: active={counts['active']} archived={counts['archived']} pruned={counts['pruned']} path={self._domains_root / domain}"
            )
        return "\n".join(lines)

    def _append_domain_note(self, entry: AdaptiveMemoryEntry) -> None:
        domain_dir = self._domains_root / entry.domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        path = domain_dir / "MEMORY.md"
        if not path.exists():
            path.write_text(f"# Domain: {entry.domain}\n\n", encoding="utf-8")

        ts = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
        line = f"- [{ts}] (imp:{entry.importance}) [{entry.memory_id}] {entry.content}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _refresh_domain_index(self) -> None:
        domains = sorted([p.name for p in self._domains_root.iterdir() if p.is_dir()])
        lines = [
            "# Memory Domains",
            "",
            f"updated_at: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]
        if not domains:
            lines.append("- (none)")
        else:
            for name in domains:
                lines.append(f"- {name}: {self._domains_root / name}/MEMORY.md")
        self._domain_index.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _rewrite(self, entries: list[AdaptiveMemoryEntry]) -> None:
        lines = [e.model_dump_json() for e in entries]
        text = "\n".join(lines)
        if text:
            text += "\n"
        self._state_path.write_text(text, encoding="utf-8")


def _extract_candidates(text: str, *, source: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    permissive = source.startswith("memory_")
    parts: list[str] = []
    for block in re.split(r"[\n]+", raw):
        for piece in re.split(r"[。]\s*", block):
            line = piece.strip(" -・\t")
            if len(line) < 8:
                continue
            if _is_question(line):
                continue
            if permissive:
                parts.append(line)
                continue
            if _looks_important(line):
                parts.append(line)

    seen: set[str] = set()
    out: list[str] = []
    for item in parts:
        norm = re.sub(r"\s+", "", item).lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item)
    return out


def _is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(h in t for h in _QUESTION_HINTS)


def _looks_important(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if any(k in t for k in _IMPORTANT_KEYWORDS):
        return True
    if re.search(r"/[-\w./]+", t) and any(v in t for v in ("保存", "場所", "ファイル", "手順", "運用")):
        return True
    if re.search(r"https?://", t):
        return True
    return False


def _infer_domain(text: str) -> str:
    best_domain = "general"
    best_score = 0
    for domain, kws in _DOMAIN_RULES:
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def _score_importance(text: str, *, source: str) -> int:
    score = 3
    t = text or ""

    if source in ("memory_pin", "memory_curated"):
        score += 1
    if any(k in t for k in ("決定", "必ず", "禁止", "上限", "期限", "手順", "復旧", "障害")):
        score += 1
    if re.search(r"https?://", t) or re.search(r"/[-\w./]+", t):
        score += 1
    if _is_question(t):
        score -= 2

    return max(1, min(5, score))


def _extract_tags(text: str) -> list[str]:
    tags: list[str] = []
    t = text or ""
    for kw in ("予算", "方針", "ルール", "デプロイ", "障害", "顧客", "モデル", "記憶", "運用"):
        if kw in t:
            tags.append(kw)
    return tags[:8]


def _dedupe_key(domain: str, text: str) -> str:
    norm = re.sub(r"\s+", "", (text or "").strip()).lower()
    return f"{domain}::{norm}"
