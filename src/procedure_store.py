"""Procedure store for verbatim multi-line operational runbooks.

This store is intentionally separate from abstract memory:
- Save full procedure blocks (steps/commands) as the source of truth.
- Support shared procedure docs for future multi-agent access.
- Provide deterministic retrieval for procedure recall requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from models import ProcedureDocument
from ndjson_store import ndjson_append, ndjson_read


@dataclass
class ProcedureIngestResult:
    created: list[ProcedureDocument] = field(default_factory=list)
    updated: list[ProcedureDocument] = field(default_factory=list)


_SHARED_HINTS = (
    "社内共有", "社内で共有", "共有ドキュメント", "全社員に共有", "共有してください", "shared",
)

_RECALL_HINTS = (
    "手順", "手順名", "再掲", "順番", "番号", "もう一度", "思い出して", "教えて",
)

_FORGET_HINTS = (
    "忘れて", "不要", "覚えなくて", "重要ではない", "長期記憶化しなくて", "記憶しない",
)

_TITLE_PATTERNS = [
    re.compile(r"手順名\s*[「\"](?P<name>[^」\"]{1,120})[」\"]"),
    re.compile(r"手順名\s*[:：]\s*(?P<name>[^\n。]{1,120})"),
    re.compile(r"(?:runbook|procedure)\s*[:：]\s*(?P<name>[^\n]{1,120})", re.IGNORECASE),
    re.compile(r"^(?P<name>[^:\n]{2,120})\s*(?:手順|runbook|procedure)\s*[:：]", re.IGNORECASE | re.MULTILINE),
]

_STEP_LINE_RE = re.compile(r"^\s*(?:\d+[\)\.．:：]|[-*・])\s*(?P<body>.+?)\s*$")
_STEP_INLINE_RE = re.compile(r"\s*(\d+[\)\.．:：])\s*")
_FENCED_BLOCK_RE = re.compile(r"```(?:bash|sh|zsh|shell)?\s*\n(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)

_COMMAND_HINTS = (
    "cd ", "git ", "pip ", "python ", "systemctl ", "docker ", "kubectl ", "curl ",
    "sudo ", "source ", ".venv/bin/", "poetry ", "npm ", "pnpm ", "uv ", "./", "/",
)


class ProcedureStore:
    """Durable store for verbatim procedures and shared runbooks."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self.base_dir = base_dir
        self.company_id = company_id
        self.company_root = base_dir / "companies" / company_id
        self.state_path = self.company_root / "state" / "procedures.ndjson"
        self.private_dir = self.company_root / "knowledge" / "procedures"
        self.shared_dir = self.company_root / "knowledge" / "shared" / "procedures"
        self.shared_index_path = self.company_root / "knowledge" / "shared" / "INDEX.md"

    def ensure_initialized(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.shared_index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self.state_path.write_text("", encoding="utf-8")
        if not self.shared_index_path.exists():
            self.shared_index_path.write_text("# Shared Docs Index\n\n", encoding="utf-8")

    def list_all(self) -> list[ProcedureDocument]:
        return ndjson_read(self.state_path, ProcedureDocument)

    def list_active(self) -> list[ProcedureDocument]:
        return [e for e in self.list_all() if e.status == "active"]

    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> ProcedureIngestResult:
        self.ensure_initialized()
        result = ProcedureIngestResult()

        raw = (text or "").strip()
        if not raw:
            return result
        if any(h in raw for h in _FORGET_HINTS):
            return result

        title = _extract_title(raw)
        steps = _extract_steps(raw)
        if not title or len(steps) < 2:
            return result

        shared = any(h in raw for h in _SHARED_HINTS)
        visibility = "shared" if shared else "private"

        active = self.list_active()
        same_name = [e for e in active if e.name == title and e.visibility == visibility]
        latest = same_name[-1] if same_name else None

        canonical_steps = [_norm_step(s) for s in steps]
        if latest and [_norm_step(s) for s in latest.steps] == canonical_steps:
            return result

        now = datetime.now(timezone.utc)
        if latest:
            self._supersede(latest.doc_id)

        version = (latest.version + 1) if latest else 1
        doc_id = uuid4().hex[:12]
        slug = _slugify(title)
        base_dir = self.shared_dir if shared else self.private_dir
        file_path = base_dir / f"{slug}.md"

        body = _render_markdown(title=title, steps=steps, visibility=visibility, source=source)
        file_path.write_text(body, encoding="utf-8")

        entry = ProcedureDocument(
            doc_id=doc_id,
            name=title,
            version=version,
            created_at=now,
            updated_at=now,
            status="active",
            visibility=visibility,
            steps=steps,
            raw_text=raw,
            source=source,
            user_id=user_id,
            task_id=task_id,
            file_path=str(file_path),
            tags=["procedure", visibility],
        )
        ndjson_append(self.state_path, entry)

        if latest:
            result.updated.append(entry)
        else:
            result.created.append(entry)

        self._refresh_shared_index()
        return result

    def find_best_for_request(self, text: str) -> ProcedureDocument | None:
        query = (text or "").strip()
        if not query:
            return None
        if not any(h in query for h in _RECALL_HINTS):
            return None

        active = self.list_active()
        if not active:
            return None

        target = _extract_title(query)
        if target:
            exact = [e for e in active if target in e.name or e.name in target]
            if exact:
                return sorted(exact, key=lambda e: e.updated_at, reverse=True)[0]

        query_tokens = _tokenize(query)
        scored: list[tuple[int, ProcedureDocument]] = []
        for doc in active:
            score = 0
            if "手順" in query and "手順" in doc.name:
                score += 2
            if any(tok in doc.name for tok in query_tokens):
                score += sum(1 for tok in query_tokens if tok in doc.name)
            if any(tok in " ".join(doc.steps) for tok in query_tokens):
                score += sum(1 for tok in query_tokens if tok in " ".join(doc.steps))
            if score > 0:
                scored.append((score, doc))

        if not scored:
            return sorted(active, key=lambda e: e.updated_at, reverse=True)[0]
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        return scored[0][1]

    def render_reply(self, doc: ProcedureDocument) -> str:
        lines = [
            f"手順名『{doc.name}』(v{doc.version}) を、保存済みSoTから再掲します。",
            "",
            "```bash",
        ]
        for step in doc.steps:
            lines.append(step)
        lines.append("```")
        lines.append("")
        lines.append(f"source_of_truth: {doc.file_path}")
        return "\n".join(lines)

    def format_library(self, *, limit: int = 8, include_steps: bool = True) -> str:
        active = sorted(self.list_active(), key=lambda e: e.updated_at, reverse=True)
        if not active:
            return "（なし）"

        lines: list[str] = []
        for doc in active[:limit]:
            ts = doc.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] {doc.name} (v{doc.version}, {doc.visibility}) -> {doc.file_path}")
            if include_steps:
                for i, step in enumerate(doc.steps[:6], start=1):
                    lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    def format_shared(self, *, limit: int = 8) -> str:
        active = [e for e in self.list_active() if e.visibility == "shared"]
        active.sort(key=lambda e: e.updated_at, reverse=True)
        if not active:
            return "（なし）"
        lines = []
        for doc in active[:limit]:
            ts = doc.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] {doc.name} (v{doc.version}) -> {doc.file_path}")
        return "\n".join(lines)

    def _refresh_shared_index(self) -> None:
        shared = [e for e in self.list_active() if e.visibility == "shared"]
        shared.sort(key=lambda e: e.updated_at, reverse=True)
        lines = [
            "# Shared Docs Index",
            "",
            f"updated_at: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]
        if not shared:
            lines.append("- (none)")
        else:
            for doc in shared:
                lines.append(f"- {doc.name} (v{doc.version}): {doc.file_path}")
        self.shared_index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _supersede(self, doc_id: str) -> None:
        entries = self.list_all()
        changed = False
        for i, entry in enumerate(entries):
            if entry.doc_id == doc_id and entry.status == "active":
                updated = entry.model_copy(deep=True)
                updated.status = "superseded"
                updated.updated_at = datetime.now(timezone.utc)
                entries[i] = updated
                changed = True
                break
        if changed:
            body = "\n".join(e.model_dump_json() for e in entries)
            if body:
                body += "\n"
            self.state_path.write_text(body, encoding="utf-8")


def _extract_title(text: str) -> str | None:
    raw = (text or "").strip()
    for pattern in _TITLE_PATTERNS:
        m = pattern.search(raw)
        if m:
            title = (m.group("name") or "").strip()
            if title:
                return title

    first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
    if first_line.startswith("手順") and len(first_line) <= 120:
        return first_line
    return None


def _extract_steps(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    # 1) fenced multi-line command block (verbatim priority)
    for match in _FENCED_BLOCK_RE.finditer(raw):
        body = (match.group("body") or "").strip()
        if not body:
            continue
        lines = [line.rstrip() for line in body.splitlines() if line.strip()]
        if len(lines) >= 2:
            return lines

    # 2) numbered/bullet steps
    steps: list[str] = []
    for line in raw.splitlines():
        m = _STEP_LINE_RE.match(line)
        if m:
            body = (m.group("body") or "").strip()
            if body:
                steps.append(body)

    if len(steps) >= 2:
        return steps

    # 3) inline numbered steps
    if re.search(r"\d+[\)\.．:：]", raw):
        chunks = _STEP_INLINE_RE.split(raw)
        # split result: [prefix, marker, body, marker, body, ...]
        rebuilt: list[str] = []
        for i in range(1, len(chunks), 2):
            if i + 1 >= len(chunks):
                break
            body = chunks[i + 1].strip()
            # trim tail by next marker if lingering
            body = re.sub(r"\s*\d+[\)\.．:：]\s*$", "", body).strip()
            if body:
                rebuilt.append(body)
        if len(rebuilt) >= 2:
            return rebuilt

    # 4) command-like lines
    cmd_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if _looks_like_command(s):
            cmd_lines.append(s)
    if len(cmd_lines) >= 2:
        return cmd_lines

    return []


def _render_markdown(*, title: str, steps: list[str], visibility: str, source: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Procedure: {title}",
        "",
        f"- visibility: {visibility}",
        f"- source: {source}",
        f"- updated_at_utc: {now}",
        "",
        "## Steps",
    ]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append("")
    return "\n".join(lines)


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_ぁ-んァ-ヶ一-龥]+", "-", text.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:80] or "procedure"


def _norm_step(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        norm = _norm_step(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item.strip())
    return out


def _looks_like_command(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if any(s.startswith(prefix) for prefix in _COMMAND_HINTS):
        return True
    if s.startswith("#"):
        return False
    if " && " in s or " | " in s:
        return True
    if re.match(r"^[A-Za-z0-9._/-]+\s+[-]{1,2}[A-Za-z0-9_-]+", s):
        return True
    return False


def _tokenize(text: str) -> list[str]:
    raw = re.sub(r"[^\wぁ-んァ-ヶ一-龥]+", " ", text)
    tokens = [t for t in raw.split() if len(t) >= 2]
    return tokens[:20]
