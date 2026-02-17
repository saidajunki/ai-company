"""Memory manager — long-term storage, recall, and summarization.

Design goals:
- Persistent: survives restarts (disk-based)
- Cheap: no external APIs required (local embeddings + SQLite FTS)
- Useful: automatic recall section in prompts + rolling summary + journal
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from constitution_store import constitution_load
from journal import JournalWriter
from memory_index import MemoryIndex
from models import (
    ConsultationEntry,
    ConversationEntry,
    CreatorReview,
    DecisionLogEntry,
    ResearchNote,
    TaskEntry,
)
from rolling_summary import RollingSummary

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int = 220) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[:max_len] + "…"


_PIN_KEYWORDS = (
    "目的",
    "北極星",
    "KPI",
    "最優先",
    "方針",
    "価値観",
    "禁止",
    "必須",
    "予算",
    "承認",
)


def _looks_pinnable(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(k in t for k in _PIN_KEYWORDS)


class MemoryManager:
    """Owns long-term memory index and summary/journal stores."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self.base_dir = base_dir
        self.company_id = company_id

        state_dir = base_dir / "companies" / company_id / "state"
        self.index = MemoryIndex(state_dir / "memory.sqlite3")
        self.summary = RollingSummary(state_dir / "rolling_summary.md")
        self.journal = JournalWriter(base_dir, company_id)

    # ------------------------------------------------------------------
    # Ingestion (incremental)
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Best-effort ingestion of existing on-disk state into the index."""
        try:
            self.ingest_all_sources()
        except Exception:
            logger.warning("Memory bootstrap failed", exc_info=True)

    def ingest_all_sources(self) -> None:
        company_root = self.base_dir / "companies" / self.company_id

        self._upsert_constitution(company_root / "constitution.yaml")
        self._upsert_vision(company_root / "vision.md")

        self._ingest_ndjson(
            company_root / "state" / "conversations.ndjson",
            source_key="conversations",
            handler=self._handle_conversation_line,
        )
        self._ingest_ndjson(
            company_root / "state" / "tasks.ndjson",
            source_key="tasks",
            handler=self._handle_task_line,
        )
        self._ingest_ndjson(
            company_root / "state" / "consultations.ndjson",
            source_key="consultations",
            handler=self._handle_consultation_line,
        )
        self._ingest_ndjson(
            company_root / "state" / "research_notes.ndjson",
            source_key="research_notes",
            handler=self._handle_research_line,
        )
        self._ingest_ndjson(
            company_root / "decisions" / "log.ndjson",
            source_key="decisions",
            handler=self._handle_decision_line,
        )
        self._ingest_ndjson(
            company_root / "state" / "creator_reviews.ndjson",
            source_key="creator_reviews",
            handler=self._handle_creator_review_line,
        )

    def _ingest_ndjson(self, path: Path, *, source_key: str, handler) -> None:
        if not path.exists():
            return
        offset = 0
        try:
            offset = self.index.get_source_offset(source_key)
        except Exception:
            offset = 0

        try:
            with open(path, "rb") as f:
                size = f.seek(0, 2)
                if offset < 0 or offset > size:
                    offset = 0
                f.seek(offset)

                while True:
                    line = f.readline()
                    if not line:
                        break
                    s = line.decode("utf-8", errors="replace").strip()
                    if not s:
                        continue
                    try:
                        handler(s)
                    except Exception:
                        logger.debug(
                            "Failed to ingest line for %s (%s)", source_key, path, exc_info=True
                        )

                new_offset = f.tell()

            self.index.set_source_offset(source_key, new_offset)
        except Exception:
            logger.warning("NDJSON ingestion failed: %s", path, exc_info=True)

    def _upsert_constitution(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            c = constitution_load(path)
            text = "\n".join(
                [
                    f"目的: {c.purpose}",
                    f"予算上限: ${c.budget.limit_usd}/{c.budget.window_minutes}分",
                    f"WIP制限: {c.work_principles.wip_limit}件",
                    f"優先KPI: Creatorスコア（優先: {c.creator_score_policy.priority}）"
                    if getattr(c, "creator_score_policy", None)
                    else "優先KPI: 未設定",
                ]
            )
            self.index.upsert(
                doc_id="constitution",
                text=text,
                source_type="constitution",
                source_id=str(path),
                importance=5,
                tags=["constitution", "pinned"],
                created_at=datetime.now(timezone.utc),
            )
        except Exception:
            logger.debug("Failed to upsert constitution", exc_info=True)

    def _upsert_vision(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            text = path.read_text(encoding="utf-8")
            self.index.upsert(
                doc_id="vision",
                text=text,
                source_type="vision",
                source_id=str(path),
                importance=5,
                tags=["vision", "pinned"],
                created_at=datetime.now(timezone.utc),
            )
        except Exception:
            logger.debug("Failed to upsert vision", exc_info=True)

    def _handle_conversation_line(self, raw: str) -> None:
        entry = ConversationEntry.model_validate_json(raw)
        h = hashlib.sha1((entry.content or "").encode("utf-8")).hexdigest()[:10]
        doc_id = f"conv:{entry.timestamp.isoformat()}:{entry.role}:{h}"
        tags = ["conversation", entry.role]
        if entry.user_id:
            tags.append("creator")
        importance = 4 if (entry.role == "user" and _looks_pinnable(entry.content)) else 3
        self.index.upsert(
            doc_id=doc_id,
            text=entry.content,
            source_type="conversation",
            source_id=entry.task_id or entry.user_id,
            created_at=entry.timestamp,
            importance=importance,
            tags=tags,
        )

    def _handle_task_line(self, raw: str) -> None:
        entry = TaskEntry.model_validate_json(raw)
        if entry.status not in ("completed", "failed"):
            return
        parts = [f"タスク[{entry.status}] {entry.description}"]
        if entry.result:
            parts.append(f"結果: {entry.result}")
        if entry.error:
            parts.append(f"エラー: {entry.error}")
        text = "\n".join(parts)
        doc_id = f"task:{entry.task_id}:{entry.updated_at.isoformat()}:{entry.status}"
        self.index.upsert(
            doc_id=doc_id,
            text=text,
            source_type="task",
            source_id=entry.task_id,
            created_at=entry.updated_at,
            importance=4 if entry.status == "failed" else 3,
            tags=["task", entry.status],
        )

    def _handle_consultation_line(self, raw: str) -> None:
        entry = ConsultationEntry.model_validate_json(raw)
        tags = ["consult", entry.status]
        importance = 5 if entry.status == "pending" else 3
        doc_id = f"consult:{entry.consultation_id}:{entry.status}:{entry.created_at.isoformat()}"
        self.index.upsert(
            doc_id=doc_id,
            text=entry.content,
            source_type="consult",
            source_id=entry.related_task_id or entry.consultation_id,
            created_at=entry.created_at,
            importance=importance,
            tags=tags,
        )

    def _handle_research_line(self, raw: str) -> None:
        note = ResearchNote.model_validate_json(raw)
        doc_id = f"research:{hashlib.sha1(note.source_url.encode('utf-8')).hexdigest()[:12]}:{note.retrieved_at.isoformat()}"
        text = f"{note.title}\n{note.source_url}\n{note.summary}"
        self.index.upsert(
            doc_id=doc_id,
            text=text,
            source_type="research",
            source_id=note.source_url,
            created_at=note.retrieved_at,
            importance=3,
            tags=["research"],
        )

    def _handle_decision_line(self, raw: str) -> None:
        d = DecisionLogEntry.model_validate_json(raw)
        rid = d.request_id or d.date.isoformat()
        h = hashlib.sha1(d.decision.encode("utf-8")).hexdigest()[:10]
        doc_id = f"decision:{rid}:{h}"
        text = "\n".join(
            [
                f"意思決定: {d.decision}",
                f"理由: {d.why}",
                f"影響: {d.scope}",
                f"見直し条件: {d.revisit}",
                f"status: {d.status or 'decided'}",
            ]
        )
        tags = ["decision", d.status or "decided"]
        self.index.upsert(
            doc_id=doc_id,
            text=text,
            source_type="decision",
            source_id=d.request_id or "",
            created_at=datetime.now(timezone.utc),
            importance=4,
            tags=tags,
        )

    def _handle_creator_review_line(self, raw: str) -> None:
        r = CreatorReview.model_validate_json(raw)
        doc_id = f"review:{r.timestamp.isoformat()}:{r.score_total_100}"
        axis = []
        if r.score_interestingness_25 is not None:
            axis.append(f"面白さ{r.score_interestingness_25}/25")
        if r.score_cost_efficiency_25 is not None:
            axis.append(f"コスト効率{r.score_cost_efficiency_25}/25")
        if r.score_realism_25 is not None:
            axis.append(f"現実性{r.score_realism_25}/25")
        if r.score_evolvability_25 is not None:
            axis.append(f"進化性{r.score_evolvability_25}/25")
        axis_text = " ".join(axis) if axis else "軸スコアなし"
        text = f"Creatorスコア: {r.score_total_100}/100 ({axis_text})\n{r.comment}".strip()
        self.index.upsert(
            doc_id=doc_id,
            text=text,
            source_type="review",
            source_id=r.user_id or "",
            created_at=r.timestamp,
            importance=4,
            tags=["review"],
        )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall_for_prompt(self, query: str, *, limit: int = 8) -> list[str]:
        """Return formatted recall snippets for inclusion in prompts."""
        try:
            hits = self.index.search(
                query,
                limit=limit,
                exclude_source_types={"constitution", "vision"},
            )
        except Exception:
            logger.warning("Memory recall failed", exc_info=True)
            return []

        out: list[str] = []
        for h in hits:
            ts = h.created_at.strftime("%Y-%m-%d %H:%M:%S")
            st = h.source_type or "memory"
            out.append(f"- [{ts}] ({st}) {_truncate(h.text)}")
        return out

    def summary_for_prompt(self) -> str:
        return self.summary.format_for_prompt(max_recent=8)

    # ------------------------------------------------------------------
    # Creator utilities (pin/recall)
    # ------------------------------------------------------------------

    def pin(self, text: str) -> str:
        """Persist a pinned memory item and index it. Returns doc_id."""
        t = (text or "").strip()
        h = hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
        doc_id = f"pin:{h}"
        self.index.upsert(
            doc_id=doc_id,
            text=t,
            source_type="pinned",
            source_id="creator",
            created_at=datetime.now(timezone.utc),
            importance=5,
            tags=["pinned"],
        )
        self.summary.update(pinned_add=[t], recent_add=[])
        return doc_id

    def search_text(self, query: str, *, limit: int = 5) -> str:
        """Return a human-readable search result string."""
        hits = self.index.search(query, limit=limit)
        if not hits:
            return "検索結果: なし"
        lines = ["検索結果:"]
        for h in hits:
            ts = h.created_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] ({h.source_type}) {_truncate(h.text)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summarization / journaling
    # ------------------------------------------------------------------

    def note_interaction(
        self,
        *,
        timestamp: datetime,
        user_id: str | None,
        request_text: str,
        response_text: str,
        snapshot_lines: list[str] | None = None,
    ) -> None:
        pinned_add: list[str] = []
        recent_add: list[str] = []

        if _looks_pinnable(request_text):
            pinned_add.append(_truncate(request_text, 300))

        recent_add.append(
            f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"request: {_truncate(request_text, 120)} / response: {_truncate(response_text, 120)}"
        )

        try:
            self.summary.update(pinned_add=pinned_add, recent_add=recent_add)
        except Exception:
            logger.warning("Failed to update rolling summary", exc_info=True)

        try:
            self.journal.append_interaction(
                timestamp=timestamp,
                user_id=user_id,
                request_text=request_text,
                response_text=response_text,
                snapshot_lines=snapshot_lines,
            )
        except Exception:
            logger.warning("Failed to append journal", exc_info=True)

