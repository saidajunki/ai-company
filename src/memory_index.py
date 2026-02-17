"""Long-term memory index (SQLite + FTS + local vector embeddings).

This module provides a lightweight, dependency-free "vector DB"-like store:
- Stores memory documents in SQLite
- Uses FTS5 (trigram tokenizer) for candidate retrieval (works for Japanese)
- Re-ranks candidates with a local hashing-based embedding (cosine similarity)

Network/API keys are not required.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_DIM = 256
_DEFAULT_EMBED_NGRAM = 3


@dataclass(frozen=True)
class MemoryHit:
    doc_id: str
    text: str
    source_type: str | None
    source_id: str | None
    created_at: datetime
    importance: int
    tags: list[str]
    score: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_local_hash(
    text: str,
    *,
    dim: int = _DEFAULT_EMBED_DIM,
    ngram: int = _DEFAULT_EMBED_NGRAM,
) -> bytes:
    """Create a deterministic local embedding as float32 bytes.

    Uses signed feature hashing over character n-grams. Cheap and works offline.
    """
    t = _normalize_text(text)
    if not t:
        return b""

    vec = [0.0] * dim

    if len(t) < ngram:
        grams = [t]
    else:
        grams = (t[i : i + ngram] for i in range(len(t) - ngram + 1))

    for gram in grams:
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little", signed=False)
        idx = v % dim
        sign = 1.0 if ((v >> 63) & 1) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]

    arr = array.array("f", vec)
    return arr.tobytes()


def _cosine_sim(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = array.array("f")
    bb = array.array("f")
    aa.frombytes(a)
    bb.frombytes(b)
    return float(sum(x * y for x, y in zip(aa, bb)))


def _safe_fts_query(query: str) -> str:
    """Best-effort sanitize for FTS5 query syntax."""
    q = (query or "").strip()
    if not q:
        return ""
    # Remove characters that are likely to be interpreted as FTS operators.
    q = re.sub(r'["`:\\^*()\[\]{}]', " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


class MemoryIndex:
    """SQLite-backed memory index with optional semantic re-ranking."""

    def __init__(
        self,
        path: Path,
        *,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        embed_ngram: int = _DEFAULT_EMBED_NGRAM,
    ) -> None:
        self._path = path
        self._embed_dim = embed_dim
        self._embed_ngram = embed_ngram
        self._lock = threading.Lock()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                logger.warning("Failed to close memory index connection", exc_info=True)

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    source_type TEXT,
                    source_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    tags TEXT NOT NULL DEFAULT '[]',
                    embedding BLOB,
                    content_hash TEXT
                );
                """
            )
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        text,
                        tags,
                        content='memory_docs',
                        content_rowid='id',
                        tokenize='trigram'
                    );
                    """
                )
            except sqlite3.OperationalError:
                logger.warning(
                    "FTS5 trigram tokenizer unavailable; falling back to unicode61"
                )
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        text,
                        tags,
                        content='memory_docs',
                        content_rowid='id',
                        tokenize='unicode61'
                    );
                    """
                )
            cur.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memory_docs_ai AFTER INSERT ON memory_docs BEGIN
                    INSERT INTO memory_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_docs_ad AFTER DELETE ON memory_docs BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, text, tags) VALUES('delete', old.id, old.text, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_docs_au AFTER UPDATE ON memory_docs BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, text, tags) VALUES('delete', old.id, old.text, old.tags);
                    INSERT INTO memory_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
                END;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS source_offsets (
                    source_key TEXT PRIMARY KEY,
                    byte_offset INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def get_source_offset(self, source_key: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT byte_offset FROM source_offsets WHERE source_key=?",
                (source_key,),
            ).fetchone()
            return int(row["byte_offset"]) if row else 0

    def set_source_offset(self, source_key: str, byte_offset: int) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO source_offsets(source_key, byte_offset, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(source_key) DO UPDATE SET
                    byte_offset=excluded.byte_offset,
                    updated_at=excluded.updated_at;
                """,
                (source_key, int(byte_offset), _now_iso()),
            )
            self._conn.commit()

    def upsert(
        self,
        *,
        doc_id: str,
        text: str,
        source_type: str | None = None,
        source_id: str | None = None,
        created_at: datetime | None = None,
        importance: int = 3,
        tags: Sequence[str] | None = None,
    ) -> bool:
        """Upsert a memory doc.

        Returns:
            True when inserted, False when updated.
        """
        normalized = _normalize_text(text)
        if not normalized:
            return False

        created_at_dt = created_at or datetime.now(timezone.utc)
        tags_json = json.dumps(list(tags or []), ensure_ascii=False)
        content_hash = _sha256_hex(normalized)
        embedding = _embed_local_hash(
            normalized, dim=self._embed_dim, ngram=self._embed_ngram
        )

        with self._lock:
            cur = self._conn.cursor()
            existing = cur.execute(
                "SELECT id FROM memory_docs WHERE doc_id=?",
                (doc_id,),
            ).fetchone()

            if existing is None:
                cur.execute(
                    """
                    INSERT INTO memory_docs(
                        doc_id, text, source_type, source_id,
                        created_at, updated_at, importance, tags,
                        embedding, content_hash
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?);
                    """,
                    (
                        doc_id,
                        normalized,
                        source_type,
                        source_id,
                        created_at_dt.isoformat(),
                        _now_iso(),
                        int(importance),
                        tags_json,
                        embedding,
                        content_hash,
                    ),
                )
                self._conn.commit()
                return True

            cur.execute(
                """
                UPDATE memory_docs SET
                    text=?,
                    source_type=?,
                    source_id=?,
                    updated_at=?,
                    importance=?,
                    tags=?,
                    embedding=?,
                    content_hash=?
                WHERE doc_id=?;
                """,
                (
                    normalized,
                    source_type,
                    source_id,
                    _now_iso(),
                    int(importance),
                    tags_json,
                    embedding,
                    content_hash,
                    doc_id,
                ),
            )
            self._conn.commit()
            return False

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        candidate_limit: int = 50,
        exclude_source_types: set[str] | None = None,
    ) -> list[MemoryHit]:
        """Search memory docs.

        Candidate retrieval uses FTS5, then re-ranks by local embedding similarity.
        """
        q = (query or "").strip()
        if not q:
            return []
        fts_q = _safe_fts_query(q)
        if not fts_q:
            return []

        exclude = exclude_source_types or set()
        q_emb = _embed_local_hash(q, dim=self._embed_dim, ngram=self._embed_ngram)

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT
                    d.doc_id,
                    d.text,
                    d.source_type,
                    d.source_id,
                    d.created_at,
                    d.importance,
                    d.tags,
                    d.embedding,
                    bm25(memory_fts) AS rank
                FROM memory_fts
                JOIN memory_docs d ON d.id = memory_fts.rowid
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?;
                """,
                (fts_q, int(candidate_limit)),
            ).fetchall()

        hits: list[MemoryHit] = []
        for r in rows:
            source_type = r["source_type"]
            if source_type and source_type in exclude:
                continue

            try:
                tags = json.loads(r["tags"] or "[]")
                if not isinstance(tags, list):
                    tags = []
                tags = [str(t) for t in tags]
            except Exception:
                tags = []

            created_at = datetime.fromisoformat(str(r["created_at"]))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            emb = r["embedding"] or b""
            sim = _cosine_sim(q_emb, emb)
            importance = int(r["importance"] or 3)
            score = sim + 0.05 * (importance - 3)

            hits.append(
                MemoryHit(
                    doc_id=str(r["doc_id"]),
                    text=str(r["text"]),
                    source_type=str(source_type) if source_type is not None else None,
                    source_id=str(r["source_id"]) if r["source_id"] is not None else None,
                    created_at=created_at,
                    importance=importance,
                    tags=tags,
                    score=float(score),
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: int(limit)]
