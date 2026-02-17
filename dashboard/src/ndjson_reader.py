"""NDJSON read-only module for the dashboard.

Provides read_ndjson for parsing NDJSON files into Pydantic models
(with invalid-line skip + logging), and resolve_latest for deduplicating
entries by ID field (keeping the last occurrence).

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


def read_ndjson(path: Path, model_class: type[T]) -> list[T]:
    """NDJSONファイルを読み取り、全エントリをリストで返す。

    ファイルが存在しない場合は空リスト。不正行はスキップしてログ記録。
    """
    if not path.exists():
        return []
    results: list[T] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                results.append(model_class.model_validate_json(stripped))
            except Exception as e:
                logger.warning(f"Skipping invalid line {line_num} in {path}: {e}")
    return results


def resolve_latest(entries: list[T], id_field: str) -> list[T]:
    """同一IDの複数エントリから最新（最後）のエントリのみを返す。"""
    seen: dict[str, T] = {}
    for entry in entries:
        key = getattr(entry, id_field)
        seen[key] = entry  # 後のエントリが上書き
    return list(seen.values())
