"""NDJSON read/write module.

Provides append-only write and bulk read for NDJSON (Newline Delimited JSON) files.
Each line is one JSON object serialized from a Pydantic model.

Requirements: 5.5, 6.2, 8.2
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ndjson_append(path: Path, obj: BaseModel) -> None:
    """Serialize a Pydantic model to JSON and append as one line.

    Creates the file and parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(obj.model_dump_json() + "\n")


def ndjson_read(path: Path, model_class: type[T]) -> list[T]:
    """Read all lines from an NDJSON file and deserialize into model instances.

    Returns an empty list if the file doesn't exist. Blank lines are skipped.
    """
    if not path.exists():
        return []

    results: list[T] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            results.append(model_class.model_validate_json(stripped))
    return results
