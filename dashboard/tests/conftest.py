"""Common test fixtures for AI Company Dashboard tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory mimicking the ai-company-data volume structure."""
    d = tmp_path / "companies" / "alpha"
    d.mkdir(parents=True)
    (d / "ledger").mkdir()
    return d


@pytest.fixture
def empty_data_dir(tmp_path: Path) -> Path:
    """Return a data directory path that does not exist (for missing-file tests)."""
    return tmp_path / "nonexistent"


def write_ndjson(path: Path, records: list[dict]) -> None:
    """Helper: write a list of dicts as NDJSON lines to a file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
