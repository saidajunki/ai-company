"""Manager state persistence and recovery.

Provides save/restore of the Manager's runtime state (WIP, ledger, decision log,
constitution, heartbeat) to/from the filesystem so that the process can resume
after a restart.

Requirements: 6.3, 7.3, 7.5
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from constitution_store import constitution_load, constitution_save
from models import (
    ConstitutionModel,
    DecisionLogEntry,
    HeartbeatState,
    LedgerEvent,
)
from ndjson_store import ndjson_append, ndjson_read


# ---------------------------------------------------------------------------
# Standard file-path helpers
# ---------------------------------------------------------------------------

def ledger_path(base_dir: Path, company_id: str) -> Path:
    """Return ``companies/<company_id>/ledger/events.ndjson``."""
    return base_dir / "companies" / company_id / "ledger" / "events.ndjson"


def decision_log_path(base_dir: Path, company_id: str) -> Path:
    """Return ``companies/<company_id>/decisions/log.ndjson``."""
    return base_dir / "companies" / company_id / "decisions" / "log.ndjson"


def heartbeat_path(base_dir: Path, company_id: str) -> Path:
    """Return ``companies/<company_id>/state/heartbeat.json``."""
    return base_dir / "companies" / company_id / "state" / "heartbeat.json"


def constitution_path(base_dir: Path, company_id: str) -> Path:
    """Return ``companies/<company_id>/constitution.yaml``."""
    return base_dir / "companies" / company_id / "constitution.yaml"


# ---------------------------------------------------------------------------
# ManagerState – aggregate of all persisted state
# ---------------------------------------------------------------------------

@dataclass
class ManagerState:
    """Aggregate of all Manager runtime state that is persisted to disk.

    Attributes:
        wip: List of currently in-progress task descriptions (max 3).
        ledger_events: Append-only cost/activity ledger.
        decision_log: Append-only decision log.
        constitution: The company constitution (may be ``None`` before first load).
        heartbeat: Latest heartbeat snapshot (may be ``None``).
    """

    wip: list[str] = field(default_factory=list)
    ledger_events: list[LedgerEvent] = field(default_factory=list)
    decision_log: list[DecisionLogEntry] = field(default_factory=list)
    constitution: Optional[ConstitutionModel] = None
    heartbeat: Optional[HeartbeatState] = None


# ---------------------------------------------------------------------------
# Restore (load from filesystem)
# ---------------------------------------------------------------------------

def restore_state(base_dir: Path, company_id: str) -> ManagerState:
    """Restore Manager state from the filesystem after a restart.

    Loads each component independently so that partial state (e.g. ledger
    exists but heartbeat does not) is recovered gracefully.

    Returns a ``ManagerState`` with whatever data was found on disk.
    """
    state = ManagerState()

    # Ledger (Req 6.3 – read existing ledger on restart)
    lp = ledger_path(base_dir, company_id)
    state.ledger_events = ndjson_read(lp, LedgerEvent)

    # Decision log
    dp = decision_log_path(base_dir, company_id)
    state.decision_log = ndjson_read(dp, DecisionLogEntry)

    # Constitution
    cp = constitution_path(base_dir, company_id)
    try:
        state.constitution = constitution_load(cp)
    except FileNotFoundError:
        state.constitution = None

    # Heartbeat
    hp = heartbeat_path(base_dir, company_id)
    if hp.exists():
        raw = hp.read_text(encoding="utf-8")
        state.heartbeat = HeartbeatState.model_validate_json(raw)

    # WIP – derived from heartbeat's current_wip if available
    if state.heartbeat is not None:
        state.wip = list(state.heartbeat.current_wip)

    return state


# ---------------------------------------------------------------------------
# Persist (save to filesystem)
# ---------------------------------------------------------------------------

def persist_state(base_dir: Path, company_id: str, state: ManagerState) -> None:
    """Persist the full Manager state to the filesystem.

    * Ledger and decision log are written as complete NDJSON files (overwrite).
    * Constitution is saved as YAML.
    * Heartbeat is saved as a single JSON file (overwrite).

    This is intended for bulk snapshots.  For incremental writes during
    normal operation, prefer ``append_ledger_event`` / ``append_decision``.
    """
    # Ledger
    lp = ledger_path(base_dir, company_id)
    _write_ndjson_bulk(lp, state.ledger_events)

    # Decision log
    dp = decision_log_path(base_dir, company_id)
    _write_ndjson_bulk(dp, state.decision_log)

    # Constitution
    if state.constitution is not None:
        cp = constitution_path(base_dir, company_id)
        constitution_save(cp, state.constitution)

    # Heartbeat
    if state.heartbeat is not None:
        save_heartbeat(base_dir, company_id, state.heartbeat)


# ---------------------------------------------------------------------------
# Incremental helpers (Req 7.5 – step-level recording for retry)
# ---------------------------------------------------------------------------

def append_ledger_event(base_dir: Path, company_id: str, event: LedgerEvent) -> None:
    """Append a single ledger event (append-only, Req 6.2)."""
    ndjson_append(ledger_path(base_dir, company_id), event)


def append_decision(base_dir: Path, company_id: str, entry: DecisionLogEntry) -> None:
    """Append a single decision log entry (append-only)."""
    ndjson_append(decision_log_path(base_dir, company_id), entry)


def save_heartbeat(base_dir: Path, company_id: str, hb: HeartbeatState) -> None:
    """Overwrite the heartbeat file with the latest state."""
    hp = heartbeat_path(base_dir, company_id)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(hb.model_dump_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_ndjson_bulk(path: Path, items: list) -> None:
    """Write a list of Pydantic models as a complete NDJSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
