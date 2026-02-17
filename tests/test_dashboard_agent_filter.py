from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import importlib
import sys
from pathlib import Path


@contextmanager
def _dashboard_imports():
    """Import dashboard modules without permanently clobbering top-level 'models' imports."""
    repo_root = Path(__file__).resolve().parents[1]
    dashboard_src = repo_root / "dashboard" / "src"

    dashboard_src_str = str(dashboard_src)
    sys.path.insert(0, dashboard_src_str)

    saved = {k: sys.modules.get(k) for k in ("models", "ndjson_reader", "data_service")}
    for k in saved:
        sys.modules.pop(k, None)

    try:
        dash_models = importlib.import_module("models")
        importlib.import_module("ndjson_reader")
        dash_data_service = importlib.import_module("data_service")
        yield dash_models, dash_data_service
    finally:
        for k in ("models", "ndjson_reader", "data_service"):
            sys.modules.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
        if dashboard_src_str in sys.path:
            sys.path.remove(dashboard_src_str)


def test_dashboard_agent_filter_hides_old_inactive_but_keeps_pinned():
    with _dashboard_imports() as (dash_models, dash_data_service):
        AgentEntry = dash_models.AgentEntry
        DataService = dash_data_service.DataService

        now = datetime.now(timezone.utc)
        old = now - timedelta(days=10)

        agents = [
            AgentEntry(
                agent_id="ceo",
                name="CEO AI",
                role="ceo",
                model="openai/gpt-4.1-mini",
                budget_limit_usd=10.0,
                status="inactive",
                created_at=old,
                updated_at=old,
            ),
            AgentEntry(
                agent_id="manager",
                name="Manager",
                role="manager",
                model="openai/gpt-4.1-mini",
                budget_limit_usd=10.0,
                status="inactive",
                created_at=old,
                updated_at=old,
            ),
            AgentEntry(
                agent_id="sub-aaaaaa",
                name="researcher#aaaaaa",
                role="researcher",
                model="openai/gpt-4.1-mini",
                budget_limit_usd=1.0,
                status="inactive",
                created_at=old,
                updated_at=old,
            ),
            AgentEntry(
                agent_id="sub-bbbbbb",
                name="developer#bbbbbb",
                role="developer",
                model="openai/gpt-4.1",
                budget_limit_usd=1.0,
                status="active",
                created_at=now,
                updated_at=now,
            ),
        ]

        filtered = DataService._filter_agents_for_dashboard(agents)
        ids = [a.agent_id for a in filtered]

        assert "ceo" in ids
        assert "manager" in ids
        assert "sub-bbbbbb" in ids
        assert "sub-aaaaaa" not in ids

