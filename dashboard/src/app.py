"""FastAPI application for the AI Company Dashboard.

Serves a unified dashboard API and static frontend files.

Requirements: 7.1, 8.4
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from data_service import DataService
from models import (
    ConsultationResponse,
    ConversationResponse,
    CostSummaryResponse,
    DashboardResponse,
    InitiativeResponse,
    TaskResponse,
    TasksSummaryResponse,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
COMPANY_ID = os.environ.get("COMPANY_ID", "alpha")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))

app = FastAPI(title="AI Company Dashboard")


@app.get("/api/dashboard")
async def get_dashboard() -> DashboardResponse:
    """ダッシュボード全体のデータを返す統合エンドポイント。"""
    service = DataService(DATA_DIR, COMPANY_ID)
    data = service.get_dashboard_data()

    initiatives = [
        InitiativeResponse(
            initiative_id=i.initiative_id,
            title=i.title,
            description=i.description,
            status=i.status,
            estimated_scores=i.estimated_scores.model_dump() if i.estimated_scores else None,
            actual_scores=i.actual_scores.model_dump() if i.actual_scores else None,
            task_ids=i.task_ids,
            created_at=i.created_at,
            updated_at=i.updated_at,
        )
        for i in data.initiatives
    ]

    recent_tasks = [
        TaskResponse(
            task_id=t.task_id,
            description=t.description,
            priority=t.priority,
            status=t.status,
            agent_id=t.agent_id,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in data.recent_tasks
    ]

    consultations = [
        ConsultationResponse(
            consultation_id=c.consultation_id,
            content=c.content,
            status=c.status,
            created_at=c.created_at,
            resolved_at=c.resolved_at,
            resolution=c.resolution,
            related_task_id=c.related_task_id,
        )
        for c in data.consultations
    ]

    conversations = [
        ConversationResponse(
            timestamp=cv.timestamp,
            role=cv.role,
            content=cv.content,
            user_id=cv.user_id,
            task_id=cv.task_id,
        )
        for cv in data.conversations
    ]

    tasks_summary = TasksSummaryResponse(
        pending=data.tasks_summary.pending,
        running=data.tasks_summary.running,
        completed=data.tasks_summary.completed,
        failed=data.tasks_summary.failed,
        total=data.tasks_summary.total,
    )

    cost = CostSummaryResponse(
        window_60min_usd=data.cost.window_60min_usd,
        total_usd=data.cost.total_usd,
        budget_limit_usd=data.cost.budget_limit_usd,
        budget_usage_percent=data.cost.budget_usage_percent,
    )

    return DashboardResponse(
        timestamp=datetime.now(timezone.utc),
        heartbeat=data.heartbeat,
        agents=data.agents,
        initiatives=initiatives,
        tasks_summary=tasks_summary,
        recent_tasks=recent_tasks,
        consultations=consultations,
        cost=cost,
        conversations=conversations,
    )


@app.get("/api/health")
async def health_check() -> dict:
    """ヘルスチェック用エンドポイント。"""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# 静的ファイル配信（API ルートの後にマウント）
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True))
