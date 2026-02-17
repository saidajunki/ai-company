"""Data models for AI Company Dashboard.

Simplified read-only copies of ai-company/src/models.py core models,
plus API response models for the dashboard endpoints.
Write-only validators (e.g. retry_count, event_fields) are removed
since the dashboard only reads data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore[assignment]

from pydantic import BaseModel, Field


# ── Core domain models (read-only copies) ──────────────────────────


class HeartbeatState(BaseModel):
    """Heartbeat ファイルのモデル."""

    updated_at: datetime
    manager_pid: int
    status: Literal["running", "idle", "waiting_approval"]
    current_wip: List[str] = Field(default_factory=list, max_length=3)
    last_report_at: Optional[datetime] = None


class AgentEntry(BaseModel):
    """エージェントレジストリエントリ."""

    agent_id: str
    name: str
    role: str
    model: str
    budget_limit_usd: float = Field(ge=0)
    status: Literal["active", "inactive"] = "active"
    created_at: datetime
    updated_at: datetime


class InitiativeScores(BaseModel):
    """イニシアチブのスコア予測/実績."""

    interestingness: int = Field(ge=0, le=25)
    cost_efficiency: int = Field(ge=0, le=25)
    realism: int = Field(ge=0, le=25)
    evolvability: int = Field(ge=0, le=25)

    @property
    def total(self) -> int:
        return (
            self.interestingness
            + self.cost_efficiency
            + self.realism
            + self.evolvability
        )


class InitiativeEntry(BaseModel):
    """ビジネスイニシアチブエントリ."""

    initiative_id: str
    title: str
    description: str
    status: Literal[
        "planned",
        "in_progress",
        "completed",
        "abandoned",
        "consulting",
    ]
    task_ids: List[str] = Field(default_factory=list)
    estimated_scores: Optional[InitiativeScores] = None
    actual_scores: Optional[InitiativeScores] = None
    retrospective: Optional[str] = None
    first_step: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TaskEntry(BaseModel):
    """自律タスクエントリ (read-only — validate_retry_count removed)."""

    task_id: str
    description: str
    priority: int = Field(default=3, ge=1, le=5)
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    result: Optional[str] = None
    error: Optional[str] = None
    agent_id: str = "ceo"
    depends_on: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_notes: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    parent_task_id: Optional[str] = None


class ConsultationEntry(BaseModel):
    """Creator への相談事項."""

    consultation_id: str
    created_at: datetime
    status: Literal["pending", "resolved"] = "pending"
    content: str
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    related_task_id: Optional[str] = None


class ConversationEntry(BaseModel):
    """会話履歴エントリ."""

    timestamp: datetime
    role: Literal["user", "assistant", "system"]
    content: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None


class LedgerEvent(BaseModel):
    """台帳イベント (read-only — validate_event_fields removed)."""

    timestamp: datetime
    event_type: Literal["llm_call", "api_call", "shell_exec", "decision", "report"]
    agent_id: str
    task_id: str

    # LLM call fields
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    unit_price_usd_per_1k_input_tokens: Optional[float] = Field(default=None, ge=0)
    unit_price_usd_per_1k_output_tokens: Optional[float] = Field(default=None, ge=0)
    price_retrieved_at: Optional[datetime] = None
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)

    # API call fields
    api_call_count: Optional[int] = Field(default=None, ge=0)
    api_unit_price_usd: Optional[float] = Field(default=None, ge=0)

    # Generic metadata
    metadata: Optional[Dict[str, Any]] = None


class Budget(BaseModel):
    """予算設定."""

    limit_usd: float = 10
    window_minutes: int = 60
    precision: str = "概算（数万円単位の誤差は許容）"


# ── API response models ────────────────────────────────────────────


class TasksSummaryResponse(BaseModel):
    """タスクステータスごとの件数サマリー."""

    pending: int
    running: int
    completed: int
    failed: int
    total: int


class CostSummaryResponse(BaseModel):
    """コスト使用状況サマリー."""

    window_60min_usd: float
    total_usd: float
    budget_limit_usd: float
    budget_usage_percent: float


class InitiativeResponse(BaseModel):
    """イニシアチブの API レスポンス."""

    initiative_id: str
    title: str
    description: str
    status: str
    estimated_scores: Optional[dict] = None
    actual_scores: Optional[dict] = None
    task_ids: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class TaskResponse(BaseModel):
    """タスクの API レスポンス."""

    task_id: str
    description: str
    priority: int
    status: str
    agent_id: str
    created_at: datetime
    updated_at: datetime


class ConsultationResponse(BaseModel):
    """相談事項の API レスポンス."""

    consultation_id: str
    content: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    related_task_id: Optional[str] = None


class ConversationResponse(BaseModel):
    """会話の API レスポンス."""

    timestamp: datetime
    role: str
    content: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None


class DashboardResponse(BaseModel):
    """GET /api/dashboard の統合レスポンス."""

    timestamp: datetime
    heartbeat: Optional[HeartbeatState] = None
    agents: List[AgentEntry] = Field(default_factory=list)
    initiatives: List[InitiativeResponse] = Field(default_factory=list)
    tasks_summary: TasksSummaryResponse
    recent_tasks: List[TaskResponse] = Field(default_factory=list)
    consultations: List[ConsultationResponse] = Field(default_factory=list)
    cost: CostSummaryResponse
    conversations: List[ConversationResponse] = Field(default_factory=list)
