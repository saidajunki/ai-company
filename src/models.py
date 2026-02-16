"""Data models (Pydantic) for AI Company.

Defines the core data structures used throughout the system:
- ConstitutionModel: 会社憲法のYAMLスキーマ
- LedgerEvent: コスト台帳イベント
- DecisionLogEntry: 意思決定ログエントリ
- HeartbeatState: Heartbeatファイル
- PricingCache: 単価キャッシュ
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# --- Constitution sub-models ---

class DisclosurePolicy(BaseModel):
    default: str = "public"
    sensitive_info_allowed: bool = True
    risk_control: str = "いつでも無効化できる・上限を設定できる運用でリスク制御"


class CreatorIntervention(BaseModel):
    scope: list[str] = Field(default_factory=lambda: [
        "契約・支払い方法の登録・アカウント作成",
        "上記に付随する承認",
    ])
    note: str = "それ以外の作業はAIが前提で進める"


class Budget(BaseModel):
    limit_usd: float = 10
    window_minutes: int = 60
    precision: str = "概算（数万円単位の誤差は許容）"


class WorkPrinciples(BaseModel):
    wip_limit: int = 3
    record_diffs: bool = True
    record_sources_and_dates: bool = True


class Amendment(BaseModel):
    proposer: str = "Manager（理由と影響を添える）"
    approver: str = "Creator（Slackリアクション）"
    process: str = "提案→承認→適用→Decision_Logに記録"


class ConstitutionModel(BaseModel):
    """会社憲法のYAMLスキーマに対応するモデル (Req 1.1, 1.4)."""

    version: int = 1
    purpose: str = "研究開発中心のAI組織"
    disclosure_policy: DisclosurePolicy = Field(default_factory=DisclosurePolicy)
    creator_intervention: CreatorIntervention = Field(default_factory=CreatorIntervention)
    budget: Budget = Field(default_factory=Budget)
    work_principles: WorkPrinciples = Field(default_factory=WorkPrinciples)
    amendment: Amendment = Field(default_factory=Amendment)


# --- Ledger Event ---

class LedgerEvent(BaseModel):
    """台帳イベント (Req 5.1, 5.2).

    event_type ごとに必須フィールドが異なる:
    - llm_call: provider, model, input_tokens, output_tokens,
                unit_price_usd_per_1k_input_tokens, unit_price_usd_per_1k_output_tokens,
                price_retrieved_at, estimated_cost_usd
    - api_call: api_call_count, api_unit_price_usd
    """

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
    metadata: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_event_fields(self) -> "LedgerEvent":
        """llm_call イベントには LLM 固有フィールドが必須."""
        if self.event_type == "llm_call":
            required = [
                "provider", "model", "input_tokens", "output_tokens",
                "unit_price_usd_per_1k_input_tokens",
                "unit_price_usd_per_1k_output_tokens",
                "price_retrieved_at", "estimated_cost_usd",
            ]
            missing = [f for f in required if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"llm_call event requires fields: {', '.join(missing)}"
                )
        return self


# --- Decision Log ---

class DecisionLogEntry(BaseModel):
    """意思決定ログエントリ (Req 8.1, 8.4)."""

    date: date
    decision: str
    why: str
    scope: str
    revisit: str
    status: Optional[Literal["decided", "proposed", "approved", "rejected"]] = None
    request_id: Optional[str] = None
    related_constitution_field: Optional[str] = None


# --- Heartbeat ---

class HeartbeatState(BaseModel):
    """Heartbeatファイルのモデル (Req 10.1, 10.2)."""

    updated_at: datetime
    manager_pid: int
    status: Literal["running", "idle", "waiting_approval"]
    current_wip: list[str] = Field(default_factory=list, max_length=3)
    last_report_at: Optional[datetime] = None


# --- Pricing Cache ---

class ModelPricing(BaseModel):
    """個別モデルの単価情報."""

    input_price_per_1k: float
    output_price_per_1k: float
    retrieved_at: datetime


class PricingCache(BaseModel):
    """単価キャッシュのモデル (Req 9.4)."""

    retrieved_at: datetime
    models: dict[str, ModelPricing] = Field(default_factory=dict)


# --- Conversation Memory (Req 1.1) ---

class ConversationEntry(BaseModel):
    """会話履歴エントリ."""

    timestamp: datetime
    role: Literal["user", "assistant", "system"]
    content: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None


# --- Task Queue (Req 3.1) ---

class TaskEntry(BaseModel):
    """自律タスクエントリ."""

    task_id: str
    description: str
    priority: int = Field(default=3, ge=1, le=5)
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    result: Optional[str] = None
    error: Optional[str] = None
    agent_id: str = "ceo"


# --- Agent Registry (Req 4.1) ---

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


# --- Service Registry (Req 5.1) ---

class ServiceEntry(BaseModel):
    """サービスレジストリエントリ."""

    name: str
    description: str
    status: Literal["active", "archived"] = "active"
    created_at: datetime
    updated_at: datetime
    agent_id: str
