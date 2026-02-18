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
from typing import Any, Dict, List, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import BaseModel, Field, model_validator


# --- Constitution sub-models ---

class DisclosurePolicy(BaseModel):
    default: str = "public"
    sensitive_info_allowed: bool = False
    risk_control: str = "APIキー/トークン/支払い情報/個人情報などの機微情報は公開しない"


class CreatorScorePolicy(BaseModel):
    """Creatorスコア運用ポリシー（当面の北極星指標）."""

    enabled: bool = True
    priority: str = "面白さ最優先"
    axis_max_score: int = 25
    total_max_score: int = 100
    axes: Dict[str, str] = Field(default_factory=lambda: {
        "面白さ": (
            "単純に面白い/興味を持てるか。人の野心や知的欲求を満たせるか。"
            "お金が稼げる/コストがかかる等はこの軸では考慮しない。"
        ),
        "コスト効率": (
            "面白さがあっても、コストがかかりすぎる/かかりそうなら減点。"
            "より安い代替手段・小さく試す方法があるか。"
        ),
        "現実性": (
            "現実世界で打ち上げる上で致命的な懸念がないか。"
            "法/規約/技術/運用/セキュリティ/信頼面のリスクを踏まえる。"
        ),
        "進化性": (
            "失敗しても会社にナレッジや有効データが残るか。"
            "過程が再利用可能で、次の施策の質を上げるか。"
        ),
    })
    creator_reply_format: str = (
        "面白さ: x/25, コスト効率: y/25, 現実性: z/25, 進化性: w/25\n"
        "コメント: （自由記述）\n"
        "指示: 継続/ピボット/停止 + 次にやること1つ"
    )


class CreatorIntervention(BaseModel):
    scope: List[str] = Field(default_factory=lambda: [
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
    approver: str = "Creator（Slack返信/リアクション）"
    process: str = "提案→承認→適用→Decision_Logに記録"


class ConstitutionModel(BaseModel):
    """会社憲法のYAMLスキーマに対応するモデル (Req 1.1, 1.4)."""

    version: int = 1
    purpose: str = "有名で面白い存在になり、収益化を目指すAI会社"
    disclosure_policy: DisclosurePolicy = Field(default_factory=DisclosurePolicy)
    creator_intervention: CreatorIntervention = Field(default_factory=CreatorIntervention)
    budget: Budget = Field(default_factory=Budget)
    work_principles: WorkPrinciples = Field(default_factory=WorkPrinciples)
    creator_score_policy: CreatorScorePolicy = Field(default_factory=CreatorScorePolicy)
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
    metadata: Optional[Dict[str, Any]] = None

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
    current_wip: List[str] = Field(default_factory=list, max_length=3)
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
    models: Dict[str, ModelPricing] = Field(default_factory=dict)


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
    source: Literal["creator", "autonomous", "initiative"] = "autonomous"
    status: Literal["pending", "running", "paused", "canceled", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    result: Optional[str] = None
    error: Optional[str] = None
    agent_id: str = "ceo"
    slack_channel: Optional[str] = None
    slack_thread_ts: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_notes: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    parent_task_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_retry_count(self) -> "TaskEntry":
        """retry_count は max_retries 以下でなければならない."""
        if self.retry_count > self.max_retries:
            raise ValueError(
                f"retry_count ({self.retry_count}) must be <= max_retries ({self.max_retries})"
            )
        return self


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


# --- Persistent Employee Registry ---

class EmployeeEntry(BaseModel):
    """永続社員AIのプロファイル."""

    employee_id: str
    name: str
    role: str
    purpose: str
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


# --- Research Note (Req 2.1, 2.2) ---

class ResearchNote(BaseModel):
    """リサーチノートモデル。Web検索結果を構造化して保存する。"""

    query: str
    source_url: str
    title: str
    snippet: str
    summary: str
    published_at: Optional[datetime] = None
    retrieved_at: datetime


# --- Creator Review (KPI loop) ---

class CreatorReview(BaseModel):
    """Creatorによるスコアリング結果（0-100）."""

    timestamp: datetime
    user_id: Optional[str] = None
    score_total_100: int = Field(ge=0, le=100)
    score_interestingness_25: Optional[int] = Field(default=None, ge=0, le=25)
    score_cost_efficiency_25: Optional[int] = Field(default=None, ge=0, le=25)
    score_realism_25: Optional[int] = Field(default=None, ge=0, le=25)
    score_evolvability_25: Optional[int] = Field(default=None, ge=0, le=25)
    comment: str = ""
    raw_text: str = ""


# --- Consultation Queue ---

class ConsultationEntry(BaseModel):
    """Creatorへの相談事項（未解決のものを保持する）."""

    consultation_id: str
    created_at: datetime
    status: Literal["pending", "resolved"] = "pending"
    content: str
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    related_task_id: Optional[str] = None


# --- Commitments / TODOs ---

class CommitmentEntry(BaseModel):
    """約束/TODO（オープンな義務）を保持する."""

    commitment_id: str
    created_at: datetime
    status: Literal["open", "done", "canceled"] = "open"
    title: str = ""
    content: str
    owner: str = "ceo"
    due_date: Optional[date] = None
    related_task_id: Optional[str] = None
    closed_at: Optional[datetime] = None
    close_note: Optional[str] = None


# --- Initiative Models (Req 3.1) ---


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
        "paused",
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


class StrategyDirection(BaseModel):
    """戦略方針の分析結果."""

    strengthen: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    pivot_suggestions: List[str] = Field(default_factory=list)
    weak_axes: List[str] = Field(default_factory=list)
    score_trends: Dict[str, float] = Field(default_factory=dict)
    summary: str = ""


# --- Policy Memory (direction/rules/budget) ---

class PolicyMemoryEntry(BaseModel):
    """会社方針・運用ルール・予算方針などの長期記憶エントリ。"""

    memory_id: str
    created_at: datetime
    category: Literal["direction", "rule", "budget", "operation", "fact"]
    status: Literal["active", "conflicted", "superseded"] = "active"
    content: str
    source: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    conflict_with: List[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)


# --- Adaptive Memory (dynamic long-term memory domains) ---

class AdaptiveMemoryEntry(BaseModel):
    """動的ドメインで管理する長期記憶エントリ。"""

    memory_id: str
    created_at: datetime
    updated_at: datetime
    domain: str
    status: Literal["active", "archived", "pruned"] = "active"
    content: str
    source: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    importance: int = Field(default=3, ge=1, le=5)
    tags: List[str] = Field(default_factory=list)
    last_accessed_at: Optional[datetime] = None


# --- Procedure SoT (verbatim runbook memory) ---

class ProcedureDocument(BaseModel):
    """複数行コマンド手順を丸ごと保存するSoTドキュメント。"""

    doc_id: str
    name: str
    version: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime
    status: Literal["active", "superseded"] = "active"
    visibility: Literal["private", "shared"] = "private"
    steps: List[str] = Field(default_factory=list)
    raw_text: str
    source: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    file_path: str
    tags: List[str] = Field(default_factory=list)
