"""Creator daily brief formatter.

Creates a fixed, scannable daily brief used for the Creator score KPI loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DailyCostSummary:
    """コスト要約（日報用）."""

    spent_usd_60m: float
    spent_usd_24h: float
    budget_limit_usd_60m: float


@dataclass
class DailyBriefData:
    """Creator日報の入力データ."""

    timestamp: datetime
    company_id: str
    planned_initiatives: list[str] = field(default_factory=list)
    active_initiatives: list[str] = field(default_factory=list)
    paused_initiatives: list[str] = field(default_factory=list)
    canceled_initiatives: list[str] = field(default_factory=list)
    consultations: list[str] = field(default_factory=list)
    cost: DailyCostSummary = field(
        default_factory=lambda: DailyCostSummary(
            spent_usd_60m=0.0,
            spent_usd_24h=0.0,
            budget_limit_usd_60m=10.0,
        )
    )
    latest_creator_score: str = ""
    creator_reply_format: str = ""


def format_daily_brief(data: DailyBriefData) -> str:
    planned_lines = (
        "\n".join(f"- {s}" for s in data.planned_initiatives)
        if data.planned_initiatives
        else "- なし"
    )
    active_lines = (
        "\n".join(f"- {s}" for s in data.active_initiatives)
        if data.active_initiatives
        else "- なし"
    )
    paused_lines = (
        "\n".join(f"- {s}" for s in data.paused_initiatives)
        if data.paused_initiatives
        else "- なし"
    )
    canceled_lines = (
        "\n".join(f"- {s}" for s in data.canceled_initiatives)
        if data.canceled_initiatives
        else "- なし"
    )
    consult_lines = (
        "\n".join(f"- {s}" for s in data.consultations)
        if data.consultations
        else "- なし"
    )

    latest_score_block = data.latest_creator_score.strip() or "（未記録）"
    reply_format_block = data.creator_reply_format.strip() or "（未設定）"

    return f"""\
## Creator日報
**時刻:** {data.timestamp.isoformat()}
**会社:** {data.company_id}

### 最近計画している施策（候補）
{planned_lines}

### 実施している施策（進行中）
{active_lines}

### 保留中の施策
{paused_lines}

### 中止した施策
{canceled_lines}

### Creatorに対する相談内容
{consult_lines}

### 最近かかったコスト
- 直近60分: ${data.cost.spent_usd_60m:.2f} / ${data.cost.budget_limit_usd_60m:.2f}
- 直近24時間: ${data.cost.spent_usd_24h:.2f}

### 直近のCreatorスコア
{latest_score_block}

### Creator返信フォーマット
{reply_format_block}"""
