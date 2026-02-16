"""Ten-minute report formatter (Req 3.1).

Generates a fixed-format Markdown report containing:
WIP, Δ(10m), Next(10m), Blockers, Cost(60m), Approvals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CostSummary:
    """60分移動窓のコストサマリー."""

    spent_usd: float
    remaining_usd: float
    limit_usd: float = 10.0
    allocation_plan: str = ""


@dataclass
class ReportData:
    """10分レポートの入力データ."""

    timestamp: datetime
    company_id: str
    wip: list[str] = field(default_factory=list)
    delta_description: str = ""
    artifact_links: str = ""
    next_plan: str = ""
    blockers: list[str] = field(default_factory=list)
    cost: CostSummary = field(default_factory=lambda: CostSummary(0.0, 10.0, 10.0))
    approvals: list[str] = field(default_factory=list)
    running_tasks: list[str] = field(default_factory=list)
    active_agents: list[str] = field(default_factory=list)
    recent_services: list[str] = field(default_factory=list)


def format_report(data: ReportData) -> str:
    """ReportData を固定テンプレートの Markdown 文字列に変換する."""
    wip_items = data.wip[:3] if data.wip else []
    wip_lines = "\n".join(f"- {item}" for item in wip_items) if wip_items else "- なし"

    blocker_lines = (
        "\n".join(f"- {b}" for b in data.blockers) if data.blockers else "- なし"
    )

    approval_lines = (
        "\n".join(f"- {a}" for a in data.approvals) if data.approvals else "- なし"
    )

    task_lines = (
        "\n".join(f"- {t}" for t in data.running_tasks)
        if data.running_tasks
        else "- なし"
    )

    agent_lines = (
        "\n".join(f"- {a}" for a in data.active_agents)
        if data.active_agents
        else "- なし"
    )

    service_lines = (
        "\n".join(f"- {s}" for s in data.recent_services)
        if data.recent_services
        else "- なし"
    )

    return f"""\
## 10分レポート
**時刻:** {data.timestamp.isoformat()}
**会社:** {data.company_id}

### WIP（最大3件）
{wip_lines}

### Δ(10m)
- {data.delta_description}
- 成果物: {data.artifact_links}

### Next(10m)
- {data.next_plan}

### Blockers
{blocker_lines}

### Cost(60m)
- 消費: ${data.cost.spent_usd:.2f} / ${data.cost.limit_usd:.2f}
- 残り: ${data.cost.remaining_usd:.2f}
- 配分予定: {data.cost.allocation_plan}

### Approvals
{approval_lines}

### 自律タスク
{task_lines}

### エージェント
{agent_lines}

### サービス
{service_lines}"""
