"""Context builder for CEO AI system prompt."""

from __future__ import annotations

from dataclasses import dataclass, field

from models import (
    CommitmentEntry,
    ConstitutionModel,
    ConversationEntry,
    CreatorReview,
    DecisionLogEntry,
    InitiativeEntry,
    ResearchNote,
    StrategyDirection,
    TaskEntry,
)


@dataclass
class TaskHistoryContext:
    """Task history context passed from Manager."""

    completed: list[TaskEntry] = field(default_factory=list)
    failed: list[TaskEntry] = field(default_factory=list)
    running: list[TaskEntry] = field(default_factory=list)
    paused: list[TaskEntry] = field(default_factory=list)
    canceled: list[TaskEntry] = field(default_factory=list)


def build_system_prompt(
    constitution: ConstitutionModel | None = None,
    wip: list[str] | None = None,
    recent_decisions: list[DecisionLogEntry] | None = None,
    budget_spent: float = 0.0,
    budget_limit: float = 10.0,
    conversation_history: list[ConversationEntry] | None = None,
    vision_text: str | None = None,
    curated_memory_text: str | None = None,
    daily_memory_text: str | None = None,
    creator_reviews: list[CreatorReview] | None = None,
    research_notes: list[ResearchNote] | None = None,
    task_history: TaskHistoryContext | None = None,
    active_initiatives: list[InitiativeEntry] | None = None,
    strategy_direction: StrategyDirection | None = None,
    model_catalog_text: str | None = None,
    rolling_summary: str | None = None,
    recalled_memories: list[str] | None = None,
    slack_thread_context: str | None = None,
    open_commitments: list[CommitmentEntry] | None = None,
    policy_memory_text: str | None = None,
    policy_timeline_text: str | None = None,
    policy_conflicts_text: str | None = None,
    **_unused,
) -> str:
    """Build system prompt with short-term + long-term memory context."""

    wip = wip or []
    recent_decisions = recent_decisions or []
    recalled_memories = recalled_memories or []

    sections: list[str] = [
        "あなたはAI会社の社長AIです。",
        "このVPS上で事業の実行・改善・運用を担う主体として行動してください。",
        "判断時は必ず『会社方針・運用ルール・予算・直近会話』を確認し、矛盾があれば<consult>でCreatorに相談してください。",
        "",
        "## 実体ファイル",
        "- システムプロンプト: /opt/apps/ai-company/src/context_builder.py の build_system_prompt()",
        "- 読み込み元: /opt/apps/ai-company/src/manager.py の process_message()",
        "- 主要ロジック: /opt/apps/ai-company/src/",
        "- 会社方針記憶: /opt/apps/ai-company/data/companies/alpha/state/policy_memory.ndjson",
        "- 再読込フラグ: /opt/apps/ai-company/data/companies/alpha/state/restart_manager.flag",
        "",
        "## 恒久方針",
        "- あなたはこのVPS内の社長であり、このVPSを使ってあらゆる事業を行っていく主体である。",
        "- あなた自身の挙動コードの変更は、あなたを強化して社長能力を上げるための行為である。",
        "",
        "## 会社方針・重要ルール（長期記憶 / 新しい順）",
        (policy_memory_text or "（なし）"),
        "",
        "## 方針衝突（未解決）",
        (policy_conflicts_text or "（衝突なし）"),
        "",
        "## 方針タイムライン（新しい順）",
        (policy_timeline_text or "（なし）"),
        "",
        "## 予算",
        f"- 消費: ${budget_spent:.2f} / ${budget_limit:.2f}",
        "- 予算方針の変更提案や上限超過リスクがある場合は<consult>で相談すること。",
        "",
        "## WIP",
        "\n".join([f"- {x}" for x in wip]) if wip else "- なし",
        "",
        "## 直近の意思決定",
        "\n".join([f"- {d.date}: {d.decision}（理由: {d.why}）" for d in recent_decisions[:8]]) if recent_decisions else "- なし",
        "",
        "## 直近会話（Slack含む / 新しい順）",
        _format_conversation_history(conversation_history),
        "",
        "## 補助メモ（要約/リコール）",
        _format_optional_text(rolling_summary, fallback="（要約なし）"),
        _format_optional_list(recalled_memories, fallback="（リコールなし）"),
        "",
        "## ビジョン",
        _format_optional_text(vision_text, fallback="（未設定）"),
        "",
        "## キュレート記憶",
        _format_optional_text(curated_memory_text, fallback="（なし）"),
        "",
        "## Daily記憶",
        _format_optional_text(daily_memory_text, fallback="（なし）"),
        "",
        _build_format_section(),
    ]

    return "\n".join(sections)


def _format_optional_text(text: str | None, *, fallback: str) -> str:
    t = (text or "").strip()
    if not t:
        return fallback
    if len(t) > 3000:
        return t[-3000:]
    return t


def _format_optional_list(items: list[str] | None, *, fallback: str) -> str:
    if not items:
        return fallback
    lines = []
    for it in items[:12]:
        s = (it or "").strip()
        if not s:
            continue
        lines.append(s if s.startswith("-") else f"- {s}")
    return "\n".join(lines) if lines else fallback


def _format_conversation_history(history: list[ConversationEntry] | None) -> str:
    if not history:
        return "- なし"
    recent = sorted(history, key=lambda e: e.timestamp, reverse=True)[:40]
    lines: list[str] = []
    for entry in recent:
        ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        content = (entry.content or "").strip().replace("\n", " ")
        if len(content) > 180:
            content = content[:180] + "…"
        lines.append(f"- [{ts}] [{entry.role}] {content}")
    return "\n".join(lines)


def _build_format_section() -> str:
    return "\n".join([
        "## 応答フォーマット",
        "以下のタグを使って応答してください:",
        "",
        "<reply>",
        "Creatorへの返信テキスト",
        "</reply>",
        "",
        "<consult>",
        "Creatorに相談したい内容（方針矛盾・予算変更・高リスク判断など）",
        "</consult>",
        "",
        "<shell>",
        "実行するシェルコマンド",
        "</shell>",
        "",
        "<publish>",
        "self_commit:message / commit:repo_path:message / create_repo:repo_name:description",
        "</publish>",
        "",
        "<memory>",
        "curated: or daily: を使って重要事項を保存",
        "</memory>",
        "",
        "<commitment>",
        "add: ... / close <id>: ...",
        "</commitment>",
        "",
        "<done>",
        "タスク完了の要約",
        "</done>",
        "",
        "重要: 会社方針とルールに反する判断はしない。矛盾時は<consult>で確認する。",
    ])
