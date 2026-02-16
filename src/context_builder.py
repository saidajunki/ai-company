"""Context Builder — LLM呼び出し用のシステムプロンプトを構築する.

Provides:
- build_system_prompt: コンテキスト情報からシステムプロンプトを構築

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.3
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import (
    ConstitutionModel,
    ConversationEntry,
    CreatorReview,
    DecisionLogEntry,
    InitiativeEntry,
    ResearchNote,
    StrategyDirection,
    TaskEntry,
)


_RESULT_TRUNCATE_LEN = 200


@dataclass
class TaskHistoryContext:
    """タスク履歴コンテキスト."""

    completed: list[TaskEntry] = field(default_factory=list)
    failed: list[TaskEntry] = field(default_factory=list)
    running: list[TaskEntry] = field(default_factory=list)


def build_system_prompt(
    constitution: ConstitutionModel | None,
    wip: list[str],
    recent_decisions: list[DecisionLogEntry],
    budget_spent: float,
    budget_limit: float,
    conversation_history: list[ConversationEntry] | None = None,
    vision_text: str | None = None,
    creator_reviews: list[CreatorReview] | None = None,
    research_notes: list[ResearchNote] | None = None,
    task_history: TaskHistoryContext | None = None,
    active_initiatives: list[InitiativeEntry] | None = None,
    strategy_direction: StrategyDirection | None = None,
) -> str:
    """コンテキスト情報からシステムプロンプトを構築する."""
    sections: list[str] = [
        "あなたはAI会社の社長AIです。以下のコンテキストに基づいて行動してください。",
    ]

    # --- 会社憲法 ---
    sections.append(_build_constitution_section(constitution))

    # --- ビジョン・事業方針 ---
    sections.append(_build_vision_section(vision_text))

    # --- 評価（Creatorスコア） ---
    sections.append(_build_creator_score_section(constitution, creator_reviews))

    # --- 現在のWIP ---
    sections.append(_build_wip_section(wip))

    # --- 直近の意思決定 ---
    sections.append(_build_decisions_section(recent_decisions))

    # --- 予算状況 ---
    sections.append(_build_budget_section(budget_spent, budget_limit))

    # --- イニシアチブ ---
    sections.append(_build_initiative_section(active_initiatives))

    # --- 戦略方針 ---
    sections.append(_build_strategy_section(strategy_direction))

    # --- リサーチノート ---
    sections.append(_build_research_section(research_notes))

    # --- タスク履歴 ---
    sections.append(_build_task_history_section(task_history))

    # --- 会話履歴 ---
    sections.append(_build_conversation_section(conversation_history))

    # --- 応答フォーマット ---
    sections.append(_build_format_section())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_constitution_section(constitution: ConstitutionModel | None) -> str:
    lines = ["## 会社憲法"]
    if constitution is None:
        lines.append("憲法未設定")
    else:
        lines.append(f"- 目的: {constitution.purpose}")
        lines.append(
            f"- 予算上限: ${constitution.budget.limit_usd}/{constitution.budget.window_minutes}分"
        )
        lines.append(f"- WIP制限: {constitution.work_principles.wip_limit}件")
    return "\n".join(lines)


def _build_vision_section(vision_text: str | None) -> str:
    lines = ["## ビジョン・事業方針"]
    if not vision_text:
        lines.append("ビジョン未設定")
    else:
        lines.append(vision_text)
    return "\n".join(lines)


def _build_wip_section(wip: list[str]) -> str:
    lines = ["## 現在のWIP"]
    if not wip:
        lines.append("現在進行中のタスクなし")
    else:
        for item in wip:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _build_decisions_section(recent_decisions: list[DecisionLogEntry]) -> str:
    lines = ["## 直近の意思決定"]
    if not recent_decisions:
        lines.append("直近の意思決定なし")
    else:
        for entry in recent_decisions[:5]:
            lines.append(f"- {entry.date}: {entry.decision}（理由: {entry.why}）")
    return "\n".join(lines)


def _build_budget_section(budget_spent: float, budget_limit: float) -> str:
    remaining = budget_limit - budget_spent
    return "\n".join([
        "## 予算状況",
        f"- 消費: ${budget_spent:.2f} / ${budget_limit:.2f}（残り: ${remaining:.2f}）",
    ])


def _build_initiative_section(
    active_initiatives: list[InitiativeEntry] | None,
) -> str:
    lines = ["## イニシアチブ"]
    if not active_initiatives:
        lines.append("アクティブなイニシアチブなし")
    else:
        for ini in active_initiatives:
            lines.append(f"- [{ini.status}] {ini.title}: {ini.description}")
    return "\n".join(lines)


def _build_strategy_section(
    strategy_direction: StrategyDirection | None,
) -> str:
    lines = ["## 戦略方針"]
    if strategy_direction is None or not strategy_direction.summary:
        lines.append("戦略方針未設定")
    else:
        lines.append(strategy_direction.summary)
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
        "<shell>",
        "実行するシェルコマンド",
        "</shell>",
        "",
        "<done>",
        "タスク完了の要約",
        "</done>",
        "",
        "<consult>",
        "Creatorに相談したい内容（質問 + 選択肢 + 推奨 + 影響 + 上限コスト）",
        "</consult>",
        "",
        "タグは複数組み合わせ可能です。タグなしの場合は全体がreplyとして扱われます。",
    ])


def _truncate(text: str, max_len: int = _RESULT_TRUNCATE_LEN) -> str:
    """文字列を指定長に切り詰める."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _build_task_history_section(
    task_history: TaskHistoryContext | None,
) -> str:
    lines = ["## タスク履歴"]
    if task_history is None or (
        not task_history.completed and not task_history.failed and not task_history.running
    ):
        lines.append("タスク履歴なし")
        return "\n".join(lines)

    if task_history.running:
        lines.append("### 実行中タスク")
        for t in task_history.running:
            lines.append(f"- [{t.task_id}] {t.description}")

    if task_history.completed:
        lines.append("### 最近完了したタスク")
        for t in task_history.completed:
            result = _truncate(t.result) if t.result else "結果なし"
            lines.append(f"- [{t.task_id}] {t.description} → {result}")

    if task_history.failed:
        lines.append("### 最近失敗したタスク")
        for t in task_history.failed:
            error = _truncate(t.error) if t.error else "原因不明"
            lines.append(f"- [{t.task_id}] {t.description} — エラー: {error}")

    return "\n".join(lines)


def _build_research_section(
    research_notes: list[ResearchNote] | None,
) -> str:
    lines = ["## リサーチノート"]
    if not research_notes:
        lines.append("リサーチノートなし")
    else:
        for note in research_notes[:10]:
            ts = note.retrieved_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] {note.source_url}\n  {note.summary}")
    return "\n".join(lines)

def _build_conversation_section(
    conversation_history: list[ConversationEntry] | None,
) -> str:
    lines = ["## 会話履歴"]
    if not conversation_history:
        lines.append("会話履歴なし")
    else:
        for entry in conversation_history:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{entry.role}] {ts}: {entry.content}")
    return "\n".join(lines)


def _build_creator_score_section(
    constitution: ConstitutionModel | None,
    creator_reviews: list[CreatorReview] | None,
) -> str:
    lines = ["## 評価（Creatorスコア）"]

    policy = getattr(constitution, "creator_score_policy", None) if constitution else None
    if not policy or not getattr(policy, "enabled", False):
        lines.append("Creatorスコア未設定")
        return "\n".join(lines)

    lines.append(f"- 優先: {policy.priority}")
    lines.append("- 採点軸: 面白さ / コスト効率 / 現実性 / 進化性（各0-25、合計0-100）")

    try:
        axes = getattr(policy, "axes", {}) or {}
        for name in ("面白さ", "コスト効率", "現実性", "進化性"):
            desc = axes.get(name, "")
            if desc:
                short = desc.replace("\n", " ").strip()
                if len(short) > 140:
                    short = short[:140] + "…"
                lines.append(f"- {name}: {short}")
    except Exception:
        pass

    if creator_reviews:
        lines.append("### 直近のCreatorレビュー")
        for r in creator_reviews[-3:]:
            ts = r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            axis = []
            if r.score_interestingness_25 is not None:
                axis.append(f"面白さ{r.score_interestingness_25}/25")
            if r.score_cost_efficiency_25 is not None:
                axis.append(f"コスト効率{r.score_cost_efficiency_25}/25")
            if r.score_realism_25 is not None:
                axis.append(f"現実性{r.score_realism_25}/25")
            if r.score_evolvability_25 is not None:
                axis.append(f"進化性{r.score_evolvability_25}/25")
            axis_text = " ".join(axis) if axis else "軸スコアなし"

            comment = (r.comment or "").replace("\n", " ").strip()
            if len(comment) > 200:
                comment = comment[:200] + "…"
            if comment:
                lines.append(f"- [{ts}] {r.score_total_100}/100 ({axis_text}) — {comment}")
            else:
                lines.append(f"- [{ts}] {r.score_total_100}/100 ({axis_text})")
    else:
        lines.append("直近レビューなし（Creator日報に対する採点を待っています）")

    return "\n".join(lines)
