"""Context Builder — LLM呼び出し用のシステムプロンプトを構築する.

Provides:
- build_system_prompt: コンテキスト情報からシステムプロンプトを構築

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.3
"""

from __future__ import annotations

from models import ConstitutionModel, DecisionLogEntry


def build_system_prompt(
    constitution: ConstitutionModel | None,
    wip: list[str],
    recent_decisions: list[DecisionLogEntry],
    budget_spent: float,
    budget_limit: float,
) -> str:
    """コンテキスト情報からシステムプロンプトを構築する."""
    sections: list[str] = [
        "あなたはAI会社の社長AIです。以下のコンテキストに基づいて行動してください。",
    ]

    # --- 会社憲法 ---
    sections.append(_build_constitution_section(constitution))

    # --- 現在のWIP ---
    sections.append(_build_wip_section(wip))

    # --- 直近の意思決定 ---
    sections.append(_build_decisions_section(recent_decisions))

    # --- 予算状況 ---
    sections.append(_build_budget_section(budget_spent, budget_limit))

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
        "タグは複数組み合わせ可能です。タグなしの場合は全体がreplyとして扱われます。",
    ])
