"""Response Parser — LLM応答からアクションを抽出する.

Provides:
- Action dataclass
- parse_response: テキストからアクションリストを抽出
- format_actions: アクションリストをテキストに変換（往復テスト用）

Requirements: 7.1, 7.2, 7.4, 7.5
"""

from __future__ import annotations

import re
from dataclasses import dataclass
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlannedSubtask:
    """<plan>タグ内の個別サブタスク."""

    index: int  # plan内の番号 (1-based)
    description: str  # サブタスクの説明
    depends_on_indices: list[int]  # 依存するplan内番号


@dataclass
class Action:
    """LLM応答から抽出されたアクション."""

    action_type: Literal[
        "shell_command",
        "reply",
        "done",
        "research",
        "publish",
        "consult",
        "delegate",
        "plan",
        "control",
        "memory",
        "commitment",
    ]
    content: str
    model: str | None = None  # delegateアクション用のモデル指定


# ---------------------------------------------------------------------------
# Tag → action_type mapping
# ---------------------------------------------------------------------------

_TAG_TO_ACTION: dict[str, Literal["shell_command", "reply", "done", "research", "publish", "consult", "delegate", "plan", "control", "memory", "commitment"]] = {
    "shell": "shell_command",
    "reply": "reply",
    "done": "done",
    "research": "research",
    "publish": "publish",
    "consult": "consult",
    "delegate": "delegate",
    "plan": "plan",
    "control": "control",
    "memory": "memory",
    "commitment": "commitment",
}

_ACTION_TO_TAG: dict[str, str] = {v: k for k, v in _TAG_TO_ACTION.items()}

# Regex that matches any of the supported tags and captures inner content.
# re.DOTALL so '.' matches newlines inside the tag body.
_TAG_PATTERN = re.compile(
    r"<(reply|shell|done|research|publish|consult|delegate|plan|control|memory|commitment)>\s*(.*?)\s*</\1>",
    re.DOTALL,
)

# Regex for extracting model= parameter from delegate content.
_DELEGATE_MODEL_PATTERN = re.compile(r'\bmodel=(\S+)')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_delegate_model(content: str) -> tuple[str, str | None]:
    """delegateコンテンツからmodel指定を抽出する.

    Returns:
        (model指定を除いたコンテンツ, モデル名 or None)
    """
    match = _DELEGATE_MODEL_PATTERN.search(content)
    if match:
        model = match.group(1)
        cleaned = content[:match.start()].rstrip() + content[match.end():]
        return cleaned.strip(), model if model else None
    return content, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_response(text: str) -> list[Action]:
    """LLM応答テキストからアクションを抽出する.

    ``<reply>``, ``<shell>``, ``<done>`` タグで囲まれたブロックを
    出現順に抽出し、Action リストとして返す。

    タグが1つも見つからない場合は、テキスト全体を単一の ``reply``
    アクションとして返す。
    """
    actions: list[Action] = []

    for match in _TAG_PATTERN.finditer(text):
        tag = match.group(1)
        content = match.group(2)
        model = None
        if tag == "delegate":
            content, model = _extract_delegate_model(content)
        actions.append(Action(
            action_type=_TAG_TO_ACTION[tag],
            content=content,
            model=model,
        ))

    # タグなしの場合は全体を reply として扱う
    if not actions:
        stripped = text.strip()
        if stripped:
            actions.append(Action(action_type="reply", content=stripped))

    return actions


def format_actions(actions: list[Action]) -> str:
    """アクションリストをLLM応答フォーマットに戻す（往復テスト用）.

    各アクションを対応するタグで囲み、改行で連結する。
    ``parse_response(format_actions(actions))`` が元のリストと等価に
    なることを保証する。
    """
    parts: list[str] = []

    for action in actions:
        tag = _ACTION_TO_TAG[action.action_type]
        content = action.content
        if action.action_type == "delegate" and action.model is not None:
            content = f"{content} model={action.model}"
        parts.append(f"<{tag}>\n{content}\n</{tag}>")

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Plan content parser
# ---------------------------------------------------------------------------

# Pattern: "N. description [depends:M,K]" or "N. description"
_PLAN_LINE_PATTERN = re.compile(
    r"^\s*(\d+)\.\s+(.+?)(?:\s*\[depends:([\d,\s]+)\])?\s*$",
)


def parse_plan_content(content: str) -> list[PlannedSubtask]:
    """<plan>タグ内のコンテンツをサブタスクリストにパースする.

    各行を ``N. 説明文 [depends:M,K]`` 形式でパースする。
    ``[depends:...]`` が省略された場合、直前のタスク (N-1) に依存する。
    最初のタスク (index=1) は依存なし。
    空行・不正行はスキップする。
    """
    subtasks: list[PlannedSubtask] = []

    for line in content.splitlines():
        match = _PLAN_LINE_PATTERN.match(line)
        if not match:
            continue

        index = int(match.group(1))
        description = match.group(2).strip()
        depends_raw = match.group(3)

        if depends_raw is not None:
            # Explicit depends: parse comma-separated indices
            depends_on = [int(d.strip()) for d in depends_raw.split(",") if d.strip()]
        elif subtasks:
            # Implicit: depend on previous task
            depends_on = [subtasks[-1].index]
        else:
            # First task: no dependencies
            depends_on = []

        subtasks.append(PlannedSubtask(
            index=index,
            description=description,
            depends_on_indices=depends_on,
        ))

    return subtasks
