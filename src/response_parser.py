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
from typing import Literal


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """LLM応答から抽出されたアクション."""

    action_type: Literal["shell_command", "reply", "done", "research", "publish", "consult", "delegate"]
    content: str


# ---------------------------------------------------------------------------
# Tag → action_type mapping
# ---------------------------------------------------------------------------

_TAG_TO_ACTION: dict[str, Literal["shell_command", "reply", "done", "research", "publish", "consult", "delegate"]] = {
    "shell": "shell_command",
    "reply": "reply",
    "done": "done",
    "research": "research",
    "publish": "publish",
    "consult": "consult",
    "delegate": "delegate",
}

_ACTION_TO_TAG: dict[str, str] = {v: k for k, v in _TAG_TO_ACTION.items()}

# Regex that matches any of the supported tags and captures inner content.
# re.DOTALL so '.' matches newlines inside the tag body.
_TAG_PATTERN = re.compile(
    r"<(reply|shell|done|research|publish|consult|delegate)>\s*(.*?)\s*</\1>",
    re.DOTALL,
)


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
        actions.append(Action(
            action_type=_TAG_TO_ACTION[tag],
            content=content,
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
        parts.append(f"<{tag}>\n{action.content}\n</{tag}>")

    return "\n\n".join(parts)
