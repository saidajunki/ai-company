"""Approval protocol message generation (Req 4.1, 4.2, 4.3).

Generates approval request messages with:
- UUID v4 request_id
- Required fields: action, reason, cost, rollback, impact
- ✅=承認 / ❌=却下 instruction
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class ApprovalRequest:
    """承認依頼の入力データ."""

    action_description: str
    reason: str
    cost_estimate: str
    rollback_procedure: str
    impact_description: str


def generate_approval_message(request: ApprovalRequest) -> tuple[str, str]:
    """承認依頼メッセージを生成する.

    Args:
        request: 承認依頼の入力データ

    Returns:
        (request_id, formatted_message) のタプル
    """
    request_id = str(uuid.uuid4())
    message = (
        f"🔔 承認依頼 [request_id: {request_id}]\n"
        f"\n"
        f"📋 何をしたいか: {request.action_description}\n"
        f"💡 なぜ必要か: {request.reason}\n"
        f"💰 上限費用: {request.cost_estimate}\n"
        f"⏪ 取り消し手順: {request.rollback_procedure}\n"
        f"📊 実行による変化: {request.impact_description}\n"
        f"\n"
        f"このメッセージへのスレッド返信で、自由記述で意思（進めて/やめて等）を返してください。\n"
        f"（互換: ✅ = 承認 / ❌ = 却下 でも可）"
    )
    return request_id, message
