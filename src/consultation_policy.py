from __future__ import annotations

import re
from dataclasses import dataclass

from models import ConstitutionModel, TaskEntry


@dataclass(frozen=True)
class ConsultationAssessment:
    is_major: bool
    reason: str


_MONEY_PATTERN = re.compile(
    r"(\$|usd\b|円|万円|千円|円\/月|\/月|月額|年額|課金|支払い|決済|請求|billing|payment)",
    re.IGNORECASE,
)

_MAJOR_DIRECTION_PATTERN = re.compile(
    r"(目的|ビジョン|方向性|北極星|KPI|価値観|方針|ブランド|社名|ピボット|撤退|停止|中止|廃止)",
)

_MAJOR_RISK_PATTERN = re.compile(
    r"(法|法律|規約|利用規約|著作権|個人情報|プライバシー|セキュリティ|脆弱性|違反|炎上|リスク|危険)",
)

_MAJOR_CONTRACT_PATTERN = re.compile(
    r"(契約|アカウント作成|登録|クレカ|カード|ドメイン購入|広告出稿|外注|サブスク|有料)",
)


def assess_creator_consultation(
    text: str,
    *,
    constitution: ConstitutionModel | None = None,
) -> ConsultationAssessment:
    """Assess whether a consultation request should be escalated to Creator.

    Policy: default to autonomy. Only escalate for direction/cost/contract/risk.
    """
    t = (text or "").strip()
    if not t:
        return ConsultationAssessment(False, "empty")

    # Constitution-defined Creator intervention scope (best-effort phrase match)
    try:
        if constitution is not None:
            for scope in constitution.creator_intervention.scope:
                s = (scope or "").strip()
                if s and s in t:
                    return ConsultationAssessment(True, "creator_scope")
    except Exception:
        pass

    if _MONEY_PATTERN.search(t):
        return ConsultationAssessment(True, "cost")
    if _MAJOR_CONTRACT_PATTERN.search(t):
        return ConsultationAssessment(True, "contract")
    if _MAJOR_DIRECTION_PATTERN.search(t):
        return ConsultationAssessment(True, "direction")
    if _MAJOR_RISK_PATTERN.search(t):
        return ConsultationAssessment(True, "risk")

    return ConsultationAssessment(False, "minor")


def should_escalate_task_failure(
    task: TaskEntry,
    *,
    constitution: ConstitutionModel | None = None,
) -> bool:
    """Return True if a permanently failed task should be escalated to Creator."""
    if task.source == "creator":
        return True
    if task.priority <= 2:
        return True
    combined = f"{task.description}\n{task.error or ''}".strip()
    return assess_creator_consultation(combined, constitution=constitution).is_major

