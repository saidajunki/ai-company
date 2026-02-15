"""Constitution amendment process (Req 2.1–2.5).

Provides three-phase amendment workflow:
1. propose_amendment: 提案作成 → Decision_Logに proposed 記録 → 承認依頼メッセージ生成
2. approve_amendment: Constitution更新(version++) → Decision_Logに approved 記録
3. reject_amendment: Constitution変更なし → Decision_Logに rejected 記録
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from approval import ApprovalRequest, generate_approval_message
from constitution_store import constitution_save
from models import ConstitutionModel, DecisionLogEntry
from ndjson_store import ndjson_append


def _decision_log_path(base_dir: Path, company_id: str) -> Path:
    """Return the standard decision log file path."""
    return base_dir / "companies" / company_id / "decisions" / "log.ndjson"


def propose_amendment(
    constitution: ConstitutionModel,
    change_description: str,
    reason: str,
    impact: str,
    base_dir: Path,
    company_id: str,
    related_field: Optional[str] = None,
) -> tuple[str, str, DecisionLogEntry]:
    """Create an amendment proposal and record it in Decision_Log.

    Records a 'proposed' entry in Decision_Log and generates an approval
    request message for Slack.

    Args:
        constitution: Current constitution (used for context, not modified).
        change_description: What change is being proposed.
        reason: Why the change is needed.
        impact: Impact of the change.
        base_dir: Root directory for company data.
        company_id: Company identifier.
        related_field: Optional constitution field being changed.

    Returns:
        (request_id, approval_message, decision_log_entry) tuple.
    """
    # Generate approval message with request_id
    approval_req = ApprovalRequest(
        action_description=f"憲法変更: {change_description}",
        reason=reason,
        cost_estimate="なし（設定変更のみ）",
        rollback_procedure="憲法を前バージョンに戻す（version を戻す）",
        impact_description=impact,
    )
    request_id, message = generate_approval_message(approval_req)

    # Record proposal in Decision_Log
    entry = DecisionLogEntry(
        date=date.today(),
        decision=change_description,
        why=reason,
        scope=impact,
        revisit="次回の憲法レビュー時",
        status="proposed",
        request_id=request_id,
        related_constitution_field=related_field,
    )
    log_path = _decision_log_path(base_dir, company_id)
    ndjson_append(log_path, entry)

    return request_id, message, entry


def approve_amendment(
    constitution: ConstitutionModel,
    proposal: DecisionLogEntry,
    constitution_path: Path,
    base_dir: Path,
    company_id: str,
) -> ConstitutionModel:
    """Approve an amendment: update Constitution (version++) and record approval.

    Args:
        constitution: Current constitution to update.
        proposal: The original proposal entry (must have status='proposed').
        constitution_path: Path to the constitution YAML file.
        base_dir: Root directory for company data.
        company_id: Company identifier.

    Returns:
        Updated ConstitutionModel with incremented version.

    Raises:
        ValueError: If proposal status is not 'proposed'.
    """
    if proposal.status != "proposed":
        raise ValueError(
            f"Cannot approve amendment with status '{proposal.status}', expected 'proposed'"
        )

    # Increment version and save
    updated = constitution.model_copy(update={"version": constitution.version + 1})
    constitution_save(constitution_path, updated)

    # Record approval in Decision_Log
    entry = DecisionLogEntry(
        date=date.today(),
        decision=proposal.decision,
        why=proposal.why,
        scope=proposal.scope,
        revisit=proposal.revisit,
        status="approved",
        request_id=proposal.request_id,
        related_constitution_field=proposal.related_constitution_field,
    )
    log_path = _decision_log_path(base_dir, company_id)
    ndjson_append(log_path, entry)

    return updated


def reject_amendment(
    proposal: DecisionLogEntry,
    base_dir: Path,
    company_id: str,
) -> DecisionLogEntry:
    """Reject an amendment: record rejection in Decision_Log (no constitution change).

    Args:
        proposal: The original proposal entry (must have status='proposed').
        base_dir: Root directory for company data.
        company_id: Company identifier.

    Returns:
        The rejection DecisionLogEntry.

    Raises:
        ValueError: If proposal status is not 'proposed'.
    """
    if proposal.status != "proposed":
        raise ValueError(
            f"Cannot reject amendment with status '{proposal.status}', expected 'proposed'"
        )

    entry = DecisionLogEntry(
        date=date.today(),
        decision=proposal.decision,
        why=proposal.why,
        scope=proposal.scope,
        revisit=proposal.revisit,
        status="rejected",
        request_id=proposal.request_id,
        related_constitution_field=proposal.related_constitution_field,
    )
    log_path = _decision_log_path(base_dir, company_id)
    ndjson_append(log_path, entry)

    return entry
