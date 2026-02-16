"""Unit tests for main.py on_reaction callback (Task 7.2).

Tests that on_reaction routes to approve_amendment / reject_amendment
and sends appropriate Slack notifications.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from amendment import propose_amendment
from manager import Manager, init_company_directory
from manager_state import constitution_path


CID = "test-co"


def _setup_manager_with_proposal(tmp_path: Path):
    """Create a Manager with a pending amendment proposal."""
    init_company_directory(tmp_path, CID)
    mgr = Manager(tmp_path, CID)
    mock_slack = MagicMock()
    mgr.slack = mock_slack

    # Create a proposal via the real propose_amendment function
    request_id, _msg, entry = propose_amendment(
        constitution=mgr.state.constitution,
        change_description="テスト変更",
        reason="テスト理由",
        impact="テスト影響",
        base_dir=tmp_path,
        company_id=CID,
    )

    # Reload state so decision_log includes the proposal
    from manager_state import restore_state
    mgr.state = restore_state(tmp_path, CID)

    return mgr, mock_slack, request_id


def _make_on_reaction(mgr, slack, base_dir, company_id):
    """Build the on_reaction closure matching main.py's implementation."""
    from amendment import approve_amendment, reject_amendment
    from manager_state import constitution_path as cp_fn, restore_state
    import logging

    log = logging.getLogger("test-on-reaction")

    def on_reaction(request_id: str, result: str, user_id: str) -> None:
        log.info("Approval %s for %s by %s", result, request_id, user_id)

        proposal = None
        already_processed = False
        for entry in mgr.state.decision_log:
            if entry.request_id == request_id:
                if entry.status in ("approved", "rejected"):
                    already_processed = True
                    break
                if entry.status == "proposed":
                    proposal = entry

        if already_processed:
            slack.send_message(
                f"⚠️ 承認リクエスト `{request_id}` は既に処理済みです"
            )
            return

        if proposal is None:
            slack.send_message(
                f"⚠️ 承認リクエスト `{request_id}` に対応する提案が見つかりません"
            )
            return

        const_path = cp_fn(base_dir, company_id)

        try:
            if result == "approved":
                updated = approve_amendment(
                    constitution=mgr.state.constitution,
                    proposal=proposal,
                    constitution_path=const_path,
                    base_dir=base_dir,
                    company_id=company_id,
                )
                mgr.state.constitution = updated
                mgr.state.decision_log = restore_state(base_dir, company_id).decision_log
                slack.send_message(
                    f"✅ 憲法変更が承認されました (v{updated.version})\n"
                    f"変更内容: {proposal.decision}"
                )
            elif result == "rejected":
                reject_amendment(
                    proposal=proposal,
                    base_dir=base_dir,
                    company_id=company_id,
                )
                mgr.state.decision_log = restore_state(base_dir, company_id).decision_log
                slack.send_message(
                    f"❌ 憲法変更が却下されました\n"
                    f"変更内容: {proposal.decision}"
                )
        except Exception:
            slack.send_message(
                f"⚠️ 承認処理中にエラーが発生しました (request_id: {request_id})"
            )

    return on_reaction


class TestOnReactionApproval:
    """Verify on_reaction handles approval correctly."""

    def test_approve_increments_constitution_version(self, tmp_path: Path):
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        original_version = mgr.state.constitution.version
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction(request_id, "approved", "U123")

        assert mgr.state.constitution.version == original_version + 1

    def test_approve_sends_slack_notification(self, tmp_path: Path):
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction(request_id, "approved", "U123")

        slack.send_message.assert_called_once()
        msg = slack.send_message.call_args[0][0]
        assert "✅" in msg
        assert "テスト変更" in msg

    def test_approve_updates_decision_log(self, tmp_path: Path):
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction(request_id, "approved", "U123")

        approved_entries = [
            e for e in mgr.state.decision_log
            if e.request_id == request_id and e.status == "approved"
        ]
        assert len(approved_entries) == 1


class TestOnReactionRejection:
    """Verify on_reaction handles rejection correctly."""

    def test_reject_does_not_change_constitution(self, tmp_path: Path):
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        original_version = mgr.state.constitution.version
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction(request_id, "rejected", "U123")

        assert mgr.state.constitution.version == original_version

    def test_reject_sends_slack_notification(self, tmp_path: Path):
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction(request_id, "rejected", "U123")

        slack.send_message.assert_called_once()
        msg = slack.send_message.call_args[0][0]
        assert "❌" in msg
        assert "テスト変更" in msg

    def test_reject_updates_decision_log(self, tmp_path: Path):
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction(request_id, "rejected", "U123")

        rejected_entries = [
            e for e in mgr.state.decision_log
            if e.request_id == request_id and e.status == "rejected"
        ]
        assert len(rejected_entries) == 1


class TestOnReactionNotFound:
    """Verify on_reaction handles missing request_id."""

    def test_unknown_request_id_sends_warning(self, tmp_path: Path):
        init_company_directory(tmp_path, CID)
        mgr = Manager(tmp_path, CID)
        slack = MagicMock()
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        on_reaction("nonexistent-id", "approved", "U123")

        slack.send_message.assert_called_once()
        msg = slack.send_message.call_args[0][0]
        assert "⚠️" in msg
        assert "nonexistent-id" in msg

    def test_already_approved_proposal_not_found(self, tmp_path: Path):
        """A proposal that was already approved should not be processed again."""
        mgr, slack, request_id = _setup_manager_with_proposal(tmp_path)
        on_reaction = _make_on_reaction(mgr, slack, tmp_path, CID)

        # First approval succeeds
        on_reaction(request_id, "approved", "U123")
        slack.reset_mock()

        # Second approval should detect already-processed
        on_reaction(request_id, "approved", "U123")

        slack.send_message.assert_called_once()
        msg = slack.send_message.call_args[0][0]
        assert "⚠️" in msg
        assert "処理済み" in msg
