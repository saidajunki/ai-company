"""Tests for the constitution amendment process (Task 7.1).

Covers all three phases:
- propose_amendment: proposal creation + Decision_Log recording
- approve_amendment: Constitution version++ + Decision_Log approved
- reject_amendment: no Constitution change + Decision_Log rejected
"""

from __future__ import annotations

from datetime import date

import pytest

from amendment import approve_amendment, propose_amendment, reject_amendment
from constitution_store import constitution_load, constitution_save
from models import ConstitutionModel, DecisionLogEntry
from ndjson_store import ndjson_read


@pytest.fixture()
def company_dir(tmp_path):
    """Set up a company directory structure with an initial constitution."""
    base_dir = tmp_path
    company_id = "test-co"
    company_root = base_dir / "companies" / company_id
    company_root.mkdir(parents=True)

    constitution = ConstitutionModel(version=1)
    const_path = company_root / "constitution.yaml"
    constitution_save(const_path, constitution)

    return base_dir, company_id, const_path, constitution


# --- propose_amendment ---


class TestProposeAmendment:
    def test_returns_request_id_and_message(self, company_dir):
        base_dir, company_id, _, constitution = company_dir
        request_id, message, entry = propose_amendment(
            constitution=constitution,
            change_description="予算上限を$20に変更",
            reason="研究規模の拡大",
            impact="コスト上限が倍増する",
            base_dir=base_dir,
            company_id=company_id,
        )
        assert request_id
        assert "承認依頼" in message
        assert "✅" in message
        assert "❌" in message
        assert request_id in message

    def test_records_proposed_in_decision_log(self, company_dir):
        base_dir, company_id, _, constitution = company_dir
        request_id, _, entry = propose_amendment(
            constitution=constitution,
            change_description="WIP制限を5に変更",
            reason="並行作業の増加",
            impact="WIP制限が緩和される",
            base_dir=base_dir,
            company_id=company_id,
            related_field="work_principles.wip_limit",
        )
        log_path = base_dir / "companies" / company_id / "decisions" / "log.ndjson"
        entries = ndjson_read(log_path, DecisionLogEntry)

        assert len(entries) == 1
        assert entries[0].status == "proposed"
        assert entries[0].request_id == request_id
        assert entries[0].decision == "WIP制限を5に変更"
        assert entries[0].related_constitution_field == "work_principles.wip_limit"

    def test_entry_has_required_fields(self, company_dir):
        base_dir, company_id, _, constitution = company_dir
        _, _, entry = propose_amendment(
            constitution=constitution,
            change_description="公開方針を非公開に変更",
            reason="セキュリティ強化",
            impact="デフォルト非公開になる",
            base_dir=base_dir,
            company_id=company_id,
        )
        assert entry.date == date.today()
        assert entry.decision == "公開方針を非公開に変更"
        assert entry.why == "セキュリティ強化"
        assert entry.scope == "デフォルト非公開になる"
        assert entry.revisit
        assert entry.status == "proposed"


# --- approve_amendment ---


class TestApproveAmendment:
    def test_increments_version(self, company_dir):
        base_dir, company_id, const_path, constitution = company_dir
        assert constitution.version == 1

        _, _, proposal = propose_amendment(
            constitution=constitution,
            change_description="テスト変更",
            reason="テスト",
            impact="なし",
            base_dir=base_dir,
            company_id=company_id,
        )
        updated = approve_amendment(
            constitution=constitution,
            proposal=proposal,
            constitution_path=const_path,
            base_dir=base_dir,
            company_id=company_id,
        )
        assert updated.version == 2

    def test_persists_updated_constitution(self, company_dir):
        base_dir, company_id, const_path, constitution = company_dir
        _, _, proposal = propose_amendment(
            constitution=constitution,
            change_description="テスト変更",
            reason="テスト",
            impact="なし",
            base_dir=base_dir,
            company_id=company_id,
        )
        approve_amendment(
            constitution=constitution,
            proposal=proposal,
            constitution_path=const_path,
            base_dir=base_dir,
            company_id=company_id,
        )
        reloaded = constitution_load(const_path)
        assert reloaded.version == 2

    def test_records_approved_in_decision_log(self, company_dir):
        base_dir, company_id, const_path, constitution = company_dir
        request_id, _, proposal = propose_amendment(
            constitution=constitution,
            change_description="テスト変更",
            reason="テスト",
            impact="なし",
            base_dir=base_dir,
            company_id=company_id,
        )
        approve_amendment(
            constitution=constitution,
            proposal=proposal,
            constitution_path=const_path,
            base_dir=base_dir,
            company_id=company_id,
        )
        log_path = base_dir / "companies" / company_id / "decisions" / "log.ndjson"
        entries = ndjson_read(log_path, DecisionLogEntry)

        assert len(entries) == 2  # proposed + approved
        assert entries[0].status == "proposed"
        assert entries[1].status == "approved"
        assert entries[1].request_id == request_id

    def test_rejects_non_proposed_status(self, company_dir):
        base_dir, company_id, const_path, constitution = company_dir
        bad_proposal = DecisionLogEntry(
            date=date.today(),
            decision="x",
            why="x",
            scope="x",
            revisit="x",
            status="approved",
        )
        with pytest.raises(ValueError, match="expected 'proposed'"):
            approve_amendment(
                constitution=constitution,
                proposal=bad_proposal,
                constitution_path=const_path,
                base_dir=base_dir,
                company_id=company_id,
            )


# --- reject_amendment ---


class TestRejectAmendment:
    def test_records_rejected_in_decision_log(self, company_dir):
        base_dir, company_id, _, constitution = company_dir
        request_id, _, proposal = propose_amendment(
            constitution=constitution,
            change_description="却下テスト",
            reason="テスト",
            impact="なし",
            base_dir=base_dir,
            company_id=company_id,
        )
        reject_amendment(
            proposal=proposal,
            base_dir=base_dir,
            company_id=company_id,
        )
        log_path = base_dir / "companies" / company_id / "decisions" / "log.ndjson"
        entries = ndjson_read(log_path, DecisionLogEntry)

        assert len(entries) == 2  # proposed + rejected
        assert entries[0].status == "proposed"
        assert entries[1].status == "rejected"
        assert entries[1].request_id == request_id

    def test_does_not_change_constitution(self, company_dir):
        base_dir, company_id, const_path, constitution = company_dir
        _, _, proposal = propose_amendment(
            constitution=constitution,
            change_description="却下テスト",
            reason="テスト",
            impact="なし",
            base_dir=base_dir,
            company_id=company_id,
        )
        reject_amendment(
            proposal=proposal,
            base_dir=base_dir,
            company_id=company_id,
        )
        reloaded = constitution_load(const_path)
        assert reloaded.version == constitution.version

    def test_rejects_non_proposed_status(self, company_dir):
        base_dir, company_id, _, _ = company_dir
        bad_proposal = DecisionLogEntry(
            date=date.today(),
            decision="x",
            why="x",
            scope="x",
            revisit="x",
            status="rejected",
        )
        with pytest.raises(ValueError, match="expected 'proposed'"):
            reject_amendment(
                proposal=bad_proposal,
                base_dir=base_dir,
                company_id=company_id,
            )


# --- Full flow ---


class TestFullAmendmentFlow:
    def test_propose_then_approve_flow(self, company_dir):
        """Req 2.1→2.2: propose → approve → version incremented + log complete."""
        base_dir, company_id, const_path, constitution = company_dir

        request_id, message, proposal = propose_amendment(
            constitution=constitution,
            change_description="予算上限を$20に変更",
            reason="研究規模の拡大",
            impact="コスト上限が倍増する",
            base_dir=base_dir,
            company_id=company_id,
            related_field="budget.limit_usd",
        )

        updated = approve_amendment(
            constitution=constitution,
            proposal=proposal,
            constitution_path=const_path,
            base_dir=base_dir,
            company_id=company_id,
        )

        # Constitution updated
        assert updated.version == 2
        assert constitution_load(const_path).version == 2

        # Decision log has both entries
        log_path = base_dir / "companies" / company_id / "decisions" / "log.ndjson"
        entries = ndjson_read(log_path, DecisionLogEntry)
        assert len(entries) == 2
        assert entries[0].status == "proposed"
        assert entries[1].status == "approved"
        assert entries[0].request_id == entries[1].request_id == request_id

    def test_propose_then_reject_flow(self, company_dir):
        """Req 2.1→2.3: propose → reject → version unchanged + log complete."""
        base_dir, company_id, const_path, constitution = company_dir

        request_id, _, proposal = propose_amendment(
            constitution=constitution,
            change_description="公開方針を非公開に変更",
            reason="セキュリティ強化",
            impact="デフォルト非公開になる",
            base_dir=base_dir,
            company_id=company_id,
        )

        reject_amendment(
            proposal=proposal,
            base_dir=base_dir,
            company_id=company_id,
        )

        # Constitution NOT changed (Req 2.5)
        assert constitution_load(const_path).version == 1

        # Decision log has both entries
        log_path = base_dir / "companies" / company_id / "decisions" / "log.ndjson"
        entries = ndjson_read(log_path, DecisionLogEntry)
        assert len(entries) == 2
        assert entries[0].status == "proposed"
        assert entries[1].status == "rejected"
        assert entries[0].request_id == entries[1].request_id == request_id
