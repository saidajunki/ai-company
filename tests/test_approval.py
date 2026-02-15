"""Unit tests for the approval protocol message generation (Req 4.1, 4.2, 4.3)."""

import re
import uuid

from approval import ApprovalRequest, generate_approval_message

UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _make_request(**overrides) -> ApprovalRequest:
    defaults = dict(
        action_description="OpenRouter APIキーを登録する",
        reason="LLM呼び出しに必要",
        cost_estimate="$0（登録のみ）",
        rollback_procedure="APIキーを無効化する",
        impact_description="外部API呼び出しが可能になる",
    )
    defaults.update(overrides)
    return ApprovalRequest(**defaults)


class TestApprovalMessageRequestId:
    """request_id must be a valid UUID v4 (Req 4.2)."""

    def test_request_id_is_valid_uuid_v4(self):
        request_id, _ = generate_approval_message(_make_request())
        assert UUID_V4_PATTERN.match(request_id), f"Invalid UUID v4: {request_id}"

    def test_request_id_is_parseable_as_uuid(self):
        request_id, _ = generate_approval_message(_make_request())
        parsed = uuid.UUID(request_id, version=4)
        assert str(parsed) == request_id

    def test_request_ids_are_unique(self):
        ids = {generate_approval_message(_make_request())[0] for _ in range(50)}
        assert len(ids) == 50

    def test_request_id_appears_in_message(self):
        request_id, message = generate_approval_message(_make_request())
        assert request_id in message


class TestApprovalMessageRequiredFields:
    """All required fields must appear in the message (Req 4.1)."""

    def test_action_description_present(self):
        req = _make_request(action_description="新しいリポジトリを作成する")
        _, message = generate_approval_message(req)
        assert "何をしたいか" in message
        assert "新しいリポジトリを作成する" in message

    def test_reason_present(self):
        req = _make_request(reason="コード管理のため")
        _, message = generate_approval_message(req)
        assert "なぜ必要か" in message
        assert "コード管理のため" in message

    def test_cost_estimate_present(self):
        req = _make_request(cost_estimate="$5/月")
        _, message = generate_approval_message(req)
        assert "上限費用" in message
        assert "$5/月" in message

    def test_cost_unknown_present(self):
        req = _make_request(cost_estimate="不明")
        _, message = generate_approval_message(req)
        assert "不明" in message

    def test_rollback_procedure_present(self):
        req = _make_request(rollback_procedure="リポジトリを削除する")
        _, message = generate_approval_message(req)
        assert "取り消し手順" in message
        assert "リポジトリを削除する" in message

    def test_impact_description_present(self):
        req = _make_request(impact_description="公開リポジトリが1つ増える")
        _, message = generate_approval_message(req)
        assert "実行による変化" in message
        assert "公開リポジトリが1つ増える" in message


class TestApprovalMessageInstruction:
    """Message must include ✅/❌ instruction (Req 4.3)."""

    def test_approval_emoji_present(self):
        _, message = generate_approval_message(_make_request())
        assert "✅" in message

    def test_rejection_emoji_present(self):
        _, message = generate_approval_message(_make_request())
        assert "❌" in message

    def test_approval_rejection_instruction(self):
        _, message = generate_approval_message(_make_request())
        assert "承認" in message
        assert "却下" in message


class TestApprovalMessageFormat:
    """Message follows the design template structure."""

    def test_header_contains_request_id_label(self):
        request_id, message = generate_approval_message(_make_request())
        assert f"[request_id: {request_id}]" in message

    def test_message_contains_all_emoji_labels(self):
        _, message = generate_approval_message(_make_request())
        assert "🔔" in message
        assert "📋" in message
        assert "💡" in message
        assert "💰" in message
        assert "⏪" in message
        assert "📊" in message
