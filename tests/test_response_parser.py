"""Tests for response_parser — research / publish タグの解析.

Requirements: 4.1, 7.1
"""

from __future__ import annotations

from response_parser import Action, format_actions, parse_response, _extract_delegate_model


# ---------------------------------------------------------------------------
# _extract_delegate_model ユニットテスト
# ---------------------------------------------------------------------------

class TestExtractDelegateModel:
    """_extract_delegate_model() のテスト."""

    def test_extract_model_from_content(self):
        content, model = _extract_delegate_model("worker:タスク説明 model=google/gemini-2.5-flash")
        assert model == "google/gemini-2.5-flash"
        assert content == "worker:タスク説明"

    def test_no_model_returns_none(self):
        content, model = _extract_delegate_model("worker:タスク説明")
        assert model is None
        assert content == "worker:タスク説明"

    def test_empty_model_value_returns_none(self):
        """空文字列のモデル名はNoneと同等に扱う."""
        # model= の後にスペースが来る場合、\S+ はマッチしないのでNone
        content, model = _extract_delegate_model("worker:タスク model= ")
        assert model is None

    def test_model_in_middle_of_content(self):
        content, model = _extract_delegate_model("worker:タスク model=openai/gpt-4o 追加情報")
        assert model == "openai/gpt-4o"
        assert "worker:タスク" in content
        assert "追加情報" in content
        assert "model=" not in content

    def test_content_preserved_without_model(self):
        original = "engineer:コードレビュー実施"
        content, model = _extract_delegate_model(original)
        assert content == original
        assert model is None


# ---------------------------------------------------------------------------
# delegate タグの model= パース
# ---------------------------------------------------------------------------

class TestDelegateModelParsing:
    """parse_response() での delegate model= パーステスト."""

    def test_delegate_with_model(self):
        text = "<delegate>worker:タスク説明 model=google/gemini-2.5-flash</delegate>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "delegate"
        assert actions[0].model == "google/gemini-2.5-flash"
        assert actions[0].content == "worker:タスク説明"

    def test_delegate_without_model(self):
        text = "<delegate>worker:タスク説明</delegate>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "delegate"
        assert actions[0].model is None
        assert actions[0].content == "worker:タスク説明"

    def test_non_delegate_tags_have_no_model(self):
        """delegate以外のタグではmodel抽出しない."""
        text = "<reply>model=some-model テスト</reply>"
        actions = parse_response(text)
        assert actions[0].model is None
        assert "model=some-model" in actions[0].content

    def test_delegate_with_complex_model_name(self):
        text = "<delegate>coder:実装 model=anthropic/claude-sonnet-4-20250514</delegate>"
        actions = parse_response(text)
        assert actions[0].model == "anthropic/claude-sonnet-4-20250514"
        assert actions[0].content == "coder:実装"

    def test_delegate_model_with_slashes(self):
        text = "<delegate>analyst:分析 model=deepseek/deepseek-chat</delegate>"
        actions = parse_response(text)
        assert actions[0].model == "deepseek/deepseek-chat"

    def test_delegate_with_model_roundtrip(self):
        """delegate + model の往復一貫性テスト (Requirements 2.5)."""
        original = [Action(action_type="delegate", content="worker:タスク説明", model="google/gemini-2.5-flash")]
        formatted = format_actions(original)
        assert "model=google/gemini-2.5-flash" in formatted
        reparsed = parse_response(formatted)
        assert len(reparsed) == 1
        assert reparsed[0].action_type == "delegate"
        assert reparsed[0].content == "worker:タスク説明"
        assert reparsed[0].model == "google/gemini-2.5-flash"
        assert reparsed == original

    def test_delegate_without_model_roundtrip(self):
        """delegate model=None の往復一貫性テスト (Requirements 2.5)."""
        original = [Action(action_type="delegate", content="worker:タスク説明")]
        formatted = format_actions(original)
        assert "model=" not in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# plan タグ
# ---------------------------------------------------------------------------

class TestPlanTag:
    """<plan> タグの解析テスト."""

    def test_parse_plan_tag(self):
        text = "<plan>1. サブタスク1\n2. サブタスク2</plan>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "plan"
        assert "サブタスク1" in actions[0].content
        assert "サブタスク2" in actions[0].content

    def test_parse_plan_with_whitespace(self):
        text = "<plan>  trimmed content  </plan>"
        actions = parse_response(text)
        assert actions[0].content == "trimmed content"

    def test_parse_plan_empty_content(self):
        text = "<plan>   </plan>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "plan"
        assert actions[0].content == ""

    def test_parse_plan_multiline(self):
        text = "<plan>\n1. タスクA\n2. タスクB [depends:1]\n</plan>"
        actions = parse_response(text)
        assert actions[0].action_type == "plan"
        assert "タスクA" in actions[0].content
        assert "タスクB" in actions[0].content

    def test_format_plan_roundtrip(self):
        original = [Action(action_type="plan", content="1. タスク1\n2. タスク2")]
        formatted = format_actions(original)
        assert "<plan>" in formatted
        assert "</plan>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original

    def test_plan_with_other_tags(self):
        text = (
            "<reply>分解します</reply>\n"
            "<plan>1. ステップ1\n2. ステップ2</plan>"
        )
        actions = parse_response(text)
        assert len(actions) == 2
        assert actions[0].action_type == "reply"
        assert actions[1].action_type == "plan"


# ---------------------------------------------------------------------------
# control タグ
# ---------------------------------------------------------------------------

class TestControlTag:
    """<control> タグの解析テスト."""

    def test_parse_control_tag(self):
        text = "<control>pause 1234abcd: 一旦保留</control>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "control"
        assert "pause 1234abcd" in actions[0].content

    def test_format_control_roundtrip(self):
        original = [Action(action_type="control", content="cancel 1234abcd: 中止")]
        formatted = format_actions(original)
        assert "<control>" in formatted
        assert "</control>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# research タグ
# ---------------------------------------------------------------------------

class TestResearchTag:
    """<research> タグの解析テスト."""

    def test_parse_research_tag(self):
        text = "<research>AI market trends 2025</research>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "research"
        assert actions[0].content == "AI market trends 2025"

    def test_parse_research_with_whitespace(self):
        text = "<research>  spaced query  </research>"
        actions = parse_response(text)
        assert actions[0].content == "spaced query"

    def test_parse_research_multiline(self):
        text = "<research>\nline1\nline2\n</research>"
        actions = parse_response(text)
        assert actions[0].action_type == "research"
        assert "line1" in actions[0].content
        assert "line2" in actions[0].content

    def test_format_research_roundtrip(self):
        original = [Action(action_type="research", content="test query")]
        formatted = format_actions(original)
        assert "<research>" in formatted
        assert "</research>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# mcp タグ
# ---------------------------------------------------------------------------

class TestMcpTag:
    """<mcp> タグの解析テスト."""

    def test_parse_mcp_tag(self):
        text = "<mcp>server: vps-monitor\nmethod: tools/list</mcp>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "mcp"
        assert "vps-monitor" in actions[0].content

    def test_parse_mcp_with_whitespace(self):
        text = "<mcp>  list_containers  </mcp>"
        actions = parse_response(text)
        assert actions[0].content == "list_containers"

    def test_format_mcp_roundtrip(self):
        original = [Action(action_type="mcp", content="server: vps-monitor\nname: list_containers\narguments: {}")]
        formatted = format_actions(original)
        assert "<mcp>" in formatted
        assert "</mcp>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# publish タグ
# ---------------------------------------------------------------------------

class TestPublishTag:
    """<publish> タグの解析テスト."""

    def test_parse_publish_tag(self):
        text = "<publish>repo-name: my-project</publish>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "publish"
        assert actions[0].content == "repo-name: my-project"

    def test_parse_publish_with_whitespace(self):
        text = "<publish>  trimmed  </publish>"
        actions = parse_response(text)
        assert actions[0].content == "trimmed"

    def test_format_publish_roundtrip(self):
        original = [Action(action_type="publish", content="deploy artifact")]
        formatted = format_actions(original)
        assert "<publish>" in formatted
        assert "</publish>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# memory タグ
# ---------------------------------------------------------------------------

class TestMemoryTag:
    """<memory> タグの解析テスト."""

    def test_parse_memory_tag(self):
        text = "<memory>curated: 価値観は面白さ最優先</memory>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "memory"
        assert "価値観" in actions[0].content

    def test_format_memory_roundtrip(self):
        original = [Action(action_type="memory", content="pin: テスト")]
        formatted = format_actions(original)
        assert "<memory>" in formatted
        assert "</memory>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# commitment タグ
# ---------------------------------------------------------------------------

class TestCommitmentTag:
    """<commitment> タグの解析テスト."""

    def test_parse_commitment_tag(self):
        text = "<commitment>add: TODO\nやること</commitment>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "commitment"
        assert "TODO" in actions[0].content

    def test_format_commitment_roundtrip(self):
        original = [Action(action_type="commitment", content="close ab12cd34: done")]
        formatted = format_actions(original)
        assert "<commitment>" in formatted
        assert "</commitment>" in formatted
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# 混合タグ
# ---------------------------------------------------------------------------

class TestMixedTags:
    """新旧タグの混在テスト."""

    def test_research_with_existing_tags(self):
        text = (
            "<reply>考え中</reply>\n"
            "<research>Python best practices</research>\n"
            "<done>完了</done>"
        )
        actions = parse_response(text)
        assert len(actions) == 3
        assert actions[0].action_type == "reply"
        assert actions[1].action_type == "research"
        assert actions[2].action_type == "done"

    def test_publish_with_shell(self):
        text = (
            "<shell>echo hello</shell>\n"
            "<publish>push to repo</publish>"
        )
        actions = parse_response(text)
        assert len(actions) == 2
        assert actions[0].action_type == "shell_command"
        assert actions[1].action_type == "publish"

    def test_all_action_types_roundtrip(self):
        original = [
            Action(action_type="shell_command", content="ls"),
            Action(action_type="reply", content="hello"),
            Action(action_type="research", content="query"),
            Action(action_type="mcp", content="server: vps-monitor\nmethod: tools/list"),
            Action(action_type="publish", content="artifact"),
            Action(action_type="consult", content="相談したいです"),
            Action(action_type="delegate", content="worker:タスク", model="google/gemini-2.5-flash"),
            Action(action_type="plan", content="1. タスク1\n2. タスク2"),
            Action(action_type="control", content="pause 1234abcd: 理由"),
            Action(action_type="memory", content="pin: テスト"),
            Action(action_type="commitment", content="add: TODO\nsomething"),
            Action(action_type="done", content="finished"),
        ]
        formatted = format_actions(original)
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# consult タグ
# ---------------------------------------------------------------------------

class TestConsultTag:
    """<consult> タグの解析テスト."""

    def test_parse_consult_tag(self):
        text = "<consult>相談: どちらが良い？</consult>"
        actions = parse_response(text)
        assert len(actions) == 1
        assert actions[0].action_type == "consult"
        assert "どちらが良い" in actions[0].content

    def test_format_consult_roundtrip(self):
        original = [Action(action_type="consult", content="相談内容")]
        formatted = format_actions(original)
        reparsed = parse_response(formatted)
        assert reparsed == original


# ---------------------------------------------------------------------------
# 既存動作の回帰テスト
# ---------------------------------------------------------------------------

class TestExistingBehavior:
    """既存タグの動作が壊れていないことを確認."""

    def test_reply_still_works(self):
        actions = parse_response("<reply>hello</reply>")
        assert actions[0].action_type == "reply"

    def test_shell_still_works(self):
        actions = parse_response("<shell>ls -la</shell>")
        assert actions[0].action_type == "shell_command"

    def test_done_still_works(self):
        actions = parse_response("<done>完了</done>")
        assert actions[0].action_type == "done"

    def test_no_tags_fallback_to_reply(self):
        actions = parse_response("plain text")
        assert len(actions) == 1
        assert actions[0].action_type == "reply"
        assert actions[0].content == "plain text"

    def test_empty_string_returns_empty(self):
        actions = parse_response("")
        assert actions == []
