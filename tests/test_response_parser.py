"""Tests for response_parser — research / publish タグの解析.

Requirements: 4.1, 7.1
"""

from __future__ import annotations

from response_parser import Action, format_actions, parse_response


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
            Action(action_type="publish", content="artifact"),
            Action(action_type="consult", content="相談したいです"),
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
