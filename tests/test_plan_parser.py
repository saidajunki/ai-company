"""Tests for parse_plan_content — <plan>タグ内コンテンツのパース.

Requirements: 2.3, 2.4
"""

from __future__ import annotations

from response_parser import PlannedSubtask, parse_plan_content


class TestParsePlanContentBasic:
    """基本的なパース動作."""

    def test_single_task_no_depends(self):
        result = parse_plan_content("1. サブタスク1")
        assert len(result) == 1
        assert result[0] == PlannedSubtask(
            index=1, description="サブタスク1", depends_on_indices=[],
        )

    def test_multiple_tasks_implicit_depends(self):
        content = "1. タスクA\n2. タスクB\n3. タスクC"
        result = parse_plan_content(content)
        assert len(result) == 3
        assert result[0].depends_on_indices == []
        assert result[1].depends_on_indices == [1]
        assert result[2].depends_on_indices == [2]

    def test_explicit_depends(self):
        content = "1. タスクA\n2. タスクB [depends:1]\n3. タスクC [depends:1,2]"
        result = parse_plan_content(content)
        assert result[1].depends_on_indices == [1]
        assert result[2].depends_on_indices == [1, 2]

    def test_indices_match_line_numbers(self):
        content = "1. First\n2. Second\n3. Third"
        result = parse_plan_content(content)
        assert [s.index for s in result] == [1, 2, 3]

    def test_descriptions_are_preserved(self):
        content = "1. Deploy the application\n2. Run integration tests [depends:1]"
        result = parse_plan_content(content)
        assert result[0].description == "Deploy the application"
        assert result[1].description == "Run integration tests"


class TestParsePlanContentEdgeCases:
    """エッジケースのテスト."""

    def test_empty_content(self):
        result = parse_plan_content("")
        assert result == []

    def test_whitespace_only(self):
        result = parse_plan_content("   \n  \n  ")
        assert result == []

    def test_skip_empty_lines(self):
        content = "1. タスクA\n\n2. タスクB\n\n3. タスクC"
        result = parse_plan_content(content)
        assert len(result) == 3

    def test_skip_invalid_lines(self):
        content = "1. タスクA\nこれは不正な行\n2. タスクB"
        result = parse_plan_content(content)
        assert len(result) == 2
        assert result[0].index == 1
        assert result[1].index == 2

    def test_depends_with_spaces(self):
        content = "1. A\n2. B [depends: 1]\n3. C [depends: 1, 2]"
        result = parse_plan_content(content)
        assert result[1].depends_on_indices == [1]
        assert result[2].depends_on_indices == [1, 2]

    def test_leading_whitespace_in_lines(self):
        content = "  1. タスクA\n  2. タスクB"
        result = parse_plan_content(content)
        assert len(result) == 2

    def test_only_invalid_lines(self):
        content = "not a task\nalso not a task"
        result = parse_plan_content(content)
        assert result == []

    def test_mixed_valid_and_invalid(self):
        content = "header text\n1. タスクA\nsome noise\n2. タスクB [depends:1]"
        result = parse_plan_content(content)
        assert len(result) == 2
        assert result[0].description == "タスクA"
        assert result[1].depends_on_indices == [1]
