"""Unit tests for the ten-minute report formatter (Req 3.1)."""

from datetime import datetime

from report_formatter import CostSummary, ReportData, format_report


REQUIRED_SECTIONS = [
    "### WIP",
    "### Δ(10m)",
    "### Next(10m)",
    "### Blockers",
    "### Cost(60m)",
    "### Approvals",
    "### 自律タスク",
    "### エージェント",
    "### サービス",
]


def _make_report_data(**overrides) -> ReportData:
    defaults = dict(
        timestamp=datetime(2025, 1, 15, 10, 0, 0),
        company_id="test-corp",
        wip=["タスクA", "タスクB"],
        delta_description="READMEを更新",
        artifact_links="https://example.com/pr/1",
        next_plan="テスト追加",
        blockers=["API鍵が未取得"],
        cost=CostSummary(spent_usd=3.50, remaining_usd=6.50, allocation_plan="LLM: $4"),
        approvals=["req-001: 外部API利用"],
    )
    defaults.update(overrides)
    return ReportData(**defaults)


class TestFormatReportSections:
    """All required sections must be present in the output."""

    def test_all_required_sections_present(self):
        report = format_report(_make_report_data())
        for section in REQUIRED_SECTIONS:
            assert section in report, f"Missing section: {section}"

    def test_header_contains_timestamp_and_company(self):
        data = _make_report_data()
        report = format_report(data)
        assert data.timestamp.isoformat() in report
        assert data.company_id in report

    def test_wip_items_rendered(self):
        data = _make_report_data(wip=["A", "B", "C"])
        report = format_report(data)
        assert "- A" in report
        assert "- B" in report
        assert "- C" in report

    def test_wip_truncated_to_three(self):
        data = _make_report_data(wip=["A", "B", "C", "D"])
        report = format_report(data)
        assert "- D" not in report

    def test_empty_wip_shows_none(self):
        data = _make_report_data(wip=[])
        report = format_report(data)
        # Between WIP header and Δ header, should show "なし"
        wip_start = report.index("### WIP")
        delta_start = report.index("### Δ(10m)")
        wip_section = report[wip_start:delta_start]
        assert "なし" in wip_section

    def test_blockers_rendered(self):
        data = _make_report_data(blockers=["ブロッカー1", "ブロッカー2"])
        report = format_report(data)
        assert "- ブロッカー1" in report
        assert "- ブロッカー2" in report

    def test_empty_blockers_shows_none(self):
        data = _make_report_data(blockers=[])
        report = format_report(data)
        blocker_start = report.index("### Blockers")
        cost_start = report.index("### Cost(60m)")
        blocker_section = report[blocker_start:cost_start]
        assert "なし" in blocker_section

    def test_empty_approvals_shows_none(self):
        data = _make_report_data(approvals=[])
        report = format_report(data)
        approval_start = report.index("### Approvals")
        approval_section = report[approval_start:]
        assert "なし" in approval_section

    def test_cost_values_formatted(self):
        data = _make_report_data(
            cost=CostSummary(spent_usd=5.25, remaining_usd=4.75, allocation_plan="均等配分")
        )
        report = format_report(data)
        assert "$5.25" in report
        assert "$4.75" in report
        assert "均等配分" in report

    def test_delta_and_next_rendered(self):
        data = _make_report_data(
            delta_description="コード修正",
            artifact_links="link1",
            next_plan="デプロイ",
        )
        report = format_report(data)
        assert "コード修正" in report
        assert "link1" in report
        assert "デプロイ" in report


    def test_running_tasks_rendered(self):
        data = _make_report_data(running_tasks=["調査タスク", "ビルドタスク"])
        report = format_report(data)
        assert "- 調査タスク" in report
        assert "- ビルドタスク" in report

    def test_empty_running_tasks_shows_none(self):
        data = _make_report_data(running_tasks=[])
        report = format_report(data)
        task_start = report.index("### 自律タスク")
        agent_start = report.index("### エージェント")
        task_section = report[task_start:agent_start]
        assert "なし" in task_section

    def test_active_agents_rendered(self):
        data = _make_report_data(active_agents=["CEO_AI", "研究員AI"])
        report = format_report(data)
        assert "- CEO_AI" in report
        assert "- 研究員AI" in report

    def test_empty_active_agents_shows_none(self):
        data = _make_report_data(active_agents=[])
        report = format_report(data)
        agent_start = report.index("### エージェント")
        service_start = report.index("### サービス")
        agent_section = report[agent_start:service_start]
        assert "なし" in agent_section

    def test_recent_services_rendered(self):
        data = _make_report_data(recent_services=["OSSツール", "データ分析API"])
        report = format_report(data)
        assert "- OSSツール" in report
        assert "- データ分析API" in report

    def test_empty_recent_services_shows_none(self):
        data = _make_report_data(recent_services=[])
        report = format_report(data)
        service_start = report.index("### サービス")
        service_section = report[service_start:]
        assert "なし" in service_section
