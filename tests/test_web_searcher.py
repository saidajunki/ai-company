"""Tests for web_searcher — WebSearcher のユニットテスト.

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

from unittest.mock import patch

from shell_executor import ShellResult
from web_searcher import SearchResult, WebSearcher


# ---------------------------------------------------------------------------
# Helper: DuckDuckGo HTML レスポンスのモック生成
# ---------------------------------------------------------------------------

def _make_ddg_html(*entries: tuple[str, str, str]) -> str:
    """DuckDuckGo HTML 検索結果のモックHTMLを生成する.

    Args:
        entries: (url, title, snippet) のタプル群
    """
    rows = []
    for url, title, snippet in entries:
        rows.append(
            f'<tr>'
            f'<td>'
            f'<a class="result__a" href="//duckduckgo.com/l/?uddg={url}&amp;rut=abc">{title}</a>'
            f'</td>'
            f'<td>'
            f'<span class="result__snippet">{snippet}</span>'
            f'</td>'
            f'</tr>'
        )
    return "<html><body><table>" + "\n".join(rows) + "</table></body></html>"


def _shell_ok(stdout: str) -> ShellResult:
    """正常終了の ShellResult を返す."""
    return ShellResult(
        command="curl ...",
        stdout=stdout,
        stderr="",
        return_code=0,
        timed_out=False,
        duration_seconds=0.5,
    )


def _shell_error(return_code: int = 1, stderr: str = "error") -> ShellResult:
    """エラー終了の ShellResult を返す."""
    return ShellResult(
        command="curl ...",
        stdout="",
        stderr=stderr,
        return_code=return_code,
        timed_out=False,
        duration_seconds=0.1,
    )


def _shell_timeout() -> ShellResult:
    """タイムアウトの ShellResult を返す."""
    return ShellResult(
        command="curl ...",
        stdout="",
        stderr="",
        return_code=-1,
        timed_out=True,
        duration_seconds=30.0,
    )


# ---------------------------------------------------------------------------
# 正常系テスト
# ---------------------------------------------------------------------------

class TestWebSearcherSuccess:
    """正常な検索結果のテスト."""

    @patch("web_searcher.execute_shell")
    def test_returns_search_results(self, mock_shell):
        html = _make_ddg_html(
            ("https%3A%2F%2Fexample.com%2Fpage1", "Example Page", "A snippet about example"),
            ("https%3A%2F%2Ftest.org%2Farticle", "Test Article", "Test snippet here"),
        )
        mock_shell.return_value = _shell_ok(html)

        ws = WebSearcher()
        results = ws.search("test query")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    @patch("web_searcher.execute_shell")
    def test_result_fields(self, mock_shell):
        html = _make_ddg_html(
            ("https%3A%2F%2Fexample.com", "My Title", "My snippet text"),
        )
        mock_shell.return_value = _shell_ok(html)

        ws = WebSearcher()
        results = ws.search("query")

        assert len(results) == 1
        assert results[0].title == "My Title"
        assert results[0].url == "https://example.com"
        assert results[0].snippet == "My snippet text"

    @patch("web_searcher.execute_shell")
    def test_max_results_limits_output(self, mock_shell):
        entries = [
            (f"https%3A%2F%2Fsite{i}.com", f"Title {i}", f"Snippet {i}")
            for i in range(10)
        ]
        html = _make_ddg_html(*entries)
        mock_shell.return_value = _shell_ok(html)

        ws = WebSearcher()
        results = ws.search("query", max_results=3)

        assert len(results) == 3

    @patch("web_searcher.execute_shell")
    def test_html_entities_decoded(self, mock_shell):
        html = _make_ddg_html(
            ("https%3A%2F%2Fexample.com", "Title &amp; More", "Snippet &lt;b&gt;bold&lt;/b&gt;"),
        )
        mock_shell.return_value = _shell_ok(html)

        ws = WebSearcher()
        results = ws.search("query")

        assert results[0].title == "Title & More"
        assert results[0].snippet == "Snippet bold"

    @patch("web_searcher.execute_shell")
    def test_html_tags_stripped(self, mock_shell):
        html = _make_ddg_html(
            ("https%3A%2F%2Fexample.com", "<b>Bold</b> Title", "Some <em>emphasis</em> here"),
        )
        mock_shell.return_value = _shell_ok(html)

        ws = WebSearcher()
        results = ws.search("query")

        assert results[0].title == "Bold Title"
        assert results[0].snippet == "Some emphasis here"


# ---------------------------------------------------------------------------
# エラー系テスト
# ---------------------------------------------------------------------------

class TestWebSearcherErrors:
    """エラー時の動作テスト."""

    @patch("web_searcher.execute_shell")
    def test_timeout_returns_empty_list(self, mock_shell):
        mock_shell.return_value = _shell_timeout()

        ws = WebSearcher()
        results = ws.search("query")

        assert results == []

    @patch("web_searcher.execute_shell")
    def test_nonzero_return_code_returns_empty_list(self, mock_shell):
        mock_shell.return_value = _shell_error(return_code=7)

        ws = WebSearcher()
        results = ws.search("query")

        assert results == []

    @patch("web_searcher.execute_shell")
    def test_empty_stdout_returns_empty_list(self, mock_shell):
        mock_shell.return_value = _shell_ok("")

        ws = WebSearcher()
        results = ws.search("query")

        assert results == []

    @patch("web_searcher.execute_shell")
    def test_unparseable_html_returns_empty_list(self, mock_shell):
        mock_shell.return_value = _shell_ok("<html><body>no results here</body></html>")

        ws = WebSearcher()
        results = ws.search("query")

        assert results == []

    @patch("web_searcher.execute_shell")
    def test_shell_exception_returns_empty_list(self, mock_shell):
        mock_shell.side_effect = RuntimeError("unexpected")

        ws = WebSearcher()
        results = ws.search("query")

        assert results == []

    @patch("web_searcher.execute_shell")
    def test_never_raises_exceptions(self, mock_shell):
        mock_shell.side_effect = OSError("network down")

        ws = WebSearcher()
        # Should not raise
        results = ws.search("query")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 初期化テスト
# ---------------------------------------------------------------------------

class TestWebSearcherInit:
    """初期化パラメータのテスト."""

    def test_default_timeout(self):
        ws = WebSearcher()
        assert ws.shell_timeout == 30

    def test_custom_timeout(self):
        ws = WebSearcher(shell_timeout=10)
        assert ws.shell_timeout == 10

    @patch("web_searcher.execute_shell")
    def test_timeout_passed_to_curl(self, mock_shell):
        mock_shell.return_value = _shell_ok("")
        ws = WebSearcher(shell_timeout=15)
        ws.search("test")

        call_args = mock_shell.call_args
        command = call_args.kwargs.get("command") or call_args[0][0]
        assert "-m 15" in command


# ---------------------------------------------------------------------------
# URL抽出テスト
# ---------------------------------------------------------------------------

class TestUrlExtraction:
    """DuckDuckGo リダイレクトURLの抽出テスト."""

    @patch("web_searcher.execute_shell")
    def test_extracts_url_from_uddg_param(self, mock_shell):
        html = _make_ddg_html(
            ("https%3A%2F%2Freal-site.com%2Fpath", "Title", "Snippet"),
        )
        mock_shell.return_value = _shell_ok(html)

        ws = WebSearcher()
        results = ws.search("query")

        assert results[0].url == "https://real-site.com/path"
