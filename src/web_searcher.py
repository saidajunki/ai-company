"""Web Searcher — DuckDuckGo HTML検索をShellExecutor経由で実行する.

ShellExecutor で curl を使い DuckDuckGo の HTML 検索結果をパースし、
構造化された SearchResult リストを返す。
エラー時・タイムアウト時は空リストを返し、例外を送出しない。

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

import html
import logging
import re
from urllib.parse import quote_plus, unquote

from pydantic import BaseModel

from shell_executor import execute_shell

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Web検索結果."""

    title: str
    url: str
    snippet: str


class WebSearcher:
    """DuckDuckGo HTML検索を実行し、構造化された結果を返す."""

    def __init__(self, shell_timeout: int = 30) -> None:
        self.shell_timeout = shell_timeout

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """検索クエリを実行し、SearchResult のリストを返す.

        Args:
            query: 検索クエリ文字列
            max_results: 返す最大結果数（デフォルト5）

        Returns:
            SearchResult のリスト。エラー時は空リスト。
        """
        try:
            return self._do_search(query, max_results)
        except Exception:
            logger.exception("WebSearcher.search() で予期しないエラー")
            return []

    def _do_search(self, query: str, max_results: int) -> list[SearchResult]:
        """実際の検索処理."""
        encoded_query = quote_plus(query)
        command = (
            f'curl -s -L -m {self.shell_timeout} '
            f'-A "Mozilla/5.0" '
            f'"https://html.duckduckgo.com/html/?q={encoded_query}"'
        )

        result = execute_shell(
            command=command,
            timeout=self.shell_timeout + 5,
        )

        if result.timed_out:
            logger.warning("Web検索がタイムアウトしました: query=%s", query)
            return []

        if result.return_code != 0:
            logger.warning(
                "Web検索コマンドがエラーを返しました: query=%s, rc=%d, stderr=%s",
                query,
                result.return_code,
                result.stderr[:200],
            )
            return []

        if not result.stdout.strip():
            return []

        return self._parse_results(result.stdout, max_results)

    def _parse_results(self, html_content: str, max_results: int) -> list[SearchResult]:
        """DuckDuckGo HTML検索結果をパースする."""
        results: list[SearchResult] = []

        # DuckDuckGo HTML の結果ブロックを抽出
        # 各結果は class="result" の div/tr 内にある
        result_blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div|span)>',
            html_content,
            re.DOTALL,
        )

        for raw_url, raw_title, raw_snippet in result_blocks:
            if len(results) >= max_results:
                break

            url = self._extract_url(raw_url)
            title = self._clean_html(raw_title).strip()
            snippet = self._clean_html(raw_snippet).strip()

            if not url or not title:
                continue

            results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results

    @staticmethod
    def _extract_url(raw_url: str) -> str:
        """DuckDuckGo のリダイレクトURLから実際のURLを抽出する."""
        # DuckDuckGo は //duckduckgo.com/l/?uddg=ENCODED_URL&... 形式を使う
        uddg_match = re.search(r'[?&]uddg=([^&]+)', raw_url)
        if uddg_match:
            return unquote(uddg_match.group(1))
        # 直接URLの場合
        url = raw_url.strip()
        if url.startswith("//"):
            url = "https:" + url
        return url

    @staticmethod
    def _clean_html(text: str) -> str:
        """HTMLタグを除去し、エンティティをデコードする."""
        # HTMLエンティティをデコード（タグ除去の前に行う）
        cleaned = html.unescape(text)
        # HTMLタグを除去
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        # 連続する空白を1つに
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
