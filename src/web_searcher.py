"""Web Searcher — Web検索を実行し、構造化された結果を返す.

デフォルトは SearxNG の JSON API を利用する（ローカル/自前で運用可能）。
環境により外部検索がブロックされることが多いため、DuckDuckGoのHTMLスクレイピングは
フォールバックとしてのみ残す。

Env:
- AI_COMPANY_WEB_SEARCH_BACKEND: "searxng" | "ddg_html" (default: searxng)
- AI_COMPANY_SEARXNG_URL: e.g. http://127.0.0.1:8088 (default)

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.error
import urllib.request
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
    """Web検索を実行し、構造化された結果を返す."""

    def __init__(
        self,
        shell_timeout: int = 30,
        *,
        backend: str | None = None,
        searxng_url: str | None = None,
    ) -> None:
        self.shell_timeout = shell_timeout
        self.backend = (backend or os.environ.get("AI_COMPANY_WEB_SEARCH_BACKEND") or "searxng").strip().lower()
        self.searxng_url = (
            searxng_url
            or os.environ.get("AI_COMPANY_SEARXNG_URL")
            or "http://127.0.0.1:8088"
        ).strip().rstrip("/")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """検索クエリを実行し、SearchResult のリストを返す.

        エラー時は空リスト（例外は送出しない）。
        """
        q = (query or "").strip()
        if not q:
            return []

        try:
            if self.backend == "ddg_html":
                return self._do_search_ddg_html(q, max_results)

            # default: searxng
            results = self._do_search_searxng(q, max_results)
            if results:
                return results

            # If searxng returns empty, don't automatically fall back to ddg
            # (it is often blocked and would waste time), unless explicitly requested.
            return []
        except Exception:
            logger.exception("WebSearcher.search() で予期しないエラー")
            return []

    # -----------------------------
    # SearxNG backend
    # -----------------------------

    def _do_search_searxng(self, query: str, max_results: int) -> list[SearchResult]:
        """SearxNG JSON APIで検索する."""
        url = (
            f"{self.searxng_url}/search?"
            f"q={quote_plus(query)}&format=json&language=ja-JP&safesearch=0"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.shell_timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            logger.warning("SearxNG検索に失敗: %s", e)
            return []

        try:
            payload = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            logger.warning("SearxNG応答のJSONパースに失敗")
            return []

        items = payload.get("results")
        if not isinstance(items, list) or not items:
            return []

        results: list[SearchResult] = []
        for it in items:
            if len(results) >= max_results:
                break
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            url0 = str(it.get("url") or "").strip()
            snippet_raw = str(it.get("content") or it.get("snippet") or "").strip()
            snippet = self._clean_html(snippet_raw)
            if not title or not url0:
                continue
            results.append(SearchResult(title=title, url=url0, snippet=snippet))
        return results

    # -----------------------------
    # DuckDuckGo HTML backend (fallback)
    # -----------------------------

    def _do_search_ddg_html(self, query: str, max_results: int) -> list[SearchResult]:
        """DuckDuckGo HTML検索結果をcurlで取得してパースする."""
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

        return self._parse_results_ddg(result.stdout, max_results)

    def _parse_results_ddg(self, html_content: str, max_results: int) -> list[SearchResult]:
        """DuckDuckGo HTML検索結果をパースする."""
        results: list[SearchResult] = []

        result_blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div|span)>'
            ,
            html_content,
            re.DOTALL,
        )

        for raw_url, raw_title, raw_snippet in result_blocks:
            if len(results) >= max_results:
                break

            url = self._extract_url_ddg(raw_url)
            title = self._clean_html(raw_title).strip()
            snippet = self._clean_html(raw_snippet).strip()

            if not url or not title:
                continue

            results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results

    @staticmethod
    def _extract_url_ddg(raw_url: str) -> str:
        """DuckDuckGo のリダイレクトURLから実際のURLを抽出する."""
        uddg_match = re.search(r"[?&]uddg=([^&]+)", raw_url)
        if uddg_match:
            return unquote(uddg_match.group(1))
        url = raw_url.strip()
        if url.startswith("//"):
            url = "https:" + url
        return url

    @staticmethod
    def _clean_html(text: str) -> str:
        """HTMLタグを除去し、エンティティをデコードする."""
        cleaned = html.unescape(text)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
