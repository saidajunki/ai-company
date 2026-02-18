"""Tests for newsroom_team helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from newsroom_team import (
    NewsCandidate,
    NewsroomTeam,
    _extract_json_object,
    _load_simple_env,
    _parse_feed_items,
)


class _DummyManager:
    def __init__(self, base_dir: Path, company_id: str = "alpha") -> None:
        self.base_dir = base_dir
        self.company_id = company_id


def test_parse_rss_items_extracts_title_link_and_date() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>AI model release</title>
      <link>https://example.com/a</link>
      <description>new model details</description>
      <pubDate>Wed, 18 Feb 2026 10:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""
    items = _parse_feed_items(xml)
    assert len(items) == 1
    assert items[0].title == "AI model release"
    assert items[0].url == "https://example.com/a"
    assert items[0].summary == "new model details"
    assert items[0].published_at is not None
    assert items[0].published_at.year == 2026


def test_parse_atom_items_extracts_entry() -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Agent update</title>
    <link href="https://example.com/agent" rel="alternate" />
    <summary>agent capabilities improved</summary>
    <updated>2026-02-18T11:30:00Z</updated>
  </entry>
</feed>
"""
    items = _parse_feed_items(xml)
    assert len(items) == 1
    assert items[0].title == "Agent update"
    assert items[0].url == "https://example.com/agent"
    assert "improved" in items[0].summary
    assert items[0].published_at is not None


def test_extract_json_object_from_mixed_text() -> None:
    text = "prefix\n{\"title\":\"t\",\"excerpt\":\"e\",\"content_html\":\"<p>x</p>\",\"tags\":[\"AI\"]}\nsuffix"
    obj = _extract_json_object(text)
    assert obj is not None
    assert obj["title"] == "t"


def test_parse_article_payload_fallback_when_json_missing(tmp_path: Path) -> None:
    manager = _DummyManager(tmp_path)
    team = NewsroomTeam(manager)
    candidate = NewsCandidate(
        source_name="Tech",
        title="AI infra update",
        url="https://example.com/source",
        summary="infra summary",
        published_at=datetime.now(timezone.utc),
    )

    payload = team._parse_article_payload("not-json text", candidate)
    assert "AIニュース解説" in payload["title"]
    assert "出典" in payload["content_html"]
    assert payload["tags"]


def test_load_simple_env_reads_key_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """# comment
A=1
B=hello
""",
        encoding="utf-8",
    )
    data = _load_simple_env(env_file)
    assert data["A"] == "1"
    assert data["B"] == "hello"
