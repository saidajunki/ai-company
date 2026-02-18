"""Newsroom Team — RSS/scraping based autonomous WordPress news posting.

Runs as an internal autonomous team:
- Collect latest candidates from configured RSS feeds.
- Ask sub-agents (researcher/writer) to produce Japanese article content.
- Publish one post to WordPress via XML-RPC.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import xmlrpc.client
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


DEFAULT_NEWSROOM_SOURCES = {
    "schedule": {
        "interval_minutes": 60,
    },
    "budgets": {
        "post_usd": 0.5,
        "research_usd": 0.2,
        "writer_usd": 0.25,
    },
    "wordpress": {
        "categories": ["ニュース", "AI", "テクノロジー"],
        "status": "publish",
    },
    "campaign": {
        "enabled": True,
        "topic": "最新のAI技術を紹介し、技術力をアピールする",
        "target_posts": 100,
        "hourly_budget_usd": 1.0,
        "total_budget_usd": 10.0,
        "max_runtime_hours": 10,
        "report_every_posts": 10,
    },
    "sources": [
        {
            "name": "TechCrunch AI",
            "rss": "https://techcrunch.com/category/artificial-intelligence/feed/",
            "keywords": ["ai", "artificial intelligence", "llm", "model", "agent"],
        },
        {
            "name": "VentureBeat AI",
            "rss": "https://venturebeat.com/category/ai/feed/",
            "keywords": ["ai", "llm", "model", "inference", "agent"],
        },
        {
            "name": "The Verge AI",
            "rss": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
            "keywords": ["ai", "artificial intelligence", "model", "robot", "agent"],
        },
    ],
}

DEFAULT_NEWSROOM_STATE = {
    "last_run_at": None,
    "last_post_at": None,
    "seen_urls": [],
    "posts": [],
    "campaign": {
        "status": "running",
        "started_at": None,
        "ended_at": None,
        "stop_reason": None,
        "last_budget_skip_at": None,
        "last_reported_posts": 0,
    },
}

DEFAULT_NEWSROOM_EMPLOYEES = (
    {
        "name": "神谷リサ",
        "role": "news-researcher",
        "purpose": "最新AIニュースの一次情報を調査し、事実整理を行う",
        "env_model": "AI_COMPANY_NEWS_RESEARCH_MODEL",
        "default_model": "openai/gpt-4.1-mini",
    },
    {
        "name": "綾瀬文香",
        "role": "news-writer",
        "purpose": "調査結果を日本語の技術解説記事として構成する",
        "env_model": "AI_COMPANY_NEWS_WRITER_MODEL",
        "default_model": "openai/gpt-4.1",
    },
)


@dataclass
class NewsCandidate:
    source_name: str
    title: str
    url: str
    summary: str
    published_at: datetime | None


class WordPressXMLRPCPublisher:
    """Minimal WordPress XML-RPC publisher."""

    def __init__(
        self,
        *,
        endpoint: str,
        username: str,
        password: str,
        default_categories: list[str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.default_categories = default_categories or ["ニュース", "AI"]

    def publish(
        self,
        *,
        title: str,
        content_html: str,
        excerpt: str,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        publish: bool = True,
    ) -> str:
        server = xmlrpc.client.ServerProxy(self.endpoint)
        blog_id = "1"

        post_data = {
            "title": title,
            "description": content_html,
            "mt_excerpt": excerpt,
            "mt_keywords": ", ".join(tags or []),
            "categories": categories or self.default_categories,
            "post_status": "publish" if publish else "draft",
        }

        post_id = server.metaWeblog.newPost(
            blog_id,
            self.username,
            self.password,
            post_data,
            publish,
        )
        return str(post_id)


class NewsroomTeam:
    """Autonomous newsroom team using sub-agents + WordPress publishing."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.company_root = manager.base_dir / "companies" / manager.company_id
        self.protocols_dir = self.company_root / "protocols"
        self.state_dir = self.company_root / "state"
        self.sources_path = self.protocols_dir / "newsroom_sources.yaml"
        self.state_path = self.state_dir / "newsroom_state.json"
        self.archive_dir = self.company_root / "knowledge" / "shared" / "newsroom"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        self.protocols_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        if not self.sources_path.exists():
            self.sources_path.write_text(
                yaml.safe_dump(DEFAULT_NEWSROOM_SOURCES, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

        if not self.state_path.exists():
            self.state_path.write_text(
                json.dumps(DEFAULT_NEWSROOM_STATE, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def tick(self) -> None:
        """Run one newsroom cycle.

        - Select one unseen latest candidate.
        - Delegate research/writing to sub-agents.
        - Publish to WordPress.
        """
        self.ensure_initialized()

        if self.manager.llm_client is None:
            logger.info("Newsroom tick skipped: llm_client is not configured")
            return

        cfg = self._load_sources_config()
        state = self._load_state()
        state = self._normalize_state(state)

        can_proceed, block_reason = self._campaign_allows_posting(cfg, state)
        if not can_proceed:
            state["last_run_at"] = _utc_now().isoformat()
            self._save_state(state)
            if block_reason:
                self._activity(f"ニュース部隊待機: {block_reason}")
            return

        if not self._interval_ready(cfg, state):
            return

        candidate = self._select_candidate(cfg, state)
        if candidate is None:
            state["last_run_at"] = _utc_now().isoformat()
            self._save_state(state)
            self._activity("ニュース部隊: 新規候補なし（RSS更新待ち）")
            return

        team = self._ensure_newsroom_employees(cfg)
        researcher_profile = team.get("news-researcher") or {}
        writer_profile = team.get("news-writer") or {}

        self._activity(
            "CEO→社員AI 委任: role=news-researcher "
            f"task=最新ニュース調査 ({_short(candidate.title, 100)})"
        )
        research_result = self.manager.sub_agent_runner.spawn(
            name=str(researcher_profile.get("name") or "news-researcher"),
            role="news-researcher",
            task_description=self._build_research_task(candidate),
            budget_limit_usd=self._research_budget(cfg),
            model=os.environ.get("AI_COMPANY_NEWS_RESEARCH_MODEL", "openai/gpt-4.1-mini"),
            ignore_wip_limit=True,
            persistent_employee=researcher_profile,
        )

        self._activity(
            "CEO→社員AI 委任: role=news-writer "
            f"task=記事作成 ({_short(candidate.title, 100)})"
        )
        writer_result = self.manager.sub_agent_runner.spawn(
            name=str(writer_profile.get("name") or "news-writer"),
            role="news-writer",
            task_description=self._build_writer_task(candidate, research_result),
            budget_limit_usd=self._writer_budget(cfg),
            model=os.environ.get("AI_COMPANY_NEWS_WRITER_MODEL", "openai/gpt-4.1"),
            ignore_wip_limit=True,
            persistent_employee=writer_profile,
        )

        article = self._parse_article_payload(writer_result, candidate)
        publisher = self._build_publisher(cfg)
        if publisher is None:
            self._activity("ニュース部隊: WordPress資格情報が不足しているため投稿停止")
            return

        categories = cfg.get("wordpress", {}).get("categories")
        publish = str(cfg.get("wordpress", {}).get("status", "publish")).lower() != "draft"

        post_id = publisher.publish(
            title=article["title"],
            content_html=article["content_html"],
            excerpt=article["excerpt"],
            tags=article.get("tags") or [],
            categories=categories if isinstance(categories, list) else None,
            publish=publish,
        )

        now = _utc_now()
        state["last_run_at"] = now.isoformat()
        state["last_post_at"] = now.isoformat()
        state["seen_urls"] = _append_unique_tail(state.get("seen_urls") or [], candidate.url, limit=800)

        posts = state.get("posts") if isinstance(state.get("posts"), list) else []
        posts.append(
            {
                "posted_at": now.isoformat(),
                "post_id": post_id,
                "title": article["title"],
                "source_url": candidate.url,
                "source_name": candidate.source_name,
            }
        )
        state["posts"] = posts[-200:]

        self._refresh_campaign_progress(cfg, state)
        self._save_state(state)
        self._archive_article(article, candidate, post_id, now)

        campaign_cfg = self._campaign_config(cfg)
        post_count = len(state.get("posts") or [])
        target_posts = int(campaign_cfg.get("target_posts", 100))
        est_cost = self._estimated_total_spent(cfg, state)
        msg = (
            "📰 ニュース投稿完了\n"
            f"- タイトル: {article['title']}\n"
            f"- WordPress post_id: {post_id}\n"
            f"- 元ニュース: {candidate.url}\n"
            f"- 進捗: {post_count}/{target_posts} 投稿\n"
            f"- 推定総コスト: ${est_cost:.2f}"
        )
        self._slack(msg)
        self._activity(f"ニュース部隊完了: title={_short(article['title'], 120)} post_id={post_id}")

    # ------------------------------------------------------------------
    # Config / State
    # ------------------------------------------------------------------

    def _load_sources_config(self) -> dict[str, Any]:
        try:
            data = yaml.safe_load(self.sources_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            logger.warning("Failed to load newsroom sources config", exc_info=True)
        return DEFAULT_NEWSROOM_SOURCES

    def _load_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            logger.warning("Failed to load newsroom state", exc_info=True)
        return dict(DEFAULT_NEWSROOM_STATE)

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(DEFAULT_NEWSROOM_STATE)
        normalized.update(state if isinstance(state, dict) else {})

        campaign = dict(DEFAULT_NEWSROOM_STATE.get("campaign") or {})
        if isinstance(state.get("campaign"), dict):  # type: ignore[union-attr]
            campaign.update(state.get("campaign"))  # type: ignore[arg-type]
        normalized["campaign"] = campaign

        posts = normalized.get("posts")
        normalized["posts"] = posts if isinstance(posts, list) else []
        seen_urls = normalized.get("seen_urls")
        normalized["seen_urls"] = seen_urls if isinstance(seen_urls, list) else []
        return normalized

    def _campaign_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        base = dict(DEFAULT_NEWSROOM_SOURCES.get("campaign") or {})
        custom = cfg.get("campaign")
        if isinstance(custom, dict):
            base.update(custom)

        env_map = {
            "enabled": os.environ.get("AI_COMPANY_NEWS_CAMPAIGN_ENABLED"),
            "target_posts": os.environ.get("AI_COMPANY_NEWS_TARGET_POSTS"),
            "hourly_budget_usd": os.environ.get("AI_COMPANY_NEWS_HOURLY_BUDGET_USD"),
            "total_budget_usd": os.environ.get("AI_COMPANY_NEWS_TOTAL_BUDGET_USD"),
            "max_runtime_hours": os.environ.get("AI_COMPANY_NEWS_MAX_RUNTIME_HOURS"),
            "report_every_posts": os.environ.get("AI_COMPANY_NEWS_REPORT_EVERY_POSTS"),
        }
        for key, raw in env_map.items():
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            if key == "enabled":
                base[key] = text.lower() not in ("0", "false", "off", "no")
            elif key in ("target_posts", "report_every_posts"):
                try:
                    base[key] = max(1, int(text))
                except Exception:
                    pass
            else:
                try:
                    base[key] = max(0.1, float(text))
                except Exception:
                    pass
        return base

    def _campaign_allows_posting(
        self,
        cfg: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[bool, str | None]:
        campaign_cfg = self._campaign_config(cfg)
        campaign_state = state.get("campaign") if isinstance(state.get("campaign"), dict) else {}
        if not isinstance(campaign_state, dict):
            campaign_state = {}
            state["campaign"] = campaign_state

        enabled = bool(campaign_cfg.get("enabled", True))
        if not enabled:
            campaign_state["status"] = "disabled"
            return False, None

        if str(campaign_state.get("status") or "").strip() == "completed":
            return False, None

        now = _utc_now()
        if not str(campaign_state.get("started_at") or "").strip():
            campaign_state["started_at"] = now.isoformat()

        post_cost = self._estimated_post_cost(cfg)
        posts = state.get("posts") if isinstance(state.get("posts"), list) else []
        post_count = len(posts)
        target_posts = _safe_int(campaign_cfg.get("target_posts"), default=100, minimum=1)
        total_budget = max(post_cost, _safe_float(campaign_cfg.get("total_budget_usd"), default=10.0, minimum=0.1))
        hourly_budget = max(post_cost, _safe_float(campaign_cfg.get("hourly_budget_usd"), default=1.0, minimum=0.1))
        max_runtime_hours = _safe_float(campaign_cfg.get("max_runtime_hours"), default=10.0, minimum=0.1)

        if post_count >= target_posts:
            self._complete_campaign(
                state,
                reason=f"目標投稿数に到達 ({post_count}/{target_posts})",
                cfg=cfg,
            )
            return False, None

        total_spent = self._estimated_total_spent(cfg, state)
        if total_spent + post_cost > total_budget + 1e-9:
            self._complete_campaign(
                state,
                reason=f"総予算上限に到達 (${total_spent:.2f}/${total_budget:.2f})",
                cfg=cfg,
            )
            return False, None

        started_at = _parse_iso_datetime(str(campaign_state.get("started_at") or ""))
        if started_at is not None and (now - started_at).total_seconds() >= max_runtime_hours * 3600:
            self._complete_campaign(
                state,
                reason=f"稼働時間上限に到達 ({max_runtime_hours:.1f}h)",
                cfg=cfg,
            )
            return False, None

        hourly_spent = self._estimated_hourly_spent(cfg, state)
        if hourly_spent + post_cost > hourly_budget + 1e-9:
            campaign_state["status"] = "throttled"
            last_skip = _parse_iso_datetime(str(campaign_state.get("last_budget_skip_at") or ""))
            campaign_state["last_budget_skip_at"] = now.isoformat()
            if last_skip is not None and (now - last_skip).total_seconds() < 300:
                return False, None
            return False, f"hourly budget guard (${hourly_spent:.2f}/${hourly_budget:.2f})"

        campaign_state["status"] = "running"
        return True, None

    def _complete_campaign(self, state: dict[str, Any], *, reason: str, cfg: dict[str, Any]) -> None:
        now = _utc_now()
        campaign_state = state.get("campaign")
        if not isinstance(campaign_state, dict):
            campaign_state = {}
            state["campaign"] = campaign_state
        already_completed = str(campaign_state.get("status") or "").strip() == "completed"
        campaign_state["status"] = "completed"
        if not str(campaign_state.get("started_at") or "").strip():
            campaign_state["started_at"] = now.isoformat()
        campaign_state["ended_at"] = now.isoformat()
        campaign_state["stop_reason"] = reason

        if already_completed:
            return

        posts = state.get("posts") if isinstance(state.get("posts"), list) else []
        target_posts = int(self._campaign_config(cfg).get("target_posts", 100))
        total_cost = self._estimated_total_spent(cfg, state)
        self._slack(
            "✅ ニュース運営キャンペーン完了\n"
            f"- 理由: {reason}\n"
            f"- 投稿数: {len(posts)}/{target_posts}\n"
            f"- 推定総コスト: ${total_cost:.2f}"
        )
        self._activity(f"ニュース部隊キャンペーン完了: {reason}")

    def _refresh_campaign_progress(self, cfg: dict[str, Any], state: dict[str, Any]) -> None:
        campaign_cfg = self._campaign_config(cfg)
        campaign_state = state.get("campaign") if isinstance(state.get("campaign"), dict) else {}
        if not isinstance(campaign_state, dict):
            campaign_state = {}
            state["campaign"] = campaign_state

        posts = state.get("posts") if isinstance(state.get("posts"), list) else []
        report_every = _safe_int(campaign_cfg.get("report_every_posts"), default=10, minimum=1)
        last_reported = _safe_int(campaign_state.get("last_reported_posts"), default=0, minimum=0)
        current = len(posts)

        if current >= last_reported + report_every:
            target_posts = _safe_int(campaign_cfg.get("target_posts"), default=100, minimum=1)
            spent = self._estimated_total_spent(cfg, state)
            self._slack(
                "📈 ニュース運営キャンペーン進捗\n"
                f"- 投稿数: {current}/{target_posts}\n"
                f"- 推定総コスト: ${spent:.2f}"
            )
            campaign_state["last_reported_posts"] = current

        if current >= _safe_int(campaign_cfg.get("target_posts"), default=100, minimum=1):
            self._complete_campaign(
                state,
                reason=f"目標投稿数に到達 ({current}/{campaign_cfg.get('target_posts', 100)})",
                cfg=cfg,
            )

    def _ensure_newsroom_employees(self, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        store = getattr(self.manager, "employee_store", None)
        if store is None:
            return out

        post_budget = self._estimated_post_cost(cfg)
        research_budget = max(0.05, min(self._research_budget(cfg), post_budget))
        writer_budget = max(0.05, min(self._writer_budget(cfg), post_budget))
        budget_by_role = {
            "news-researcher": research_budget,
            "news-writer": writer_budget,
        }

        for profile in DEFAULT_NEWSROOM_EMPLOYEES:
            model = (
                os.environ.get(str(profile.get("env_model") or ""), str(profile.get("default_model") or ""))
                or str(profile.get("default_model") or "openai/gpt-4.1-mini")
            ).strip() or "openai/gpt-4.1-mini"
            role = str(profile.get("role") or "").strip()
            if not role:
                continue
            name = str(profile.get("name") or role).strip() or role
            purpose = str(profile.get("purpose") or role).strip() or role
            budget = budget_by_role.get(role, max(0.05, post_budget))
            try:
                entry, created = store.ensure_active(
                    name=name,
                    role=role,
                    purpose=purpose,
                    model=model,
                    budget_limit_usd=budget,
                )
                out[role] = entry.model_dump()
                if created:
                    self._activity(
                        f"CEO→社員AI 雇用: name={entry.name} role={entry.role} model={entry.model} budget=${entry.budget_limit_usd:.2f}"
                    )
            except Exception:
                logger.warning("Failed to ensure newsroom employee: %s", role, exc_info=True)
        return out

    def _estimated_post_cost(self, cfg: dict[str, Any]) -> float:
        research = self._research_budget(cfg)
        writer = self._writer_budget(cfg)
        return max(0.1, research + writer)

    def _estimated_total_spent(self, cfg: dict[str, Any], state: dict[str, Any]) -> float:
        posts = state.get("posts") if isinstance(state.get("posts"), list) else []
        return len(posts) * self._estimated_post_cost(cfg)

    def _estimated_hourly_spent(self, cfg: dict[str, Any], state: dict[str, Any]) -> float:
        posts = state.get("posts") if isinstance(state.get("posts"), list) else []
        if not posts:
            return 0.0
        now = _utc_now()
        threshold = now - timedelta(hours=1)
        count = 0
        for post in posts:
            if not isinstance(post, dict):
                continue
            posted_at = _parse_iso_datetime(str(post.get("posted_at") or ""))
            if posted_at is None:
                continue
            if posted_at >= threshold:
                count += 1
        return count * self._estimated_post_cost(cfg)

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _interval_ready(self, cfg: dict[str, Any], state: dict[str, Any]) -> bool:
        interval_minutes = self._interval_minutes(cfg)
        last_post_raw = str(state.get("last_post_at") or "").strip()
        if not last_post_raw:
            return True
        try:
            last_post = datetime.fromisoformat(last_post_raw.replace("Z", "+00:00"))
            if last_post.tzinfo is None:
                last_post = last_post.replace(tzinfo=timezone.utc)
        except Exception:
            return True

        return (_utc_now() - last_post).total_seconds() >= interval_minutes * 60

    def _select_candidate(self, cfg: dict[str, Any], state: dict[str, Any]) -> NewsCandidate | None:
        seen = set(state.get("seen_urls") or [])
        sources = cfg.get("sources") if isinstance(cfg.get("sources"), list) else []

        all_items: list[NewsCandidate] = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            source_name = str(src.get("name") or "source").strip() or "source"
            rss = str(src.get("rss") or "").strip()
            keywords = [str(k).strip().lower() for k in (src.get("keywords") or []) if str(k).strip()]
            if not rss:
                continue

            xml_text = self._fetch_text(rss)
            if not xml_text:
                continue

            for item in _parse_feed_items(xml_text):
                if not item.url:
                    continue
                if item.url in seen:
                    continue
                if keywords and not _contains_any((item.title + "\n" + item.summary).lower(), keywords):
                    continue
                all_items.append(NewsCandidate(
                    source_name=source_name,
                    title=item.title,
                    url=item.url,
                    summary=item.summary,
                    published_at=item.published_at,
                ))

        if not all_items:
            return None

        all_items.sort(
            key=lambda x: x.published_at or datetime(1970, 1, 1, tzinfo=timezone.utc),
            reverse=True,
        )
        return all_items[0]

    # ------------------------------------------------------------------
    # Sub-agent task builders
    # ------------------------------------------------------------------

    def _build_research_task(self, candidate: NewsCandidate) -> str:
        article_text = self._fetch_article_text(candidate.url)
        return "\n".join([
            "【ニュース調査タスク】",
            "- 目的: 海外最新ニュースを日本語で正確に要点整理する",
            "- 禁止: 事実の捏造、未確認の断定、出典の誤記",
            "",
            f"source_name: {candidate.source_name}",
            f"source_title: {candidate.title}",
            f"source_url: {candidate.url}",
            f"source_summary: {_short(candidate.summary, 600)}",
            "",
            "page_excerpt:",
            article_text,
            "",
            "出力形式: <done> で次のYAMLだけを返す",
            "headline_ja: ...",
            "facts:",
            "  - ...",
            "  - ...",
            "impact: ...",
            "cautions: ...",
            "",
            "不明点がある場合は『不明』と書くこと。",
        ])

    def _build_writer_task(self, candidate: NewsCandidate, research_result: str) -> str:
        return "\n".join([
            "【ニュース記事作成タスク】",
            "- 目的: 日本語の解説記事をWordPress投稿用に作る",
            "- 制約: 誇張しない、断定しすぎない、出典URLを明記",
            "",
            f"source_name: {candidate.source_name}",
            f"source_title: {candidate.title}",
            f"source_url: {candidate.url}",
            "",
            "research_memo:",
            research_result.strip(),
            "",
            "<done>で JSON オブジェクトのみ返すこと（前後テキスト禁止）:",
            '{"title":"...","excerpt":"...","content_html":"...","tags":["AI","Tech"]}',
            "",
            "content_html には以下を含める:",
            "- 見出し<h2>を2〜4個",
            "- 400〜900語程度の本文",
            "- 最後に出典セクション（source_url）",
        ])

    # ------------------------------------------------------------------
    # Article parse / publish helpers
    # ------------------------------------------------------------------

    def _parse_article_payload(self, text: str, candidate: NewsCandidate) -> dict[str, Any]:
        payload = _extract_json_object(text)
        if not payload:
            logger.warning("News writer output was not valid JSON; fallback template used")
            fallback_body = _simple_paragraph_html(text)
            if not fallback_body:
                fallback_body = f"<p>{_escape_html(candidate.summary or candidate.title)}</p>"
            return {
                "title": f"【AIニュース解説】{candidate.title}",
                "excerpt": _short(candidate.summary or candidate.title, 140),
                "content_html": (
                    f"<h2>概要</h2>{fallback_body}"
                    f"<h2>出典</h2><p><a href=\"{_escape_html(candidate.url)}\">{_escape_html(candidate.url)}</a></p>"
                ),
                "tags": ["AI", "Tech"],
            }

        title = str(payload.get("title") or "").strip() or f"【AIニュース解説】{candidate.title}"
        excerpt = str(payload.get("excerpt") or "").strip() or _short(candidate.summary or title, 140)
        content_html = str(payload.get("content_html") or "").strip()
        if not content_html:
            md = str(payload.get("body_markdown") or "").strip()
            content_html = _markdown_to_html(md)

        if "出典" not in content_html:
            content_html += (
                "<h2>出典</h2>"
                f"<p><a href=\"{_escape_html(candidate.url)}\">{_escape_html(candidate.url)}</a></p>"
            )

        tags = payload.get("tags")
        if not isinstance(tags, list):
            tags = ["AI", "Tech"]
        tags = [str(t).strip() for t in tags if str(t).strip()][:8] or ["AI", "Tech"]

        return {
            "title": title,
            "excerpt": excerpt,
            "content_html": content_html,
            "tags": tags,
        }

    def _build_publisher(self, cfg: dict[str, Any]) -> WordPressXMLRPCPublisher | None:
        endpoint = (
            os.environ.get("AI_COMPANY_WP_XMLRPC_URL")
            or str(cfg.get("wordpress", {}).get("xmlrpc_url") or "")
            or "https://app.babl.tech/xmlrpc.php"
        ).strip()

        username = (os.environ.get("AI_COMPANY_WP_USERNAME") or os.environ.get("WP_ADMIN_USER") or "").strip()
        password = (os.environ.get("AI_COMPANY_WP_PASSWORD") or os.environ.get("WP_ADMIN_PASSWORD") or "").strip()

        if not username or not password:
            wp_env = Path(os.environ.get("AI_COMPANY_WP_ENV_FILE", "/opt/apps/services/wordpress-app/.env"))
            if wp_env.exists():
                env_map = _load_simple_env(wp_env)
                username = username or str(env_map.get("WP_ADMIN_USER") or "").strip()
                password = password or str(env_map.get("WP_ADMIN_PASSWORD") or "").strip()

        if not endpoint or not username or not password:
            return None

        categories = cfg.get("wordpress", {}).get("categories")
        cats = [str(c).strip() for c in categories if str(c).strip()] if isinstance(categories, list) else None
        return WordPressXMLRPCPublisher(
            endpoint=endpoint,
            username=username,
            password=password,
            default_categories=cats or ["ニュース", "AI"],
        )

    def _archive_article(
        self,
        article: dict[str, Any],
        candidate: NewsCandidate,
        post_id: str,
        now: datetime,
    ) -> None:
        stamp = now.strftime("%Y%m%d-%H%M%S")
        path = self.archive_dir / f"{stamp}-{post_id}.md"
        lines = [
            f"# {article['title']}",
            "",
            f"- posted_at: {now.isoformat()}",
            f"- wordpress_post_id: {post_id}",
            f"- source_name: {candidate.source_name}",
            f"- source_url: {candidate.url}",
            "",
            "## Excerpt",
            article.get("excerpt", ""),
            "",
            "## Content(HTML)",
            article.get("content_html", ""),
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_text(url: str, timeout: int = 20) -> str | None:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (ai-company newsroom)",
                "Accept": "application/rss+xml, application/atom+xml, text/xml, application/xml, text/html",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            logger.warning("Newsroom fetch failed: %s (%s)", url, exc)
            return None
        except Exception:
            logger.warning("Newsroom fetch failed: %s", url, exc_info=True)
            return None

        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode(errors="ignore")

    def _fetch_article_text(self, url: str) -> str:
        body = self._fetch_text(url, timeout=20)
        if not body:
            return "(本文取得失敗)"

        body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.IGNORECASE | re.DOTALL)

        title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
        title = _clean_html(title_match.group(1)) if title_match else ""

        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.IGNORECASE | re.DOTALL)
        texts: list[str] = []
        for p in paragraphs:
            t = _clean_html(p)
            if len(t) >= 50:
                texts.append(t)
            if len("\n".join(texts)) >= 2600:
                break

        out: list[str] = []
        if title:
            out.append(f"title: {title}")
        if texts:
            out.append("paragraphs:")
            out.extend(f"- {t}" for t in texts[:12])
        else:
            out.append("paragraphs: (取得失敗)")

        return "\n".join(out)[:3800]

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interval_minutes(cfg: dict[str, Any]) -> int:
        env = (os.environ.get("AI_COMPANY_NEWS_TEAM_INTERVAL_MINUTES") or "").strip()
        if env:
            try:
                return max(5, int(env))
            except Exception:
                pass
        try:
            return max(5, int(cfg.get("schedule", {}).get("interval_minutes", 60)))
        except Exception:
            return 60

    @staticmethod
    def _post_budget(cfg: dict[str, Any]) -> float:
        env = (os.environ.get("AI_COMPANY_NEWS_POST_BUDGET_USD") or "").strip()
        if env:
            try:
                return max(0.1, float(env))
            except Exception:
                pass
        try:
            return max(0.1, float(cfg.get("budgets", {}).get("post_usd", 0.5)))
        except Exception:
            return 0.5

    def _research_budget(self, cfg: dict[str, Any]) -> float:
        total = self._post_budget(cfg)
        try:
            cfg_budget = float(cfg.get("budgets", {}).get("research_usd", 0.2))
        except Exception:
            cfg_budget = 0.2
        return max(0.05, min(cfg_budget, total * 0.6))

    def _writer_budget(self, cfg: dict[str, Any]) -> float:
        total = self._post_budget(cfg)
        research = self._research_budget(cfg)
        remaining = max(0.05, total - research)
        try:
            cfg_budget = float(cfg.get("budgets", {}).get("writer_usd", 0.25))
        except Exception:
            cfg_budget = 0.25
        return max(0.05, min(cfg_budget, remaining))

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _slack(self, text: str) -> None:
        try:
            fn = getattr(self.manager, "_slack_send", None)
            if callable(fn):
                fn(text)
        except Exception:
            logger.warning("Failed to send newsroom Slack message", exc_info=True)

    def _activity(self, text: str) -> None:
        try:
            fn = getattr(self.manager, "_activity_log", None)
            if callable(fn):
                fn(text)
        except Exception:
            logger.warning("Failed to write newsroom activity", exc_info=True)


# ----------------------------------------------------------------------
# Feed parsing helpers
# ----------------------------------------------------------------------


@dataclass
class _RawFeedItem:
    title: str
    url: str
    summary: str
    published_at: datetime | None


def _parse_feed_items(xml_text: str) -> list[_RawFeedItem]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    tag = _strip_ns(root.tag).lower()
    if tag == "rss":
        return _parse_rss(root)
    if tag == "feed":
        return _parse_atom(root)

    # unknown XML root
    return []


def _parse_rss(root: ET.Element) -> list[_RawFeedItem]:
    channel = root.find("channel")
    if channel is None:
        return []

    out: list[_RawFeedItem] = []
    for item in channel.findall("item"):
        title = _clean_html(_text_of(item.find("title")))
        url = _clean_html(_text_of(item.find("link")))
        summary = _clean_html(_text_of(item.find("description")))
        pub_raw = _text_of(item.find("pubDate"))
        published_at = _parse_date(pub_raw)
        if title and url:
            out.append(_RawFeedItem(title=title, url=url, summary=summary, published_at=published_at))
    return out


def _parse_atom(root: ET.Element) -> list[_RawFeedItem]:
    ns = "{http://www.w3.org/2005/Atom}"
    out: list[_RawFeedItem] = []

    for entry in root.findall(f"{ns}entry"):
        title = _clean_html(_text_of(entry.find(f"{ns}title")))

        link = ""
        for link_elem in entry.findall(f"{ns}link"):
            href = (link_elem.attrib.get("href") or "").strip()
            rel = (link_elem.attrib.get("rel") or "alternate").strip().lower()
            if href and rel in ("", "alternate"):
                link = href
                break
            if href and not link:
                link = href

        summary = _clean_html(_text_of(entry.find(f"{ns}summary")))
        if not summary:
            summary = _clean_html(_text_of(entry.find(f"{ns}content")))

        pub_raw = (
            _text_of(entry.find(f"{ns}published"))
            or _text_of(entry.find(f"{ns}updated"))
        )
        published_at = _parse_date(pub_raw)

        if title and link:
            out.append(_RawFeedItem(title=title, url=link, summary=summary, published_at=published_at))

    return out


def _parse_date(text: str) -> datetime | None:
    s = (text or "").strip()
    if not s:
        return None

    # RFC822 / RFC2822 (RSS pubDate)
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # ISO-8601 (Atom)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _text_of(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    text = "".join(elem.itertext())
    return text.strip()


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _contains_any(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return any(k in text for k in keywords)


def _append_unique_tail(items: list[str], value: str, *, limit: int) -> list[str]:
    out = [x for x in items if x != value]
    out.append(value)
    return out[-limit:]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _end = decoder.raw_decode((text or "")[m.start():])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _clean_html(text: str) -> str:
    s = unescape(text or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _escape_html(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _simple_paragraph_html(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("<")]
    if not lines:
        return ""
    return "".join(f"<p>{_escape_html(ln)}</p>" for ln in lines[:12])


def _markdown_to_html(markdown_text: str) -> str:
    lines = (markdown_text or "").splitlines()
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("### "):
            out.append(f"<h3>{_escape_html(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{_escape_html(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h2>{_escape_html(s[2:])}</h2>")
        elif s.startswith("- "):
            # simple list fallback as paragraph
            out.append(f"<p>• {_escape_html(s[2:])}</p>")
        else:
            out.append(f"<p>{_escape_html(s)}</p>")
    return "\n".join(out)


def _load_simple_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    except Exception:
        return {}
    return out


def _short(text: str, limit: int) -> str:
    s = " ".join((text or "").split())
    if len(s) > limit:
        return s[:limit] + "…"
    return s


def _safe_int(value: Any, *, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(value))
    except Exception:
        return max(minimum, default)


def _safe_float(value: Any, *, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(value))
    except Exception:
        return max(minimum, default)


def _parse_iso_datetime(text: str) -> datetime | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
