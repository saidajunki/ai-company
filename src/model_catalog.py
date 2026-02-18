"""モデルカタログ: PricingCacheからモデル情報を抽出しカテゴリを付与する.

加えて、サブエージェント（社員AI）のモデル選定ロジックを提供する。
- 役割(role)優先
- roleが曖昧な場合はタスク本文のヒューリスティクス
- 最終的に env で上書き可能

注意: OpenRouter側の利用可否はモデルIDとアカウント設定に依存する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from models import PricingCache


@dataclass
class ModelInfo:
    """モデルの情報とカテゴリ."""

    model_id: str
    input_price_per_1k: float
    output_price_per_1k: float
    categories: list[str] = field(default_factory=list)


DEFAULT_MODEL_CATEGORIES: dict[str, list[str]] = {
    # Anthropic
    "anthropic/claude-sonnet-4": ["coding", "analysis"],
    "anthropic/claude-3.5-sonnet": ["coding", "analysis"],
    "anthropic/claude-3.5-haiku": ["fast", "cheap", "general"],
    "anthropic/claude-3.7-sonnet": ["coding", "analysis"],
    # Google
    "google/gemini-2.5-flash": ["fast", "cheap", "general"],
    "google/gemini-2.5-pro": ["coding", "analysis"],
    # OpenAI
    "openai/gpt-4.1-mini": ["fast", "cheap", "general"],
    "openai/gpt-4.1": ["coding", "analysis"],
    "openai/gpt-4.1-nano": ["fast", "cheap"],
    "openai/gpt-4o-mini": ["fast", "cheap", "general"],
    "openai/gpt-4o": ["coding", "analysis"],
    # Others
    "deepseek/deepseek-chat": ["coding", "cheap"],
}


# ---------------------------------------------------------------------------
# Role-based model selection for sub-agents
# ---------------------------------------------------------------------------

# roleキーワード → タスク種別(class)
# Order matters: first match wins.
ROLE_CLASS_MAP: list[tuple[list[str], str]] = [
    (  # Research / writing — cheap & fast is fine
        ["researcher", "research", "調査", "リサーチ"],
        "research",
    ),
    (
        ["writer", "ライター", "執筆", "レポート", "文章"],
        "writing",
    ),
    (  # Infra / ops
        ["infra", "ops", "sre", "devops", "運用", "インフラ"],
        "infra",
    ),
    (  # Coding / engineering
        [
            "coder",
            "engineer",
            "developer",
            "software-engineer",
            "web-developer",
            "コーダー",
            "エンジニア",
            "開発",
            "実装",
        ],
        "coding",
    ),
    (
        ["analyst", "分析", "アナリスト", "strategist", "戦略"],
        "analysis",
    ),
    (  # Default worker
        ["worker", "assistant", "アシスタント"],
        "general",
    ),
]


def _env_model(key: str, default: str) -> str:
    v = (os.environ.get(key) or "").strip()
    return v if v else default


def _model_by_class(*, fallback_model: str) -> dict[str, str]:
    # NOTE: default values are conservative (won't break existing deployments).
    general = _env_model("AI_COMPANY_MODEL_GENERAL", "openai/gpt-4.1-mini")
    coding = _env_model("AI_COMPANY_MODEL_CODING", "openai/gpt-4.1")
    analysis = _env_model("AI_COMPANY_MODEL_ANALYSIS", coding)
    infra = _env_model("AI_COMPANY_MODEL_INFRA", coding)
    research = _env_model("AI_COMPANY_MODEL_RESEARCH", general)
    writing = _env_model("AI_COMPANY_MODEL_WRITING", research)

    def _or_fallback(x: str) -> str:
        x = (x or "").strip()
        return x if x else fallback_model

    return {
        "general": _or_fallback(general),
        "coding": _or_fallback(coding),
        "analysis": _or_fallback(analysis),
        "infra": _or_fallback(infra),
        "research": _or_fallback(research),
        "writing": _or_fallback(writing),
    }


def _looks_like_coding_task(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    keywords = (
        "code",
        "coding",
        "implement",
        "bug",
        "fix",
        "refactor",
        "python",
        "typescript",
        "node",
        "docker",
        "compose",
        "kubernetes",
        "traefik",
        "nginx",
        "systemd",
        "sql",
        "postgres",
        "mysql",
        "wordpress",
        "deploy",
        "release",
        "ci",
        "github",
        "commit",
        "pr",
        "実装",
        "修正",
        "バグ",
        "デバッグ",
        "デプロイ",
        "設定",
    )
    return any(k in t for k in keywords)


def _looks_like_research_task(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    keywords = (
        "research",
        "investigate",
        "compare",
        "benchmark",
        "docs",
        "documentation",
        "調査",
        "リサーチ",
        "比較",
        "一次情報",
        "仕様",
        "公式",
    )
    return any(k in t for k in keywords)


def select_model_for_role(role: str, fallback_model: str, *, task_description: str | None = None) -> str:
    """ロール名とタスク本文に基づいてサブエージェント用モデルを選択する.

    Priorities:
    1) roleキーワード → class
    2) roleが曖昧/未マッチなら task_description のヒューリスティクス
    3) fallback_model

    Env overrides:
    - AI_COMPANY_MODEL_GENERAL (default openai/gpt-4.1-mini)
    - AI_COMPANY_MODEL_CODING (default openai/gpt-4.1)
    - AI_COMPANY_MODEL_ANALYSIS (default = CODING)
    - AI_COMPANY_MODEL_INFRA (default = CODING)
    - AI_COMPANY_MODEL_RESEARCH (default = GENERAL)
    - AI_COMPANY_MODEL_WRITING (default = RESEARCH)
    """
    models = _model_by_class(fallback_model=fallback_model)

    role_lower = (role or "").lower()
    for keywords, cls in ROLE_CLASS_MAP:
        for kw in keywords:
            if kw.lower() in role_lower:
                return models.get(cls, models["general"])

    desc = task_description or ""
    if _looks_like_coding_task(desc):
        return models["coding"]
    if _looks_like_research_task(desc):
        return models["research"]

    return fallback_model


def _match_categories(
    model_id: str,
    known_categories: dict[str, list[str]],
) -> list[str]:
    """プレフィックスマッチでモデルIDに対応するカテゴリを返す."""
    for prefix, cats in known_categories.items():
        if model_id.startswith(prefix):
            return list(cats)
    return []


def build_model_catalog(
    pricing_cache: PricingCache | None,
    known_categories: dict[str, list[str]] | None = None,
) -> list[ModelInfo]:
    """PricingCacheからモデルカタログを生成する."""
    if pricing_cache is None or not pricing_cache.models:
        return []

    if known_categories is None:
        known_categories = DEFAULT_MODEL_CATEGORIES

    catalog: list[ModelInfo] = []
    for model_id, pricing in pricing_cache.models.items():
        catalog.append(
            ModelInfo(
                model_id=model_id,
                input_price_per_1k=pricing.input_price_per_1k,
                output_price_per_1k=pricing.output_price_per_1k,
                categories=_match_categories(model_id, known_categories),
            )
        )
    return catalog


def format_model_catalog_for_prompt(
    catalog: list[ModelInfo],
    max_models: int = 15,
) -> str:
    """モデルカタログをシステムプロンプト用テキストに変換する."""
    if not catalog:
        return ""

    sorted_catalog = sorted(
        catalog,
        key=lambda m: m.input_price_per_1k + m.output_price_per_1k,
    )

    limited = sorted_catalog[:max_models]

    lines = ["利用可能なモデル一覧（料金順）:"]
    for m in limited:
        cats = f" [{', '.join(m.categories)}]" if m.categories else ""
        lines.append(
            f"- {m.model_id} (${m.input_price_per_1k:.2f}/${m.output_price_per_1k:.2f} per 1K tokens){cats}"
        )
    return "\n".join(lines)
