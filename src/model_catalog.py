"""モデルカタログ: PricingCacheからモデル情報を抽出しカテゴリを付与する."""

from __future__ import annotations

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
    "anthropic/claude-sonnet": ["coding", "analysis"],
    "google/gemini-2.5-flash": ["fast", "cheap", "general"],
    "google/gemini-2.5-pro": ["coding", "analysis"],
    "openai/gpt-4.1-mini": ["fast", "cheap", "general"],
    "openai/gpt-4.1": ["coding", "analysis"],
    "openai/gpt-4.1-nano": ["fast", "cheap"],
    "openai/gpt-4o-mini": ["fast", "cheap", "general"],
    "openai/gpt-4o": ["coding", "analysis"],
    "deepseek/deepseek-chat": ["coding", "cheap"],
}


# ---------------------------------------------------------------------------
# Role-based model selection for sub-agents
# ---------------------------------------------------------------------------

# Maps sub-agent role keywords to preferred model (OpenRouter model ID).
# If the role matches a keyword, the corresponding model is used instead of
# falling back to the CEO model.  Order matters: first match wins.
ROLE_MODEL_MAP: list[tuple[list[str], str]] = [
    # Research / writing — cheap & fast is fine
    (["researcher", "research", "調査", "リサーチ"], "openai/gpt-4.1-mini"),
    (["writer", "ライター", "執筆", "レポート"], "openai/gpt-4.1-mini"),
    # Coding / engineering — needs stronger model
    (["coder", "engineer", "developer", "コーダー", "エンジニア", "開発"], "openai/gpt-4.1"),
    (["analyst", "分析", "アナリスト", "strategist", "戦略"], "openai/gpt-4.1"),
    # Default worker — cheap
    (["worker", "assistant", "アシスタント"], "openai/gpt-4.1-mini"),
]


def select_model_for_role(role: str, fallback_model: str) -> str:
    """ロール名に基づいてサブエージェント用モデルを選択する.

    Args:
        role: サブエージェントの役割名
        fallback_model: マッチしなかった場合のフォールバックモデル

    Returns:
        選択されたモデルID
    """
    role_lower = role.lower()
    for keywords, model_id in ROLE_MODEL_MAP:
        for kw in keywords:
            if kw in role_lower:
                return model_id
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

    # 料金の安い順（input + output）でソート
    sorted_catalog = sorted(
        catalog,
        key=lambda m: m.input_price_per_1k + m.output_price_per_1k,
    )

    # max_modelsで制限
    limited = sorted_catalog[:max_models]

    lines = ["利用可能なモデル一覧（料金順）:"]
    for m in limited:
        cats = f" [{', '.join(m.categories)}]" if m.categories else ""
        lines.append(
            f"- {m.model_id} (${m.input_price_per_1k:.2f}/${m.output_price_per_1k:.2f} per 1K tokens){cats}"
        )
    return "\n".join(lines)

