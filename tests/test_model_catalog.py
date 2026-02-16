"""model_catalog モジュールのユニットテスト."""

from datetime import datetime, timezone

from model_catalog import (
    DEFAULT_MODEL_CATEGORIES,
    ModelInfo,
    build_model_catalog,
)
from models import ModelPricing, PricingCache


def _make_pricing(input_p: float = 0.5, output_p: float = 1.0) -> ModelPricing:
    return ModelPricing(
        input_price_per_1k=input_p,
        output_price_per_1k=output_p,
        retrieved_at=datetime.now(tz=timezone.utc),
    )


def _make_cache(models: dict[str, ModelPricing]) -> PricingCache:
    return PricingCache(
        retrieved_at=datetime.now(tz=timezone.utc),
        models=models,
    )


# --- build_model_catalog with valid PricingCache ---


class TestBuildModelCatalogValid:
    def test_returns_correct_number_of_models(self):
        cache = _make_cache(
            {
                "anthropic/claude-sonnet-4-20250514": _make_pricing(),
                "google/gemini-2.5-flash-preview": _make_pricing(),
                "openai/gpt-4o": _make_pricing(),
            }
        )
        catalog = build_model_catalog(cache)
        assert len(catalog) == 3

    def test_each_model_has_correct_pricing(self):
        cache = _make_cache(
            {
                "anthropic/claude-sonnet-4-20250514": _make_pricing(0.3, 1.5),
                "google/gemini-2.5-flash-preview": _make_pricing(0.1, 0.2),
            }
        )
        catalog = build_model_catalog(cache)
        by_id = {m.model_id: m for m in catalog}

        claude = by_id["anthropic/claude-sonnet-4-20250514"]
        assert claude.input_price_per_1k == 0.3
        assert claude.output_price_per_1k == 1.5

        gemini = by_id["google/gemini-2.5-flash-preview"]
        assert gemini.input_price_per_1k == 0.1
        assert gemini.output_price_per_1k == 0.2

    def test_known_models_get_categories_via_prefix_match(self):
        cache = _make_cache(
            {
                "anthropic/claude-sonnet-4-20250514": _make_pricing(),
                "google/gemini-2.5-flash-preview": _make_pricing(),
                "deepseek/deepseek-chat-v3": _make_pricing(),
            }
        )
        catalog = build_model_catalog(cache)
        by_id = {m.model_id: m for m in catalog}

        assert by_id["anthropic/claude-sonnet-4-20250514"].categories == ["coding", "analysis"]
        assert by_id["google/gemini-2.5-flash-preview"].categories == ["fast", "cheap", "general"]
        assert by_id["deepseek/deepseek-chat-v3"].categories == ["coding", "cheap"]

    def test_unknown_models_get_empty_categories(self):
        cache = _make_cache(
            {
                "meta/llama-3-70b": _make_pricing(),
                "mistral/mixtral-8x7b": _make_pricing(),
            }
        )
        catalog = build_model_catalog(cache)
        for model in catalog:
            assert model.categories == []


# --- None / empty PricingCache ---


class TestBuildModelCatalogEmpty:
    def test_none_pricing_cache_returns_empty(self):
        assert build_model_catalog(None) == []

    def test_empty_models_returns_empty(self):
        cache = _make_cache({})
        assert build_model_catalog(cache) == []


# --- custom known_categories override ---


class TestBuildModelCatalogCustomCategories:
    def test_custom_categories_override_defaults(self):
        cache = _make_cache({"custom/model-v1": _make_pricing()})
        custom = {"custom/model": ["special"]}
        catalog = build_model_catalog(cache, known_categories=custom)
        assert catalog[0].categories == ["special"]


# --- format_model_catalog_for_prompt ---

from model_catalog import format_model_catalog_for_prompt


class TestFormatModelCatalogForPrompt:
    def test_empty_catalog_returns_empty_string(self):
        assert format_model_catalog_for_prompt([]) == ""

    def test_models_sorted_by_total_cost(self):
        catalog = [
            ModelInfo("expensive/model", 3.0, 15.0, ["coding"]),
            ModelInfo("cheap/model", 0.1, 0.2, ["fast"]),
            ModelInfo("mid/model", 1.0, 2.0, ["general"]),
        ]
        result = format_model_catalog_for_prompt(catalog)
        lines = result.strip().split("\n")
        # Header + 3 models
        assert len(lines) == 4
        assert "cheap/model" in lines[1]
        assert "mid/model" in lines[2]
        assert "expensive/model" in lines[3]

    def test_max_models_limits_output(self):
        catalog = [
            ModelInfo(f"model/{i}", float(i), float(i), [])
            for i in range(10)
        ]
        result = format_model_catalog_for_prompt(catalog, max_models=3)
        lines = result.strip().split("\n")
        # Header + 3 models
        assert len(lines) == 4

    def test_categories_displayed_correctly(self):
        catalog = [
            ModelInfo("google/gemini-2.5-flash", 0.10, 0.20, ["fast", "cheap", "general"]),
        ]
        result = format_model_catalog_for_prompt(catalog)
        assert "[fast, cheap, general]" in result

    def test_model_with_no_categories(self):
        catalog = [
            ModelInfo("unknown/model", 0.5, 1.0, []),
        ]
        result = format_model_catalog_for_prompt(catalog)
        assert "unknown/model" in result
        assert "[$" not in result  # no brackets for empty categories

    def test_header_present(self):
        catalog = [ModelInfo("a/b", 0.1, 0.2, [])]
        result = format_model_catalog_for_prompt(catalog)
        assert result.startswith("利用可能なモデル一覧（料金順）:")

    def test_price_formatting(self):
        catalog = [ModelInfo("test/model", 0.1, 0.2, [])]
        result = format_model_catalog_for_prompt(catalog)
        assert "$0.10/$0.20 per 1K tokens" in result
