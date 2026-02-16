"""Unit tests for pricing cache read/write and fallback logic."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from typing import Dict, Optional

from models import ModelPricing, PricingCache
from pricing import (
    get_model_pricing,
    get_pricing_with_fallback,
    load_pricing_cache,
    pricing_cache_path,
    save_pricing_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minute: int = 0) -> datetime:
    return datetime(2025, 7, 1, 12, minute, 0, tzinfo=timezone.utc)


def _model_pricing(**kw) -> ModelPricing:
    defaults = dict(input_price_per_1k=0.005, output_price_per_1k=0.015, retrieved_at=_ts())
    defaults.update(kw)
    return ModelPricing(**defaults)


def _cache(models: Optional[Dict[str, ModelPricing]] = None) -> PricingCache:
    return PricingCache(
        retrieved_at=_ts(),
        models=models or {},
    )


# ---------------------------------------------------------------------------
# pricing_cache_path
# ---------------------------------------------------------------------------

class TestPricingCachePath:
    def test_returns_expected_path(self, tmp_path: Path):
        result = pricing_cache_path(tmp_path, "acme")
        expected = tmp_path / "companies" / "acme" / "pricing" / "openrouter.json"
        assert result == expected


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_round_trip_empty_models(self, tmp_path: Path):
        path = pricing_cache_path(tmp_path, "co1")
        cache = _cache()
        save_pricing_cache(path, cache)
        loaded = load_pricing_cache(path)
        assert loaded is not None
        assert loaded.retrieved_at == cache.retrieved_at
        assert loaded.models == {}

    def test_round_trip_with_models(self, tmp_path: Path):
        path = pricing_cache_path(tmp_path, "co1")
        mp = _model_pricing(input_price_per_1k=0.01, output_price_per_1k=0.02)
        cache = _cache({"gpt-4o": mp})
        save_pricing_cache(path, cache)
        loaded = load_pricing_cache(path)
        assert loaded is not None
        assert "gpt-4o" in loaded.models
        assert loaded.models["gpt-4o"].input_price_per_1k == 0.01
        assert loaded.models["gpt-4o"].output_price_per_1k == 0.02

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = pricing_cache_path(tmp_path, "new-co")
        save_pricing_cache(path, _cache())
        assert path.exists()


# ---------------------------------------------------------------------------
# load_pricing_cache – edge cases
# ---------------------------------------------------------------------------

class TestLoadPricingCache:
    def test_missing_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        assert load_pricing_cache(path) is None

    def test_invalid_json_returns_none(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all", encoding="utf-8")
        assert load_pricing_cache(path) is None

    def test_invalid_schema_returns_none(self, tmp_path: Path):
        path = tmp_path / "bad_schema.json"
        path.write_text('{"foo": "bar"}', encoding="utf-8")
        assert load_pricing_cache(path) is None


# ---------------------------------------------------------------------------
# get_model_pricing
# ---------------------------------------------------------------------------

class TestGetModelPricing:
    def test_returns_pricing_when_present(self):
        mp = _model_pricing()
        cache = _cache({"claude-3": mp})
        assert get_model_pricing(cache, "claude-3") == mp

    def test_returns_none_when_model_absent(self):
        cache = _cache({"claude-3": _model_pricing()})
        assert get_model_pricing(cache, "gpt-4o") is None

    def test_returns_none_when_cache_is_none(self):
        assert get_model_pricing(None, "gpt-4o") is None

    def test_returns_none_for_empty_cache(self):
        assert get_model_pricing(_cache(), "gpt-4o") is None


# ---------------------------------------------------------------------------
# get_pricing_with_fallback (Req 9.3)
# ---------------------------------------------------------------------------

class TestGetPricingWithFallback:
    def test_cache_hit(self):
        mp = _model_pricing(input_price_per_1k=0.005)
        cache = _cache({"gpt-4o": mp})
        pricing, source = get_pricing_with_fallback(cache, "gpt-4o")
        assert source == "cache"
        assert pricing.input_price_per_1k == 0.005

    def test_fallback_to_previous_cache(self):
        current = _cache()  # empty
        prev_mp = _model_pricing(input_price_per_1k=0.007)
        prev = _cache({"gpt-4o": prev_mp})
        pricing, source = get_pricing_with_fallback(
            current, "gpt-4o", previous_cache=prev,
        )
        assert source == "fallback_previous"
        assert pricing.input_price_per_1k == 0.007

    def test_fallback_default_when_no_cache(self):
        pricing, source = get_pricing_with_fallback(None, "gpt-4o")
        assert source == "fallback_default"
        assert pricing.input_price_per_1k == 0.01
        assert pricing.output_price_per_1k == 0.03

    def test_fallback_default_when_model_missing(self):
        cache = _cache({"claude-3": _model_pricing()})
        pricing, source = get_pricing_with_fallback(cache, "unknown-model")
        assert source == "fallback_default"

    def test_custom_fallback_values(self):
        pricing, source = get_pricing_with_fallback(
            None, "x", fallback_input=0.05, fallback_output=0.10,
        )
        assert source == "fallback_default"
        assert pricing.input_price_per_1k == 0.05
        assert pricing.output_price_per_1k == 0.10

    def test_fallback_has_retrieved_at(self):
        """Fallback pricing includes retrieved_at (Req 9.4)."""
        pricing, _ = get_pricing_with_fallback(None, "x")
        assert pricing.retrieved_at is not None

    def test_current_cache_takes_priority_over_previous(self):
        """Current cache is preferred even when previous cache also has the model."""
        current_mp = _model_pricing(input_price_per_1k=0.001)
        prev_mp = _model_pricing(input_price_per_1k=0.999)
        current = _cache({"m": current_mp})
        prev = _cache({"m": prev_mp})
        pricing, source = get_pricing_with_fallback(
            current, "m", previous_cache=prev,
        )
        assert source == "cache"
        assert pricing.input_price_per_1k == 0.001
