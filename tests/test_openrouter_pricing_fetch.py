"""Unit tests for OpenRouter pricing fetch/refresh."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import respx

from pricing import (
    OPENROUTER_MODELS_URL,
    fetch_openrouter_pricing_cache,
    refresh_openrouter_pricing_cache,
)


class TestFetchOpenRouterPricingCache:
    @respx.mock
    def test_parses_models_pricing_per_1k(self):
        respx.get(OPENROUTER_MODELS_URL).respond(
            200,
            json={
                "data": [
                    {
                        "id": "vendor/model-a",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    },
                    {
                        "id": "vendor/model-b",
                        "pricing": {"prompt": "0.0", "completion": "0.0"},
                    },
                ]
            },
        )

        cache = fetch_openrouter_pricing_cache(timeout_s=5.0)
        assert "vendor/model-a" in cache.models
        mp = cache.models["vendor/model-a"]
        assert mp.input_price_per_1k == 0.001  # 0.000001 * 1000
        assert mp.output_price_per_1k == 0.002  # 0.000002 * 1000

        assert "vendor/model-b" in cache.models


class TestRefreshOpenRouterPricingCache:
    @respx.mock
    def test_refresh_writes_cache_file(self, tmp_path: Path):
        respx.get(OPENROUTER_MODELS_URL).respond(
            200,
            json={
                "data": [
                    {
                        "id": "m",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    }
                ]
            },
        )
        path = tmp_path / "openrouter.json"

        cache = refresh_openrouter_pricing_cache(path, force=True)
        assert cache is not None
        assert path.exists()

    @respx.mock
    def test_refresh_keeps_existing_on_failure(self, tmp_path: Path):
        # Write an existing cache first
        existing = (
            '{"retrieved_at":"2025-01-01T00:00:00+00:00","models":{"m":{"input_price_per_1k":0.1,"output_price_per_1k":0.2,"retrieved_at":"2025-01-01T00:00:00+00:00"}}}'
        )
        path = tmp_path / "openrouter.json"
        path.write_text(existing, encoding="utf-8")

        # Simulate API failure
        respx.get(OPENROUTER_MODELS_URL).respond(500, text="oops")

        cache = refresh_openrouter_pricing_cache(path, force=True)
        assert cache is not None
        assert "m" in cache.models
        assert cache.models["m"].input_price_per_1k == 0.1

