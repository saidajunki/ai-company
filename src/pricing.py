"""OpenRouter pricing cache and fallback logic.

Provides read/write for the pricing cache file and fallback handling
when pricing data is unavailable for a model.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from models import ModelPricing, PricingCache

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def pricing_cache_path(base_dir: Path, company_id: str) -> Path:
    """Return ``companies/<company_id>/pricing/openrouter.json``."""
    return base_dir / "companies" / company_id / "pricing" / "openrouter.json"


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_pricing_cache(path: Path, cache: PricingCache) -> None:
    """Write *cache* as JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cache.model_dump_json(indent=2), encoding="utf-8")


def load_pricing_cache(path: Path) -> Optional[PricingCache]:
    """Read a :class:`PricingCache` from *path*.

    Returns ``None`` when the file does not exist or contains invalid JSON.
    """
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return PricingCache.model_validate_json(raw)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fetch (startup / unknown-model refresh)
# ---------------------------------------------------------------------------

def fetch_openrouter_pricing_cache(
    *,
    api_key: str | None = None,
    timeout_s: float = 20.0,
) -> PricingCache:
    """Fetch a fresh pricing snapshot from OpenRouter models API."""
    headers = {
        "User-Agent": "ai-company/0.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()

    now = datetime.now(timezone.utc)
    models: dict[str, ModelPricing] = {}

    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        pricing = item.get("pricing") or {}
        if not model_id or not isinstance(pricing, dict):
            continue

        prompt = pricing.get("prompt")
        completion = pricing.get("completion")
        if prompt is None or completion is None:
            continue

        try:
            prompt_per_token = float(prompt)
            completion_per_token = float(completion)
        except (TypeError, ValueError):
            continue

        models[str(model_id)] = ModelPricing(
            input_price_per_1k=prompt_per_token * 1000.0,
            output_price_per_1k=completion_per_token * 1000.0,
            retrieved_at=now,
        )

    return PricingCache(
        retrieved_at=now,
        models=models,
    )


def refresh_openrouter_pricing_cache(
    path: Path,
    *,
    api_key: str | None = None,
    max_age_hours: float = 24.0,
    force: bool = False,
) -> Optional[PricingCache]:
    """Refresh pricing cache if missing or stale.

    Returns the newest available cache (freshly fetched or existing), or None.
    """
    now = datetime.now(timezone.utc)
    existing = load_pricing_cache(path)

    if not force and existing is not None:
        age_hours = (now - existing.retrieved_at).total_seconds() / 3600.0
        if age_hours <= max_age_hours:
            return existing

    try:
        fresh = fetch_openrouter_pricing_cache(api_key=api_key)
        save_pricing_cache(path, fresh)
        return fresh
    except Exception:
        # Network/API failure → keep existing if available
        return existing

# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------

def get_model_pricing(
    cache: Optional[PricingCache],
    model_name: str,
) -> Optional[ModelPricing]:
    """Look up pricing for *model_name* in *cache*.

    Returns ``None`` when *cache* is ``None`` or the model is not present.
    """
    if cache is None:
        return None
    return cache.models.get(model_name)


# ---------------------------------------------------------------------------
# Fallback (Req 9.3)
# ---------------------------------------------------------------------------

def get_pricing_with_fallback(
    cache: Optional[PricingCache],
    model_name: str,
    *,
    fallback_input: float = 0.01,
    fallback_output: float = 0.03,
    previous_cache: Optional[PricingCache] = None,
) -> tuple[ModelPricing, str]:
    """Return ``(pricing, source)`` for *model_name*.

    Resolution order:

    1. Current cache hit → source ``"cache"``
    2. Previous cache hit → source ``"fallback_previous"``
    3. Default fallback prices → source ``"fallback_default"``

    The *source* string is intended to be recorded in the Ledger as the
    fallback rationale (Req 9.3).
    """
    # 1. Try current cache
    pricing = get_model_pricing(cache, model_name)
    if pricing is not None:
        return pricing, "cache"

    # 2. Try previous cache
    prev_pricing = get_model_pricing(previous_cache, model_name)
    if prev_pricing is not None:
        return prev_pricing, "fallback_previous"

    # 3. Default fallback
    now = datetime.now(timezone.utc)
    fallback = ModelPricing(
        input_price_per_1k=fallback_input,
        output_price_per_1k=fallback_output,
        retrieved_at=now,
    )
    return fallback, "fallback_default"
