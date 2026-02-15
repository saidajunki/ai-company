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

from models import ModelPricing, PricingCache


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
