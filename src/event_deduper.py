from __future__ import annotations

import threading
from datetime import datetime, timezone


class EventDeduper:
    """Simple TTL-based event de-duplication helper (thread-safe)."""

    def __init__(self, *, ttl_seconds: int = 900) -> None:
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_process(self, key: str) -> bool:
        """Return True if key is new within TTL, otherwise False."""
        if not key:
            return True
        if self._ttl_seconds == 0:
            return True

        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - self._ttl_seconds

        with self._lock:
            if self._seen:
                old = [k for k, t in self._seen.items() if t < cutoff]
                for k in old:
                    self._seen.pop(k, None)

            if key in self._seen:
                return False
            self._seen[key] = now
            return True

