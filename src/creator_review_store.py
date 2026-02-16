"""Creator review store backed by NDJSON storage."""

from __future__ import annotations

from pathlib import Path

from models import CreatorReview
from ndjson_store import ndjson_append, ndjson_read


class CreatorReviewStore:
    """Creatorスコアの永続化と取得を担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        self._path = base_dir / "companies" / company_id / "state" / "creator_reviews.ndjson"

    def save(self, review: CreatorReview) -> None:
        ndjson_append(self._path, review)

    def load_all(self) -> list[CreatorReview]:
        return ndjson_read(self._path, CreatorReview)

    def recent(self, limit: int = 5) -> list[CreatorReview]:
        reviews = ndjson_read(self._path, CreatorReview)
        return reviews[-limit:]

    def latest(self) -> CreatorReview | None:
        reviews = self.recent(limit=1)
        return reviews[0] if reviews else None

