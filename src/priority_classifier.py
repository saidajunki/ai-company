"""PriorityClassifier — タスクの説明文と発生源から優先度を判定する.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations


class PriorityClassifier:
    """タスクの説明文と発生源から優先度を判定する."""

    # 緊急・重要キーワード → priority 2
    URGENT_KEYWORDS: list[str] = [
        "緊急", "障害", "修正", "セキュリティ", "バグ",
        "urgent", "critical", "fix", "security", "bug",
    ]

    # ジャンク・不要キーワード → priority 5
    JUNK_KEYWORDS: list[str] = [
        "ジョーク", "冗談", "遊び", "ネタ", "テスト投稿",
        "joke", "fun", "meme", "lol",
    ]

    @classmethod
    def classify(cls, description: str, source: str) -> int:
        """タスクの優先度を判定する.

        Args:
            description: タスクの説明文
            source: タスクの発生源

        Returns:
            優先度（1-5）
        """
        if source == "creator":
            return 1
        if source == "initiative":
            return 2

        desc_lower = description.lower()

        # ジャンク判定
        if any(kw in desc_lower for kw in cls.JUNK_KEYWORDS):
            return 5

        # 緊急判定
        if any(kw in desc_lower for kw in cls.URGENT_KEYWORDS):
            return 2

        # デフォルト
        return 3
