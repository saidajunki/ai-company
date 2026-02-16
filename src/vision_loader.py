"""Vision document loader.

Reads the company vision from a markdown file, falling back to a default
vision text when the file does not exist.

Requirements: 2.1, 2.2, 2.3, 2.4
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_VISION = """\
あなたは研究開発中心のAI組織の社長AIです。
シェルで完結しやすい活動に寄せてください。
活動例: OSS/プロトタイプの作成・検証・公開、情報収集→要約→公開、継続的改善"""


class VisionLoader:
    """ビジョンドキュメントの読み込みを担当する."""

    def __init__(self, base_dir: Path, company_id: str) -> None:
        """ビジョンファイルパス: companies/<id>/vision.md"""
        self._path = base_dir / "companies" / company_id / "vision.md"

    def load(self) -> str:
        """ビジョンテキストを返す。ファイルがなければデフォルトを返す."""
        if self._path.exists():
            return self._path.read_text(encoding="utf-8")
        return DEFAULT_VISION
