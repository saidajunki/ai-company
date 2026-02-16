"""Artifact verification for false-completion prevention.

タスク完了時に、LLMが主張するファイル・ディレクトリの実在を検証する。
会話ログと <done> 結果からファイルパスを抽出し、os.path.exists で確認する。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# 対象とする拡張子
_KNOWN_EXTENSIONS = (
    ".py", ".txt", ".json", ".yaml", ".yml", ".md", ".sh",
    ".js", ".ts", ".html", ".css", ".csv", ".xml", ".toml",
    ".cfg", ".ini", ".log", ".mp3", ".wav", ".png", ".jpg",
    ".gif", ".pdf", ".zip", ".tar", ".gz",
)

# 拡張子パターン（正規表現用）
_EXT_PATTERN = "|".join(re.escape(ext) for ext in _KNOWN_EXTENSIONS)

# Unix 絶対パス: /foo/bar.py  /etc/config  (URLは除外)
_ABS_PATH_RE = re.compile(
    r"(?<![a-zA-Z0-9_/])(/(?:[a-zA-Z0-9_.\-]+/)*[a-zA-Z0-9_.\-]+(?:" + _EXT_PATTERN + r"))"
)

# ./ で始まる相対パス: ./src/main.py
_DOT_SLASH_RE = re.compile(
    r"(?<![a-zA-Z0-9_])(\.\/(?:[a-zA-Z0-9_.\-]+\/)*[a-zA-Z0-9_.\-]+)"
)

# 拡張子付き相対パス: src/main.py, config.yaml
_REL_PATH_RE = re.compile(
    r"(?<![a-zA-Z0-9_/.\-])([a-zA-Z0-9_][a-zA-Z0-9_.\-]*(?:/[a-zA-Z0-9_.\-]+)*(?:"
    + _EXT_PATTERN
    + r"))"
)

# URL パターン（除外用）
_URL_RE = re.compile(r"https?://\S+")

# バージョン番号パターン（除外用）: Python 3.12, v1.2.3
_VERSION_RE = re.compile(
    r"(?:[vV]?\d+\.\d+(?:\.\d+)*)"
)

# スコア形式パターン（除外用）: 0.5/1.0, 3/5
_SCORE_RE = re.compile(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?")


@dataclass
class ArtifactVerificationResult:
    """成果物検証の結果."""

    verified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def all_exist(self) -> bool:
        """全ての成果物が存在するか."""
        return len(self.missing) == 0


class ArtifactVerifier:
    """タスク成果物の実在を検証する."""

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir

    def extract_file_paths(self, text: str) -> list[str]:
        """テキストからファイルパスを抽出する.

        Unix形式のパス（/で始まる、または./で始まる）と
        相対パス（拡張子付き）を抽出する。
        URLやバージョン番号などの誤検出は除外する。
        """
        # まず URL を除去してから抽出する
        cleaned = _URL_RE.sub("", text)

        paths: list[str] = []
        seen: set[str] = set()

        for pattern in (_ABS_PATH_RE, _DOT_SLASH_RE, _REL_PATH_RE):
            for match in pattern.finditer(cleaned):
                candidate = match.group(1)
                if candidate in seen:
                    continue
                if self._is_false_positive(candidate, cleaned, match.start(1)):
                    continue
                seen.add(candidate)
                paths.append(candidate)

        return paths

    def verify(self, paths: list[str]) -> ArtifactVerificationResult:
        """ファイルパスの実在を検証する.

        各パスに対して os.path.exists で確認。
        相対パスは work_dir を基準に解決する。
        """
        verified: list[str] = []
        missing: list[str] = []

        for p in paths:
            resolved = Path(p) if os.path.isabs(p) else self._work_dir / p
            if os.path.exists(resolved):
                verified.append(p)
            else:
                missing.append(p)

        return ArtifactVerificationResult(verified=verified, missing=missing)

    @staticmethod
    def _is_false_positive(candidate: str, text: str, start: int) -> bool:
        """誤検出かどうかを判定する."""
        # バージョン番号チェック: 数字.数字 のみで構成されるもの
        stripped = candidate.lstrip("/").lstrip("./")
        if _VERSION_RE.fullmatch(stripped):
            return True

        # スコア形式チェック: 候補がスコアパターンの一部かどうか
        # 前後のコンテキストを確認
        context_start = max(0, start - 10)
        context_end = min(len(text), start + len(candidate) + 10)
        context = text[context_start:context_end]
        if _SCORE_RE.search(context):
            # スコアパターンに含まれる場合は除外
            for m in _SCORE_RE.finditer(context):
                score_text = m.group()
                if candidate in score_text or stripped in score_text:
                    return True

        return False
