"""Unit tests for VisionLoader."""

from pathlib import Path

import pytest

from vision_loader import DEFAULT_VISION, VisionLoader


@pytest.fixture
def loader(tmp_path: Path) -> VisionLoader:
    return VisionLoader(base_dir=tmp_path, company_id="test-co")


class TestLoadFromFile:
    def test_returns_file_content_when_exists(self, tmp_path: Path):
        vision_dir = tmp_path / "companies" / "test-co"
        vision_dir.mkdir(parents=True)
        vision_file = vision_dir / "vision.md"
        vision_file.write_text("カスタムビジョン", encoding="utf-8")

        loader = VisionLoader(base_dir=tmp_path, company_id="test-co")
        assert loader.load() == "カスタムビジョン"

    def test_preserves_multiline_content(self, tmp_path: Path):
        vision_dir = tmp_path / "companies" / "test-co"
        vision_dir.mkdir(parents=True)
        content = "行1\n行2\n行3"
        (vision_dir / "vision.md").write_text(content, encoding="utf-8")

        loader = VisionLoader(base_dir=tmp_path, company_id="test-co")
        assert loader.load() == content


class TestDefaultVision:
    def test_returns_default_when_file_missing(self, loader: VisionLoader):
        assert loader.load() == DEFAULT_VISION

    def test_default_contains_shell_policy(self, loader: VisionLoader):
        result = loader.load()
        assert "シェルで完結しやすい活動に寄せてください" in result

    def test_default_contains_activity_examples(self, loader: VisionLoader):
        result = loader.load()
        assert "OSS" in result
        assert "情報収集" in result
        assert "継続的改善" in result

    def test_default_contains_ceo_role(self, loader: VisionLoader):
        result = loader.load()
        assert "社長AI" in result


class TestFilePath:
    def test_uses_correct_path(self, tmp_path: Path):
        loader = VisionLoader(base_dir=tmp_path, company_id="my-corp")
        expected = tmp_path / "companies" / "my-corp" / "vision.md"
        assert loader._path == expected
