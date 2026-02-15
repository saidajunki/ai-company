"""Unit tests for constitution_store module.

Tests YAML serialization/deserialization of ConstitutionModel.
Requirements: 1.2, 1.4
"""

from pathlib import Path

import pytest

from constitution_store import constitution_load, constitution_save, get_constitution_path
from models import ConstitutionModel


class TestGetConstitutionPath:
    def test_returns_correct_path(self, tmp_path: Path) -> None:
        result = get_constitution_path(tmp_path, "acme-corp")
        assert result == tmp_path / "companies" / "acme-corp" / "constitution.yaml"

    def test_different_company_ids(self, tmp_path: Path) -> None:
        p1 = get_constitution_path(tmp_path, "alpha")
        p2 = get_constitution_path(tmp_path, "beta")
        assert p1 != p2
        assert "alpha" in str(p1)
        assert "beta" in str(p2)


class TestConstitutionSave:
    def test_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "constitution.yaml"
        constitution_save(path, ConstitutionModel())
        assert path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "dir" / "constitution.yaml"
        constitution_save(path, ConstitutionModel())
        assert path.exists()

    def test_file_is_valid_yaml(self, tmp_path: Path) -> None:
        import yaml

        path = tmp_path / "constitution.yaml"
        constitution_save(path, ConstitutionModel())
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "version" in data


class TestConstitutionLoad:
    def test_returns_correct_model(self, tmp_path: Path) -> None:
        path = tmp_path / "constitution.yaml"
        original = ConstitutionModel(version=5, purpose="テスト用")
        constitution_save(path, original)
        loaded = constitution_load(path)
        assert loaded.version == 5
        assert loaded.purpose == "テスト用"

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            constitution_load(path)


class TestRoundTrip:
    def test_default_model_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "constitution.yaml"
        original = ConstitutionModel()
        constitution_save(path, original)
        loaded = constitution_load(path)
        assert loaded == original

    def test_custom_model_round_trip(self, tmp_path: Path) -> None:
        from models import Budget

        path = tmp_path / "constitution.yaml"
        original = ConstitutionModel(
            version=3,
            purpose="カスタム目的",
            budget=Budget(limit_usd=20.0),
        )
        constitution_save(path, original)
        loaded = constitution_load(path)
        assert loaded == original
