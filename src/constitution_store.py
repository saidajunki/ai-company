"""Constitution YAML read/write module.

Provides serialization/deserialization of ConstitutionModel to/from YAML files.
Standard path: companies/<company_id>/constitution.yaml

Requirements: 1.2, 1.4
"""

from __future__ import annotations

from pathlib import Path

import yaml

from models import ConstitutionModel


def get_constitution_path(base_dir: Path, company_id: str) -> Path:
    """Return the standard constitution file path.

    Path format: base_dir/companies/<company_id>/constitution.yaml
    """
    return base_dir / "companies" / company_id / "constitution.yaml"


def constitution_save(path: Path, constitution: ConstitutionModel) -> None:
    """Serialize ConstitutionModel to YAML and write to file.

    Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = constitution.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def constitution_load(path: Path) -> ConstitutionModel:
    """Read YAML file and deserialize into ConstitutionModel.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Constitution file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConstitutionModel.model_validate(data)
