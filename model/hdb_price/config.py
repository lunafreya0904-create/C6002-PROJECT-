from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML and attach resolved project paths."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = path.parent.parent
    project = config["project"]
    project["root_dir"] = str(root)
    for key in ("data_path", "artifacts_dir", "reports_dir"):
        candidate = Path(project[key])
        if not candidate.is_absolute():
            candidate = root / candidate
        project[key] = str(candidate.resolve())
    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create model and report output directories."""
    Path(config["project"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["project"]["reports_dir"]).mkdir(parents=True, exist_ok=True)

