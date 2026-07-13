"""Central config loader. Everything path- or hyperparameter-related lives
in configs/*.yaml, not hardcoded inside scripts - this is the single place
that reads them.
"""
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(config_dir: str = "configs") -> dict:
    config_dir = Path(config_dir)

    data_yaml = _load_yaml(config_dir / "data_config.yaml")
    model_yaml = _load_yaml(config_dir / "model_config.yaml")
    pipeline_yaml = _load_yaml(config_dir / "pipeline_config.yaml")

    data_cfg = {
        **data_yaml["data"],
        "selected_features": data_yaml["selected_features"],
        "dropped_features": data_yaml.get("dropped_features", []),
    }

    return {
        "data": data_cfg,
        "models": model_yaml["models"],
        "pipeline": pipeline_yaml,
    }
