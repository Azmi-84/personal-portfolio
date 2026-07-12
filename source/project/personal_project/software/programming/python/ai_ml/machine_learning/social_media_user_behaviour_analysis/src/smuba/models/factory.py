"""Resolves a model_name string (from configs/model_config.yaml) to an
actual estimator class. This is what replaces having a separate folder
per algorithm - adding a 7th algorithm later means adding a YAML entry,
not a new directory with 2 new files.
"""
import importlib


def _resolve_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_estimator_class(model_name: str, task_type: str, model_cfg: dict):
    try:
        entry = model_cfg[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Available: {list(model_cfg.keys())}"
        ) from exc
    return _resolve_class(entry[task_type])


def get_search_space(model_name: str, model_cfg: dict) -> dict:
    return model_cfg[model_name]["search_space"]
