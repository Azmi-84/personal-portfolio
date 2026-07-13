"""Turns the human decision recorded in configs/data_config.yaml
(selected_features, informed by the fg-data-profiling report and written
up in docs/eda_decisions.md) into the actual X/y split used for training.
"""
import pandas as pd


def select_features(df: pd.DataFrame, data_cfg: dict):
    target = data_cfg["target_column"]
    selected = data_cfg["selected_features"]

    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(
            f"selected_features in data_config.yaml reference columns not in the data: {missing}"
        )

    X = df[selected].copy()
    y = df[target].copy()
    return X, y
