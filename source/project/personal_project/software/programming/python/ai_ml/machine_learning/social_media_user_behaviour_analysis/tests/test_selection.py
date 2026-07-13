import pandas as pd
import pytest

from smuba.features.selection import select_features


def test_select_features_returns_X_and_y():
    df = pd.DataFrame({"age": [20, 30], "usage": [100, 200], "target": [0, 1]})
    X, y = select_features(df, {"target_column": "target", "selected_features": ["age", "usage"]})

    assert list(X.columns) == ["age", "usage"]
    assert list(y) == [0, 1]


def test_select_features_raises_on_missing_column():
    df = pd.DataFrame({"age": [20, 30], "target": [0, 1]})
    with pytest.raises(ValueError):
        select_features(df, {"target_column": "target", "selected_features": ["age", "usage"]})
