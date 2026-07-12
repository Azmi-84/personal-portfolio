import pytest

from smuba.models.factory import get_estimator_class, get_search_space

MODEL_CFG = {
    "random_forest": {
        "classification": "sklearn.ensemble.RandomForestClassifier",
        "regression": "sklearn.ensemble.RandomForestRegressor",
        "search_space": {"n_estimators": [100, 300]},
    }
}


def test_get_estimator_class_resolves_classification():
    cls = get_estimator_class("random_forest", "classification", MODEL_CFG)
    assert cls.__name__ == "RandomForestClassifier"


def test_get_estimator_class_unknown_model_raises():
    with pytest.raises(ValueError):
        get_estimator_class("not_a_model", "classification", MODEL_CFG)


def test_get_search_space():
    assert get_search_space("random_forest", MODEL_CFG) == {"n_estimators": [100, 300]}
