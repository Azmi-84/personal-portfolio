"""Generic Optuna-based hyperparameter search - one function that works
for ANY model_name in model_config.yaml, replacing the 12 separate
hyperparameters_optimization.py files.
"""
import optuna
from sklearn.model_selection import cross_val_score

from smuba.features.pipeline_builder import build_pipeline
from smuba.models.factory import get_estimator_class, get_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        if isinstance(spec, list) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))
        elif isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        else:
            params[name] = spec
    return params


def _instantiate(estimator_cls, params: dict, random_state: int):
    supports_seed = "random_state" in estimator_cls().get_params()
    return estimator_cls(**params, random_state=random_state) if supports_seed else estimator_cls(**params)


def tune_hyperparameters(
    X_train, y_train, model_name: str, model_cfg: dict, task_type: str,
    use_pca: bool, pca_variance_threshold: float, n_trials: int, random_state: int,
):
    estimator_cls = get_estimator_class(model_name, task_type, model_cfg)
    search_space = get_search_space(model_name, model_cfg)
    scoring = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, search_space)
        estimator = _instantiate(estimator_cls, params, random_state)
        pipeline = build_pipeline(X_train, estimator, use_pca, pca_variance_threshold)
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_estimator = _instantiate(estimator_cls, best_params, random_state)
    best_pipeline = build_pipeline(X_train, best_estimator, use_pca, pca_variance_threshold)
    return best_params, best_pipeline
