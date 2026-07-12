"""The single entry point for BOTH initial training and retraining.

Retraining isn't a separate system - it's this same pipeline run again on
new data with --promote set, so a "champion" (current Production model in
the MLflow registry) is compared against the "challenger" (freshly trained
model) and only replaces it if it wins on `promotion_metric`.

Usage:
    smuba-train --data data/raw/social_media_user_behaviour.csv --model random_forest --promote
    smuba-train --data data/raw/new_batch_2026_08.csv --model random_forest --promote   # retraining
"""
import argparse

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from smuba.config import load_config
from smuba.data.clean import clean_data
from smuba.data.ingest import load_raw_data
from smuba.data.validate import validate_schema
from smuba.features.selection import select_features
from smuba.models.evaluate import compute_metrics
from smuba.models.tune import tune_hyperparameters
from smuba.utils.logger import get_logger

logger = get_logger(__name__)


def _current_production_metric(client: MlflowClient, registered_model_name: str, metric_name: str):
    """Returns the promotion metric of whatever is currently in the
    'Production' stage, or None if nothing has been promoted yet."""
    try:
        versions = client.get_latest_versions(registered_model_name, stages=["Production"])
    except Exception:
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get(metric_name)


def _maybe_promote(pipe_cfg: dict, new_metrics: dict) -> None:
    client = MlflowClient()
    name = pipe_cfg["registered_model_name"]
    metric_name = pipe_cfg["promotion_metric"]
    lower_is_better = pipe_cfg["lower_is_better"]

    current_best = _current_production_metric(client, name, metric_name)
    new_value = new_metrics[metric_name]

    should_promote = (
        current_best is None
        or (lower_is_better and new_value < current_best)
        or (not lower_is_better and new_value > current_best)
    )

    latest = client.get_latest_versions(name, stages=["None"])[0]

    if should_promote:
        client.transition_model_version_stage(
            name=name, version=latest.version, stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Promoted version %s to Production (%s: %s -> %.4f)",
            latest.version, metric_name,
            f"{current_best:.4f}" if current_best is not None else "none yet",
            new_value,
        )
    else:
        logger.info(
            "Challenger did NOT beat Production (%s: %.4f vs current %.4f). "
            "New version stays unstaged for manual review.",
            metric_name, new_value, current_best,
        )


def run(
    data_path: str,
    model_name: str,
    config_dir: str = "configs",
    use_pca=None,
    n_trials=None,
    promote: bool = False,
) -> str:
    cfg = load_config(config_dir)
    data_cfg, model_cfg, pipe_cfg = cfg["data"], cfg["models"], cfg["pipeline"]

    mlflow.set_tracking_uri(pipe_cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(pipe_cfg["experiment_name"])

    use_pca = pipe_cfg["use_pca"] if use_pca is None else use_pca
    n_trials = pipe_cfg["n_optuna_trials"] if n_trials is None else n_trials
    task_type = data_cfg["task_type"]

    logger.info("Loading data from %s", data_path)
    df = load_raw_data(data_path)
    validate_schema(df, data_cfg)
    df = clean_data(df, data_cfg)
    X, y = select_features(df, data_cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"],
    )

    with mlflow.start_run(run_name=f"{model_name}{'_pca' if use_pca else ''}") as run_ctx:
        mlflow.log_params({
            "model_name": model_name,
            "use_pca": use_pca,
            "task_type": task_type,
            "n_train_rows": len(X_train),
            "n_features": X_train.shape[1],
            "data_path": data_path,
        })

        best_params, pipeline = tune_hyperparameters(
            X_train, y_train, model_name=model_name, model_cfg=model_cfg,
            task_type=task_type, use_pca=use_pca,
            pca_variance_threshold=pipe_cfg["pca_variance_threshold"],
            n_trials=n_trials, random_state=data_cfg["random_state"],
        )
        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, task_type)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline, artifact_path="model",
            registered_model_name=pipe_cfg["registered_model_name"],
        )

        run_id = run_ctx.info.run_id
        logger.info("Run %s finished. Metrics: %s", run_id, metrics)

    if promote:
        _maybe_promote(pipe_cfg, metrics)

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Train or retrain a SMUBA model.")
    parser.add_argument("--data", required=True, help="Path to CSV data (initial or new batch).")
    parser.add_argument("--model", required=True, help="Model key from configs/model_config.yaml")
    parser.add_argument("--use-pca", action="store_true", default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--promote", action="store_true",
                         help="Compare the new model against current Production and promote if it wins.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()

    run(
        data_path=args.data, model_name=args.model, config_dir=args.config_dir,
        use_pca=args.use_pca, n_trials=args.n_trials, promote=args.promote,
    )


if __name__ == "__main__":
    main()
