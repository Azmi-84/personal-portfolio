# Architecture

## Pipeline flow

1. **Profile** (`smuba-profile`) - fg-data-profiling generates
   `reports/eda/profiling_report.html` from raw data.
2. **Decide** (human step) - review the report, record feature
   include/exclude decisions and WHY in `docs/eda_decisions.md`, then
   reflect the final list in `configs/data_config.yaml` (`selected_features`).
3. **Validate** (`smuba.data.validate`) - schema/sanity checks run before
   every training or retraining pass. This is what makes step 5 safe to
   re-run on new data later without silently corrupting the model.
4. **Clean & select** (`smuba.data.clean`, `smuba.features.selection`) -
   deterministic, config-driven cleaning and column selection.
5. **Train** (`smuba-train`) - builds a single sklearn Pipeline
   (preprocessing -> optional PCA -> estimator), tunes hyperparameters
   with Optuna, evaluates on a held-out test set, and logs everything
   (params, metrics, the fitted pipeline) to MLflow.
6. **Register & promote** - the trained pipeline is registered as a new
   version of the `smuba-model` in the MLflow Model Registry. With
   `--promote`, it's automatically compared against whatever is currently
   in the `Production` stage on `promotion_metric`, and promoted only if
   it wins (champion/challenger pattern).
7. **Serve** (`smuba.serving.gradio_app` / `smuba.serving.api`) - both
   load whatever is currently staged `Production` directly from the
   registry, so promoting a model is the same action as "deploying" it.
8. **Retrain** (`smuba-retrain`) - literally step 5-6 run again on new
   data. There's no separate retraining system to maintain.

## Why MLflow instead of hand-rolled pickle versioning

- Every run's params/metrics/artifacts are queryable (`mlflow ui`), so you
  can answer "why did we pick this model" months later.
- The Model Registry's stages (`None` / `Staging` / `Production` /
  `Archived`) give you champion/challenger and rollback for free.
- It's a name recruiters and interviewers recognize.

## Why one `train.py`, not one per algorithm

Adding a 7th algorithm should mean adding one entry to
`configs/model_config.yaml`, not a new folder with 2 new files. The
`model_name` + `use_pca` combination is just config; the code path is
identical for every algorithm because they're all scikit-learn-compatible
estimators wrapped in the same `Pipeline`.
