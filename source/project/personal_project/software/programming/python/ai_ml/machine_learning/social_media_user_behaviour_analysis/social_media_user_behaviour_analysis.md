# Social Media User Behaviour Analysis

Industry-style ML pipeline: profile -> decide features -> validate/clean
-> train (with hyperparameter tuning) -> track & register in MLflow ->
serve -> retrain on new data via the same command.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

## Workflow

1. Put your raw CSV at `data/raw/social_media_user_behaviour.csv`.

2. Generate the EDA report:
   ```bash
   make profile
   ```
   Open `reports/eda/profiling_report.html`, decide which features to
   keep, write the reasoning in `docs/eda_decisions.md`, then set
   `selected_features` / `target_column` / `task_type` in
   `configs/data_config.yaml`.

3. Start the local MLflow server (tracking UI + model registry):
   ```bash
   make mlflow-server
   ```
   Leave this running; view runs at http://127.0.0.1:5000.

4. Train a model (also registers it, and promotes it to `Production` if
   it beats whatever's currently there):
   ```bash
   make train MODEL=random_forest
   ```
   Available `MODEL` values are the keys in `configs/model_config.yaml`:
   `random_forest`, `gradient_boosting`, `adaptive_boosting`,
   `k_nearest_neighbors`, `extreme_gradient_boosting`,
   `light_gradient_boosting_machine`.

   Toggle PCA by setting `use_pca: true` in `configs/pipeline_config.yaml`,
   or per-run with `smuba-train ... --use-pca`.

5. Serve the current Production model:
   ```bash
   make serve-gradio   # demo UI
   make serve-api      # REST API on :8000
   ```

6. **Retrain later**, whenever new labeled data shows up - same pipeline,
   new data, run manually:
   ```bash
   make retrain MODEL=random_forest DATA=data/raw/new_batch_2026_08.csv
   ```
   The new model is only promoted to `Production` if it beats the current
   one on `promotion_metric` (set in `configs/pipeline_config.yaml`). If it
   loses, it's still logged and registered - just not staged - so nothing
   is thrown away.

## Project layout

```
.
├── configs/                  # all paths, features, hyperparameters, MLflow settings
├── data/{raw,interim,processed}/
├── docs/                     # architecture notes + EDA decisions log
├── notebooks/                # exploration ONLY - nothing here is imported by the pipeline
├── reports/eda/              # fg-data-profiling HTML output
├── src/smuba/                # the actual installable package
│   ├── data/                 # ingest, validate, clean, profiling
│   ├── features/             # selection, pipeline_builder (preprocessing + optional PCA)
│   ├── models/                # factory (name -> estimator), tune (Optuna), evaluate
│   ├── pipelines/             # training_pipeline.py (= retrain.py's engine), retrain.py
│   └── serving/               # gradio_app.py, api.py
├── tests/                    # pytest
└── mlruns/, mlflow.db         # local MLflow tracking store + registry (gitignored)
```

## Tests & CI

```bash
make test
make lint
```
GitHub Actions (`.github/workflows/ci.yml`) runs both on every push/PR.

## TODOs before this actually runs on your data

- [ ] Set `target_column` and `task_type` in `configs/data_config.yaml`
- [ ] Fill in the real `selected_features` list after reviewing the
      profiling report
- [ ] Replace the placeholder pandera schema sketch in
      `src/smuba/data/validate.py` with your real column checks
- [ ] Add dataset-specific cleaning logic in `src/smuba/data/clean.py`
      (label normalization, value clipping, etc.)
