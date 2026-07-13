"""Explicit, documented retraining entry point.

Retraining IS the training pipeline, run again on new data with --promote
set - this module exists so that workflow has its own obvious name/command
(smuba-retrain) instead of being buried inside a generic --promote flag
someone might forget to pass.

Example:
    smuba-retrain --data data/raw/new_batch_2026_08.csv --model random_forest
"""
import argparse

from smuba.pipelines.training_pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the SMUBA model on new data; promotes automatically if it wins."
    )
    parser.add_argument("--data", required=True, help="Path to the new/updated CSV data.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--use-pca", action="store_true", default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()

    run(
        data_path=args.data, model_name=args.model, config_dir=args.config_dir,
        use_pca=args.use_pca, n_trials=args.n_trials, promote=True,
    )


if __name__ == "__main__":
    main()
