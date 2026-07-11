import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def _(np):
    np.random.seed(42)
    return


@app.cell
def _(pd):
    raw_dataset = pd.read_csv("./01_Data/01_Raw_Data/instagram_usage_lifestyle.csv")
    return (raw_dataset,)


@app.cell
def _(raw_dataset):
    sample_raw_dataset = raw_dataset.sample(frac=0.001, random_state=42).reset_index(drop=True)
    return (sample_raw_dataset,)


@app.cell
def _(sample_raw_dataset):
    sample_raw_dataset
    return


if __name__ == "__main__":
    app.run()
