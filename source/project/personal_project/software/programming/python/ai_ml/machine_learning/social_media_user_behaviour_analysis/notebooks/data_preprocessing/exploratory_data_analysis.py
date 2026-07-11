import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from data_profiling import ProfileReport


@app.cell
def _():
    dataset = pd.read_csv("01_Data/01_Raw_Data/Instagram_Usage_Lifestyle.csv")
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    profile = ProfileReport(
        df=dataset,
        title="User Behaviour Analysis on Social Media (Instagram).",
        explorative=True,  # In-depth analysis not just a summary for EDA
        minimal=False,  # Generate comprehensive and in-depth analysis not just summary
        progress_bar=True,  # Show progress while generating the report
        correlations={
            "pearson": {"calculate": True},
            "kendall": {"calculate": True},
            "spearman": {"calculate": True},
        },
        memory_deep=True,
    )
    return (profile,)


@app.cell
def _(profile):
    output_path = "./04_Reports/01_EDA/User_Behaviour_Analysis.html"
    profile.to_file(output_path)
    return


if __name__ == "__main__":
    app.run()
