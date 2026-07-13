# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.3",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    import sys
    import numpy as np
    import scipy as sp
    import pandas as pd
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt

    from pathlib import Path


@app.cell
def _():
    sys.path.append(str(Path(__file__).resolve().parent))
    # sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    return


@app.cell
def _():
    filepath = "machine_learning_course/projects/youtube_video_analysis/YouTubeDataset_withChannelElapsed.csv"
    df = pd.read_csv(filepath)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    # Descriptive Analysis for Numerical Features

    df.describe().T
    return


@app.cell
def _(df):
    # Check Null

    df.isnull().sum()
    return


@app.cell
def _(df):
    # Numerical, Categorical and Ratio Features Selection

    numerical_features = df.select_dtypes(include="number").columns.tolist()
    categorical_features = df.select_dtypes(exclude="number").columns.tolist()
    ratio_features = df.columns[df.columns.str.contains('/')].tolist()

    # Remove Ratio Features from Numerical and Categorical Features
    numerical_features = [_features for _features in numerical_features if _features not in ratio_features]
    categorical_features = [_features for _features in categorical_features if _features not in ratio_features]

    mo.ui.tabs({
        "Numerical Features": numerical_features,
        "Categorical Features": categorical_features,
        "Ratio Features": ratio_features
    })
    return (numerical_features,)


@app.cell
def _(df, numerical_features):
    # Visualize Distribution of Numerical Features

    _df_long = df[numerical_features].melt(var_name="feature", value_name="value")
    _g = sns.FacetGrid(_df_long, col="feature", col_wrap=3, sharex=False, sharey=False)
    _g.map(sns.histplot, "value", bins="auto", kde=True, stat="density", alpha=0.5)
    plt.show()
    return


@app.cell
def _(df, numerical_features):
    # Box Plot for Indentifying Outliers

    _df_long = df[numerical_features].melt(var_name="feature", value_name="value")

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="feature", y="value", data=_df_long)
    plt.title("Boxplot for Numerical Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    from machine_learning_course.exploratory_data_analysis.exploratory_data_analysis import OutlierDetector
    return (OutlierDetector,)


@app.cell
def _(OutlierDetector, df, numerical_features):
    count = OutlierDetector.outliers_count(df=df, features=numerical_features)
    print(count)
    return (count,)


@app.cell
def _(OutlierDetector, count):
    OutlierDetector.outliers_plot(outliers_counts_dict=count)
    return


if __name__ == "__main__":
    app.run()
