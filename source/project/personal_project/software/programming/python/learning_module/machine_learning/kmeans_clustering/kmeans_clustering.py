import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium", app_title="K-Means Clustering")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    return KMeans, StandardScaler, load_iris, pd, sns


@app.cell
def _(load_iris):
    data = load_iris()
    data
    return (data,)


@app.cell
def _(data, pd):
    X = pd.DataFrame(data.data, columns=data.feature_names)
    X
    return (X,)


@app.cell
def _(data, pd):
    y =pd.DataFrame (data.target, columns=["species_id"])
    y
    return


@app.cell
def _(StandardScaler, X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return (X_scaled,)


@app.cell
def _():
    k = 3
    return (k,)


@app.cell
def _(KMeans, k):
    kmeas = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=300,
        random_state=42
    )
    return (kmeas,)


@app.cell
def _(X_scaled, kmeas):
    kmeas.fit(X_scaled)
    return


@app.cell
def _(kmeas):
    labels = kmeas.labels_
    labels[:75]
    return (labels,)


@app.cell
def _(data, labels, pd):
    pd.DataFrame((data.target == labels).astype(int), columns=["Matching"]) # 0: false, 1: true
    return


@app.cell
def _(X, labels, sns):
    X_pair = X.copy()
    X_pair['cluster'] = labels
    
    # pairplot shows how all features interact
    g = sns.pairplot(X_pair, hue="cluster", palette="viridis", diag_kind="kde")
    g.fig.suptitle("Pairwise Cluster Comparison", y=1.02)
    return


@app.cell
def _():
    # Have to implement the intertia calculation
    return


if __name__ == "__main__":
    app.run()
