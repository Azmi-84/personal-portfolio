import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    return PCA, StandardScaler, load_wine, np, pd, plt


@app.cell
def _(load_wine, pd):
    data = load_wine()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="class")

    print(X.shape, y.shape)
    return (X,)


@app.cell
def _(X):
    X.head(5)
    return


@app.cell
def _(X):
    X.describe().T
    return


@app.cell
def _(StandardScaler, X):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    print(X_scaled[:5])
    return (X_scaled,)


@app.cell
def _(PCA, X_scaled, np):
    pca_full = PCA()

    X_pca_full = pca_full.fit_transform(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    print(explained)
    print(cum_explained)
    return (cum_explained,)


@app.cell
def _(cum_explained, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(cum_explained, marker="o")
    plt.title("Cumulative Explained Variance of Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=0.90, color="g", linestyle="--", label="90% Variance")
    plt.axhline(y=0.95, color="b", linestyle="--", label="95% Variance")
    plt.axhline(y=0.97, color="y", linestyle="--", label="95% Variance")
    plt.axhline(y=0.99, color="r", linestyle="--", label="97% Variance")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np):
    # Selection of k90 and k95

    def components_for_threashold(cum_variance, threashold):
        return int(np.argmax(cum_variance>= threashold)+1)
    return (components_for_threashold,)


@app.cell
def _(components_for_threashold, cum_explained):
    k90 = components_for_threashold(cum_variance=cum_explained, threashold=0.90)
    k95 = components_for_threashold(cum_variance=cum_explained, threashold=0.95)
    k97 = components_for_threashold(cum_variance=cum_explained, threashold=0.97)
    k99 = components_for_threashold(cum_variance=cum_explained, threashold=0.99)

    print(k90, k95, k97, k99)
    return (k95,)


@app.cell
def _(PCA, X_scaled, k95):
    pca_95 = PCA(n_components=k95)

    X_pca_95 = pca_95.fit_transform(X_scaled)
    print(X_pca_95.shape)
    return (pca_95,)


@app.cell
def _(X, k95, pca_95, pd):
    loadings = pd.DataFrame(
        pca_95.components_.T,
        index = X.columns,
        columns = [f"PC{i+1}" for i in range(k95)]
    )

    loadings
    return (loadings,)


@app.cell
def _(pd):
    def top_loadings_for_pca(loadings_df, pc_name, top_n=5):
        s = loadings_df[pc_name].abs().sort_values(ascending = False).head(top_n)
        return pd.DataFrame({"features": s.index, "abs_loading": s.values})
    return (top_loadings_for_pca,)


@app.cell
def _(loadings, top_loadings_for_pca):
    top_pca1 = top_loadings_for_pca(loadings_df=loadings, pc_name="PC1", top_n=5)
    top_pca1
    return


@app.cell
def _(loadings, top_loadings_for_pca):
    top_pca2 = top_loadings_for_pca(loadings_df=loadings, pc_name="PC2", top_n=5)
    top_pca2
    return


@app.cell
def _(loadings, top_loadings_for_pca):
    top_pca3 = top_loadings_for_pca(loadings_df=loadings, pc_name="PC3", top_n=5)
    top_pca3
    return


if __name__ == "__main__":
    app.run()
