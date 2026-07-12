"""Builds ONE sklearn Pipeline (preprocessing -> optional PCA -> estimator).

This is the key simplification vs. the original tree: "with PCA" vs.
"without PCA" is a boolean flag here, not two parallel folders of 12 files
each. The whole pipeline - including PCA - gets logged to MLflow as a
single versioned artifact, so there's never a mismatch between "which
scaler/PCA goes with which model.pkl".
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X, estimator, use_pca: bool, pca_variance_threshold: float = 0.95) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    steps = [("preprocessor", preprocessor)]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance_threshold, random_state=42)))
    steps.append(("estimator", estimator))

    return Pipeline(steps)
