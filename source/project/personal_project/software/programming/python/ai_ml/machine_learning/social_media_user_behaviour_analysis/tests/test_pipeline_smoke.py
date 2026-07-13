import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from smuba.features.pipeline_builder import build_pipeline


def test_pipeline_fits_and_predicts_on_mixed_dtypes():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "age": rng.integers(15, 60, size=50),
        "platform": rng.choice(["Instagram", "TikTok", "YouTube"], size=50),
    })
    y = rng.integers(0, 2, size=50)

    pipeline = build_pipeline(X, LogisticRegression(max_iter=1000), use_pca=False)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
