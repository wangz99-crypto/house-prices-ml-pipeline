from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipelines import get_pipeline


SAMPLE = Path("tests/data/sample_train.csv")


def test_ridge_predictions_same_after_save_load():
    # 1) load sample data
    df = pd.read_csv(SAMPLE)
    assert "SalePrice" in df.columns

    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    # keep test fast and deterministic
    X_small = X.iloc[:50].copy()

    # 2) train model
    model = get_pipeline("ridge", seed=42)
    model.fit(X, y)

    pred_before = model.predict(X_small)

    # 3) save + load
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model_path = tmp / "ridge.joblib"

        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)

        pred_after = loaded.predict(X_small)

    # 4) assert predictions are identical (or extremely close)
    # Ridge should be essentially bit-level stable on same platform.
    assert pred_before.shape == pred_after.shape
    assert np.allclose(pred_before, pred_after, rtol=0.0, atol=1e-12), (
        f"Predictions changed after save/load. "
        f"max_abs_diff={np.max(np.abs(pred_before - pred_after))}"
    )
