from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipelines import get_pipeline
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)
SAMPLE = Path("tests/data/sample_train.csv")


def _fit_save_load_predict(model_name: str, X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    model = get_pipeline(model_name, seed=seed)
    model.fit(X, y)

    pred_before = model.predict(X)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = tmp / f"{model_name}.joblib"
        joblib.dump(model, p)

        loaded = joblib.load(p)
        pred_after = loaded.predict(X)

    return pred_before, pred_after


def test_tree_models_predictions_close_after_save_load():
    df = pd.read_csv(SAMPLE)
    y = np.log1p(df["SalePrice"]).values
    X = df.drop(columns=["SalePrice"])

    # 小样本，CI 稳定 + 快
    X_small = X.iloc[:80].copy()
    y_small = y[:80]

    for name in ["extratrees", "lgbm", "xgb"]:
        pred_before, pred_after = _fit_save_load_predict(
            name, X_small, y_small, seed=42
        )

        assert pred_before.shape == pred_after.shape

        # ⚠️ 注意：这里用的是 pred_before，不是 pred_before_
        assert np.allclose(
            pred_before,
            pred_after,
            rtol=0.0,
            atol=1e-6,
        ), (
            f"{name} predictions changed after save/load. "
            f"max_abs_diff={np.max(np.abs(pred_before - pred_after))}"
        )
