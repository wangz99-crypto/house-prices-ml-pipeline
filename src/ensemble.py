# src/ensemble.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge

def blend_mean(test_preds: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(test_preds), axis=0)

def blend_weighted(test_preds: list[np.ndarray], cv_rmses: list[float]) -> np.ndarray:
    # weight = 1 / rmse
    w = 1.0 / np.array(cv_rmses, dtype=float)
    w = w / w.sum()
    stacked = np.vstack(test_preds)  # (n_models, n_samples)
    return (stacked.T @ w).astype(float)

def stacking_ridge(oof_preds: list[np.ndarray], y: np.ndarray, test_preds: list[np.ndarray], seed: int = 42) -> np.ndarray:
    """
    Classic stacking:
    - meta features = OOF predictions of base models
    - meta model = Ridge to reduce overfit
    """
    X_meta = np.vstack(oof_preds).T
    X_test_meta = np.vstack(test_preds).T

    meta = Ridge(alpha=1.0, random_state=seed)
    meta.fit(X_meta, y)
    return meta.predict(X_test_meta)
