# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 18:12:16 2026

@author: 23517
"""

from __future__ import annotations

from typing import List
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _as_2d(pred_list: List[np.ndarray]) -> np.ndarray:
    """Stack [n_models] arrays of shape (n_samples,) -> (n_samples, n_models)."""
    if len(pred_list) == 0:
        raise ValueError("pred_list is empty")
    arrs = [np.asarray(a).reshape(-1) for a in pred_list]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("All prediction arrays must have the same length")
    return np.vstack(arrs).T  # (n_samples, n_models)


def blend_mean(test_preds: List[np.ndarray]) -> np.ndarray:
    """
    Simple mean blend in log-space.
    Input: list of (n_test,) arrays
    Output: (n_test,) array
    """
    X = _as_2d(test_preds)
    return X.mean(axis=1)


def blend_weighted(test_preds: List[np.ndarray], rmses: List[float], eps: float = 1e-12) -> np.ndarray:
    """
    Weighted blend in log-space, weight = 1 / rmse.
    Lower RMSE -> higher weight.
    """
    X = _as_2d(test_preds)
    if len(rmses) != X.shape[1]:
        raise ValueError(f"len(rmses) must equal n_models. Got {len(rmses)} vs {X.shape[1]}")

    r = np.asarray(rmses, dtype=float)
    r = np.maximum(r, eps)
    w = 1.0 / r
    w = w / w.sum()
    return X @ w


def stacking_ridge(
    oof_preds: List[np.ndarray],
    y_log: np.ndarray,
    test_preds: List[np.ndarray],
    seed: int = 42,
    alpha: float = 1.0,
    meta_folds: int = 5,
) -> np.ndarray:
    """
    Ridge stacking in log-space.

    Train meta-model on OOF predictions (train rows), then predict test from test predictions.
    - oof_preds: list of arrays, each (n_train,)
    - y_log: (n_train,)
    - test_preds: list of arrays, each (n_test,)
    """
    X_meta = _as_2d(oof_preds)        # (n_train, n_models)
    X_test_meta = _as_2d(test_preds)  # (n_test, n_models)
    y_log = np.asarray(y_log).reshape(-1)

    if X_meta.shape[0] != len(y_log):
        raise ValueError(f"OOF rows must match y_log length. Got {X_meta.shape[0]} vs {len(y_log)}")

    # Optional: train meta model via CV to reduce overfit (still no leakage)
    # We will fit one final model on full meta-train after checking CV quickly.
    kf = KFold(n_splits=meta_folds, shuffle=True, random_state=seed)
    _ = []
    for tr, va in kf.split(X_meta):
        m = Ridge(alpha=alpha)
        m.fit(X_meta[tr], y_log[tr])
        pred_va = m.predict(X_meta[va])
        _.append(pred_va)  # placeholder, you can compute rmse if you want

    final = Ridge(alpha=alpha)
    final.fit(X_meta, y_log)
    pred_test_log = final.predict(X_test_meta)
    return pred_test_log
