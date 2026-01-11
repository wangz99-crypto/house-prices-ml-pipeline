from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def kfold_oof_predict(
    model,
    X,
    y,
    X_test,
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Train model with KFold, return:
    - oof_pred: out-of-fold predictions for train
    - test_pred: averaged predictions for test across folds
    - overall_rmse: RMSE on full OOF
    - fold_scores: per-fold RMSE
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_train = len(X)
    oof = np.zeros(n_train, dtype=float)

    # running average to save memory
    test_pred = np.zeros(len(X_test), dtype=float)

    fold_scores = []

    # helper: slice pandas or numpy safely
    def _take(a, idx):
        return a.iloc[idx] if hasattr(a, "iloc") else a[idx]

    base_est = model() if callable(model) else model

    for fold, (tr, va) in enumerate(kf.split(range(n_train)), start=1):
        X_tr, X_va = _take(X, tr), _take(X, va)
        y_tr, y_va = _take(y, tr), _take(y, va)

        m = clone(base_est)   # always fresh estimator each fold
        m.fit(X_tr, y_tr)

        pred_va = m.predict(X_va)
        oof[va] = pred_va

        score = rmse(y_va, pred_va)
        fold_scores.append(score)
        if verbose:
            print(f"[Fold {fold}] RMSE: {score:.5f}")

        pred_test = m.predict(X_test)
        test_pred += pred_test / n_splits

    overall = rmse(y, oof)
    if verbose:
        print(
            f"[CV] RMSE: {overall:.5f} | "
            f"folds mean: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}"
        )

    return oof, test_pred, overall, fold_scores

