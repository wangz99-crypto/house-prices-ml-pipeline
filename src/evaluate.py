# src/evaluate.py
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def kfold_oof_predict(
    model,
    X,
    y,
    X_test,
    n_splits: int = 5,
    seed: int = 42,
):
    """
    Train model with KFold, return:
    - oof_pred: out-of-fold predictions for train
    - test_pred: averaged predictions for test across folds
    - fold_scores: per-fold RMSE
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    test_pred_folds = []
    fold_scores = []

    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        m = model() if callable(model) else model
        m.fit(X_tr, y_tr)

        pred_va = m.predict(X_va)
        oof[va] = pred_va

        score = rmse(y_va, pred_va)
        fold_scores.append(score)
        print(f"[Fold {fold}] RMSE: {score:.5f}")

        pred_test = m.predict(X_test)
        test_pred_folds.append(pred_test)

    test_pred = np.mean(np.vstack(test_pred_folds), axis=0)
    overall = rmse(y, oof)
    print(f"[CV] RMSE: {overall:.5f} | folds mean: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    return oof, test_pred, overall, fold_scores

