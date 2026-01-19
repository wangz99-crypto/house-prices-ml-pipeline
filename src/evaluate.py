from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# ------------------------------------------------------------
# small helpers
# ------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _take(a, idx):
    return a.iloc[idx] if hasattr(a, "iloc") else a[idx]


def _unwrap_estimator(obj):
    """
    Recursively unwrap common wrappers to get the "real" underlying estimator.
    Works with:
      - sklearn Pipeline / ColumnTransformer
      - custom wrappers with .estimator / .model
      - sklearn Stacking/Voting style estimators
    """
    cur = obj
    visited = set()

    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        # 1) sklearn Pipeline: take last step estimator
        if hasattr(cur, "steps") and isinstance(cur.steps, list) and cur.steps:
            cur = cur.steps[-1][1]
            continue

        # 2) common wrapper attribute names
        for attr in ("estimator_", "estimator", "model_", "model", "regressor_", "regressor"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
        else:
            # no wrapper attrs matched
            break

    return cur


def _get_best_iter(est) -> int | None:
    """
    Robust best-iteration getter for LightGBM / XGBoost.
    Returns int or None.
    """
    m = _unwrap_estimator(est)

    # ---------- LightGBM sklearn API ----------
    # LGBMRegressor exposes best_iteration_ after early stopping
    for attr in ("best_iteration_", "best_iteration"):
        if hasattr(m, attr):
            v = getattr(m, attr)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)

    # LightGBM wrapper sometimes keeps Booster in booster_
    if hasattr(m, "booster_") and m.booster_ is not None:
        b = m.booster_
        if hasattr(b, "best_iteration"):
            v = b.best_iteration
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)

    # ---------- XGBoost sklearn API ----------
    # Depending on version, could be best_iteration or best_iteration_
    for attr in ("best_iteration", "best_iteration_"):
        if hasattr(m, attr):
            v = getattr(m, attr)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)

    # XGBoost: sometimes best_iteration is stored on booster
    if hasattr(m, "get_booster"):
        try:
            booster = m.get_booster()
            # Not always present, but try
            if booster is not None and hasattr(booster, "best_iteration"):
                v = booster.best_iteration
                if isinstance(v, (int, np.integer)) and v > 0:
                    return int(v)
        except Exception:
            pass

    return None



def _fit_pipeline_with_es_if_possible(
    est,
    X_tr,
    y_tr,
    X_va,
    y_va,
    *,
    use_early_stopping: bool,
    early_stopping_rounds: int,
):
    """
    Fit estimator. If early stopping is requested AND estimator is a pipeline-like
    (AsRegressor wrapping a Pipeline with steps: shared -> prep -> model),
    do pipeline-aware ES by transforming eval_set through the same steps.

    Otherwise, fall back to plain fit().
    """
    if not use_early_stopping:
        est.fit(X_tr, y_tr)
        return

    # --- Try to unpack your project pipeline structure ---
    try:
        base = est                      # AsRegressor
        pipe = base.estimator           # sklearn Pipeline template (unfitted)
        steps = dict(pipe.named_steps)

        shared = steps["shared"]
        prep = steps["prep"]
        model_step = steps["model"]
    except Exception:
        # Not expected structure -> fallback
        est.fit(X_tr, y_tr)
        return

    # ---- Fit shared + prep on train, transform train/val ----
    X_tr_s = shared.fit_transform(X_tr, y_tr)
    X_va_s = shared.transform(X_va)

    X_tr_m = prep.fit_transform(X_tr_s, y_tr)
    X_va_m = prep.transform(X_va_s)

    # ---- Fit underlying model with early stopping ----
    module = model_step.__class__.__module__

    # 1) XGBoost sklearn API
    if module.startswith("xgboost"):
        # Try without 'verbose' to avoid version conflicts
        model_step.fit(
            X_tr_m,
            y_tr,
            eval_set=[(X_va_m, y_va)],
            early_stopping_rounds=early_stopping_rounds,
        )

    # 2) LightGBM sklearn API (most compatible via callbacks)
    elif module.startswith("lightgbm"):
        import lightgbm as lgb

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        ]

        model_step.fit(
            X_tr_m,
            y_tr,
            eval_set=[(X_va_m, y_va)],
            callbacks=callbacks,
        )

    else:
        # Unknown model type -> fallback
        est.fit(X_tr, y_tr)
        return

    # ---- Rebuild a fitted pipeline inside AsRegressor so predict() works ----
    from sklearn.pipeline import Pipeline

    fitted_pipe = Pipeline([
        ("shared", shared),
        ("prep", prep),
        ("model", model_step),
    ])

    base.estimator_ = fitted_pipe



def kfold_oof_predict(
    model,
    X,
    y,
    X_test,
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
    *,
    use_early_stopping: bool = False,
    early_stopping_rounds: int = 200,
    record_best_iter: bool = False,
):
    """
    Train model with KFold, return:
      - oof_pred: out-of-fold predictions for train
      - test_pred: averaged predictions for test across folds
      - overall_rmse: RMSE on full OOF
      - fold_scores: per-fold RMSE

    Optional (if record_best_iter=True):
      - fold_meta: list of dicts (rmse, best_iter)
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_train = len(X)
    oof = np.zeros(n_train, dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    fold_scores = []
    fold_meta = []

    for fold, (tr, va) in enumerate(kf.split(range(n_train)), start=1):
        X_tr, X_va = _take(X, tr), _take(X, va)
        y_tr, y_va = _take(y, tr), _take(y, va)

        # ✅ IMPORTANT: fresh estimator each fold
        m = model() if callable(model) else clone(model)

        _fit_pipeline_with_es_if_possible(
            m, X_tr, y_tr, X_va, y_va,
            use_early_stopping=use_early_stopping,
            early_stopping_rounds=early_stopping_rounds,
        )

        pred_va = m.predict(X_va)
        oof[va] = pred_va

        score = rmse(y_va, pred_va)
        fold_scores.append(score)

        best_iter = _get_best_iter(m) if record_best_iter else None


        if verbose:
            msg = f"[Fold {fold}] RMSE: {score:.5f}"
            if record_best_iter and best_iter is not None:
                msg += f" | best_iter={best_iter}"
            print(msg)

        if record_best_iter:
            fold_meta.append({"fold": fold, "rmse": float(score), "best_iter": best_iter})

        pred_test = m.predict(X_test)
        test_pred += pred_test / n_splits

    overall = rmse(y, oof)

    if verbose:
        print(
            f"[CV] RMSE: {overall:.5f} | "
            f"folds mean: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}"
        )
        if record_best_iter:
            iters = [d["best_iter"] for d in fold_meta if d.get("best_iter") is not None]
            if iters:
                print(
                    f"[CV] best_iter mean: {np.mean(iters):.1f} ± {np.std(iters):.1f} "
                    f"(n={len(iters)}/{n_splits})"
                )

    if record_best_iter:
        return oof, test_pred, overall, fold_scores, fold_meta

    return oof, test_pred, overall, fold_scores
