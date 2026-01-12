# tests/unit/test_model_contract_generic.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines import get_pipeline


CONTRACT_DIR = Path("tests/contracts")


def load_contract(model_name: str) -> dict:
    path = CONTRACT_DIR / f"{model_name}_contract.json"
    assert path.exists(), f"Missing contract file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_schema_and_predictions_ridge():
    _assert_contract_predictions_within_range("ridge")


def test_contract_schema_and_predictions_lgbm():
    _assert_contract_predictions_within_range("lgbm")


def _assert_contract_predictions_within_range(model_name: str) -> None:
    payload = load_contract(model_name)

    # ---- schema (match your make_contract.py output) ----
    assert payload["model_name"] == model_name
    features = payload["feature_columns"]
    X_rows = payload["golden_X"]
    pred_lo = payload["expected"]["pred_lo"]
    pred_hi = payload["expected"]["pred_hi"]

    assert isinstance(features, list) and len(features) >= 20
    assert isinstance(X_rows, list) and len(X_rows) >= 5
    assert len(pred_lo) == len(X_rows)
    assert len(pred_hi) == len(X_rows)

    # ---- build X with stable column order ----
    X = pd.DataFrame(X_rows).reindex(columns=features)

    # ---- run predictions ----
    pipe = get_pipeline(model_name, seed=int(payload.get("seed", 42)))

    # IMPORTANT:
    # contract values were created from training on full train set
    # so in test we must also fit on full train set to reproduce behavior.
    # We'll load raw train data via pandas from tests/data/sample_train.csv? -> not enough.
    # Instead we use your existing project loader: load_train_test + split_xy
    from src.data import load_train_test, split_xy

    train_df, _ = load_train_test()
    X_train, y_train_raw = split_xy(train_df)
    y_train = np.log1p(y_train_raw)

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X)

    # ---- assert each pred within saved range ----
    preds = np.asarray(preds, dtype=float)
    lo = np.asarray(pred_lo, dtype=float)
    hi = np.asarray(pred_hi, dtype=float)

    # ---- allow tolerance policy (more robust across small code changes) ----
    policy = payload.get("tolerance_policy", {}) or {}
    rel = float(policy.get("relative", 0.0))
    abs_tol = float(policy.get("absolute", 0.0))

    # center-based tolerance: tol = rel * |center| + abs
    center = (lo + hi) / 2.0
    tol = rel * np.abs(center) + abs_tol

    lo2 = lo - tol
    hi2 = hi + tol

    ok = np.all(preds >= lo2) and np.all(preds <= hi2)
    if not ok:
        # helpful debug: show the worst violations
        below = np.min(preds - lo2)
        above = np.max(preds - hi2)
        raise AssertionError(
            f"{model_name} contract failed (with tolerance_policy).\n"
            f"min(pred-lo2)={float(below):.6f}, max(pred-hi2)={float(above):.6f}\n"
            f"policy: rel={rel}, abs={abs_tol}"
        )
