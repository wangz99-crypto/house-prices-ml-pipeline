# tests/unit/test_model_contract_generic.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines import get_pipeline


ROOT = Path(__file__).resolve().parents[2]   # repo root (tests/unit/ -> repo)
CONTRACT_DIR = ROOT / "tests" / "contracts"
SAMPLE_TRAIN = ROOT / "tests" / "data" / "sample_train.csv"


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

    # ---- schema ----
    assert payload["model_name"] == model_name
    features = payload["feature_columns"]
    X_rows = payload["golden_X"]
    pred_lo = payload["expected"]["pred_lo"]
    pred_hi = payload["expected"]["pred_hi"]

    assert isinstance(features, list) and len(features) >= 20
    assert isinstance(X_rows, list) and len(X_rows) >= 5
    assert len(pred_lo) == len(X_rows)
    assert len(pred_hi) == len(X_rows)

    # ---- build X (stable column order) ----
    X = pd.DataFrame(X_rows).reindex(columns=features)

    # ---- fit pipeline on sample data (CI-safe) ----
    pipe = get_pipeline(model_name, seed=int(payload.get("seed", 42)))

    assert SAMPLE_TRAIN.exists(), f"Missing sample data: {SAMPLE_TRAIN}"
    df_train = pd.read_csv(SAMPLE_TRAIN)

    y_train = np.log1p(df_train["SalePrice"])
    X_train = df_train.drop(columns=["SalePrice"])

    pipe.fit(X_train, y_train)

    # ---- predict ----
    preds_arr = np.asarray(pipe.predict(X), dtype=float)
    lo_arr = np.asarray(pred_lo, dtype=float)
    hi_arr = np.asarray(pred_hi, dtype=float)

    assert np.all(preds_arr >= lo_arr) and np.all(preds_arr <= hi_arr), (
        f"{model_name} contract failed.\n"
        f"min(pred-lo)={float(np.min(preds_arr - lo_arr)):.6f}, "
        f"max(pred-hi)={float(np.max(preds_arr - hi_arr)):.6f}"
    )