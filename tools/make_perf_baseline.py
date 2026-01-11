from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main(seed=42, n_splits=3):
    data_path = ROOT / "tests" / "data" / "sample_train.csv"
    df = pd.read_csv(data_path)

    y = np.log1p(df["SalePrice"].values)
    X = df.drop(columns=["SalePrice"])

    # 为了让测试稳定：只用“最稳的做法”
    # 1) one-hot
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = Ridge(alpha=10.0, random_state=seed)
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict(X_va)

    score = rmse(y, oof)

    out = {
        "dataset": "tests/data/sample_train.csv",
        "target": "log1p(SalePrice)",
        "model": "Ridge(alpha=10.0)",
        "cv": {"n_splits": n_splits, "seed": seed},
        "rmse": score,
        "tolerance": 0.015,  # 允许变差的幅度（你可以调）
    }

    out_path = ROOT / "tests" / "baselines" / "perf_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Baseline saved to: {out_path}\nRMSE={score:.6f}")

if __name__ == "__main__":
    main()
