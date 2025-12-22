# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .models import MODEL_FACTORY
from .evaluate import kfold_oof_predict
from .data import load_train_test, split_xy, ID_COL
from .features import build_features

TARGET = "SalePrice"

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # split
    X_raw, y_raw = split_xy(train_df)  # X includes Id unless you removed it; we will drop ID_COL below
    if ID_COL in X_raw.columns:
        X_raw = X_raw.drop(columns=[ID_COL])
    if ID_COL in test_df.columns:
        test_raw = test_df.drop(columns=[ID_COL])
    else:
        test_raw = test_df.copy()

    # build features jointly to keep consistent columns
    all_df = pd.concat([X_raw, test_raw], axis=0, ignore_index=True)
    all_feat = build_features(all_df)

    X = all_feat.iloc[: len(X_raw)].copy()
    X_test = all_feat.iloc[len(X_raw):].copy()

    # IMPORTANT: for sklearn/xgb, categories must be numeric
    # convert category -> codes, keep NaN as -1
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes.replace({-1: -1}).astype(int)
            X_test[c] = X_test[c].cat.codes.replace({-1: -1}).astype(int)

    # y log transform
    y = np.log1p(y_raw.values.astype(float))
    return X, y, X_test

def run_one(model_name: str, X, y, X_test, reports_dir: Path, models_dir: Path, seed: int = 42):
    model_fn = MODEL_FACTORY[model_name]

    oof, test_pred, cv_rmse, fold_scores = kfold_oof_predict(
        model=model_fn,
        X=X,
        y=pd.Series(y),
        X_test=X_test,
        n_splits=5,
        seed=seed,
    )

    # save artifacts
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    np.save(reports_dir / f"oof_{model_name}.npy", oof)
    np.save(reports_dir / f"testpred_{model_name}.npy", test_pred)

    joblib.dump(
        {"model_name": model_name, "model_factory": model_name, "feature_columns": list(X.columns)},
        models_dir / f"{model_name}.joblib"
    )

    return {
        "model": model_name,
        "cv_rmse_log": float(cv_rmse),
        "fold_scores_log": [float(x) for x in fold_scores],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all"] + sorted(MODEL_FACTORY.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"

    train_df, test_df = load_train_test()
    X, y, X_test = prepare_features(train_df, test_df)

    results = []
    models_to_run = sorted(MODEL_FACTORY.keys()) if args.model == "all" else [args.model]

    for name in models_to_run:
        print(f"\n=== Training: {name} ===")
        res = run_one(name, X, y, X_test, reports_dir, models_dir, seed=args.seed)
        results.append(res)

    # metrics table
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to: {metrics_path}")

    # also save a csv summary for convenience
    dfm = pd.DataFrame([{"model": r["model"], "cv_rmse_log": r["cv_rmse_log"]} for r in results]).sort_values("cv_rmse_log")
    csv_path = reports_dir / "metrics.csv"
    dfm.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")

if __name__ == "__main__":
    main()
