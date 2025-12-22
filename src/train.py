from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor

from .data import load_kaggle_house_prices, handle_missing_values
from .features import create_features, build_preprocessor


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cross_val_rmse(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> Dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        s = rmse(y_va.values, pred)
        scores.append(s)
        print(f"Fold {fold}: RMSE={s:.5f}")
    return {"rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)), "folds": scores}


def build_model(random_state: int = 42) -> Pipeline:
    # Feature engineering happens outside, then we one-hot + model
    reg = LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    # Preprocessor will be injected later (needs columns)
    return Pipeline(steps=[("preprocess", None), ("model", reg)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Folder containing train.csv and test.csv")
    parser.add_argument("--out-dir", type=str, default="models", help="Where to save trained model")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Where to save metrics and figures")
    parser.add_argument("--target", type=str, default="SalePrice")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    ds = load_kaggle_house_prices(args.data_dir)
    train = handle_missing_values(ds.train)
    train = create_features(train)

    y = np.log1p(train[args.target].astype(float))
    X = train.drop(columns=[args.target])

    pre = build_preprocessor(X)
    model = Pipeline(steps=[
        ("preprocess", pre),
        ("model", LGBMRegressor(
            n_estimators=6000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    cv_summary = cross_val_rmse(model, X, y, n_splits=args.cv, seed=args.seed)
    print(f"CV RMSE: {cv_summary['rmse_mean']:.5f} ± {cv_summary['rmse_std']:.5f}")

    # Fit final model on all training data
    model.fit(X, y)

    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "target_transform": "log1p(SalePrice)",
        "cv": cv_summary,
        "model": "LightGBM (LGBMRegressor) + OneHotEncoder",
        "artifacts": {"model_path": str(model_path)},
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save a quick “in-sample” diagnostic plot (not a real test metric, but nice for report)
    pred = model.predict(X)
    plt.figure()
    plt.scatter(y, pred, s=10)
    plt.xlabel("y_true (log1p)")
    plt.ylabel("y_pred (log1p)")
    plt.title("Train fit: y_true vs y_pred (diagnostic)")
    fig_path = reports_dir / "figures" / "train_fit_scatter.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved model -> {model_path}")
    print(f"Saved metrics -> {reports_dir / 'metrics.json'}")
    print(f"Saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
