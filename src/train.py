from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from .config import default_paths
from .data import load_train_test, split_xy, ID_COL
from .evaluate import kfold_oof_predict
from .pipelines import get_pipeline, PIPELINES


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prepare raw X/y/test for sklearn Pipelines.
    - drop Id column (not a feature)
    - y uses log1p (Kaggle standard for this competition)
    """
    X_raw, y_raw = split_xy(train_df)

    if ID_COL in X_raw.columns:
        X_raw = X_raw.drop(columns=[ID_COL])

    X_test = test_df.copy()
    if ID_COL in X_test.columns:
        X_test = X_test.drop(columns=[ID_COL])

    y = np.log1p(y_raw.astype(float).values)
    return X_raw, y, X_test


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_metrics(reports_dir: Path, model_name: str, cv_rmse: float, fold_scores: list[float]):
    payload = {
        "model": model_name,
        "cv_rmse": float(cv_rmse),
        "fold_rmse": [float(x) for x in fold_scores],
    }
    out = reports_dir / f"{model_name}_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one(model_name: str, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
            reports_dir: Path, models_dir: Path, seed: int, n_splits: int = 5):
    print(f"\n=== Training: {model_name} ===")

    # model factory: each fold should get a fresh estimator
    model_fn = lambda: get_pipeline(model_name, seed=seed)

    oof, test_pred, cv_rmse, fold_scores = kfold_oof_predict(
        model=model_fn,
        X=X,
        y=y,
        X_test=X_test,
        n_splits=n_splits,
        seed=seed,
    )

    # Save OOF + test predictions (log-space)
    np.save(reports_dir / f"{model_name}_oof.npy", oof)
    np.save(reports_dir / f"{model_name}_test_pred.npy", test_pred)

    # Save metrics
    save_metrics(reports_dir, model_name, cv_rmse, fold_scores)

    # Fit full model and save for reproducibility
    final_model = get_pipeline(model_name, seed=seed)
    final_model.fit(X, y)
    joblib.dump(final_model, models_dir / f"{model_name}.joblib")

    print(f"[{model_name}] CV RMSE (log-space): {cv_rmse:.6f}")
    return {
        "model": model_name,
        "cv_rmse": cv_rmse,
        "fold_scores": fold_scores,
        "oof_path": str(reports_dir / f"{model_name}_oof.npy"),
        "test_pred_path": str(reports_dir / f"{model_name}_test_pred.npy"),
        "model_path": str(models_dir / f"{model_name}.joblib"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        help=f"one of {sorted(PIPELINES.keys())} or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to directory containing train.csv/test.csv. If omitted, uses config.default_paths().data_raw")
    args = parser.parse_args()

    paths = default_paths()
    reports_dir = paths.reports_dir
    models_dir = paths.models_dir
    ensure_dirs(reports_dir, models_dir)

    # Load data
    if args.data_dir:
        train_df, test_df = load_train_test(Path(args.data_dir))
    else:
        train_df, test_df = load_train_test()  # data.py should support default or you can pass paths.data_raw

    # Prepare raw features/target
    X, y, X_test = prepare_features(train_df, test_df)

    # Select models
    if args.model == "all":
        to_run = sorted(PIPELINES.keys())
    else:
        if args.model not in PIPELINES:
            raise ValueError(f"Unknown model: {args.model}. Choose from {sorted(PIPELINES.keys())} or 'all'")
        to_run = [args.model]

    # Train
    results = []
    for name in to_run:
        res = run_one(
            model_name=name,
            X=X,
            y=y,
            X_test=X_test,
            reports_dir=reports_dir,
            models_dir=models_dir,
            seed=args.seed,
            n_splits=args.folds,
        )
        results.append(res)

    # Save summary
    summary_path = reports_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved CV summary to: {summary_path}")


if __name__ == "__main__":
    main()

