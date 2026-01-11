from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .config import default_paths
from .data import load_train_test, split_xy, ID_COL
from .evaluate import kfold_oof_predict
from .pipelines import get_pipeline, PIPELINES
from .registry import make_run_dir, write_json, fingerprint_dataframe


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prepare raw X/y/test for sklearn Pipelines.
    - drop Id column (not a feature)
    - y uses log1p (Kaggle standard)
    """
    X_raw, y_raw = split_xy(train_df)

    if ID_COL in X_raw.columns:
        X_raw = X_raw.drop(columns=[ID_COL])

    X_test = test_df.copy()
    if ID_COL in X_test.columns:
        X_test = X_test.drop(columns=[ID_COL])

    y = np.log1p(y_raw.astype(float))
    return X_raw, y, X_test


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_metrics_compat(reports_dir: Path, model_name: str, cv_rmse: float, fold_scores: list[float]):
    payload = {
        "model": model_name,
        "cv_rmse": float(cv_rmse),
        "fold_rmse": [float(x) for x in fold_scores],
    }
    out = reports_dir / f"{model_name}_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
    registry_dir: Path,
    seed: int,
    n_splits: int = 5,
):
    print(f"\n=== Training: {model_name} ===")

    # Make sure output dirs exist
    ensure_dirs(reports_dir, models_dir, registry_dir)

    # Create a registry run folder for this training run
    run = make_run_dir(registry_dir, model_name)

    # model factory: each fold gets a fresh estimator
    model_fn = lambda: get_pipeline(model_name, seed=seed)

    # KFold OOF predictions + averaged test predictions
    oof, test_pred, cv_rmse, fold_scores = kfold_oof_predict(
        model=model_fn,
        X=X,
        y=y,
        X_test=X_test,
        n_splits=n_splits,
        seed=seed,
    )

    # ------------------------------
    # Save OOF + test preds
    #   1) compat location (reports/)
    #   2) registry location (models/registry/<model>/<run_id>/)
    # ------------------------------
    oof_path_reports = reports_dir / f"{model_name}_oof.npy"
    test_pred_path_reports = reports_dir / f"{model_name}_test_pred.npy"

    np.save(oof_path_reports, oof)
    np.save(test_pred_path_reports, test_pred)

    np.save(run.run_dir / "oof.npy", oof)
    np.save(run.run_dir / "test_pred.npy", test_pred)

    # ------------------------------
    # Save metrics (compat + registry)
    # ------------------------------
    save_metrics_compat(reports_dir, model_name, cv_rmse, fold_scores)

    write_json(
        run.run_dir / "metrics.json",
        {
            "model": model_name,
            "cv_rmse": float(cv_rmse),
            "fold_rmse": [float(x) for x in fold_scores],
        },
    )

    # ------------------------------
    # Fit full model on all data and save artifacts
    # ------------------------------
    final_model = get_pipeline(model_name, seed=seed)
    final_model.fit(X, y)

    # old-style single model file (models/)
    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(final_model, model_path)

    # registry-style model file
    registry_model_path = run.run_dir / "model.joblib"
    joblib.dump(final_model, registry_model_path)

    # ------------------------------
    # Lineage / args / fingerprints
    # ------------------------------
    write_json(
        run.run_dir / "train_args.json",
        {
            "seed": int(seed),
            "n_splits": int(n_splits),
            "model_name": model_name,
        },
    )

    write_json(
        run.run_dir / "data_fingerprint.json",
        {
            "X": fingerprint_dataframe(X),
            "y_rows": int(len(y)),
        },
    )

    (run.run_dir / "pipeline_repr.txt").write_text(str(final_model), encoding="utf-8")

    print(f"[{model_name}] CV RMSE (log-space): {cv_rmse:.6f} | model_id={run.model_id}")

    # ✅ 关键：测试需要 oof_path，所以必须返回
    return {
        "model": model_name,
        "model_id": run.model_id,
        "run_dir": str(run.run_dir),
        "cv_rmse": float(cv_rmse),
        "fold_scores": [float(x) for x in fold_scores],
        "oof_path": str(oof_path_reports),
        "test_pred_path": str(test_pred_path_reports),
        "model_path": str(model_path),
        "registry_model_path": str(registry_model_path),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        help=f"one of {sorted(PIPELINES.keys())} or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing train.csv/test.csv (default: data/raw)")
    args = parser.parse_args()

    paths = default_paths()
    reports_dir = paths.reports_dir
    models_dir = paths.models_dir
    registry_dir = paths.models_registry
    ensure_dirs(reports_dir, models_dir, registry_dir)

    # Load data
    data_dir = Path(args.data_dir) if args.data_dir else paths.data_raw
    train_df, test_df = load_train_test(data_dir)

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
            registry_dir=registry_dir,
            seed=args.seed,
            n_splits=args.folds,
        )
        results.append(res)

    # Save metrics.csv for ensembling
    metrics_rows = [{"model": r["model"], "cv_rmse_log": float(r["cv_rmse"])} for r in results]
    metrics_path = reports_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    # Save summary
    summary_path = reports_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved CV summary to: {summary_path}")

if __name__ == "__main__":
    main()
