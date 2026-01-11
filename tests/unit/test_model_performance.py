import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import run_one


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "data" / "sample_train.csv"
BASELINE = ROOT / "tests" / "baselines" / "perf_baseline.json"


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_ridge_rmse_not_regressing():
    # 1) load sample data
    df = pd.read_csv(SAMPLE)
    assert "SalePrice" in df.columns

    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    # 2) create temp dirs (avoid writing into repo during tests/CI)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reports_dir = tmp / "reports"
        models_dir = tmp / "models"
        registry_dir = tmp / "registry"

        res = run_one(
            model_name="ridge",
            X=X,
            y=y,
            X_test=X.iloc[:10].copy(),  # dummy, we only care about oof
            reports_dir=reports_dir,
            models_dir=models_dir,
            registry_dir=registry_dir,
            seed=42,
            n_splits=3,  # keep CI fast & stable
        )

        oof = np.load(res["oof_path"])
        score = rmse(y.values, oof)

    # 3) compare with baseline
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    base_rmse = float(baseline["rmse"])

    # allow slight fluctuation (CI machines / deps can cause tiny drift)
    tolerance = 0.01
    assert score <= base_rmse + tolerance, (
        f"RMSE regressed: {score:.6f} > {(base_rmse + tolerance):.6f} "
        f"(baseline={base_rmse:.6f}, tol={tolerance:.3f})"
    )
