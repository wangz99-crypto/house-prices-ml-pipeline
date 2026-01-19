from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from src.train import run_one # type: ignore

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]           # tests/unit/... -> repo root
SAMPLE = REPO_ROOT / "tests" / "data" / "sample_train.csv"



def test_model_can_be_loaded_and_predicts():
    # 1) load sample data
    df = pd.read_csv(SAMPLE)
    assert "SalePrice" in df.columns

    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    # 2) temp dirs (avoid writing to repo)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reports_dir = tmp / "reports"
        models_dir = tmp / "models"
        registry_dir = tmp / "registry"

        # 3) train once (small CV for CI speed)
        res = run_one(
            model_name="ridge",
            X=X,
            y=y,
            X_test=X.iloc[:10].copy(),  # dummy
            reports_dir=reports_dir,
            models_dir=models_dir,
            registry_dir=registry_dir,
            seed=42,
            n_splits=3,
        )

        # 4) load the saved model
        assert "model_path" in res, f"run_one() must return model_path, got keys={list(res.keys())}"
        model_path = Path(res["model_path"])
        assert model_path.exists(), f"model file not found: {model_path}"

        model = joblib.load(model_path)

        # 5) predict on a small batch
        X_batch = X.iloc[:20].copy()
        pred = model.predict(X_batch)

        # 6) sanity checks
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (len(X_batch),)
        assert np.isfinite(pred).all()
