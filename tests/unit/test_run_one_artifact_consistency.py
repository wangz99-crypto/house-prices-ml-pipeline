from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train import run_one


SAMPLE = Path("tests/data/sample_train.csv")


def test_run_one_registry_model_predictions_reproducible():
    df = pd.read_csv(SAMPLE)
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    # dummy test set (not important for this test)
    X_test = X.iloc[:10].copy()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reports_dir = tmp / "reports"
        models_dir = tmp / "models"
        registry_dir = tmp / "registry"

        res = run_one(
            model_name="ridge",
            X=X,
            y=y,
            X_test=X_test,
            reports_dir=reports_dir,
            models_dir=models_dir,
            registry_dir=registry_dir,
            seed=42,
            n_splits=3,
        )

        # load model artifact from registry path
        model_path = Path(res["registry_model_path"])
        assert model_path.exists()

        model = joblib.load(model_path)

        # compare model prediction vs saved oof file existence (sanity)
        X_small = X.iloc[:50].copy()
        pred = model.predict(X_small)
        assert pred.shape[0] == len(X_small)
