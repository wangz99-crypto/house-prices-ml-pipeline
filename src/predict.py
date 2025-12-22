from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import handle_missing_values
from .features import create_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model.joblib")
    parser.add_argument("--test-path", type=str, required=True, help="Path to Kaggle test.csv")
    parser.add_argument("--out-path", type=str, default="reports/submission.csv", help="Where to write submission CSV")
    parser.add_argument("--id-col", type=str, default="Id")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)

    test = pd.read_csv(args.test_path)
    test = handle_missing_values(test)
    test = create_features(test)

    ids = test[args.id_col].copy() if args.id_col in test.columns else pd.Series(np.arange(len(test)))
    X_test = test.drop(columns=["SalePrice"], errors="ignore")

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)  # invert log1p

    submission = pd.DataFrame({args.id_col: ids, "SalePrice": pred})
    submission.to_csv(out_path, index=False)

    print(f"Saved submission -> {out_path}")


if __name__ == "__main__":
    main()
