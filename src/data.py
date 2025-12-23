from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import default_paths


# Public constants expected by train.py / predict.py
ID_COL = "Id"
TARGET_COL = "SalePrice"


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load Kaggle House Prices train/test CSVs from a directory."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.csv at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test.csv at: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return Dataset(train=train, test=test)


def load_train_test(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader used by CLI pipeline.

    - If data_dir is None, load from config.default_paths().data_raw
    - Returns (train_df, test_df) so callers can do: train_df, test_df = load_train_test()
    """
    if data_dir is None:
        data_dir = default_paths().data_raw
    ds = load_dataset(data_dir)
    return ds.train, ds.test


def split_xy(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split train dataframe into X and y."""
    if TARGET_COL not in train_df.columns:
        raise KeyError(f"Missing target column: {TARGET_COL}")

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    return X, y


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    House Prices: pragmatic missing-value handling (Kaggle-style).

    Principles:
    - "Missing means feature not present" -> fill with 'None' for certain categoricals.
    - Numeric counts/areas missing because feature absent -> fill with 0.
    - Some fields: fill by mode or grouped medians.
    """
    df = df.copy()

    # 1) Missing == not present (categoricals)
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1",
        "MasVnrType",
    ]
    for c in none_cols:
        if c in df.columns:
            df[c] = df[c].fillna("None")

    # 2) Numeric: missing == 0 (feature absent)
    zero_cols = [
        "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
    ]
    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 3) LotFrontage: fill by Neighborhood median (stronger than global median)
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(
            df.groupby("Neighborhood")["LotFrontage"].transform("median")
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # 4) GarageYrBlt: if no garage, use YearBuilt (keeps age logic consistent)
    if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # 5) Mode fills (low missing rate in Kaggle data)
    mode_cols = [
        "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "KitchenQual",
        "Functional", "SaleType", "Electrical",
    ]
    for c in mode_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

    return df
