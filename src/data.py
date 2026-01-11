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
    Convenience loader:
    - If data_dir is None, load from config.default_paths().data_raw
    - Returns (train_df, test_df)
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
