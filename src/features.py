from __future__ import annotations
from .data import handle_missing_values
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering designed for strong tabular regressors.

    We keep original columns (tree models can learn splits), and add a small set
    of high-signal composites:
    - TotalSF, TotalBathrooms, TotalPorchSF
    - Age, RemodAge, IsRemodeled
    - Quality-weighted size features
    - Binary amenity flags
    - Log1p versions of highly skewed numeric features (added, not replaced)
    """
    df = df.copy()
    # ensure missing is handled before feature ops
    df = handle_missing_values(df)

    # Basic composites
    if set(["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]).issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if set(["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]).issubset(df.columns):
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    porch_cols = [c for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"] if c in df.columns]
    if porch_cols:
        df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

    # Age features
    if set(["YrSold", "YearBuilt"]).issubset(df.columns):
        df["Age"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    if set(["YrSold", "YearRemodAdd"]).issubset(df.columns):
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int) if "YearBuilt" in df.columns else 0

    # Quality-weighted size (helps capture “nice big house” effect)
    if set(["OverallQual", "GrLivArea"]).issubset(df.columns):
        df["Qual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    if set(["OverallQual", "TotalBsmtSF"]).issubset(df.columns):
        df["Qual_x_BsmtSF"] = df["OverallQual"] * df["TotalBsmtSF"]

    # Amenity flags (binary)
    for col, flag in [
        ("PoolArea", "HasPool"),
        ("GarageArea", "HasGarage"),
        ("TotalBsmtSF", "HasBasement"),
        ("Fireplaces", "HasFireplace"),
    ]:
        if col in df.columns:
            df[flag] = (df[col] > 0).astype(int)

    # Add log1p variants for skewed numerics (safe list)
    skew_cols = [c for c in ["LotArea", "GrLivArea", "TotalSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF"] if c in df.columns]
    for c in skew_cols:
        df[f"{c}_log1p"] = np.log1p(df[c].clip(lower=0))

    return df


@dataclass(frozen=True)
class EncodedData:
    X: np.ndarray
    feature_names: List[str]
    transformer: ColumnTransformer


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals, pass through numeric features."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre


def fit_transform(preprocessor: ColumnTransformer, X_df: pd.DataFrame) -> EncodedData:
    X = preprocessor.fit_transform(X_df)
    feature_names = []

    # Get feature names
    cat = preprocessor.named_transformers_["cat"]
    cat_cols = preprocessor.transformers_[0][2]
    num_cols = preprocessor.transformers_[1][2]

    if hasattr(cat, "get_feature_names_out"):
        feature_names.extend(cat.get_feature_names_out(cat_cols).tolist())
    else:
        feature_names.extend([f"cat_{i}" for i in range(X.shape[1] - len(num_cols))])

    feature_names.extend(num_cols)
    return EncodedData(X=X, feature_names=feature_names, transformer=preprocessor)


def transform(preprocessor: ColumnTransformer, X_df: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X_df)
