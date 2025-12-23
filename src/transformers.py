from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """House Prices missing-value handling.

    Notes on leakage:
    - Any statistics used to fill missing values (e.g., group medians) are learned in `fit()`
      on the training fold, then applied in `transform()` for both train/valid/test.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # LotFrontage: median by Neighborhood (fallback to global median)
        if "LotFrontage" in X.columns:
            self.lotfrontage_global_median_ = float(pd.to_numeric(X["LotFrontage"], errors="coerce").median())
        else:
            self.lotfrontage_global_median_ = 0.0

        if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
            lf = pd.to_numeric(X["LotFrontage"], errors="coerce")
            self.lotfrontage_by_neighborhood_ = (
                pd.DataFrame({"Neighborhood": X["Neighborhood"], "LotFrontage": lf})
                .groupby("Neighborhood")["LotFrontage"]
                .median()
            )
        else:
            self.lotfrontage_by_neighborhood_ = pd.Series(dtype=float)

        # Mode fills that should also be learned from train only (optional but safer)
        self.mszoning_mode_ = X["MSZoning"].mode(dropna=True).iloc[0] if "MSZoning" in X.columns else None
        self.utilities_mode_ = X["Utilities"].mode(dropna=True).iloc[0] if "Utilities" in X.columns else None
        self.exterior1st_mode_ = X["Exterior1st"].mode(dropna=True).iloc[0] if "Exterior1st" in X.columns else None
        self.exterior2nd_mode_ = X["Exterior2nd"].mode(dropna=True).iloc[0] if "Exterior2nd" in X.columns else None
        self.kitchenqual_mode_ = X["KitchenQual"].mode(dropna=True).iloc[0] if "KitchenQual" in X.columns else None
        self.functional_mode_ = X["Functional"].mode(dropna=True).iloc[0] if "Functional" in X.columns else None
        self.sale_type_mode_ = X["SaleType"].mode(dropna=True).iloc[0] if "SaleType" in X.columns else None
        self.electrical_mode_ = X["Electrical"].mode(dropna=True).iloc[0] if "Electrical" in X.columns else None

        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        none_cols = [
            "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1",
            "MasVnrType",
        ]
        for c in none_cols:
            if c in df.columns:
                df[c] = df[c].fillna("None")

        zero_cols = [
            "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
            "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
        ]
        for c in zero_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        # LotFrontage: use training-fold medians only
        if "LotFrontage" in df.columns:
            lf = pd.to_numeric(df["LotFrontage"], errors="coerce")
            if "Neighborhood" in df.columns and len(self.lotfrontage_by_neighborhood_) > 0:
                mapped = df["Neighborhood"].map(self.lotfrontage_by_neighborhood_)
                lf = lf.fillna(mapped)
            lf = lf.fillna(self.lotfrontage_global_median_)
            df["LotFrontage"] = lf

        # Mode fills (train-only)
        def _fill_mode(col, mode_val):
            if col in df.columns and mode_val is not None:
                df[col] = df[col].fillna(mode_val)

        _fill_mode("MSZoning", self.mszoning_mode_)
        _fill_mode("Utilities", self.utilities_mode_)
        _fill_mode("Exterior1st", self.exterior1st_mode_)
        _fill_mode("Exterior2nd", self.exterior2nd_mode_)
        _fill_mode("KitchenQual", self.kitchenqual_mode_)
        _fill_mode("Functional", self.functional_mode_)
        _fill_mode("SaleType", self.sale_type_mode_)
        _fill_mode("Electrical", self.electrical_mode_)

        return df


class FeatureEngineerV2(BaseEstimator, TransformerMixin):
    """Your engineered features (tree + linear friendly)."""

    def __init__(self, enable_logs: bool = True):
        self.enable_logs = enable_logs

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        # ---------- Time / age ----------
        if {"YrSold", "YearBuilt"}.issubset(df.columns):
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
            df["IsNewHouse"] = (df["HouseAge"] <= 5).astype(int)

        if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
            df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

        if {"YearBuilt", "YearRemodAdd"}.issubset(df.columns):
            df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

        # ---------- Areas ----------
        for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]:
            if col not in df.columns:
                df[col] = 0

        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        # Bathrooms
        for col in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
            if col not in df.columns:
                df[col] = 0

        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

        # Porches
        for col in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]:
            if col not in df.columns:
                df[col] = 0
        df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

        # ---------- Amenity flags ----------
        for col in ["TotalBsmtSF", "GarageArea", "Fireplaces", "PoolArea", "WoodDeckSF", "MasVnrArea"]:
            if col not in df.columns:
                df[col] = 0

        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)
        df["HasDeck"] = (df["WoodDeckSF"] > 0).astype(int)
        df["HasMasonryVeneer"] = (df["MasVnrArea"] > 0).astype(int)

        df["LuxuryAmenityScore"] = (
            df["HasPool"] + df["HasMasonryVeneer"] + df["HasDeck"] + df["HasPorch"] + df["HasFireplace"]
        )

        # ---------- Interactions ----------
        for col in ["OverallQual", "GrLivArea"]:
            if col not in df.columns:
                df[col] = 0

        df["QualGrLiv"] = df["OverallQual"] * df["GrLivArea"]
        df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
        df["QualGarage"] = df["OverallQual"] * df["GarageArea"]

        # ---------- Threshold / buckets ----------
        df["IsHighQuality"] = (df["OverallQual"] >= 7).astype(int)
        df["IsLargeHouse"] = (df["GrLivArea"] >= 2000).astype(int)
        df["IsLuxury"] = ((df["OverallQual"] >= 8) & (df["GrLivArea"] >= 2500)).astype(int)

        if "GrLivArea" in df.columns:
            df["GrLivAreaBin"] = pd.cut(
                df["GrLivArea"],
                bins=[-np.inf, 1200, 2000, np.inf],
                labels=["small", "mid", "large"],
            ).astype(str)

        # ---------- log features ----------
        if self.enable_logs:
            log_cols = ["LotArea", "LotFrontage", "GrLivArea", "TotalBsmtSF", "MasVnrArea", "GarageArea", "1stFlrSF", "2ndFlrSF"]
            for c in log_cols:
                if c in df.columns:
                    df[c + "_log"] = np.log1p(df[c])

        # Neighborhood × Quality interaction (categorical)
        if {"Neighborhood", "OverallQual"}.issubset(df.columns):
            df["Neighborhood_Qual"] = df["Neighborhood"].astype(str) + "_" + df["OverallQual"].astype(str)

        return df

