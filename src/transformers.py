from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Kaggle House Prices missing-value handling (pragmatic version)."""

    def fit(self, X: pd.DataFrame, y=None):
        # 未来要做“更严格不泄漏”：把 groupby median/mode 存在这里
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

        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            df["LotFrontage"] = df["LotFrontage"].fillna(
                df.groupby("Neighborhood")["LotFrontage"].transform("median")
            )
            df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

        if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
            df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

        mode_cols = [
            "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "KitchenQual",
            "Functional", "SaleType", "Electrical",
        ]
        for c in mode_cols:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

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

