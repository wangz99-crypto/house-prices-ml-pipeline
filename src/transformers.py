from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _safe_mode(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if len(m) else None


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """House Prices missing-value handling (train-fold safe)."""

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = getattr(X, "columns", None)

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

        # Train-only modes (safe)
        self.mszoning_mode_ = _safe_mode(X["MSZoning"]) if "MSZoning" in X.columns else None
        self.utilities_mode_ = _safe_mode(X["Utilities"]) if "Utilities" in X.columns else None
        self.exterior1st_mode_ = _safe_mode(X["Exterior1st"]) if "Exterior1st" in X.columns else None
        self.exterior2nd_mode_ = _safe_mode(X["Exterior2nd"]) if "Exterior2nd" in X.columns else None
        self.kitchenqual_mode_ = _safe_mode(X["KitchenQual"]) if "KitchenQual" in X.columns else None
        self.functional_mode_ = _safe_mode(X["Functional"]) if "Functional" in X.columns else None
        self.sale_type_mode_ = _safe_mode(X["SaleType"]) if "SaleType" in X.columns else None
        self.electrical_mode_ = _safe_mode(X["Electrical"]) if "Electrical" in X.columns else None

        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ["lotfrontage_global_median_", "lotfrontage_by_neighborhood_"])

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

        # GarageYrBlt: if missing, use YearBuilt (keeps age logic consistent)
        if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
            g = pd.to_numeric(df["GarageYrBlt"], errors="coerce")
            yb = pd.to_numeric(df["YearBuilt"], errors="coerce")
            df["GarageYrBlt"] = g.fillna(yb)

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
    def set_output(self, *, transform=None):
        # pandas DataFrame
        return self

class FeatureEngineerV2(BaseEstimator, TransformerMixin):
    """Engineered features (tree + linear friendly)."""
    def __init__(self, enable_logs: bool = True):
        self.enable_logs = enable_logs

    def fit(self, X: pd.DataFrame, y=None):
        # fitted flag
        self.is_fitted_ = True

        
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = getattr(X, "columns", None)

        return self

    def transform(self, X: pd.DataFrame):
        
        df = X.copy()



        # ---------- Time / age ----------
        if {"YrSold", "YearBuilt"}.issubset(df.columns):
            ys = pd.to_numeric(df["YrSold"], errors="coerce")
            yb = pd.to_numeric(df["YearBuilt"], errors="coerce")
            df["HouseAge"] = (ys - yb).clip(lower=0)
            df["IsNewHouse"] = (df["HouseAge"] <= 5).astype(int)

        if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
            ys = pd.to_numeric(df["YrSold"], errors="coerce")
            yr = pd.to_numeric(df["YearRemodAdd"], errors="coerce")
            df["RemodAge"] = (ys - yr).clip(lower=0)

        if {"YearBuilt", "YearRemodAdd"}.issubset(df.columns):
            yb = pd.to_numeric(df["YearBuilt"], errors="coerce")
            yr = pd.to_numeric(df["YearRemodAdd"], errors="coerce")
            df["IsRemodeled"] = (yb != yr).astype(int)

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
        for col in ["GarageArea", "Fireplaces", "PoolArea", "WoodDeckSF", "MasVnrArea"]:
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
            b = pd.cut(
                df["GrLivArea"],
                bins=[-np.inf, 1200, 2000, np.inf],
                labels=["small", "mid", "large"],
            )
            df["GrLivAreaBin"] = b.astype("object").where(~b.isna(), "missing")

        # ---------- log features ----------
        if self.enable_logs:
            log_cols = ["LotArea", "LotFrontage", "GrLivArea", "TotalBsmtSF", "MasVnrArea", "GarageArea", "1stFlrSF", "2ndFlrSF"]
            for c in log_cols:
                if c in df.columns:
                    vals = pd.to_numeric(df[c], errors="coerce").clip(lower=0)
                    df[c + "_log"] = np.log1p(vals)

        # Neighborhood × Quality interaction (categorical)
        if {"Neighborhood", "OverallQual"}.issubset(df.columns):
            df["Neighborhood_Qual"] = df["Neighborhood"].astype(str) + "_" + df["OverallQual"].astype(str)

        return df


