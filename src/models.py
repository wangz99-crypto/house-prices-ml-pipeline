# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

def make_ridge(seed: int = 42):
    return Ridge(alpha=12.0, random_state=seed)

def make_extratrees(seed: int = 42):
    return ExtraTreesRegressor(
        n_estimators=2000,
        random_state=seed,
        n_jobs=-1,
        max_features="sqrt",   # FIX: sklearn no longer accepts "auto"
        min_samples_leaf=1,
        min_samples_split=2,
    )

def make_lgbm(seed: int = 42):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=seed,
        n_jobs=-1,
    )

def make_xgb(seed: int = 42):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=6000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )

MODEL_FACTORY: Dict[str, Callable[[], object]] = {
    "ridge": make_ridge,
    "extratrees": make_extratrees,
    "lgbm": make_lgbm,
    "xgb": make_xgb,
}

