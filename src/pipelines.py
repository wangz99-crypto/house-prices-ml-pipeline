from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, StackingRegressor


from .transformers import MissingValueHandler, FeatureEngineerV2


def shared_pipeline() -> Pipeline:
    return Pipeline([
        ("missing", MissingValueHandler()),
        ("feat", FeatureEngineerV2(enable_logs=True)),
    ])


def preprocessor_for_linear() -> ColumnTransformer:
    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_include=object)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_sel),
            ("cat", cat_pipe, cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


def preprocessor_for_trees() -> ColumnTransformer:
    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_include=object)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_sel),
            ("cat", cat_pipe, cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct

def make_ridge(seed: int = 42):
    # NOTE: Ridge doesn't accept random_state (only some solvers do), so remove it.
    return Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_linear()),
        ("model", Ridge(alpha=12.0)),
    ])


def make_extratrees(seed: int = 42):
    return Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_trees()),
        ("model", ExtraTreesRegressor(
            n_estimators=1500,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
        )),
    ])


def make_xgb(seed: int = 42):
    import xgboost as xgb
    return Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_trees()),
        ("model", xgb.XGBRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])


def make_lgbm(seed: int = 42):
    import lightgbm as lgb

    return Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_trees()),             
        ("model", lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )),
    ])




def make_voting_mean(seed: int = 42):
    estimators = [
        ("ridge", make_ridge(seed=seed)),
        ("extratrees", make_extratrees(seed=seed)),
        ("xgb", make_xgb(seed=seed)),
        ("lgbm", make_lgbm(seed=seed)),
    ]
    return VotingRegressor(
        estimators=estimators,
        weights=None,   # 显式写出来：就是均值
        n_jobs=-1,
    )






def make_stacking(seed: int = 42):
    estimators = [
        ("ridge", make_ridge(seed=seed)),
        ("extratrees", make_extratrees(seed=seed)),
        ("xgb", make_xgb(seed=seed)),
        ("lgbm", make_lgbm(seed=seed)),
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    final_estimator = Ridge(alpha=1.0)  

    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,                
        passthrough=False,
        n_jobs=-1,
    )



PIPELINES = {
    "ridge": make_ridge,
    "extratrees": make_extratrees,
    "xgb": make_xgb,
    "lgbm": make_lgbm,
    "voting_mean": make_voting_mean,
    "stacking": make_stacking,
}


def get_pipeline(model_name: str, seed: int = 42):
    if model_name not in PIPELINES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {sorted(PIPELINES.keys())}")
    return PIPELINES[model_name](seed=seed)


