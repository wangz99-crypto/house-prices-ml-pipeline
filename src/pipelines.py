from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, StackingRegressor

from .transformers import MissingValueHandler, FeatureEngineerV2


# ------------------------------------------------------------
# Wrapper: make any estimator look like a proper "regressor"
# Works across sklearn versions (tags / _estimator_type changes)
# ------------------------------------------------------------

class AsRegressor(BaseEstimator, RegressorMixin):
    """
    Wrap an estimator (e.g., Pipeline) and make sklearn meta-estimators
    recognize it as a regressor (tags-based).
    """
    _estimator_type = "regressor"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True):
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")

        est_params = {}
        for k in list(params.keys()):
            if k.startswith("estimator__"):
                est_params[k[len("estimator__"):]] = params.pop(k)

        if est_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**est_params)

        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def __getattr__(self, name):
        """
        Delegate attribute access safely without recursion.
        DO NOT use hasattr(self, ...) inside __getattr__.
        """
        # try fitted estimator_ first (if exists)
        try:
            est_fitted = object.__getattribute__(self, "estimator_")
        except AttributeError:
            est_fitted = None

        if est_fitted is not None:
            try:
                return getattr(est_fitted, name)
            except AttributeError:
                pass

        # fallback to base estimator
        est = object.__getattribute__(self, "estimator")
        return getattr(est, name)


   

def shared_pipeline() -> Pipeline:
    return Pipeline([
        ("missing", MissingValueHandler()),
        ("feat", FeatureEngineerV2(enable_logs=True)),
    ])


def preprocessor_for_linear() -> ColumnTransformer:
    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_exclude=np.number)  

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_sel),
            ("cat", cat_pipe, cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



def preprocessor_for_trees() -> ColumnTransformer:
    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_exclude=np.number)  

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_sel),
            ("cat", cat_pipe, cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



def make_ridge(seed: int = 42, *, alpha: float = 12.0):
    pipe = Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_linear()),
        ("model", Ridge(alpha=alpha)),
    ])
    return AsRegressor(pipe)



def make_extratrees(seed: int = 42):
    pipe = Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_trees()),
        ("model", ExtraTreesRegressor(
            n_estimators=1500,
            random_state=seed,
            n_jobs=1,
            max_features="sqrt",
        )),
    ])
    return AsRegressor(pipe)


def make_xgb(seed: int = 42):
    import xgboost as xgb
    pipe = Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_trees()),
        ("model", xgb.XGBRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
        )),
    ])
    return AsRegressor(pipe)


def make_lgbm(seed: int = 42):
    import lightgbm as lgb
    pipe = Pipeline([
        ("shared", shared_pipeline()),
        ("prep", preprocessor_for_trees()),
        ("model", lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )),
    ])
    return AsRegressor(pipe)


def make_voting_mean(seed: int = 42):
    estimators = [
        ("ridge", make_ridge(seed=seed, alpha=6.0)),   
        ("extratrees", make_extratrees(seed=seed)),
        ("xgb", make_xgb(seed=seed)),
        ("lgbm", make_lgbm(seed=seed)),
    ]
    return VotingRegressor(
        estimators=estimators,
        weights=None,
        n_jobs=1,
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
        n_jobs=1,     
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
