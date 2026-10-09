"""
PrognosX — Baseline Model Definitions
========================================
Phase 6. Defines the four baseline RUL regressors named in the README:

    Linear Regression
    Random Forest Regressor
    XGBoost Regressor
    LightGBM

Kept as a single factory function (`get_baseline_models`) so notebooks
and `train.py` share one definition of "what a baseline model is" and
hyperparameters don't drift between them.
"""

from typing import Dict

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.config.config import RANDOM_STATE


def get_baseline_models() -> Dict[str, object]:
    """Return a dict of {model_name: unfitted sklearn-compatible regressor}."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }
