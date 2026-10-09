"""
PrognosX — Model Training
============================
Phase 6. Trains every baseline model returned by
`src.models.baseline.get_baseline_models()` on a given feature matrix,
and persists each fitted model + its feature list to `models/rul/`.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.config.config import RUL_MODELS_DIR
from src.models.baseline import get_baseline_models
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Fit every requested baseline model on (X_train, y_train).

    Parameters
    ----------
    model_names : subset of {"linear_regression", "random_forest",
        "xgboost", "lightgbm"}. None trains all four.

    Returns
    -------
    dict of {model_name: fitted_model}
    """
    try:
        all_models = get_baseline_models()
        selected = (
            all_models
            if model_names is None
            else {k: v for k, v in all_models.items() if k in model_names}
        )

        fitted = {}
        for name, model in selected.items():
            logger.info("Training %s on %d samples, %d features...",
                        name, X_train.shape[0], X_train.shape[1])
            model.fit(X_train, y_train)
            fitted[name] = model
            logger.info("Finished training %s", name)

        return fitted
    except Exception as e:
        raise PrognosXException(e, sys)


def save_model(
    model,
    model_name: str,
    feature_columns: List[str],
    variant: str = "FD001",
    output_dir: Path = RUL_MODELS_DIR,
) -> Path:
    """Persist a fitted model + the exact feature column order it expects.

    Storing `feature_columns` alongside the model is essential: at
    inference time (Phase 11/12), the caller must reproduce the same
    column order the model was fit on.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_{variant}.joblib"
        payload = {"model": model, "feature_columns": feature_columns}
        joblib.dump(payload, path)
        logger.info("Saved model '%s' to %s", model_name, path)
        return path
    except Exception as e:
        raise PrognosXException(e, sys)


def train_and_save_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variant: str = "FD001",
    model_names: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Convenience wrapper: train every baseline model and save each one."""
    try:
        fitted = train_baseline_models(X_train, y_train, model_names)
        saved_paths = {}
        for name, model in fitted.items():
            saved_paths[name] = save_model(
                model, name, list(X_train.columns), variant
            )
        return saved_paths
    except Exception as e:
        raise PrognosXException(e, sys)
