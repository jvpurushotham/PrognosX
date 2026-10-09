"""
PrognosX — Production Prediction
====================================
Loads a model saved by `src.models.train.save_model` and generates RUL
predictions, ensuring the input dataframe's columns are reordered to
exactly match what the model was trained on.

Phase 6/7 scope: baseline (and, once trained, deep) RUL models only.
Full production inference (failure risk + survival + API wiring) is
Phase 11/12 work.
"""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from src.config.config import RUL_MODELS_DIR
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_name: str, variant: str = "FD001", model_dir: Path = RUL_MODELS_DIR):
    """Load a saved {model, feature_columns} payload."""
    try:
        path = model_dir / f"{model_name}_{variant}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model found at {path}. Train and save it first "
                f"via src.models.train.save_model()."
            )
        payload = joblib.load(path)
        logger.info(
            "Loaded model '%s' (%d features) from %s",
            model_name, len(payload["feature_columns"]), path,
        )
        return payload["model"], payload["feature_columns"]
    except Exception as e:
        raise PrognosXException(e, sys)


def predict_rul(
    model_name: str,
    X: pd.DataFrame,
    variant: str = "FD001",
    model_dir: Path = RUL_MODELS_DIR,
) -> np.ndarray:
    """Load `model_name` and predict RUL for the rows in `X`.

    `X` must contain (at least) all the columns the model was trained
    on — extra columns are ignored, and the correct subset/order is
    reconstructed automatically from the saved feature list.
    """
    try:
        model, feature_columns = load_model(model_name, variant, model_dir)
        missing = set(feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")

        X_ordered = X[feature_columns]
        predictions = model.predict(X_ordered)
        return predictions
    except Exception as e:
        raise PrognosXException(e, sys)
