"""
PrognosX — Model Evaluation
==============================
Phase 6. Implements the RUL metrics named in the README:

    MAE
    RMSE
    R²
    NASA/C-MAPSS asymmetric scoring function

The NASA scoring function penalizes LATE predictions (predicting a
higher RUL than actual — i.e. underestimating urgency) more heavily
than EARLY predictions (predicting a lower RUL than actual), because in
a real maintenance setting, a late warning is operationally worse than
an early, overly-cautious one.

    d = predicted_RUL - true_RUL

    score = sum( exp(-d/13)  - 1  for d < 0  )      (early prediction)
          + sum( exp( d/10)  - 1  for d >= 0 )      (late prediction)

Lower is better. It is NOT an average — it's a summed score across all
predictions, so it is only comparable across runs with the same number
of samples.
"""

import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA/C-MAPSS asymmetric scoring function (PHM08 challenge)."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    early = d[d < 0]
    late = d[d >= 0]
    score = np.sum(np.exp(-early / 13) - 1) + np.sum(np.exp(late / 10) - 1)
    return float(score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R², and the NASA score for a set of predictions."""
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
            "NASA_score": nasa_score(y_true, y_pred),
            "n_samples": int(len(y_true)),
        }
        return metrics
    except Exception as e:
        raise PrognosXException(e, sys)


def evaluate_model(model, X, y_true) -> Dict[str, float]:
    """Predict with `model` and compute metrics against `y_true`."""
    try:
        y_pred = model.predict(X)
        return compute_metrics(y_true, y_pred)
    except Exception as e:
        raise PrognosXException(e, sys)


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Turn a {model_name: metrics_dict} mapping into a sorted comparison
    table (best RMSE first).
    """
    try:
        comparison = pd.DataFrame(results).T
        comparison = comparison.sort_values("RMSE")
        logger.info("Model comparison:\n%s", comparison.to_string())
        return comparison
    except Exception as e:
        raise PrognosXException(e, sys)
