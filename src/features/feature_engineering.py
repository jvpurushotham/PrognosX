"""
PrognosX — Feature Engineering
=================================
Phase 5. Generates, per README:

  - Rolling statistics (mean, std, min, max)
  - Lag features
  - Trend features (slope, rate of change, pct change)
  - Degradation indicators (cumulative change, EMA, health index)
  - Feature selection (drop constant/near-constant sensors,
    prune highly-correlated duplicates)

All functions operate PER ENGINE (grouped by `unit_number`) so that
rolling/lag windows never bleed across different engines' trajectories.
"""

import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config.config import (
    CONSTANT_SENSOR_STD_THRESHOLD,
    EMA_SPAN,
    HIGH_CORRELATION_THRESHOLD,
    LAG_STEPS,
    ROLLING_WINDOWS,
    TREND_WINDOW,
)
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = ROLLING_WINDOWS,
    id_column: str = "unit_number",
) -> pd.DataFrame:
    """Add rolling mean/std/min/max for each column, for each window size.

    Windows are computed with `min_periods=1` so the first few cycles of
    an engine (where a full window isn't yet available) still get a
    value instead of NaN — using whatever history exists so far.
    """
    try:
        df = df.copy()
        grouped = df.groupby(id_column)

        new_cols = {}
        for window in windows:
            for col in columns:
                roll = grouped[col].rolling(window=window, min_periods=1)
                new_cols[f"{col}_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
                new_cols[f"{col}_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
                new_cols[f"{col}_roll_min_{window}"] = roll.min().reset_index(level=0, drop=True)
                new_cols[f"{col}_roll_max_{window}"] = roll.max().reset_index(level=0, drop=True)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        logger.info(
            "Added rolling features for %d columns x %d windows (%d new columns)",
            len(columns), len(windows), len(new_cols),
        )
        return df
    except Exception as e:
        raise PrognosXException(e, sys)


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = LAG_STEPS,
    id_column: str = "unit_number",
) -> pd.DataFrame:
    """Add Sensor(t-lag) columns. Missing early values (before enough
    history exists) are back-filled with the engine's first observed
    value, so early cycles don't introduce NaNs into downstream models.
    """
    try:
        df = df.copy()
        grouped = df.groupby(id_column)

        new_cols = {}
        for lag in lags:
            for col in columns:
                shifted = grouped[col].shift(lag)
                new_cols[f"{col}_lag_{lag}"] = shifted

        lag_df = pd.DataFrame(new_cols, index=df.index)
        # back-fill within each engine group so early rows aren't NaN
        groups = df[id_column]
        lag_df = lag_df.groupby(groups).transform(lambda s: s.bfill())

        df = pd.concat([df, lag_df], axis=1)
        logger.info(
            "Added lag features for %d columns x %d lags (%d new columns)",
            len(columns), len(lags), len(new_cols),
        )
        return df
    except Exception as e:
        raise PrognosXException(e, sys)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope of `series` against cycle index."""
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(y):
        if len(y) < 2:
            return 0.0
        y = np.asarray(y)
        xs = np.arange(len(y))
        xs_mean = xs.mean()
        d = ((xs - xs_mean) ** 2).sum()
        if d == 0:
            return 0.0
        return ((xs - xs_mean) * (y - y.mean())).sum() / d

    return series.rolling(window=window, min_periods=2).apply(slope, raw=True)


def add_trend_features(
    df: pd.DataFrame,
    columns: List[str],
    window: int = TREND_WINDOW,
    id_column: str = "unit_number",
) -> pd.DataFrame:
    """Add slope, rate-of-change, and pct-change trend features.

    - `{col}_slope_{window}`: rolling linear-regression slope over the
      last `window` cycles (positive = increasing, negative = decreasing).
    - `{col}_roc`: first difference (Sensor(t) - Sensor(t-1)).
    - `{col}_pct_change`: percentage change vs. the previous cycle.
    """
    try:
        df = df.copy()
        grouped = df.groupby(id_column)

        new_cols = {}
        for col in columns:
            new_cols[f"{col}_slope_{window}"] = (
                grouped[col].apply(lambda s: _rolling_slope(s, window))
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )
            new_cols[f"{col}_roc"] = grouped[col].diff().fillna(0.0)
            pct = grouped[col].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            new_cols[f"{col}_pct_change"] = pct

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        logger.info("Added trend features for %d columns", len(columns))
        return df
    except Exception as e:
        raise PrognosXException(e, sys)


def add_degradation_features(
    df: pd.DataFrame,
    columns: List[str],
    ema_span: int = EMA_SPAN,
    id_column: str = "unit_number",
) -> pd.DataFrame:
    """Add cumulative-change, exponential-moving-average, and a simple
    per-sensor health-index feature.

    - `{col}_cumsum_change`: cumulative sum of cycle-to-cycle changes —
      captures total drift since the engine started operating.
    - `{col}_ema_{span}`: exponential moving average (recent-weighted
      smoothing, complements the plain rolling mean).
    - `{col}_health_index`: normalized distance from the engine's own
      starting value, scaled to roughly [0, 1] using that engine's
      observed range. 0 = at starting condition, 1 = at the most
      extreme value seen so far.
    """
    try:
        df = df.copy()
        grouped = df.groupby(id_column)

        new_cols = {}
        for col in columns:
            diffs = grouped[col].diff().fillna(0.0)
            new_cols[f"{col}_cumsum_change"] = diffs.groupby(df[id_column]).cumsum()
            new_cols[f"{col}_ema_{ema_span}"] = (
                grouped[col].apply(lambda s: s.ewm(span=ema_span, adjust=False).mean())
                .reset_index(level=0, drop=True)
            )

            first_val = grouped[col].transform("first")
            running_max_dev = grouped[col].apply(
                lambda s: (s - s.iloc[0]).abs().cummax()
            ).reset_index(level=0, drop=True)
            deviation = (df[col] - first_val).abs()
            health = deviation / running_max_dev.replace(0, np.nan)
            new_cols[f"{col}_health_index"] = health.fillna(0.0).clip(0, 1)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        logger.info("Added degradation features for %d columns", len(columns))
        return df
    except Exception as e:
        raise PrognosXException(e, sys)


def select_features(
    df: pd.DataFrame,
    candidate_columns: List[str],
    std_threshold: float = CONSTANT_SENSOR_STD_THRESHOLD,
    corr_threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> List[str]:
    """Return a filtered feature list with constant/near-constant and
    highly-redundant columns removed.

    Two passes:
      1. Drop any column whose standard deviation is below `std_threshold`
         (carries essentially no signal).
      2. Among the survivors, for every pair with |correlation| >
         `corr_threshold`, drop the second column of the pair (keeps
         the first-seen one) to reduce redundancy.
    """
    try:
        std = df[candidate_columns].std()
        non_constant = std[std >= std_threshold].index.tolist()
        dropped_constant = sorted(set(candidate_columns) - set(non_constant))

        corr = df[non_constant].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [
            col for col in upper.columns if any(upper[col] > corr_threshold)
        ]
        selected = [c for c in non_constant if c not in to_drop]

        logger.info(
            "Feature selection: %d candidates -> %d selected "
            "(%d constant/near-constant dropped, %d redundant dropped)",
            len(candidate_columns), len(selected), len(dropped_constant), len(to_drop),
        )
        return selected
    except Exception as e:
        raise PrognosXException(e, sys)


def build_feature_pipeline(
    df: pd.DataFrame,
    sensor_columns: List[str],
    rolling_windows: List[int] = ROLLING_WINDOWS,
    lags: List[int] = LAG_STEPS,
    trend_window: int = TREND_WINDOW,
    ema_span: int = EMA_SPAN,
    id_column: str = "unit_number",
) -> pd.DataFrame:
    """Orchestrate the full Phase 5 feature-engineering pipeline in one call:
    rolling -> lag -> trend -> degradation features, all on `sensor_columns`.
    """
    try:
        df = add_rolling_features(df, sensor_columns, rolling_windows, id_column)
        df = add_lag_features(df, sensor_columns, lags, id_column)
        df = add_trend_features(df, sensor_columns, trend_window, id_column)
        df = add_degradation_features(df, sensor_columns, ema_span, id_column)
        logger.info(
            "Feature pipeline complete: %d total columns", df.shape[1]
        )
        return df
    except Exception as e:
        raise PrognosXException(e, sys)
