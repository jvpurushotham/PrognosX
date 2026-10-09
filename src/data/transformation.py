"""
PrognosX — Data Transformation
=================================
Phase 4 (RUL Engineering) + supporting utilities used by Phase 5-7:

  - add_rul_labels_train()      Compute the RUL regression target for every
                                 row of a training trajectory.
  - attach_test_rul()           Attach NASA's provided ground-truth RUL to
                                 the LAST observed cycle of each test engine
                                 (that is the only point C-MAPSS test sets
                                 are scored on).
  - engine_train_val_split()    Split TRAIN engines (by unit_number, not by
                                 row) into a training and validation subset,
                                 so no engine's trajectory leaks across the
                                 split.
  - make_simulated_partial_trajectories()
                                 Realistic-validation trick (per README):
                                 truncate full run-to-failure training
                                 trajectories at a random cycle to mimic the
                                 "stops before failure" shape of the real
                                 test set.
  - create_sequences()          Turn a feature dataframe into fixed-length
                                 sliding-window sequences for LSTM/GRU/TCN/
                                 Transformer models (Phase 7).

RUL definition
--------------
Per the README:

    RUL = Final Failure Cycle - Current Cycle

This module computes that **raw** RUL, and additionally a **clipped**
version (`RUL_CLIP` in src/config/config.py) using the standard piecewise
linear RUL convention from the C-MAPSS literature. Both are kept as
separate columns (`RUL` and `RUL_clipped`) so callers can choose which
target to train against; Phase 6 baseline models use the clipped target by
default (it consistently improves regression performance on this dataset),
while the raw target remains available for anyone who wants to reproduce
the literal README definition exactly.
"""

import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config.config import RANDOM_STATE, RUL_CLIP, VALIDATION_ENGINE_FRACTION
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_rul_labels_train(df: pd.DataFrame, clip_at: Optional[int] = RUL_CLIP) -> pd.DataFrame:
    """Add `RUL` (raw) and `RUL_clipped` columns to a training dataframe.

    For each engine, RUL at a given cycle = (that engine's final observed
    cycle) - (current cycle). Since training trajectories run to failure,
    the final observed cycle IS the failure point.

    Parameters
    ----------
    df : training dataframe with `unit_number` and `time_in_cycles`
    clip_at : ceiling for the clipped RUL target. Pass None to skip
        computing `RUL_clipped` entirely.
    """
    try:
        df = df.copy()
        max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycle - df["time_in_cycles"]

        if clip_at is not None:
            df["RUL_clipped"] = df["RUL"].clip(upper=clip_at)

        logger.info(
            "Added RUL labels: raw RUL range=[%d, %d]%s",
            df["RUL"].min(),
            df["RUL"].max(),
            f", clipped at {clip_at}" if clip_at is not None else "",
        )
        return df
    except Exception as e:
        raise PrognosXException(e, sys)


def attach_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """Attach NASA's ground-truth RUL to the last observed cycle of each
    test engine.

    C-MAPSS test trajectories stop before failure; `RUL_df` (from
    RUL_FD00X.txt) gives the true remaining life at the point each test
    trajectory was cut off. This is only meaningful for the LAST row of
    each engine — earlier rows do not have a ground-truth RUL from this
    file (their true RUL would need to be derived from cycle offsets).

    Returns
    -------
    A copy of `test_df` with an additional `RUL` column, populated only on
    each engine's final row (NaN elsewhere).
    """
    try:
        test_df = test_df.copy()
        last_cycle_idx = test_df.groupby("unit_number")["time_in_cycles"].idxmax()

        rul_lookup = rul_df.set_index("unit_number")["RUL"]

        test_df["RUL"] = np.nan
        for idx in last_cycle_idx:
            engine_id = test_df.loc[idx, "unit_number"]
            if engine_id in rul_lookup.index:
                test_df.loc[idx, "RUL"] = rul_lookup.loc[engine_id]

        n_labeled = test_df["RUL"].notna().sum()
        logger.info(
            "Attached test RUL to %d/%d engines (final-cycle rows only)",
            n_labeled,
            test_df["unit_number"].nunique(),
        )
        return test_df
    except Exception as e:
        raise PrognosXException(e, sys)


def engine_train_val_split(
    df: pd.DataFrame,
    val_fraction: float = VALIDATION_ENGINE_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training dataframe into train/validation by ENGINE, not row.

    Uses sklearn's GroupShuffleSplit keyed on `unit_number` so that every
    row belonging to a given engine ends up entirely in either the train
    or the validation split — never split across both (this is the
    "Avoiding Data Leakage" requirement from the README).
    """
    try:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=random_state
        )
        train_idx, val_idx = next(
            splitter.split(df, groups=df["unit_number"])
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_engines = set(train_df["unit_number"].unique())
        val_engines = set(val_df["unit_number"].unique())
        overlap = train_engines & val_engines
        if overlap:
            # Should be impossible with GroupShuffleSplit, but assert
            # explicitly since this is the most important invariant here.
            raise ValueError(f"Engine leakage detected between splits: {overlap}")

        logger.info(
            "Engine-level split: %d train engines / %d val engines "
            "(%.0f%% held out)",
            len(train_engines),
            len(val_engines),
            val_fraction * 100,
        )
        return train_df, val_df
    except Exception as e:
        raise PrognosXException(e, sys)


def make_simulated_partial_trajectories(
    df: pd.DataFrame, random_state: int = RANDOM_STATE, min_fraction: float = 0.3
) -> pd.DataFrame:
    """Truncate each engine's full run-to-failure trajectory at a random
    cycle, mimicking the "stops before failure" shape of the real test
    set (per README's "Realistic Validation Strategy").

    This should be applied to a VALIDATION split only (never to the data
    a model is trained on), so that validation performance reflects the
    same partial-trajectory conditions the model will see at inference
    time.

    `df` must already have RUL labels attached (see add_rul_labels_train).
    """
    try:
        rng = np.random.default_rng(random_state)
        truncated = []
        for engine_id, group in df.groupby("unit_number"):
            group = group.sort_values("time_in_cycles")
            n = len(group)
            min_len = max(1, int(n * min_fraction))
            cutoff = rng.integers(min_len, n + 1)
            truncated.append(group.iloc[:cutoff])

        result = pd.concat(truncated, ignore_index=True)
        logger.info(
            "Created simulated partial trajectories for %d engines "
            "(avg retained length: %.1f%% of full life)",
            df["unit_number"].nunique(),
            100 * len(result) / len(df),
        )
        return result
    except Exception as e:
        raise PrognosXException(e, sys)


def create_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    sequence_length: int,
    id_column: str = "unit_number",
    time_column: str = "time_in_cycles",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-length sliding-window sequences for sequence models
    (LSTM/GRU/Temporal CNN/Transformer — Phase 7).

    For each engine, produces overlapping windows of `sequence_length`
    consecutive cycles. Engines shorter than `sequence_length` are
    left-padded by repeating their first row, so short trajectories are
    not silently dropped.

    Returns
    -------
    X : np.ndarray of shape (n_samples, sequence_length, n_features)
    y : np.ndarray of shape (n_samples,) — label at the LAST cycle of each window
    engine_ids : np.ndarray of shape (n_samples,) — which engine each window belongs to
        (needed later to keep train/val splits engine-consistent)
    """
    try:
        X, y, engine_ids = [], [], []

        for engine_id, group in df.groupby(id_column):
            group = group.sort_values(time_column)
            values = group[feature_columns].to_numpy(dtype=np.float32)
            labels = group[label_column].to_numpy(dtype=np.float32)
            n = len(group)

            if n < sequence_length:
                pad_width = sequence_length - n
                pad = np.repeat(values[[0]], pad_width, axis=0)
                values = np.vstack([pad, values])
                labels = np.concatenate([np.repeat(labels[0], pad_width), labels])
                n = sequence_length

            for end in range(sequence_length, n + 1):
                start = end - sequence_length
                X.append(values[start:end])
                y.append(labels[end - 1])
                engine_ids.append(engine_id)

        X = np.stack(X)
        y = np.array(y, dtype=np.float32)
        engine_ids = np.array(engine_ids)

        logger.info(
            "Created %d sequences of length %d with %d features",
            X.shape[0],
            sequence_length,
            X.shape[2],
        )
        return X, y, engine_ids
    except Exception as e:
        raise PrognosXException(e, sys)


def create_test_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int,
    id_column: str = "unit_number",
    time_column: str = "time_in_cycles",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ONE sequence per test engine, taken from its last
    `sequence_length` cycles (left-padded if shorter). This mirrors how
    the real inference scenario works: predict RUL at the last observed
    cycle for a partial trajectory.

    Returns
    -------
    X : np.ndarray of shape (n_engines, sequence_length, n_features)
    engine_ids : np.ndarray of shape (n_engines,)
    """
    try:
        X, engine_ids = [], []

        for engine_id, group in df.groupby(id_column):
            group = group.sort_values(time_column)
            values = group[feature_columns].to_numpy(dtype=np.float32)
            n = len(group)

            if n < sequence_length:
                pad_width = sequence_length - n
                pad = np.repeat(values[[0]], pad_width, axis=0)
                values = np.vstack([pad, values])
            else:
                values = values[-sequence_length:]

            X.append(values)
            engine_ids.append(engine_id)

        X = np.stack(X)
        engine_ids = np.array(engine_ids)
        logger.info("Created %d test sequences (one per engine)", X.shape[0])
        return X, engine_ids
    except Exception as e:
        raise PrognosXException(e, sys)
