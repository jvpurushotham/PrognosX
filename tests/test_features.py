"""
Tests for src/data/transformation.py (Phase 4 — RUL Engineering)
and src/features/feature_engineering.py (Phase 5 — Feature Engineering)

All tests use small synthetic C-MAPSS-shaped dataframes so they run
without the real dataset present.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.config import COLUMN_NAMES, SENSOR_COLUMNS
from src.data.transformation import (
    add_rul_labels_train,
    attach_test_rul,
    create_sequences,
    create_test_sequences,
    engine_train_val_split,
    make_simulated_partial_trajectories,
)
from src.features.feature_engineering import (
    add_degradation_features,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    build_feature_pipeline,
    select_features,
)


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(0)
    rows = []
    for engine_id in range(1, 11):
        n_cycles = int(rng.integers(20, 50))
        base = rng.random(21) * 50
        for cycle in range(1, n_cycles + 1):
            drift = (cycle / n_cycles) * rng.random(21) * 5
            sensors = base + drift + rng.normal(0, 0.2, 21)
            settings = rng.normal(0, 0.01, 3)
            rows.append([engine_id, cycle] + list(settings) + list(sensors))
    return pd.DataFrame(rows, columns=COLUMN_NAMES)


# ----------------------------------------------------------------
# Phase 4 — RUL Engineering
# ----------------------------------------------------------------
def test_rul_generation_hits_zero_at_last_cycle(synthetic_train_df):
    df = add_rul_labels_train(synthetic_train_df)
    last_rows = df.loc[df.groupby("unit_number")["time_in_cycles"].idxmax()]
    assert (last_rows["RUL"] == 0).all()


def test_rul_generation_is_monotonically_decreasing(synthetic_train_df):
    df = add_rul_labels_train(synthetic_train_df)
    for _, group in df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles")
        assert (group["RUL"].diff().dropna() <= 0).all()


def test_rul_clipping_respects_ceiling(synthetic_train_df):
    df = add_rul_labels_train(synthetic_train_df, clip_at=15)
    assert df["RUL_clipped"].max() <= 15
    assert (df["RUL_clipped"] == df["RUL"].clip(upper=15)).all()


def test_engine_split_has_no_leakage(synthetic_train_df):
    df = add_rul_labels_train(synthetic_train_df)
    train_df, val_df = engine_train_val_split(df, val_fraction=0.3)
    train_engines = set(train_df["unit_number"].unique())
    val_engines = set(val_df["unit_number"].unique())
    assert train_engines.isdisjoint(val_engines)
    assert len(train_engines) + len(val_engines) == df["unit_number"].nunique()


def test_attach_test_rul_only_labels_last_cycle(synthetic_train_df):
    test_df = synthetic_train_df.copy()
    rul_df = pd.DataFrame({
        "unit_number": test_df["unit_number"].unique(),
        "RUL": np.arange(1, test_df["unit_number"].nunique() + 1),
    })
    labeled = attach_test_rul(test_df, rul_df)

    for engine_id, group in labeled.groupby("unit_number"):
        non_null = group["RUL"].notna()
        assert non_null.sum() == 1
        assert non_null.iloc[-1]  # the last row (by original order) is the max cycle


def test_simulated_partial_trajectories_are_shorter():
    rng = np.random.default_rng(1)
    rows = [[1, c] + [0.0] * 24 for c in range(1, 51)]
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df["RUL"] = 50 - df["time_in_cycles"]

    truncated = make_simulated_partial_trajectories(df, random_state=1, min_fraction=0.3)
    assert len(truncated) <= len(df)
    assert len(truncated) >= int(len(df) * 0.3)


def test_create_sequences_shape(synthetic_train_df):
    df = add_rul_labels_train(synthetic_train_df)
    feat_cols = SENSOR_COLUMNS[:4]
    X, y, engine_ids = create_sequences(df, feat_cols, "RUL_clipped", sequence_length=10)
    assert X.shape[1] == 10
    assert X.shape[2] == len(feat_cols)
    assert len(y) == len(X) == len(engine_ids)


def test_create_sequences_pads_short_engines():
    # engine with fewer cycles than sequence_length must still produce 1 sequence
    rows = [[1, c] + [0.0] * 24 for c in range(1, 4)]  # only 3 cycles
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df["RUL_clipped"] = [2, 1, 0]
    X, y, engine_ids = create_sequences(df, SENSOR_COLUMNS[:2], "RUL_clipped", sequence_length=10)
    assert X.shape == (1, 10, 2)


def test_create_test_sequences_one_per_engine(synthetic_train_df):
    feat_cols = SENSOR_COLUMNS[:3]
    X, engine_ids = create_test_sequences(synthetic_train_df, feat_cols, sequence_length=10)
    assert X.shape[0] == synthetic_train_df["unit_number"].nunique()
    assert X.shape[1:] == (10, len(feat_cols))


# ----------------------------------------------------------------
# Phase 5 — Feature Engineering
# ----------------------------------------------------------------
def test_rolling_features_no_nans(synthetic_train_df):
    df = add_rolling_features(synthetic_train_df, SENSOR_COLUMNS[:2], windows=[5])
    roll_cols = [c for c in df.columns if "_roll_" in c]
    assert len(roll_cols) == 2 * 4  # mean/std/min/max per sensor
    assert df[roll_cols].isna().sum().sum() == 0


def test_lag_features_no_nans(synthetic_train_df):
    df = add_lag_features(synthetic_train_df, SENSOR_COLUMNS[:2], lags=[1, 3])
    lag_cols = [c for c in df.columns if "_lag_" in c]
    assert df[lag_cols].isna().sum().sum() == 0


def test_trend_features_columns_created(synthetic_train_df):
    df = add_trend_features(synthetic_train_df, SENSOR_COLUMNS[:2], window=5)
    for col in SENSOR_COLUMNS[:2]:
        assert f"{col}_slope_5" in df.columns
        assert f"{col}_roc" in df.columns
        assert f"{col}_pct_change" in df.columns
    assert not df.isna().any().any()


def test_degradation_features_health_index_bounded(synthetic_train_df):
    df = add_degradation_features(synthetic_train_df, SENSOR_COLUMNS[:2])
    for col in SENSOR_COLUMNS[:2]:
        health = df[f"{col}_health_index"]
        assert health.min() >= 0
        assert health.max() <= 1


def test_build_feature_pipeline_produces_expected_columns(synthetic_train_df):
    df = build_feature_pipeline(synthetic_train_df, SENSOR_COLUMNS[:2])
    assert df.shape[1] > synthetic_train_df.shape[1]
    assert df.isna().sum().sum() == 0


def test_select_features_drops_constant_column(synthetic_train_df):
    df = synthetic_train_df.copy()
    df["constant_sensor"] = 5.0
    candidates = SENSOR_COLUMNS[:3] + ["constant_sensor"]
    selected = select_features(df, candidates)
    assert "constant_sensor" not in selected


def test_select_features_drops_redundant_column(synthetic_train_df):
    df = synthetic_train_df.copy()
    df["duplicate_sensor"] = df[SENSOR_COLUMNS[0]] + 1e-9  # near-perfect correlation
    candidates = [SENSOR_COLUMNS[0], "duplicate_sensor"]
    selected = select_features(df, candidates, corr_threshold=0.99)
    assert len(selected) == 1
