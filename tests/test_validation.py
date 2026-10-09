"""
Tests for src/data/validation.py
"""

import pandas as pd
import pytest

from src.config.config import COLUMN_NAMES
from src.data.validation import DataValidation
from src.entity.config_entity import DataValidationConfig


def _make_valid_df():
    rows = []
    for engine_id in [1, 2]:
        for cycle in range(1, 6):
            rows.append([engine_id, cycle] + [0.5] * 3 + list(range(21)))
    return pd.DataFrame(rows, columns=COLUMN_NAMES)


def test_validation_passes_on_clean_data():
    df = _make_valid_df()
    validator = DataValidation(DataValidationConfig(variant="FD001"))
    report = validator.validate(df, dataset_name="train")

    assert report["schema_ok"] is True
    assert report["has_missing_values"] is False
    assert report["duplicate_engine_cycle_rows"] == 0
    assert report["engines_with_cycle_ordering_issues"] == []
    assert report["passed"] is True


def test_validation_detects_missing_values():
    df = _make_valid_df()
    df.loc[0, "sensor_1"] = None
    validator = DataValidation(DataValidationConfig(variant="FD001"))
    report = validator.validate(df, dataset_name="train")

    assert report["has_missing_values"] is True
    assert report["passed"] is False


def test_validation_detects_duplicate_engine_cycle():
    df = _make_valid_df()
    dup_row = df.iloc[[0]]
    df = pd.concat([df, dup_row], ignore_index=True)

    validator = DataValidation(DataValidationConfig(variant="FD001"))
    report = validator.validate(df, dataset_name="train")

    assert report["duplicate_engine_cycle_rows"] == 2
    assert report["passed"] is False


def test_validation_detects_cycle_ordering_issue():
    df = _make_valid_df()
    # Break ordering for engine 1: drop cycle 3
    df = df[~((df["unit_number"] == 1) & (df["time_in_cycles"] == 3))]

    validator = DataValidation(DataValidationConfig(variant="FD001"))
    report = validator.validate(df, dataset_name="train")

    assert 1 in report["engines_with_cycle_ordering_issues"]
    assert report["passed"] is False


def test_validation_flags_constant_sensor():
    df = _make_valid_df()
    df["sensor_1"] = 42.0  # constant across all rows

    validator = DataValidation(DataValidationConfig(variant="FD001"))
    report = validator.validate(df, dataset_name="train")

    assert "sensor_1" in report["constant_sensors"]


def test_save_report_writes_json(tmp_path):
    df = _make_valid_df()
    cfg = DataValidationConfig(
        variant="FD001", report_path=tmp_path / "validation_report_FD001.json"
    )
    validator = DataValidation(cfg)
    report = validator.validate(df)
    validator.save_report(report, suffix="train")

    expected_path = tmp_path / "validation_report_FD001_train.json"
    assert expected_path.exists()
