"""
Tests for src/data/ingestion.py

These tests generate a small synthetic C-MAPSS-formatted file so
they can run in CI without requiring the real NASA dataset to be
present.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.config import COLUMN_NAMES
from src.data.ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


@pytest.fixture
def synthetic_raw_dir(tmp_path):
    """Create a tiny synthetic FD001-style train/test/RUL set."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    rng = np.random.default_rng(42)
    n_engines = 3
    rows = []
    for engine_id in range(1, n_engines + 1):
        n_cycles = rng.integers(15, 25)
        for cycle in range(1, n_cycles + 1):
            row = [engine_id, cycle] + list(rng.random(3)) + list(rng.random(21))
            rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMN_NAMES)

    # C-MAPSS raw files: space separated, no header, no index,
    # with a trailing double space (mimics the real file quirk).
    train_path = raw_dir / "train_FD001.txt"
    df.to_csv(train_path, sep=" ", header=False, index=False)

    test_path = raw_dir / "test_FD001.txt"
    df.to_csv(test_path, sep=" ", header=False, index=False)

    rul_path = raw_dir / "RUL_FD001.txt"
    rul_values = pd.DataFrame({"RUL": rng.integers(5, 50, size=n_engines)})
    rul_values.to_csv(rul_path, sep=" ", header=False, index=False)

    return raw_dir


def test_ingestion_loads_expected_columns(synthetic_raw_dir, tmp_path):
    config = DataIngestionConfig(
        variant="FD001",
        raw_dir=synthetic_raw_dir,
        processed_dir=tmp_path / "processed",
    )
    ingestion = DataIngestion(config)

    train_df = ingestion.load_train()

    assert list(train_df.columns) == COLUMN_NAMES
    assert train_df["unit_number"].nunique() == 3
    assert train_df.dtypes["unit_number"] == "int64"
    assert train_df.dtypes["time_in_cycles"] == "int64"


def test_ingestion_loads_rul_file(synthetic_raw_dir, tmp_path):
    config = DataIngestionConfig(
        variant="FD001",
        raw_dir=synthetic_raw_dir,
        processed_dir=tmp_path / "processed",
    )
    ingestion = DataIngestion(config)
    rul_df = ingestion.load_rul()

    assert list(rul_df.columns) == ["unit_number", "RUL"]
    assert len(rul_df) == 3


def test_ingestion_persists_parquet(synthetic_raw_dir, tmp_path):
    processed_dir = tmp_path / "processed"
    config = DataIngestionConfig(
        variant="FD001", raw_dir=synthetic_raw_dir, processed_dir=processed_dir
    )
    ingestion = DataIngestion(config)
    data = ingestion.run(persist=True)

    assert config.processed_train_path.exists()
    assert config.processed_test_path.exists()
    assert config.processed_rul_path.exists()
    assert not data["train"].empty


def test_ingestion_raises_on_missing_file(tmp_path):
    config = DataIngestionConfig(
        variant="FD001",
        raw_dir=tmp_path / "does_not_exist",
        processed_dir=tmp_path / "processed",
    )
    ingestion = DataIngestion(config)

    with pytest.raises(Exception):
        ingestion.load_train()
