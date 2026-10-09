"""
PrognosX — Configuration Entities
====================================
Typed dataclasses that describe the inputs/outputs of each
pipeline stage. Keeping these separate from config.py lets
pipelines (train/inference) build stage-specific config objects
without importing global state everywhere.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.config.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_FILENAMES,
)


@dataclass
class DataIngestionConfig:
    """Configuration for loading a single C-MAPSS dataset variant."""

    variant: str  # e.g. "FD001"
    raw_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR

    @property
    def train_path(self) -> Path:
        return self.raw_dir / RAW_FILENAMES[self.variant]["train"]

    @property
    def test_path(self) -> Path:
        return self.raw_dir / RAW_FILENAMES[self.variant]["test"]

    @property
    def rul_path(self) -> Path:
        return self.raw_dir / RAW_FILENAMES[self.variant]["rul"]

    @property
    def processed_train_path(self) -> Path:
        return self.processed_dir / f"train_{self.variant}.parquet"

    @property
    def processed_test_path(self) -> Path:
        return self.processed_dir / f"test_{self.variant}.parquet"

    @property
    def processed_rul_path(self) -> Path:
        return self.processed_dir / f"RUL_{self.variant}.parquet"


@dataclass
class DataValidationConfig:
    """Configuration for validating an ingested C-MAPSS dataframe."""

    variant: str
    expected_columns: List[str] = field(default_factory=list)
    report_path: Path = None  # type: ignore

    def __post_init__(self):
        if self.report_path is None:
            self.report_path = (
                PROCESSED_DATA_DIR / f"validation_report_{self.variant}.json"
            )
