"""
PrognosX — Data Ingestion
============================
Responsible for:
  - Locating raw C-MAPSS files (train/test/RUL) for a given
    dataset variant (FD001-FD004)
  - Reading the whitespace-delimited .txt files
  - Assigning the correct column names (26 columns:
    unit, cycle, 3 settings, 21 sensors)
  - Persisting a clean, typed copy to data/processed/ as parquet

This module does NOT interpret or validate the data — that is
the responsibility of validation.py.
"""

import sys

import pandas as pd

from src.config.config import COLUMN_NAMES
from src.entity.config_entity import DataIngestionConfig
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Loads a single C-MAPSS dataset variant (e.g. FD001)."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _read_raw_txt(self, path) -> pd.DataFrame:
        """Read a whitespace-delimited C-MAPSS file and assign column names."""
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"Raw file not found: {path}. "
                    f"Place the C-MAPSS files inside data/raw/CMAPSS/ "
                    f"before running ingestion."
                )

            # C-MAPSS files are space-delimited with trailing whitespace,
            # which produces two extra empty (NaN) columns if not handled.
            df = pd.read_csv(path, sep=r"\s+", header=None)
            df = df.dropna(axis=1, how="all")  # drop trailing empty columns

            if df.shape[1] != len(COLUMN_NAMES):
                raise ValueError(
                    f"Unexpected column count in {path.name}: "
                    f"got {df.shape[1]}, expected {len(COLUMN_NAMES)}"
                )

            df.columns = COLUMN_NAMES
            df["unit_number"] = df["unit_number"].astype(int)
            df["time_in_cycles"] = df["time_in_cycles"].astype(int)

            logger.info(
                "Loaded %s -> shape=%s, engines=%d",
                path.name,
                df.shape,
                df["unit_number"].nunique(),
            )
            return df

        except Exception as e:
            raise PrognosXException(e, sys)

    def _read_rul_txt(self, path) -> pd.DataFrame:
        """Read the RUL ground-truth file for the test set (1 value per engine)."""
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"RUL file not found: {path}. "
                    f"Place the C-MAPSS files inside data/raw/CMAPSS/ "
                    f"before running ingestion."
                )
            df = pd.read_csv(path, sep=r"\s+", header=None)
            df = df.dropna(axis=1, how="all")
            df.columns = ["RUL"]
            df["unit_number"] = df.index + 1  # RUL rows are ordered by engine id
            logger.info("Loaded %s -> shape=%s", path.name, df.shape)
            return df[["unit_number", "RUL"]]
        except Exception as e:
            raise PrognosXException(e, sys)

    def load_train(self) -> pd.DataFrame:
        return self._read_raw_txt(self.config.train_path)

    def load_test(self) -> pd.DataFrame:
        return self._read_raw_txt(self.config.test_path)

    def load_rul(self) -> pd.DataFrame:
        return self._read_rul_txt(self.config.rul_path)

    def run(self, persist: bool = True) -> dict:
        """Load train/test/RUL for this variant and optionally persist as parquet.

        Returns
        -------
        dict with keys "train", "test", "rul" -> pandas DataFrames
        """
        try:
            train_df = self.load_train()
            test_df = self.load_test()
            rul_df = self.load_rul()

            if persist:
                self.config.processed_dir.mkdir(parents=True, exist_ok=True)
                train_df.to_parquet(self.config.processed_train_path, index=False)
                test_df.to_parquet(self.config.processed_test_path, index=False)
                rul_df.to_parquet(self.config.processed_rul_path, index=False)
                logger.info(
                    "Persisted processed files for %s to %s",
                    self.config.variant,
                    self.config.processed_dir,
                )

            return {"train": train_df, "test": test_df, "rul": rul_df}

        except Exception as e:
            raise PrognosXException(e, sys)


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.data.ingestion
    cfg = DataIngestionConfig(variant="FD001")
    ingestion = DataIngestion(cfg)
    data = ingestion.run()
    print(data["train"].head())
    print(data["rul"].head())
