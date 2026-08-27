"""
PrognosX — Data Validation
==============================
Checks (per README, src/data/ responsibilities):
  - Missing values
  - Duplicates
  - Data types
  - Sensor ranges
  - Engine IDs
  - Cycle ordering
  - Unexpected values

Produces a JSON validation report and raises no exception on
failed checks by default (failures are recorded, not fatal) —
callers can decide whether to halt the pipeline based on the
returned report.
"""

import json
import sys

import pandas as pd

from src.config.config import COLUMN_NAMES, SENSOR_COLUMNS
from src.entity.config_entity import DataValidationConfig
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidation:
    """Runs structural and sanity checks on an ingested C-MAPSS dataframe."""

    def __init__(self, config: DataValidationConfig):
        self.config = config
        if not self.config.expected_columns:
            self.config.expected_columns = COLUMN_NAMES

    def validate(self, df: pd.DataFrame, dataset_name: str = "train") -> dict:
        try:
            report = {"dataset": dataset_name, "variant": self.config.variant}

            # 1. Schema check
            missing_cols = set(self.config.expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(self.config.expected_columns)
            report["schema_ok"] = not missing_cols and not extra_cols
            report["missing_columns"] = sorted(missing_cols)
            report["extra_columns"] = sorted(extra_cols)

            # 2. Missing values
            null_counts = df.isnull().sum()
            report["has_missing_values"] = bool(null_counts.sum() > 0)
            report["missing_value_columns"] = (
                null_counts[null_counts > 0].to_dict()
            )

            # 3. Duplicate rows (same engine + cycle repeated)
            if {"unit_number", "time_in_cycles"}.issubset(df.columns):
                dup_mask = df.duplicated(
                    subset=["unit_number", "time_in_cycles"], keep=False
                )
                report["duplicate_engine_cycle_rows"] = int(dup_mask.sum())
            else:
                report["duplicate_engine_cycle_rows"] = None

            # 4. Data types
            report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            non_numeric = [
                col
                for col in df.columns
                if col not in ("unit_number",) and not pd.api.types.is_numeric_dtype(df[col])
            ]
            report["non_numeric_columns"] = non_numeric

            # 5. Engine IDs — should be positive integers, contiguous-ish
            if "unit_number" in df.columns:
                engine_ids = sorted(df["unit_number"].unique().tolist())
                report["num_engines"] = len(engine_ids)
                report["engine_id_min"] = int(min(engine_ids))
                report["engine_id_max"] = int(max(engine_ids))
                report["engine_ids_positive"] = bool(min(engine_ids) > 0)

            # 6. Cycle ordering — cycles within each engine must be
            #    strictly increasing by 1, starting at 1
            cycle_issues = []
            if {"unit_number", "time_in_cycles"}.issubset(df.columns):
                for engine_id, group in df.groupby("unit_number"):
                    cycles = group["time_in_cycles"].to_numpy()
                    expected = list(range(1, len(cycles) + 1))
                    if list(cycles) != expected:
                        cycle_issues.append(int(engine_id))
            report["engines_with_cycle_ordering_issues"] = cycle_issues

            # 7. Sensor ranges — flag sensors that are constant (zero variance)
            #    NOTE: constant sensors are not necessarily errors, but are
            #    useful to flag for feature selection later.
            constant_sensors = []
            for sensor in SENSOR_COLUMNS:
                if sensor in df.columns and df[sensor].nunique(dropna=False) <= 1:
                    constant_sensors.append(sensor)
            report["constant_sensors"] = constant_sensors

            # Overall pass/fail (soft — informational, not blocking)
            report["passed"] = (
                report["schema_ok"]
                and not report["has_missing_values"]
                and report["duplicate_engine_cycle_rows"] in (0, None)
                and not cycle_issues
            )

            logger.info(
                "Validation for %s [%s]: passed=%s, engines=%s, "
                "constant_sensors=%s",
                self.config.variant,
                dataset_name,
                report["passed"],
                report.get("num_engines"),
                constant_sensors,
            )

            return report

        except Exception as e:
            raise PrognosXException(e, sys)

    def save_report(self, report: dict, suffix: str = "") -> None:
        try:
            path = self.config.report_path
            if suffix:
                path = path.with_name(path.stem + f"_{suffix}" + path.suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Validation report saved to %s", path)
        except Exception as e:
            raise PrognosXException(e, sys)


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.data.validation
    from src.data.ingestion import DataIngestion
    from src.entity.config_entity import DataIngestionConfig

    ingestion = DataIngestion(DataIngestionConfig(variant="FD001"))
    data = ingestion.run(persist=False)

    validator = DataValidation(DataValidationConfig(variant="FD001"))
    train_report = validator.validate(data["train"], dataset_name="train")
    validator.save_report(train_report, suffix="train")
    print(json.dumps(train_report, indent=2, default=str))
