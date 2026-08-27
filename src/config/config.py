"""
PrognosX — Central Configuration
=================================
Single source of truth for:
  - Filesystem paths
  - C-MAPSS column names
  - Dataset variant metadata (FD001-FD004)

Everything downstream (ingestion, validation, feature engineering,
notebooks) should import from here instead of hard-coding paths
or column names.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------
# Base directories
# ----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # project root

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = Path(os.getenv("DATA_RAW_DIR", DATA_DIR / "raw" / "CMAPSS"))
PROCESSED_DATA_DIR = Path(os.getenv("DATA_PROCESSED_DIR", DATA_DIR / "processed"))
EXTERNAL_DATA_DIR = DATA_DIR / "external"

LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

MODELS_DIR = ROOT_DIR / "models"
RUL_MODELS_DIR = MODELS_DIR / "rul"
FAILURE_MODELS_DIR = MODELS_DIR / "failure"
SURVIVAL_MODELS_DIR = MODELS_DIR / "survival"

# Make sure the directories that must always exist, do.
for _dir in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    LOG_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    RUL_MODELS_DIR,
    FAILURE_MODELS_DIR,
    SURVIVAL_MODELS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------
# C-MAPSS column schema
# ----------------------------------------------------------------
# Column 1            -> unit / engine number
# Column 2            -> operating cycle
# Columns 3-5         -> operational settings 1-3
# Columns 6-26        -> sensor measurements 1-21
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

COLUMN_NAMES = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# ----------------------------------------------------------------
# Dataset variant metadata (from README)
# ----------------------------------------------------------------
DATASET_VARIANTS = {
    "FD001": {
        "train_engines": 100,
        "test_engines": 100,
        "operating_conditions": 1,
        "fault_modes": 1,
        "description": "1 operating condition, 1 fault mode (HPC degradation)",
    },
    "FD002": {
        "train_engines": 260,
        "test_engines": 259,
        "operating_conditions": 6,
        "fault_modes": 1,
        "description": "6 operating conditions, 1 fault mode (HPC degradation)",
    },
    "FD003": {
        "train_engines": 100,
        "test_engines": 100,
        "operating_conditions": 1,
        "fault_modes": 2,
        "description": "1 operating condition, 2 fault modes (HPC + Fan degradation)",
    },
    "FD004": {
        "train_engines": 248,
        "test_engines": 249,
        "operating_conditions": 6,
        "fault_modes": 2,
        "description": "6 operating conditions, 2 fault modes (HPC + Fan degradation)",
    },
}

# Filenames expected inside data/raw/CMAPSS/ (from the NASA archive)
RAW_FILENAMES = {
    variant: {
        "train": f"train_{variant}.txt",
        "test": f"test_{variant}.txt",
        "rul": f"RUL_{variant}.txt",
    }
    for variant in DATASET_VARIANTS
}

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
