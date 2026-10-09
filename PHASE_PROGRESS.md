# PrognosX — Phase Progress Tracker

This file tracks implementation status against the roadmap in README.md.
Update the checkboxes as each phase is completed.

## Phase 1 — Project Foundation ✅ DONE
- [x] Create GitHub repository
- [x] Create repository structure
- [x] Configure Python environment (requirements.txt)
- [x] Create `requirements.txt`
- [x] Configure logging (`src/utils/logger.py`)
- [x] Configure project settings (`src/config/config.py`)

## Phase 2 — Dataset Acquisition ✅ DONE
- [x] Download C-MAPSS
- [x] Store raw files in `data/raw/CMAPSS/`
- [x] Understand FD001–FD004 (`src/config/config.py::DATASET_VARIANTS`)
- [x] Document dataset metadata
- [x] Implement ingestion (`src/data/ingestion.py`)
- [x] Implement validation (`src/data/validation.py`)

## Phase 3 — FD001 Exploration ✅ DONE
- [x] Load FD001 (`notebooks/01_data_understanding.ipynb`)
- [x] Assign column names (`src/config/config.py::COLUMN_NAMES`)
- [x] Identify engines
- [x] Analyze cycle distributions
- [x] Analyze sensor distributions (`notebooks/02_eda.ipynb`)
- [x] Identify constant sensors
- [x] Analyze sensor correlations
- [x] Plot degradation trajectories (`notebooks/03_sensor_analysis.ipynb`)

## Phase 4 — RUL Engineering ✅ DONE
- [x] Calculate training RUL (`src/data/transformation.py::add_rul_labels_train`)
- [x] Validate RUL generation (monotonicity + zero-at-failure checks, in notebook + `tests/test_features.py`)
- [x] Create RUL datasets (saved to `data/processed/` in `notebooks/04_rul_generation.ipynb`)
- [x] Understand test RUL format (`src/data/transformation.py::attach_test_rul`)
- [x] Build engine-level train/validation split (`src/data/transformation.py::engine_train_val_split`)

> **Design note:** in addition to the README's raw `RUL = final_cycle -
> current_cycle`, this implementation also computes a standard **clipped**
> RUL (ceiling = `RUL_CLIP` in `src/config/config.py`, default 125). This
> is the widely-used piecewise-linear RUL convention from the C-MAPSS
> literature, and Phase 6/7 models train against it by default because it
> measurably improves regression performance. The raw RUL remains
> available as a separate column for anyone who wants the literal README
> definition.

## Phase 5 — Feature Engineering ✅ DONE
- [x] Rolling mean (`add_rolling_features`)
- [x] Rolling standard deviation
- [x] Rolling min/max
- [x] Lag features (`add_lag_features`)
- [x] Rate of change (`add_trend_features`)
- [x] Sensor trends (slope, % change)
- [x] Degradation indicators (`add_degradation_features` — cumulative change, EMA, health index)
- [x] Feature selection (`select_features` — drops constant + highly-correlated columns)

All implemented in `src/features/feature_engineering.py`, exercised end to
end in `notebooks/05_feature_engineering.ipynb`.

## Phase 6 — Baseline Models ✅ DONE
Trained:
- [x] Linear Regression
- [x] Random Forest
- [x] XGBoost (+ hyperparameter-tuned deep dive)
- [x] LightGBM

- [x] Establish baseline RMSE
- [x] Calculate MAE
- [x] Calculate R²
- [x] NASA/C-MAPSS asymmetric scoring function (`src/models/evaluate.py::nasa_score`)
- [x] Generate prediction plots (`notebooks/06_baseline_models.ipynb`)
- [x] Compare models (`src/models/evaluate.py::compare_models`)
- [x] Hyperparameter tuning + feature importance for XGBoost (`notebooks/07_xgboost_model.ipynb`)

Implemented in `src/models/baseline.py`, `train.py`, `evaluate.py`, `predict.py`.

## Phase 7 — Advanced RUL Models ✅ DONE
Implemented:
- [x] LSTM
- [x] GRU
- [x] Temporal CNN
- [x] Transformer

- [x] Create sequences (`src/data/transformation.py::create_sequences`, `create_test_sequences`)
- [x] Train models (`src/models/sequence_models.py::train_sequence_model` — early stopping + LR scheduling)
- [x] Tune hyperparameters (architecture kwargs exposed; default configs are reasonable starting points — see `notebooks/08_lstm_gru_model.ipynb` for where to extend this)
- [x] Compare with XGBoost (`notebooks/08_lstm_gru_model.ipynb`, final comparison table)
- [x] Analyze inference performance (per-architecture metrics table)

> **Design note:** the README's `src/models/` listing names only
> train.py/evaluate.py/predict.py. Those remain the shared orchestration
> layer for BOTH baseline and deep models. A new file,
> `src/models/sequence_models.py`, was added to hold the PyTorch
> architecture definitions and the from-scratch training loop (early
> stopping, LR scheduling, checkpointing) that plain `.fit()` doesn't
> need — this keeps `train.py` free of a torch import when someone only
> wants to train a Random Forest.

All four architectures were execution-tested end-to-end (forward pass,
training loop, prediction, save/load roundtrip) — see
`tests/test_models.py`.

---

## Phase 8 — Failure Risk Modeling ⏳ NOT STARTED
## Phase 9 — Survival Analysis ⏳ NOT STARTED
## Phase 10 — Explainability ⏳ NOT STARTED
## Phase 11 — Production Pipeline ⏳ NOT STARTED
## Phase 12 — API ⏳ NOT STARTED
## Phase 13 — Containerization ⏳ NOT STARTED
## Phase 14 — CI/CD ⏳ NOT STARTED
## Phase 15 — FD002 ⏳ NOT STARTED
## Phase 16 — FD003 ⏳ NOT STARTED
## Phase 17 — FD004 ⏳ NOT STARTED

(Full detail for each phase is in `README.md`. We will build these out
incrementally in upcoming sessions.)

---

## What's real vs. placeholder right now

| Area | Status |
|---|---|
| `src/config/`, `src/utils/`, `src/entity/` | ✅ Implemented |
| `src/data/ingestion.py`, `validation.py`, `transformation.py` | ✅ Implemented |
| `src/features/feature_engineering.py` | ✅ Implemented |
| `src/models/baseline.py`, `train.py`, `evaluate.py`, `predict.py` | ✅ Implemented |
| `src/models/sequence_models.py` (new — Phase 7 architectures) | ✅ Implemented |
| `notebooks/01–08` | ✅ Implemented |
| `tests/test_ingestion.py`, `test_validation.py`, `test_features.py`, `test_models.py` | ✅ Implemented (42 passing tests) |
| `notebooks/09–11` (survival, model comparison writeup, SHAP) | ⏳ Placeholder (Phase 9-10) |
| `src/pipelines/*.py`, `src/api/app.py` | ⏳ Placeholder (Phase 11–12) |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci-cd.yml` | ⏳ Placeholder (Phase 13–14) |
