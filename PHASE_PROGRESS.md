# PrognosX — Phase Progress Tracker

This file tracks implementation status against the roadmap in README.md.
Update the checkboxes as each phase is completed.

## Phase 1 — Project Foundation DONE
- [x] Create GitHub repository (do this after downloading this zip — see SETUP.md)
- [x] Create repository structure
- [x] Configure Python environment (requirements.txt)
- [x] Create `requirements.txt`
- [x] Configure logging (`src/utils/logger.py`)
- [x] Configure project settings (`src/config/config.py`)

## Phase 2 — Dataset Acquisition DONE (code) / PENDING (your action)
- [ ] Download C-MAPSS  <-- **YOU need to place the files** (see SETUP.md)
- [ ] Store raw files in `data/raw/CMAPSS/`
- [x] Understand FD001–FD004 (`src/config/config.py::DATASET_VARIANTS`)
- [x] Document dataset metadata
- [x] Implement ingestion (`src/data/ingestion.py`)
- [x] Implement validation (`src/data/validation.py`)

## Phase 3 — FD001 Exploration DONE (code, notebooks ready to run)
- [x] Load FD001 (`notebooks/01_data_understanding.ipynb`)
- [x] Assign column names (`src/config/config.py::COLUMN_NAMES`)
- [x] Identify engines
- [x] Analyze cycle distributions
- [x] Analyze sensor distributions (`notebooks/02_eda.ipynb`)
- [x] Identify constant sensors
- [x] Analyze sensor correlations
- [x] Plot degradation trajectories (`notebooks/03_sensor_analysis.ipynb`)

> Notebooks 01–03 are fully coded but **not yet executed** — they need the
> real C-MAPSS files in `data/raw/CMAPSS/` to run. Once you upload the
> dataset, run them top-to-bottom in Jupyter.

---

## Phase 4 — RUL Engineering NOT STARTED
## Phase 5 — Feature Engineering NOT STARTED
## Phase 6 — Baseline Models NOT STARTED
## Phase 7 — Advanced RUL Models NOT STARTED
## Phase 8 — Failure Risk Modeling NOT STARTED
## Phase 9 — Survival Analysis NOT STARTED
## Phase 10 — Explainability NOT STARTED
## Phase 11 — Production Pipeline NOT STARTED
## Phase 12 — API NOT STARTED
## Phase 13 — Containerization NOT STARTED
## Phase 14 — CI/CD NOT STARTED
## Phase 15 — FD002 NOT STARTED
## Phase 16 — FD003 NOT STARTED
## Phase 17 — FD004 NOT STARTED

(Full detail for each phase is in `README.md`. We will build these out
incrementally in upcoming sessions.)
