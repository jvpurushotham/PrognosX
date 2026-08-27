# PrognosX — Setup Guide (Phases 1–3)

This guide walks you through getting this scaffold running locally, up to
and including Phase 3 (FD001 Exploration).

## 1. Unzip and open the project

```bash
unzip PrognosX.zip
cd PrognosX
```

## 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # installs `src` as an editable package
```

## 3. Place the C-MAPSS dataset

Based on your `CMAPSSData` folder (14 items), copy **all of these files**
into `data/raw/CMAPSS/`:

```
data/raw/CMAPSS/
├── Damage Propagation Modeling.pdf   (reference paper — optional, informational only)
├── readme.txt                        (NASA's original readme — optional)
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
├── RUL_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
└── train_FD004.txt
```

Only `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` are required
for Phase 3. The FD002–FD004 files can go in now too — they'll be used in
Phases 15–17, and the ingestion code already supports all four variants.

**Do not rename these files** — `src/config/config.py` expects these exact
filenames.

## 4. Verify ingestion + validation work

Run the built-in smoke tests directly from the command line:

```bash
python -m src.data.ingestion
python -m src.data.validation
```

If everything is wired correctly, you'll see the first rows of the FD001
train set, and a JSON validation report printed to the console. Processed
parquet copies will appear in `data/processed/`, and a log file will
appear in `logs/`.

## 5. Run the automated tests

`tests/test_ingestion.py` and `tests/test_validation.py` use **synthetic**
data (not the real dataset), so they'll pass even before you add the real
files — this confirms the code itself is correct:

```bash
pytest -v
```

You should see all tests in `test_ingestion.py` and `test_validation.py`
pass; the other test files are intentionally skipped (marked as
placeholders for later phases).

## 6. Run the Phase 3 notebooks

```bash
jupyter notebook notebooks/
```

Run these three, in order, top to bottom:

1. **01_data_understanding.ipynb** — loads FD001, checks shape/engine
   counts against the documented dataset characteristics, runs
   validation, saves a JSON report to `data/processed/`.
2. **02_eda.ipynb** — operational settings + all 21 sensor distributions,
   flags near-constant sensors, plots the sensor correlation heatmap.
3. **03_sensor_analysis.ipynb** — plots per-engine and overlaid
   degradation trajectories, ranks sensors by correlation with
   engine life fraction, flags noisy sensors, and saves a summary CSV to
   `reports/metrics/sensor_analysis_summary_FD001.csv`.

## 7. Push to GitHub (Phase 1 checklist item)

```bash
git init
git add .
git commit -m "PrognosX: Phases 1-3 (foundation, ingestion, FD001 exploration)"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

`.gitignore` is already configured to exclude the raw/processed data,
logs, and generated report artifacts, so your repo stays lightweight.

## What's already implemented vs. placeholder

| Area | Status |
|---|---|
| `src/config/`, `src/utils/`, `src/entity/` |  Implemented |
| `src/data/ingestion.py` | Implemented (all 4 variants) |
| `src/data/validation.py` | Implemented |
| `notebooks/01–03` | Implemented (run once data is in place) |
| `tests/test_ingestion.py`, `tests/test_validation.py` | Implemented |
| `src/data/transformation.py` | Placeholder (Phase 5) |
| `src/features/feature_engineering.py` | Placeholder (Phase 5) |
| `src/models/*.py` | Placeholder (Phase 6+) |
| `src/pipelines/*.py`, `src/api/app.py` | Placeholder (Phase 11–12) |
| `notebooks/04–11` | Placeholder (Phase 4+) |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci-cd.yml` | Placeholder (Phase 13–14) |

See `PHASE_PROGRESS.md` for the full checklist.
