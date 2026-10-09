# PrognosX — Running Phases 4–7

This assumes you've already completed the Phase 1–3 setup in `SETUP.md`
(virtual environment created, real C-MAPSS files in `data/raw/CMAPSS/`,
`pip install -r requirements.txt` run at least once before).

## 1. Re-install dependencies

Phases 4–7 add scikit-learn, XGBoost, LightGBM, and PyTorch. Pull the
updated `requirements.txt`:

```bash
source .venv/bin/activate   # if not already active
pip install -r requirements.txt
```

`torch` is a large package (500MB+). If your machine has a GPU and you
want CUDA acceleration, install PyTorch separately following the
selector at https://pytorch.org/get-started/locally/ **before** running
`pip install -r requirements.txt` — pip will then see it's already
satisfied and skip the plain CPU build.

## 2. Run the notebooks in order

Each notebook writes files that the next one reads, so run them
top-to-bottom, in this order:

```bash
jupyter notebook notebooks/
```

1. **04_rul_generation.ipynb**
   Computes raw + clipped RUL targets, validates them, attaches
   ground-truth RUL to the test set, and builds a leak-free engine-level
   train/validation split. Writes:
   - `data/processed/train_FD001_rul_split_train.parquet`
   - `data/processed/train_FD001_rul_split_val.parquet`
   - `data/processed/test_FD001_rul_labeled.parquet`

2. **05_feature_engineering.ipynb**
   Builds rolling/lag/trend/degradation features on non-constant
   sensors, then runs feature selection. Writes:
   - `data/processed/train_FD001_features.parquet`
   - `data/processed/val_FD001_features.parquet`
   - `data/processed/test_FD001_features.parquet`
   - `data/processed/selected_features_FD001.json`

3. **06_baseline_models.ipynb**
   Trains Linear Regression, Random Forest, XGBoost, and LightGBM;
   evaluates each on the validation split; saves the best-performing
   models. Writes:
   - `models/rul/{model_name}_FD001.joblib` (one per model)
   - `reports/metrics/baseline_model_comparison_FD001.csv`

4. **07_xgboost_model.ipynb**
   Hyperparameter-tunes XGBoost with `RandomizedSearchCV` (using
   `GroupKFold` on engine ID, so no engine leaks across CV folds),
   compares tuned vs. default, and plots feature importance. **Overwrites**
   `models/rul/xgboost_FD001.joblib` with the tuned version.

5. **08_lstm_gru_model.ipynb**
   Builds sliding-window sequences, trains LSTM/GRU/Temporal
   CNN/Transformer, and produces a final comparison table against the
   tuned XGBoost baseline. Writes:
   - `models/rul/{lstm,gru,temporal_cnn,transformer}_FD001.pt`
   - `reports/metrics/full_model_comparison_FD001.csv`

**Expect notebook 8 to take the longest** — training four deep models is
much slower than the tree-based baselines, especially on CPU. If it's
too slow on your machine, reduce `epochs` in the training cell (defaults
to 30 with early stopping at patience=5) or the dataset size.

## 3. Run the automated tests

```bash
pytest -v
```

You should see **42 passing tests** and 2 skipped placeholders
(`test_api.py`, `test_pipeline.py` — reserved for Phase 11/12). The new
tests in `test_features.py` and `test_models.py` cover:

- RUL generation correctness (hits zero at failure, monotonically
  decreasing, clipping behaves correctly)
- Engine-level split has zero leakage
- Every feature-engineering function produces no NaNs and behaves as
  documented
- All four baseline models train, evaluate, save, and reload correctly
- All four deep architectures produce correctly-shaped output, train
  without error, and save/reload with identical predictions

These use small synthetic datasets, so they pass without your real
C-MAPSS files present — they're testing the *code*, not the model
quality on real data.

## 4. Command-line smoke tests (optional)

Every module can also be sanity-checked directly:

```bash
python -m src.data.ingestion       # loads FD001, prints head
python -m src.data.validation      # runs validation, prints JSON report
```

There isn't a single-command smoke test for the full Phase 4-7 pipeline
by design — it's meant to be run through the notebooks, where you can
inspect each intermediate result (plots, tables) before moving to the
next step.

## What changed vs. the literal README file tree

- **`src/models/sequence_models.py` was added.** It wasn't in the
  original `Repository Structure` listing, but Phase 7 needs somewhere
  to put PyTorch architecture definitions and a training loop (early
  stopping, LR scheduling) that a plain scikit-learn `.fit()` call
  doesn't need. `train.py`/`evaluate.py`/`predict.py` remain the shared
  orchestration layer for both baseline and deep models.
- **RUL has two flavors**: `RUL` (raw, exactly as the README defines it)
  and `RUL_clipped` (ceiling at `RUL_CLIP` in `src/config/config.py`,
  default 125). Phase 6/7 models train against the clipped version by
  default — see the design note in `PHASE_PROGRESS.md` for why.

See `PHASE_PROGRESS.md` for the full phase-by-phase checklist.
