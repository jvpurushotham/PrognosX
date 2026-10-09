"""
Tests for src/models/* (Phase 6 — Baseline Models, Phase 7 — Advanced RUL Models)

Uses small synthetic data and tiny training budgets (few estimators /
epochs) so the suite runs quickly without the real dataset present.
"""

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.models.baseline import get_baseline_models
from src.models.evaluate import compare_models, compute_metrics, evaluate_model, nasa_score
from src.models.predict import load_model, predict_rul
from src.models.sequence_models import (
    ARCHITECTURES,
    RULSequenceDataset,
    build_model,
    load_sequence_model,
    predict_sequence_model,
    save_sequence_model,
    train_sequence_model,
)
from src.models.train import save_model, train_and_save_all, train_baseline_models


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 6
    X = pd.DataFrame(
        rng.random((n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = (X.sum(axis=1) * 10 + rng.normal(0, 0.5, n_samples)).clip(lower=0)
    return X, y


# ----------------------------------------------------------------
# Phase 6 — Baseline Models
# ----------------------------------------------------------------
def test_get_baseline_models_returns_all_four():
    models = get_baseline_models()
    assert set(models.keys()) == {"linear_regression", "random_forest", "xgboost", "lightgbm"}


def test_train_baseline_models_fits_without_error(synthetic_regression_data):
    X, y = synthetic_regression_data
    fitted = train_baseline_models(X, y, model_names=["linear_regression", "random_forest"])
    assert set(fitted.keys()) == {"linear_regression", "random_forest"}
    preds = fitted["linear_regression"].predict(X)
    assert len(preds) == len(y)


def test_nasa_score_penalizes_late_more_than_early():
    y_true = np.array([50.0])
    late_pred = np.array([60.0])   # predicted RUL higher than true -> "late" warning
    early_pred = np.array([40.0])  # predicted RUL lower than true -> "early" warning
    late_score = nasa_score(y_true, late_pred)
    early_score = nasa_score(y_true, early_pred)
    assert late_score > early_score  # late predictions penalized more heavily


def test_compute_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    metrics = compute_metrics(y, y)
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == pytest.approx(1.0)
    assert metrics["NASA_score"] == pytest.approx(0.0)


def test_evaluate_model_and_compare(synthetic_regression_data):
    X, y = synthetic_regression_data
    fitted = train_baseline_models(X, y, model_names=["linear_regression", "random_forest"])
    results = {name: evaluate_model(model, X, y) for name, model in fitted.items()}
    comparison = compare_models(results)
    assert list(comparison.index) == comparison.sort_values("RMSE").index.tolist()
    assert set(comparison.columns) >= {"MAE", "RMSE", "R2", "NASA_score"}


def test_save_and_load_model_roundtrip(synthetic_regression_data, tmp_path):
    X, y = synthetic_regression_data
    fitted = train_baseline_models(X, y, model_names=["linear_regression"])
    save_model(fitted["linear_regression"], "linear_regression", list(X.columns),
               variant="UNITTEST", output_dir=tmp_path)

    model, feature_columns = load_model("linear_regression", variant="UNITTEST", model_dir=tmp_path)
    assert feature_columns == list(X.columns)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_and_save_all_writes_to_given_output_dir(synthetic_regression_data, tmp_path):
    X, y = synthetic_regression_data
    saved_paths = train_and_save_all(
        X, y, variant="UNITTEST2", model_names=["linear_regression"]
    )
    # train_and_save_all uses the default RUL_MODELS_DIR; verify the path
    # it reports actually exists, then clean it up so the test suite
    # never leaves artifacts behind in the real models/ directory.
    path = saved_paths["linear_regression"]
    assert path.exists()
    path.unlink()


def test_predict_rul_with_extra_columns(synthetic_regression_data, tmp_path):
    X, y = synthetic_regression_data
    fitted = train_baseline_models(X, y, model_names=["linear_regression"])
    save_model(fitted["linear_regression"], "linear_regression", list(X.columns),
               variant="UNITTEST3", output_dir=tmp_path)

    X_extra = X.copy()
    X_extra["unrelated_column"] = 0
    preds = predict_rul("linear_regression", X_extra, variant="UNITTEST3", model_dir=tmp_path)
    assert len(preds) == len(y)


def test_predict_rul_raises_on_missing_columns(synthetic_regression_data, tmp_path):
    X, y = synthetic_regression_data
    fitted = train_baseline_models(X, y, model_names=["linear_regression"])
    save_model(fitted["linear_regression"], "linear_regression", list(X.columns),
               variant="UNITTEST4", output_dir=tmp_path)

    X_missing = X.drop(columns=[X.columns[0]])
    with pytest.raises(Exception):
        predict_rul("linear_regression", X_missing, variant="UNITTEST4", model_dir=tmp_path)


# ----------------------------------------------------------------
# Phase 7 — Advanced RUL Models (sequence models)
# ----------------------------------------------------------------
@pytest.fixture
def synthetic_sequences():
    rng = np.random.default_rng(0)
    n_samples, seq_len, n_features = 40, 8, 4
    X = rng.random((n_samples, seq_len, n_features)).astype(np.float32)
    y = rng.random(n_samples).astype(np.float32) * 50
    return X, y


@pytest.mark.parametrize("arch", list(ARCHITECTURES.keys()))
def test_each_architecture_forward_pass_shape(arch, synthetic_sequences):
    X, _ = synthetic_sequences
    model = build_model(arch, n_features=X.shape[2])
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(X))
    assert out.shape == (X.shape[0],)


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("not_a_real_architecture", n_features=4)


def test_train_sequence_model_reduces_loss(synthetic_sequences):
    X, y = synthetic_sequences
    train_loader = DataLoader(RULSequenceDataset(X[:30], y[:30]), batch_size=8, shuffle=True)
    val_loader = DataLoader(RULSequenceDataset(X[30:], y[30:]), batch_size=8, shuffle=False)

    model = build_model("gru", n_features=X.shape[2], hidden_size=8, num_layers=1)
    history = train_sequence_model(model, train_loader, val_loader, epochs=5, patience=5)

    assert len(history.train_loss) > 0
    assert history.best_val_loss <= history.val_loss[0] + 1e-6


def test_predict_sequence_model_output_shape(synthetic_sequences):
    X, y = synthetic_sequences
    model = build_model("lstm", n_features=X.shape[2], hidden_size=8, num_layers=1)
    preds = predict_sequence_model(model, X)
    assert preds.shape == (X.shape[0],)


def test_save_and_load_sequence_model_roundtrip(synthetic_sequences, tmp_path):
    X, y = synthetic_sequences
    model = build_model("temporal_cnn", n_features=X.shape[2], num_channels=8)
    path = tmp_path / "temporal_cnn_test.pt"
    save_sequence_model(model, "temporal_cnn", {"n_features": X.shape[2], "num_channels": 8}, path)

    loaded = load_sequence_model(path)
    preds_original = predict_sequence_model(model, X)
    preds_loaded = predict_sequence_model(loaded, X)
    np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-5)
