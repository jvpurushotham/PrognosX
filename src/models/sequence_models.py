"""
PrognosX — Sequence Models (Deep Learning)
=============================================
Phase 7 — Advanced RUL Models: LSTM, GRU, Temporal CNN, Transformer.

NOTE ON REPOSITORY STRUCTURE: the README's `src/models/` listing names
only train.py / evaluate.py / predict.py. Those three files remain the
orchestration layer (shared by baseline AND deep models). This file adds
the PyTorch architecture definitions and a from-scratch training loop
(early stopping, LR scheduling, checkpointing) that plain sklearn-style
`.fit()` doesn't need — kept separate so train.py doesn't have to import
torch just to train a Random Forest.

All four architectures take the same input shape and expose the same
`forward(x) -> (batch,)` interface so they're interchangeable in the
training/evaluation code and in the Phase 6-vs-7 comparison notebook.

Input shape: (batch, sequence_length, n_features)
Output shape: (batch,)  — predicted RUL for the LAST cycle of each window
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config.config import RANDOM_STATE
from src.utils.exception import PrognosXException
from src.utils.logger import get_logger

logger = get_logger(__name__)

torch.manual_seed(RANDOM_STATE)


# ======================================================================
# Dataset
# ======================================================================
class RULSequenceDataset(Dataset):
    """Wraps (X, y) numpy arrays produced by
    `src.data.transformation.create_sequences` for use with a DataLoader.
    """

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ======================================================================
# Architectures
# ======================================================================
class LSTMRegressor(nn.Module):
    """Stacked LSTM -> take last timestep's hidden state -> MLP head."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class GRURegressor(nn.Module):
    """Stacked GRU -> take last timestep's hidden state -> MLP head."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class TemporalCNNRegressor(nn.Module):
    """1D-convolutional (temporal CNN) regressor.

    Convolves over the time axis with increasing dilation, then
    global-average-pools over time before the MLP head — this gives the
    network a receptive field covering the whole input window without
    needing recurrence.
    """

    def __init__(self, n_features: int, num_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, num_channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(num_channels, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        # Conv1d expects (batch, channels, time) — x arrives as (batch, time, features)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    """Transformer encoder over the time axis -> mean-pool -> MLP head."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)


ARCHITECTURES = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "temporal_cnn": TemporalCNNRegressor,
    "transformer": TransformerRegressor,
}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'. Choose from {list(ARCHITECTURES)}")
    return ARCHITECTURES[name](n_features=n_features, **kwargs)


# ======================================================================
# Training loop
# ======================================================================
@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    best_epoch: int
    best_val_loss: float


def train_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    device: Optional[str] = None,
) -> TrainingHistory:
    """Train `model` with Adam + MSE loss and early stopping on validation
    loss. Restores the best-epoch weights into `model` before returning.
    """
    try:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        criterion = nn.MSELoss()

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    running_val_loss += loss.item() * X_batch.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)

            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                epoch, epochs, train_loss, val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, patience,
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        logger.info("Training complete. Best epoch=%d, best val_loss=%.4f", best_epoch, best_val_loss)
        return TrainingHistory(train_losses, val_losses, best_epoch, best_val_loss)
    except Exception as e:
        raise PrognosXException(e, sys)


def predict_sequence_model(model: nn.Module, X: np.ndarray, device: Optional[str] = None, batch_size: int = 256) -> np.ndarray:
    """Run inference for a fitted sequence model over a numpy array of
    sequences, in batches (avoids loading the whole test set onto the
    device at once).
    """
    try:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        dataset = RULSequenceDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                predictions.append(preds.cpu().numpy())

        return np.concatenate(predictions)
    except Exception as e:
        raise PrognosXException(e, sys)


def save_sequence_model(model: nn.Module, architecture: str, model_kwargs: Dict, path: Path) -> None:
    """Save model weights + the constructor kwargs needed to rebuild it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"architecture": architecture, "kwargs": model_kwargs, "state_dict": model.state_dict()},
            path,
        )
        logger.info("Saved %s model to %s", architecture, path)
    except Exception as e:
        raise PrognosXException(e, sys)


def load_sequence_model(path: Path) -> nn.Module:
    """Rebuild and load weights for a model saved by `save_sequence_model`."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["architecture"], **checkpoint["kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        logger.info("Loaded %s model from %s", checkpoint["architecture"], path)
        return model
    except Exception as e:
        raise PrognosXException(e, sys)
