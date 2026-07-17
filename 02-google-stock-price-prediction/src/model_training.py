"""Simple RNN architecture, training, and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import keras
from keras import callbacks, layers, optimizers


def build_simple_rnn_model(
    window_size: int = 10,
    n_features: int = 1,
    rnn_units: int = 16,
    dropout_rate: float = 0.05,
    learning_rate: float = 0.001,
):
    """Build the portfolio's primary Simple RNN regression model."""
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, n_features), name="return_sequence"),
            layers.SimpleRNN(rnn_units, activation="tanh", name="simple_rnn"),
            layers.Dropout(dropout_rate, name="dropout"),
            layers.Dense(max(8, rnn_units // 2), activation="relu", name="dense_hidden"),
            layers.Dense(1, activation="linear", name="predicted_scaled_return"),
        ],
        name="google_stock_simple_rnn",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        jit_compile=False,
    )
    return model


def train_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    max_epochs: int = 80,
    batch_size: int = 16,
    patience: int = 10,
):
    """Train chronologically with a dedicated validation period and no shuffling."""
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=1e-4,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        callbacks=[early_stopping],
    )
    return history


def save_training_artifacts(
    model,
    feature_scaler,
    target_scaler,
    metadata: dict,
    model_dir: str | Path,
) -> dict:
    """Save model, scalers, and metadata for inference and deployment."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": model_dir / "google_stock_rnn_model.keras",
        "feature_scaler": model_dir / "feature_scaler.joblib",
        "target_scaler": model_dir / "target_scaler.joblib",
        "metadata": model_dir / "model_metadata.json",
    }
    model.save(paths["model"])
    joblib.dump(feature_scaler, paths["feature_scaler"])
    joblib.dump(target_scaler, paths["target_scaler"])
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}
