"""Keras SimpleRNN model construction, training, and persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# TensorFlow is the deployment default. Set KERAS_BACKEND=jax before running
# to use Keras 3 with JAX in another environment.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import joblib
import keras
from keras import callbacks, layers


def build_simple_rnn_model(
    lookback: int,
    n_features: int,
    rnn_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.05,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build the portfolio's primary trainable SimpleRNN regression model."""
    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, n_features), name="sequence_input"),
            layers.SimpleRNN(
                rnn_units,
                activation="tanh",
                return_sequences=False,
                name="simple_rnn",
            ),
            layers.Dropout(dropout_rate, name="dropout"),
            layers.Dense(dense_units, activation="relu", name="dense_features"),
            layers.Dense(1, activation="linear", name="forecast_output"),
        ],
        name="electricity_simple_rnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_simple_rnn(
    model: keras.Model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    epochs: int = 60,
    batch_size: int = 64,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Train chronologically with early stopping; sequence order is not shuffled."""
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-5,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=training_callbacks,
        verbose=verbose,
    )


def save_artifacts(
    model: keras.Model,
    scaler: Any,
    metadata: dict[str, Any],
    model_dir: str | Path,
) -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / "electricity_rnn_model.keras")
    joblib.dump(scaler, model_path / "scaler.pkl")
    (model_path / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
