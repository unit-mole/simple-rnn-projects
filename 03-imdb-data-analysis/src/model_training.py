"""Simple RNN training with fair baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import (
    CHUNK_LENGTH,
    CHUNK_STRIDE,
    MAX_CHUNKS,
    MAX_VOCABULARY,
    RANDOM_SEED,
)
from .model_evaluation import (
    evaluate_binary_classifier,
    select_threshold,
)
from .sequence_generation import (
    aggregate_chunk_probabilities,
    create_batch_chunks,
)
from .text_preprocessing import VocabularyTokenizer, clean_text


@dataclass
class TrainingResult:
    model: object
    tokenizer: VocabularyTokenizer
    history: pd.DataFrame
    threshold: float
    threshold_table: pd.DataFrame
    validation_probabilities: np.ndarray
    validation_metrics: dict
    test_probabilities: np.ndarray
    test_metrics: dict
    baseline_probabilities: np.ndarray
    baseline_metrics: dict
    baseline_vectorizer: TfidfVectorizer
    baseline_classifier: LogisticRegression
    majority_class: int
    majority_probabilities: np.ndarray
    majority_metrics: dict
    model_fit_reviews: int
    validation_reviews: int
    best_epoch: int


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    import keras

    keras.utils.set_random_seed(seed)


def build_simple_rnn_model(
    vocabulary_size: int,
    chunk_length: int = CHUNK_LENGTH,
    embedding_dimension: int = 80,
    rnn_units: int = 48,
) -> object:
    """Build the primary Embedding → SimpleRNN classifier."""
    import keras
    from keras import layers

    inputs = keras.Input(
        shape=(chunk_length,),
        dtype="int32",
        name="token_sequence",
    )
    x = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        mask_zero=True,
        name="embedding",
    )(inputs)
    x = layers.SimpleRNN(
        rnn_units,
        activation="tanh",
        name="simple_rnn",
    )(x)
    x = layers.Dense(
        32,
        activation="relu",
        name="dense_hidden",
    )(x)
    x = layers.Dropout(
        0.30,
        name="dropout",
    )(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="sentiment_output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def _expand_review_labels(
    labels: np.ndarray,
    review_ids: np.ndarray,
) -> np.ndarray:
    return np.asarray(labels, dtype=np.float32)[review_ids]


def train_simple_rnn(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    validation_size: float = 0.20,
    epochs: int = 25,
    batch_size: int = 128,
) -> TrainingResult:
    """Train with a review-level split and untouched final test set."""
    import keras

    required = {"review", "label"}
    for name, frame in {
        "train": train_frame,
        "test": test_frame,
    }.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} frame is missing columns: {sorted(missing)}"
            )

    set_global_seed()

    train_part, validation_part = train_test_split(
        train_frame,
        test_size=validation_size,
        random_state=RANDOM_SEED,
        stratify=train_frame["label"],
    )

    tokenizer = VocabularyTokenizer(MAX_VOCABULARY).fit(
        train_part["review"]
    )

    x_train, train_review_ids, _ = create_batch_chunks(
        tokenizer,
        train_part["review"].tolist(),
        CHUNK_LENGTH,
        CHUNK_STRIDE,
        MAX_CHUNKS,
    )
    y_train = _expand_review_labels(
        train_part["label"].to_numpy(),
        train_review_ids,
    )

    x_validation, validation_review_ids, _ = create_batch_chunks(
        tokenizer,
        validation_part["review"].tolist(),
        CHUNK_LENGTH,
        CHUNK_STRIDE,
        MAX_CHUNKS,
    )
    y_validation_chunks = _expand_review_labels(
        validation_part["label"].to_numpy(),
        validation_review_ids,
    )

    model = build_simple_rnn_model(
        vocabulary_size=len(tokenizer.word_index),
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            min_delta=1e-3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-5,
        ),
    ]

    fitted = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation_chunks,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    validation_chunk_probabilities = model.predict(
        x_validation,
        batch_size=256,
        verbose=0,
    ).reshape(-1)
    validation_probabilities = aggregate_chunk_probabilities(
        validation_chunk_probabilities,
        validation_review_ids,
        len(validation_part),
    )
    threshold, threshold_table = select_threshold(
        validation_part["label"].to_numpy(),
        validation_probabilities,
    )
    validation_metrics = evaluate_binary_classifier(
        validation_part["label"].to_numpy(),
        validation_probabilities,
        threshold,
    )

    x_test, test_review_ids, _ = create_batch_chunks(
        tokenizer,
        test_frame["review"].tolist(),
        CHUNK_LENGTH,
        CHUNK_STRIDE,
        MAX_CHUNKS,
    )
    test_chunk_probabilities = model.predict(
        x_test,
        batch_size=256,
        verbose=0,
    ).reshape(-1)
    test_probabilities = aggregate_chunk_probabilities(
        test_chunk_probabilities,
        test_review_ids,
        len(test_frame),
    )
    test_metrics = evaluate_binary_classifier(
        test_frame["label"].to_numpy(),
        test_probabilities,
        threshold,
    )

    # Fair baseline: use the same model-fitting partition as the RNN.
    baseline_vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    x_baseline_train = baseline_vectorizer.fit_transform(
        train_part["review"]
    )
    x_baseline_test = baseline_vectorizer.transform(
        test_frame["review"]
    )
    baseline_classifier = LogisticRegression(
        max_iter=1_000,
        C=2.0,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )
    baseline_classifier.fit(
        x_baseline_train,
        train_part["label"],
    )
    baseline_probabilities = baseline_classifier.predict_proba(
        x_baseline_test
    )[:, 1]
    baseline_metrics = evaluate_binary_classifier(
        test_frame["label"].to_numpy(),
        baseline_probabilities,
        0.5,
    )

    majority_class = int(train_part["label"].mode().iloc[0])
    majority_probabilities = np.full(
        len(test_frame),
        float(majority_class),
    )
    majority_metrics = evaluate_binary_classifier(
        test_frame["label"].to_numpy(),
        majority_probabilities,
        0.5,
    )

    history = pd.DataFrame(fitted.history)
    history.insert(
        0,
        "epoch",
        np.arange(1, len(history) + 1),
    )
    best_epoch = int(
        history.loc[history["val_loss"].idxmin(), "epoch"]
    )

    return TrainingResult(
        model=model,
        tokenizer=tokenizer,
        history=history,
        threshold=threshold,
        threshold_table=threshold_table,
        validation_probabilities=validation_probabilities,
        validation_metrics=validation_metrics,
        test_probabilities=test_probabilities,
        test_metrics=test_metrics,
        baseline_probabilities=baseline_probabilities,
        baseline_metrics=baseline_metrics,
        baseline_vectorizer=baseline_vectorizer,
        baseline_classifier=baseline_classifier,
        majority_class=majority_class,
        majority_probabilities=majority_probabilities,
        majority_metrics=majority_metrics,
        model_fit_reviews=int(len(train_part)),
        validation_reviews=int(len(validation_part)),
        best_epoch=best_epoch,
    )
