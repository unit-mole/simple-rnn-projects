"""Simple RNN training with class weights and fair baselines."""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

from .config import MAX_SEQUENCE_LENGTH, MAX_VOCABULARY, RANDOM_SEED
from .model_evaluation import evaluate_binary_classifier, select_threshold
from .sequence_generation import texts_to_padded_sequences
from .text_preprocessing import VocabularyTokenizer, clean_text, tokenize

@dataclass
class TrainingResult:
    model: object
    tokenizer: VocabularyTokenizer
    history: pd.DataFrame
    threshold: float
    threshold_table: pd.DataFrame
    test_probabilities: np.ndarray
    test_metrics: dict
    logistic_metrics: dict
    naive_bayes_metrics: dict
    majority_metrics: dict
    class_weights: dict
    train_rows: int
    validation_rows: int
    test_rows: int

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed)
    import keras
    keras.utils.set_random_seed(seed)

def build_simple_rnn_model(
    vocabulary_size: int,
    embedding_dimension: int = 64,
    rnn_units: int = 48,
    dense_units: int = 24,
    dropout_rate: float = 0.30,
) -> object:
    import keras
    from keras import layers
    inputs = keras.Input(
        shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="token_sequence"
    )
    x = layers.Embedding(
        input_dim=vocabulary_size, output_dim=embedding_dimension,
        mask_zero=True, name="embedding",
    )(inputs)
    x = layers.SimpleRNN(rnn_units, activation="tanh", name="simple_rnn")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spam_probability")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

def split_sms_dataset(frame: pd.DataFrame):
    train, temporary = train_test_split(
        frame, test_size=0.30, random_state=RANDOM_SEED, stratify=frame["label"]
    )
    validation, test = train_test_split(
        temporary, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temporary["label"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )

def train_sms_models(
    frame: pd.DataFrame,
    epochs: int = 18,
    batch_size: int = 64,
) -> TrainingResult:
    import keras
    required={"message","clean_message","label"}
    missing=required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training frame is missing: {sorted(missing)}")
    set_global_seed()
    train,validation,test=split_sms_dataset(frame)
    tokenizer=VocabularyTokenizer(MAX_VOCABULARY).fit(train["clean_message"])
    x_train=texts_to_padded_sequences(
        tokenizer,train["clean_message"].tolist(),MAX_SEQUENCE_LENGTH
    )
    x_validation=texts_to_padded_sequences(
        tokenizer,validation["clean_message"].tolist(),MAX_SEQUENCE_LENGTH
    )
    x_test=texts_to_padded_sequences(
        tokenizer,test["clean_message"].tolist(),MAX_SEQUENCE_LENGTH
    )
    y_train=train["label"].to_numpy(dtype=int)
    y_validation=validation["label"].to_numpy(dtype=int)
    y_test=test["label"].to_numpy(dtype=int)

    classes=np.array([0,1])
    weights=compute_class_weight(class_weight="balanced",classes=classes,y=y_train)
    class_weights={int(label):float(weight) for label,weight in zip(classes,weights)}

    model=build_simple_rnn_model(len(tokenizer.word_index))
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",patience=4,min_delta=1e-3,restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",patience=2,factor=.5,min_lr=1e-5
        ),
    ]
    fitted=model.fit(
        x_train,y_train,validation_data=(x_validation,y_validation),
        epochs=epochs,batch_size=batch_size,class_weight=class_weights,
        callbacks=callbacks,shuffle=True,verbose=1,
    )
    validation_probabilities=model.predict(
        x_validation,batch_size=256,verbose=0
    ).reshape(-1)
    threshold,threshold_table=select_threshold(
        y_validation,validation_probabilities
    )
    test_probabilities=model.predict(x_test,batch_size=256,verbose=0).reshape(-1)
    test_metrics=evaluate_binary_classifier(y_test,test_probabilities,threshold)

    tfidf=TfidfVectorizer(
        preprocessor=clean_text,tokenizer=tokenize,token_pattern=None,
        max_features=10_000,ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True
    )
    x_tfidf_train=tfidf.fit_transform(train["message"])
    x_tfidf_test=tfidf.transform(test["message"])

    logistic=LogisticRegression(
        max_iter=1000,class_weight="balanced",C=2.0,
        solver="liblinear",random_state=RANDOM_SEED
    )
    logistic.fit(x_tfidf_train,y_train)
    logistic_probabilities=logistic.predict_proba(x_tfidf_test)[:,1]
    logistic_metrics=evaluate_binary_classifier(y_test,logistic_probabilities,.5)

    nb=MultinomialNB(alpha=.5)
    nb.fit(x_tfidf_train,y_train)
    nb_probabilities=nb.predict_proba(x_tfidf_test)[:,1]
    nb_metrics=evaluate_binary_classifier(y_test,nb_probabilities,.5)

    majority_class=int(train["label"].mode().iloc[0])
    majority_probabilities=np.full(len(test),float(majority_class))
    majority_metrics=evaluate_binary_classifier(y_test,majority_probabilities,.5)

    history=pd.DataFrame(fitted.history)
    history.insert(0,"epoch",np.arange(1,len(history)+1))
    return TrainingResult(
        model=model,tokenizer=tokenizer,history=history,threshold=threshold,
        threshold_table=threshold_table,test_probabilities=test_probabilities,
        test_metrics=test_metrics,logistic_metrics=logistic_metrics,
        naive_bayes_metrics=nb_metrics,majority_metrics=majority_metrics,
        class_weights=class_weights,train_rows=len(train),
        validation_rows=len(validation),test_rows=len(test),
    )
