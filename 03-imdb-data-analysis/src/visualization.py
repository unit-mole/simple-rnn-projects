"""Reusable matplotlib visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def sentiment_distribution_figure(predictions: pd.DataFrame):
    counts = (
        predictions["predicted_sentiment"]
        .value_counts()
        .reindex(["negative", "positive"], fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(6.5, 4))
    axis.bar(counts.index, counts.values)
    axis.set_title("Predicted Sentiment Distribution")
    axis.set_xlabel("Sentiment")
    axis.set_ylabel("Reviews")
    for index, value in enumerate(counts.values):
        axis.text(index, value, str(int(value)), ha="center", va="bottom")
    figure.tight_layout()
    return figure


def confusion_matrix_figure(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix)
    axis.set_title("Confusion Matrix")
    axis.set_xticks([0, 1], ["Predicted negative", "Predicted positive"])
    axis.set_yticks([0, 1], ["Actual negative", "Actual positive"])
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure
