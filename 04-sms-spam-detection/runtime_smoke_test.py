"""Load the deployed artifacts and run a minimal inference smoke test."""

from __future__ import annotations

import numpy as np

from src.spam_detection_pipeline import (
    load_artifacts,
    predict_messages,
)


def main() -> int:
    artifacts = load_artifacts()
    examples = [
        (
            "Congratulations! Claim your free "
            "cash prize now."
        ),
        (
            "Are we still meeting outside the "
            "station at 6 PM?"
        ),
    ]
    predictions, reports = predict_messages(
        examples,
        artifacts,
    )

    if len(predictions) != len(examples):
        raise AssertionError(
            "Inference returned an unexpected row count."
        )
    if len(reports) != len(examples):
        raise AssertionError(
            "Sequence diagnostics are incomplete."
        )

    probabilities = predictions[
        "spam_probability"
    ].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise AssertionError(
            "Inference produced a non-finite probability."
        )
    if not (
        (probabilities >= 0.0)
        & (probabilities <= 1.0)
    ).all():
        raise AssertionError(
            "Inference probability is outside [0, 1]."
        )

    required_columns = {
        "predicted_class",
        "spam_probability",
        "decision_confidence",
        "confidence_band",
        "interpretation",
    }
    missing = required_columns.difference(
        predictions.columns
    )
    if missing:
        raise AssertionError(
            f"Inference output is missing: {sorted(missing)}"
        )

    print(
        predictions[
            [
                "predicted_class",
                "spam_probability",
                "decision_confidence",
            ]
        ].to_string(index=False)
    )
    print("Runtime smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
