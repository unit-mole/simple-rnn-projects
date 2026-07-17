"""Interactive IMDb movie-review sentiment application."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    COMPARISON_PATH,
    ERROR_ANALYSIS_PATH,
    LIVE_APP_URL,
    MAX_BATCH_REVIEWS,
    MAX_REVIEW_CHARACTERS,
    MAX_UPLOAD_BYTES,
    METRICS_PATH,
    SAMPLE_DATA_PATH,
)
from src.data_preprocessing import clean_review_frame
from src.sentiment_pipeline import load_artifacts, predict_reviews
from src.visualization import sentiment_distribution_figure


st.set_page_config(
    page_title="IMDb Sentiment Analysis — Simple RNN",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.caption(
    "Sequence-based positive/negative review classification using an "
    "Embedding layer and a Simple Recurrent Neural Network."
)

st.warning(
    "Portfolio demonstration: predictions are probabilistic and may be "
    "incorrect for sarcasm, mixed sentiment, unusual wording, or context "
    "outside the IMDb review domain."
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def load_project_tables():
    sample = pd.read_csv(SAMPLE_DATA_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)
    metrics = json.loads(
        Path(METRICS_PATH).read_text(encoding="utf-8")
    )
    errors = pd.read_csv(ERROR_ANALYSIS_PATH)
    return sample, comparison, metrics, errors


try:
    artifacts = get_artifacts()
    (
        sample_frame,
        comparison_frame,
        metric_data,
        error_frame,
    ) = load_project_tables()
except Exception as exc:
    st.error(f"Application artifacts could not be loaded: {exc}")
    st.stop()


chunk_length = int(
    artifacts.model_metadata.get("chunk_length", 80)
)
decision_threshold = float(
    artifacts.model_metadata.get("decision_threshold", 0.43)
)
training_pool_reviews = int(
    artifacts.model_metadata.get("training_pool_reviews", 2_000)
)
model_fit_reviews = int(
    artifacts.model_metadata.get("model_fit_reviews", 1_600)
)
validation_reviews = int(
    artifacts.model_metadata.get("validation_reviews", 400)
)
test_reviews = int(
    artifacts.model_metadata.get("test_reviews", 600)
)


with st.sidebar:
    st.header("Application Mode")
    mode = st.radio(
        "Choose a workflow",
        [
            "Single Review",
            "Sample Reviews",
            "CSV Upload",
            "Model Performance",
        ],
    )
    st.divider()
    st.metric(
        "Input window",
        f"{chunk_length} tokens/chunk",
    )
    st.metric(
        "Decision threshold",
        f"{decision_threshold:.2f}",
    )

    with st.expander("Model scope"):
        st.write(
            "Training pool:",
            f"{training_pool_reviews:,} reviews",
        )
        st.write(
            "Model-fitting partition:",
            f"{model_fit_reviews:,} reviews",
        )
        st.write(
            "Validation partition:",
            f"{validation_reviews:,} reviews",
        )
        st.write(
            "Untouched test partition:",
            f"{test_reviews:,} reviews",
        )

    st.caption(
        "Long reviews are divided into overlapping chunks. "
        "The final score is the mean positive-sentiment probability "
        "across those chunks."
    )


def prediction_column_config() -> dict:
    return {
        "positive_probability": st.column_config.ProgressColumn(
            "Positive Probability",
            min_value=0.0,
            max_value=1.0,
            format="%.1f%%",
        ),
        "confidence": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0.0,
            max_value=1.0,
            format="%.1f%%",
        ),
    }


def display_prediction(
    result: pd.Series,
    report: dict,
) -> None:
    sentiment = str(result["predicted_sentiment"]).title()
    probability = float(result["positive_probability"])
    confidence = float(result["confidence"])

    first, second, third = st.columns(3)
    first.metric("Predicted Sentiment", sentiment)
    second.metric(
        "Positive Probability",
        f"{probability:.1%}",
    )
    third.metric(
        "Confidence",
        f"{confidence:.1%}",
        help="Probability assigned to the predicted class.",
    )

    if result["confidence_band"] == "Low":
        st.warning(
            "Low-confidence prediction: review the result carefully."
        )
    else:
        st.success(
            f"{result['confidence_band']} confidence prediction"
        )

    st.write(result["interpretation"])

    with st.expander(
        "Processed Review and Sequence Details"
    ):
        st.write(result["cleaned_review"])
        st.json(report)


if mode == "Single Review":
    st.subheader("Classify One Movie Review")
    review = st.text_area(
        "Enter or paste a movie review",
        value=(
            "The performances were excellent and the story remained "
            "engaging until the final scene."
        ),
        height=180,
        max_chars=MAX_REVIEW_CHARACTERS,
    )
    st.caption(
        f"Maximum review length: {MAX_REVIEW_CHARACTERS:,} characters."
    )

    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.error("Enter a non-empty movie review.")
        else:
            result_frame, reports = predict_reviews(
                [review],
                artifacts,
            )
            display_prediction(
                result_frame.iloc[0],
                reports[0],
            )

elif mode == "Sample Reviews":
    st.subheader("Score the Included Sample Reviews")
    st.info(
        "The model is binary. Rows marked as mixed are deliberate "
        "stress tests rather than a third model class."
    )
    st.dataframe(
        sample_frame,
        use_container_width=True,
        hide_index=True,
    )

    selected_indices = st.multiselect(
        "Choose sample rows",
        options=sample_frame.index.tolist(),
        default=sample_frame.index[:6].tolist(),
        format_func=lambda index: (
            f"{index + 1}: "
            f"{sample_frame.loc[index, 'review'][:70]}..."
        ),
    )

    if st.button("Score Selected Reviews", type="primary"):
        if not selected_indices:
            st.error("Select at least one sample review.")
        else:
            selected = sample_frame.loc[
                selected_indices
            ].copy()
            result, _ = predict_reviews(
                selected["review"].tolist(),
                artifacts,
            )
            result.insert(
                1,
                "illustrative_tone",
                selected["illustrative_tone"].to_numpy(),
            )
            st.dataframe(
                result[
                    [
                        "review",
                        "illustrative_tone",
                        "predicted_sentiment",
                        "positive_probability",
                        "confidence",
                        "confidence_band",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config=prediction_column_config(),
            )
            st.pyplot(
                sentiment_distribution_figure(result),
                clear_figure=True,
            )

            st.download_button(
                "Download Sample Predictions",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name=(
                    "imdb_sample_sentiment_predictions.csv"
                ),
                mime="text/csv",
            )

elif mode == "CSV Upload":
    st.subheader("Batch Sentiment Classification")
    st.write(
        "Upload a CSV containing a text column named `review`, "
        "`text`, `review_text`, `comment`, or `content`."
    )

    st.download_button(
        "Download CSV Template",
        data=(
            sample_frame[["review"]]
            .head(6)
            .to_csv(index=False)
            .encode("utf-8")
        ),
        file_name="imdb_review_upload_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error(
                f"The file exceeds the "
                f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB "
                "deployment limit."
            )
            st.stop()

        try:
            uploaded_frame = pd.read_csv(uploaded)
            cleaned, quality_report = clean_review_frame(
                uploaded_frame
            )
        except Exception as exc:
            st.error(
                f"The uploaded file could not be processed: {exc}"
            )
        else:
            st.write("Data-quality report")
            st.json(quality_report)
            st.dataframe(
                cleaned.head(25),
                use_container_width=True,
                hide_index=True,
            )

            if len(cleaned) > MAX_BATCH_REVIEWS:
                st.error(
                    f"For deployment stability, upload no more than "
                    f"{MAX_BATCH_REVIEWS:,} reviews."
                )
            elif st.button(
                "Generate Batch Predictions",
                type="primary",
            ):
                result, _ = predict_reviews(
                    cleaned["review"].tolist(),
                    artifacts,
                )

                if "label" in cleaned.columns:
                    result["actual_label"] = (
                        cleaned["label"].to_numpy()
                    )
                    result["actual_sentiment"] = (
                        cleaned["sentiment"].to_numpy()
                    )
                    result["is_correct"] = (
                        result["predicted_label"]
                        == result["actual_label"]
                    )

                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric(
                    "Reviews Scored",
                    f"{len(result):,}",
                )
                metric_b.metric(
                    "Positive Predictions",
                    (
                        f"{(result['predicted_label'] == 1).mean():.1%}"
                    ),
                )
                metric_c.metric(
                    "Low-confidence Results",
                    (
                        f"{(result['confidence_band'] == 'Low').sum():,}"
                    ),
                )

                if "is_correct" in result.columns:
                    st.metric(
                        "Uploaded-label accuracy",
                        f"{result['is_correct'].mean():.1%}",
                    )

                st.pyplot(
                    sentiment_distribution_figure(result),
                    clear_figure=True,
                )
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                    column_config=prediction_column_config(),
                )
                st.download_button(
                    "Download Scored CSV",
                    data=result.to_csv(index=False).encode(
                        "utf-8"
                    ),
                    file_name="imdb_sentiment_predictions.csv",
                    mime="text/csv",
                )

elif mode == "Model Performance":
    st.subheader("Held-Out Model Performance")

    first, second, third, fourth = st.columns(4)
    first.metric(
        "Accuracy",
        f"{float(metric_data['accuracy']):.1%}",
    )
    second.metric(
        "Precision",
        f"{float(metric_data['precision']):.1%}",
    )
    third.metric(
        "Recall",
        f"{float(metric_data['recall']):.1%}",
    )
    fourth.metric(
        "F1-score",
        f"{float(metric_data['f1']):.1%}",
    )

    fifth, sixth, seventh, eighth = st.columns(4)
    fifth.metric(
        "Specificity",
        f"{float(metric_data['specificity']):.1%}",
    )
    sixth.metric(
        "ROC-AUC",
        f"{float(metric_data['roc_auc']):.3f}",
    )
    seventh.metric(
        "PR-AUC",
        f"{float(metric_data['pr_auc']):.3f}",
    )
    eighth.metric(
        "Test Reviews",
        f"{int(metric_data['test_rows']):,}",
    )

    st.subheader("Baseline Comparison")
    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    st.dataframe(
        comparison_frame[display_columns],
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(
        comparison_frame.set_index("model")[
            ["accuracy", "f1"]
        ]
    )

    st.info(
        "For a fair comparison, the TF-IDF baseline is trained on "
        "the same 1,600-review model-fitting partition as the "
        "Simple RNN. It still performs better on the test set, "
        "showing that greater model complexity does not guarantee "
        "stronger generalization."
    )

    st.subheader("Evaluation Visuals")
    image_paths = [
        (
            "Confusion matrix",
            PROJECT_ROOT / "outputs" / "confusion_matrix.png",
        ),
        (
            "ROC curve",
            PROJECT_ROOT / "outputs" / "roc_curve.png",
        ),
        (
            "Precision–recall curve",
            PROJECT_ROOT
            / "outputs"
            / "precision_recall_curve.png",
        ),
        (
            "Training accuracy",
            PROJECT_ROOT / "outputs" / "training_accuracy.png",
        ),
        (
            "Training loss",
            PROJECT_ROOT / "outputs" / "training_loss.png",
        ),
    ]
    for title, image_path in image_paths:
        st.markdown(f"#### {title}")
        st.image(
            str(image_path),
            use_container_width=True,
        )

    st.subheader("Error Analysis Sample")
    st.dataframe(
        error_frame.head(20),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Common failure modes include sarcasm, mixed sentiment, "
        "long reviews with conflicting evidence, rare vocabulary, "
        "and context that exceeds the memory capacity of a basic "
        "Simple RNN."
    )

st.divider()
st.caption(
    "This application is an educational NLP portfolio project. "
    "The output should be interpreted as an estimated sentiment "
    "score, not as a definitive judgment about a reviewer or film."
)
st.caption(f"Live application: {LIVE_APP_URL}")
