"""Interactive SMS spam-detection application."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from src.config import (
    COMPARISON_PATH, ERROR_ANALYSIS_PATH, MAX_BATCH_MESSAGES,
    MAX_MESSAGE_CHARACTERS, MAX_UPLOAD_BYTES, METRICS_PATH, SAMPLE_DATA_PATH,
)
from src.data_preprocessing import clean_sms_frame
from src.spam_detection_pipeline import load_artifacts,predict_messages
from src.visualization import class_distribution_figure

st.set_page_config(
    page_title="SMS Spam Detection — Simple RNN",page_icon="📱",layout="wide"
)
st.title("📱 SMS Spam Detection using Simple RNN")
st.caption(
    "Sequence-based spam-versus-ham classification using an Embedding layer "
    "and a Simple Recurrent Neural Network."
)
st.warning(
    "Responsible-use and privacy notice: This project is for educational and "
    "portfolio demonstration only. Do not upload private, sensitive, or personal "
    "messages. Predictions may be incorrect and must not be the sole basis for "
    "blocking, filtering, monitoring, or taking action on real communications."
)

@st.cache_resource
def get_artifacts():
    return load_artifacts()

@st.cache_data
def load_project_tables():
    sample=pd.read_csv(SAMPLE_DATA_PATH)
    comparison=pd.read_csv(COMPARISON_PATH)
    metrics=json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))
    errors=pd.read_csv(ERROR_ANALYSIS_PATH)
    return sample,comparison,metrics,errors

try:
    artifacts=get_artifacts()
    sample_frame,comparison_frame,metric_data,error_frame=load_project_tables()
except Exception as exc:
    st.error(f"Application artifacts could not be loaded: {exc}")
    st.stop()

threshold=float(artifacts.model_metadata.get("decision_threshold",.255))
max_length=int(artifacts.model_metadata.get("maximum_sequence_length",50))

with st.sidebar:
    st.header("Application Mode")
    mode=st.radio(
        "Choose a workflow",
        ["Single Message","Sample Messages","CSV Upload","Model Performance"],
    )
    st.divider()
    st.metric("Maximum sequence",f"{max_length} tokens")
    st.metric("Spam threshold",f"{threshold:.3f}")
    st.caption(
        "The threshold was selected on validation data to balance spam precision "
        "and recall. The test set remained untouched."
    )

def display_prediction(result: pd.Series,report: dict) -> None:
    predicted_class=str(result["predicted_class"]).upper()
    probability=float(result["spam_probability"])
    confidence=float(result["decision_confidence"])
    first,second,third=st.columns(3)
    first.metric("Predicted Class",predicted_class)
    second.metric("Spam Probability",f"{probability:.1%}")
    third.metric(
        "Decision Confidence",f"{confidence:.1%}",
        help="Normalized distance from the validation-selected threshold.",
    )
    if result["confidence_band"]=="Low":
        st.warning("Low-confidence result: review the message manually.")
    else:
        st.success(f"{result['confidence_band']} decision confidence")
    st.write(result["interpretation"])
    st.info(
        "Visible surface cues (not model feature attribution): "
        + str(result["surface_cues"])
    )
    with st.expander("Processed Message and Sequence Details"):
        st.write(result["clean_message"])
        st.json(report)

if mode=="Single Message":
    st.subheader("Classify One SMS Message")
    message=st.text_area(
        "Enter or paste an SMS message",
        value="Congratulations! You have won a free prize. Reply WIN now to claim.",
        height=160,max_chars=MAX_MESSAGE_CHARACTERS,
    )
    st.caption(f"Maximum message length: {MAX_MESSAGE_CHARACTERS:,} characters.")
    if st.button("Predict SMS Class",type="primary"):
        if not message.strip():
            st.error("Enter a non-empty SMS message.")
        else:
            result,reports=predict_messages([message],artifacts)
            display_prediction(result.iloc[0],reports[0])

elif mode=="Sample Messages":
    st.subheader("Score Privacy-Safe Sample Messages")
    st.info(
        "The illustrative category is not used by the model. Ambiguous rows "
        "deliberately test messages containing ordinary and spam-like wording."
    )
    st.dataframe(sample_frame,use_container_width=True,hide_index=True)
    indices=st.multiselect(
        "Choose sample rows",options=sample_frame.index.tolist(),
        default=sample_frame.index[:6].tolist(),
        format_func=lambda i:f"{i+1}: {sample_frame.loc[i,'message'][:70]}...",
    )
    if st.button("Score Selected Messages",type="primary"):
        if not indices:
            st.error("Select at least one sample message.")
        else:
            selected=sample_frame.loc[indices].copy()
            result,_=predict_messages(selected["message"].tolist(),artifacts)
            result.insert(
                1,"illustrative_category",
                selected["illustrative_category"].to_numpy(),
            )
            display_frame=result.copy()
            display_frame["spam_probability_pct"]=display_frame["spam_probability"]*100
            display_frame["decision_confidence_pct"]=display_frame["decision_confidence"]*100
            st.dataframe(
                display_frame[
                    ["message","illustrative_category","predicted_class",
                     "spam_probability_pct","decision_confidence_pct","confidence_band"]
                ],
                use_container_width=True,hide_index=True,
                column_config={
                    "spam_probability_pct":st.column_config.ProgressColumn(
                        "Spam Probability",min_value=0,max_value=100,format="%.1f%%"
                    ),
                    "decision_confidence_pct":st.column_config.ProgressColumn(
                        "Decision Confidence",min_value=0,max_value=100,format="%.1f%%"
                    ),
                },
            )
            st.pyplot(class_distribution_figure(result),clear_figure=True)
            st.download_button(
                "Download Sample Predictions",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="sms_sample_predictions.csv",mime="text/csv",
            )

elif mode=="CSV Upload":
    st.subheader("Batch SMS Spam Detection")
    st.write(
        "Upload a CSV containing `message`, `text`, `sms`, `body`, `content`, or `v2`."
    )
    st.download_button(
        "Download CSV Template",
        data=sample_frame[["message"]].head(6).to_csv(index=False).encode("utf-8"),
        file_name="sms_spam_upload_template.csv",mime="text/csv",
    )
    uploaded=st.file_uploader("Upload CSV",type=["csv"])
    if uploaded is not None:
        if uploaded.size>MAX_UPLOAD_BYTES:
            st.error("The file exceeds the 5 MB deployment limit.")
            st.stop()
        try:
            uploaded_frame=pd.read_csv(uploaded)
            cleaned,quality_report=clean_sms_frame(uploaded_frame)
        except Exception as exc:
            st.error(f"The uploaded file could not be processed: {exc}")
        else:
            st.write("Data-quality report")
            st.json(quality_report)
            st.dataframe(cleaned.head(25),use_container_width=True,hide_index=True)
            if len(cleaned)>MAX_BATCH_MESSAGES:
                st.error(f"Upload no more than {MAX_BATCH_MESSAGES:,} messages.")
            elif st.button("Generate Batch Predictions",type="primary"):
                result,_=predict_messages(cleaned["message"].tolist(),artifacts)
                if "label" in cleaned.columns:
                    result["actual_label"]=cleaned["label"].to_numpy()
                    result["actual_class"]=cleaned["class_name"].to_numpy()
                    result["is_correct"]=result["predicted_label"]==result["actual_label"]
                a,b,c=st.columns(3)
                a.metric("Messages Scored",f"{len(result):,}")
                b.metric("Spam Predictions",f"{(result['predicted_label']==1).mean():.1%}")
                c.metric(
                    "Low-confidence Results",
                    f"{(result['confidence_band']=='Low').sum():,}",
                )
                if "is_correct" in result.columns:
                    st.metric("Uploaded-label accuracy",f"{result['is_correct'].mean():.1%}")
                st.pyplot(class_distribution_figure(result),clear_figure=True)
                st.dataframe(result,use_container_width=True,hide_index=True)
                st.download_button(
                    "Download Scored CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="sms_spam_predictions.csv",mime="text/csv",
                )

elif mode=="Model Performance":
    st.subheader("Held-Out Model Performance")
    a,b,c,d=st.columns(4)
    a.metric("Accuracy",f"{float(metric_data['accuracy']):.1%}")
    b.metric("Spam Precision",f"{float(metric_data['precision']):.1%}")
    c.metric("Spam Recall",f"{float(metric_data['recall']):.1%}")
    d.metric("Spam F1-score",f"{float(metric_data['f1']):.1%}")
    e,f,g,h=st.columns(4)
    e.metric("Specificity",f"{float(metric_data['specificity']):.1%}")
    f.metric("ROC-AUC",f"{float(metric_data['roc_auc']):.3f}")
    g.metric("PR-AUC",f"{float(metric_data['pr_auc']):.3f}")
    h.metric("Test Messages",f"{int(metric_data['test_rows']):,}")

    st.subheader("Model Comparison")
    columns=["model","accuracy","precision","recall","f1","roc_auc","pr_auc"]
    st.dataframe(comparison_frame[columns],use_container_width=True,hide_index=True)
    st.bar_chart(comparison_frame.set_index("model")[["precision","recall","f1"]])
    st.info(
        "TF-IDF + Logistic Regression is strongest on this test split. "
        "The Simple RNN remains the primary portfolio architecture because "
        "this repository focuses on recurrent sequence modeling."
    )

    st.subheader("Evaluation Visuals")
    visual_files=[
        ("Confusion matrix","confusion_matrix.png"),
        ("ROC curve","roc_curve.png"),
        ("Precision–recall curve","precision_recall_curve.png"),
        ("Threshold analysis","threshold_analysis.png"),
        ("Training accuracy","training_accuracy.png"),
        ("Training loss","training_loss.png"),
    ]
    for title,filename in visual_files:
        st.markdown(f"#### {title}")
        st.image(str(PROJECT_ROOT/"outputs"/filename),use_container_width=True)

    st.subheader("Error Analysis Sample")
    st.dataframe(error_frame.head(20),use_container_width=True,hide_index=True)
    st.caption(
        "Common errors include legitimate promotional messages, spam written like "
        "ordinary conversation, short messages, ambiguous numbers or links, and unseen vocabulary."
    )

st.divider()
st.caption(
    "Educational portfolio project only. Do not use this demo as the sole basis "
    "for filtering or monitoring real messages."
)
