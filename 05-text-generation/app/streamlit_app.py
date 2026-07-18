"""Interactive Streamlit demo for the saved character-level Simple RNN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation import generated_text_metrics
from src.text_generator import generate_text, load_generation_bundle

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROMPTS_PATH = PROJECT_ROOT / "data" / "sample_prompts.csv"

st.set_page_config(
    page_title="Simple RNN Text Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Load the committed model once per Streamlit session."""
    return load_generation_bundle(MODEL_DIR)


@st.cache_data(show_spinner=False)
def load_sample_prompts() -> pd.DataFrame:
    if PROMPTS_PATH.exists():
        return pd.read_csv(PROMPTS_PATH)
    return pd.DataFrame(
        {
            "prompt_name": ["Alice", "Rabbit"],
            "seed_prompt": ["Alice was beginning", "The White Rabbit"],
        }
    )


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


st.title("Character-Level Text Generation using Simple RNN")
st.caption(
    "Next-character prediction with a saved PyTorch Simple RNN, temperature and top-k "
    "sampling, quantitative evaluation, and an interactive deployment-ready interface."
)

st.warning(
    "**Responsible use:** This educational model may generate inaccurate, repetitive, biased, "
    "or nonsensical text. Do not use it for harmful, misleading, private, sensitive, or "
    "high-stakes content. Human review is required before using any output."
)

try:
    bundle = load_model_bundle()
except Exception as exc:
    st.error(
        "The saved model artifacts could not be loaded. Run `python train_model.py` from the "
        f"project folder and commit the refreshed `models/` files. Details: {exc}"
    )
    st.stop()

prompts_df = load_sample_prompts()
metadata = bundle.metadata

with st.sidebar:
    st.header("Generation Settings")
    prompt_mode = st.radio("Seed input", ["Sample prompt", "Custom prompt"])
    if prompt_mode == "Sample prompt":
        selected_name = st.selectbox("Choose a prompt", prompts_df["prompt_name"].tolist())
        seed_prompt = prompts_df.loc[
            prompts_df["prompt_name"] == selected_name, "seed_prompt"
        ].iloc[0]
        st.text_area("Seed prompt", value=seed_prompt, height=90, disabled=True)
    else:
        seed_prompt = st.text_area(
            "Seed prompt",
            value="Alice was beginning",
            height=90,
            help="The final characters of this text become the recurrent context window.",
        )

    generation_length = st.slider("Generated characters", 100, 1000, 300, 50)
    temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower values are safer and more repetitive; higher values increase randomness.",
    )
    top_k = st.slider(
        "Top-k candidates",
        min_value=5,
        max_value=min(50, int(metadata["vocabulary_size"])),
        value=min(20, int(metadata["vocabulary_size"])),
        step=1,
        help="Sampling is restricted to the k most likely next characters.",
    )
    random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    generate_clicked = st.button("Generate Text", type="primary", use_container_width=True)

    st.divider()
    st.caption("The app loads a pre-trained model and never retrains during startup.")


generation_tab, performance_tab, overview_tab = st.tabs(
    ["✍️ Generate Text", "📊 Model Performance", "🧠 Project Overview"]
)

with generation_tab:
    left, right = st.columns([2, 1])

    with right:
        st.subheader("Model Information")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Vocabulary", f"{metadata['vocabulary_size']} chars")
        metric_b.metric("Context", f"{metadata['sequence_length']} chars")
        metric_c, metric_d = st.columns(2)
        metric_c.metric("Validation loss", f"{metadata['validation_loss']:.3f}")
        metric_d.metric("Perplexity", f"{metadata['validation_perplexity']:.2f}")
        st.markdown(
            "**Architecture**  \n"
            f"Character IDs → Embedding ({metadata['embedding_dim']}) → "
            f"Simple RNN ({metadata['rnn_units']}) → Dropout → Vocabulary logits"
        )
        with st.expander("How temperature works", expanded=True):
            st.write(
                "Lower temperature concentrates probability on likely characters and usually "
                "creates safer but more repetitive output. Higher temperature increases variety "
                "and also increases spelling, punctuation, and grammar errors."
            )

    with left:
        st.subheader("Generated Output")
        if generate_clicked:
            if not str(seed_prompt).strip():
                st.warning("Enter a seed prompt before generating text.")
            else:
                with st.spinner("Generating one character at a time..."):
                    generated = generate_text(
                        bundle,
                        seed_text=str(seed_prompt),
                        generation_length=generation_length,
                        temperature=temperature,
                        top_k=top_k,
                        random_seed=int(random_seed),
                    )
                st.text_area("Generated text", value=generated, height=360)
                metrics = generated_text_metrics(generated)
                metric_columns = st.columns(3)
                metric_columns[0].metric("Characters", metrics["characters"])
                metric_columns[1].metric(
                    "Unique trigram ratio", f"{metrics['unique_trigram_ratio']:.1%}"
                )
                metric_columns[2].metric(
                    "Repeated trigram ratio", f"{metrics['repeated_trigram_ratio']:.1%}"
                )
                st.download_button(
                    "Download Generated Text",
                    data=generated.encode("utf-8"),
                    file_name="simple_rnn_generated_text.txt",
                    mime="text/plain",
                )
        else:
            st.info("Choose a prompt and sampling settings, then click **Generate Text**.")
            sample_path = OUTPUT_DIR / "generated_text_samples.txt"
            if sample_path.exists():
                sample = sample_path.read_text(encoding="utf-8").split("=" * 80)[0].strip()
                with st.expander("View a saved generation example"):
                    st.text(sample)

    st.markdown(
        "**Interpretation:** The network predicts one character from a fixed-length context. "
        "The output reflects patterns in the training corpus; it does not retrieve facts or "
        "possess human-level language understanding."
    )

with performance_tab:
    st.subheader("Held-out Validation Results")
    result_columns = st.columns(4)
    result_columns[0].metric("Validation loss", f"{metadata['validation_loss']:.4f}")
    result_columns[1].metric("Next-character accuracy", f"{metadata['validation_accuracy']:.2%}")
    result_columns[2].metric("Perplexity", f"{metadata['validation_perplexity']:.2f}")
    result_columns[3].metric("Validation sequences", f"{metadata['validation_sequences']:,}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Training and validation learning curves")
        training_curve = OUTPUT_DIR / "training_curve.png"
        if training_curve.exists():
            st.image(str(training_curve), use_container_width=True)
        else:
            st.info("Run `python train_model.py` to create the learning-curve image.")
    with chart_right:
        st.markdown("#### Temperature comparison")
        temperature_chart = OUTPUT_DIR / "temperature_comparison.png"
        if temperature_chart.exists():
            st.image(str(temperature_chart), use_container_width=True)
        else:
            st.info("Run `python train_model.py` to create the temperature chart.")

    st.markdown("#### Baseline comparison")
    baseline_df = load_csv(OUTPUT_DIR / "baseline_comparison.csv")
    if not baseline_df.empty:
        display_columns = [
            column
            for column in ["model", "summary", "unique_trigram_ratio", "repeated_trigram_ratio"]
            if column in baseline_df.columns
        ]
        st.dataframe(baseline_df[display_columns], use_container_width=True, hide_index=True)
    else:
        st.info("The baseline comparison is generated during model training.")

    st.caption(
        "Loss and perplexity quantify next-character prediction. Generated samples and repetition "
        "metrics are still required because low loss alone does not guarantee readable text."
    )

with overview_tab:
    st.subheader("Sequence-Modeling Workflow")
    st.code(
        """Public-domain text corpus
        ↓
Unicode and whitespace normalization
        ↓
Chronological train/validation split
        ↓
Character vocabulary and integer encoding
        ↓
Fixed 50-character input windows
        ↓
Embedding → Simple RNN → Dropout → Dense logits
        ↓
Temperature and top-k sampling
        ↓
Interactive Streamlit inference""",
        language="text",
    )

    scope_left, scope_right = st.columns(2)
    with scope_left:
        st.markdown("#### What this project demonstrates")
        st.markdown(
            "- Character-level NLP preprocessing\n"
            "- Leakage-aware sequence construction\n"
            "- Simple RNN architecture and training\n"
            "- Temperature and top-k sampling\n"
            "- Perplexity and qualitative evaluation\n"
            "- Saved-model inference and Streamlit deployment"
        )
    with scope_right:
        st.markdown("#### Honest limitations")
        st.markdown(
            "- Weak long-term memory\n"
            "- Small, single-domain corpus\n"
            "- Possible repetition and incomplete words\n"
            "- Strong sensitivity to sampling settings\n"
            "- Not a factual or production language model"
        )

    st.info(
        "The project is intentionally positioned as an educational generative sequence-modeling "
        "system that explains the foundations preceding LSTM, GRU, and Transformer models."
    )
