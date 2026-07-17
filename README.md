# Simple RNN Projects

A structured portfolio of six Simple Recurrent Neural Network projects covering time-series forecasting, financial sequence modeling, Natural Language Processing, text classification, text generation, and representation learning.

**Portfolio status:** 4 completed and deployed projects · 2 planned projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how Simple Recurrent Neural Networks can be applied to practical sequential-data problems. Each completed project is developed as an end-to-end case study containing:

- a clearly defined business or analytical problem;
- reproducible data preparation and sequence generation;
- leakage-aware training, validation, and test design;
- a trainable Keras `SimpleRNN` as the primary model;
- task-appropriate baseline comparison and evaluation;
- saved model and preprocessing artifacts;
- reusable inference code;
- an interactive Streamlit demonstration;
- automated tests and project-specific GitHub Actions CI;
- local-run and deployment guidance; and
- an honest discussion of assumptions, limitations, and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Data Analytics, Business Intelligence, Quality Analytics, and Analytics Engineering roles.

---

## Project Roadmap

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Electricity Consumption Forecasting](01-electricity-consumption-forecasting/) | Time-series regression and demand forecasting | [Live Demo](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/) |
| 2 | [Google Stock Price Prediction](02-google-stock-price-prediction/) | Financial time-series forecasting | [Live Demo](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/) |
| 3 | [IMDb Movie Review Sentiment Analysis](03-imdb-data-analysis/) | NLP sentiment analysis and binary sequence classification | [Live Demo](https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/) |
| 4 | [SMS Spam Detection](04-sms-spam-detection/) | Imbalanced NLP binary classification and message filtering | [Live Demo](https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/) |
| 5 | [Text Generation](05-text-generation/) | Character- or word-level sequence generation | Planned |
| 6 | [Word Embedding](06-word-embedding/) | Representation learning and semantic analysis | Planned |

---

## Completed Projects

### 1. Electricity Consumption Forecasting

An end-to-end demand-forecasting project that uses the previous 24 hourly observations and calendar features to predict near-term electricity consumption.

Key capabilities:

- chronological 70% / 15% / 15% train, validation, and test split;
- training-only scaling and leakage prevention;
- Keras `SimpleRNN` sequence model;
- naive, seasonal-naive, and moving-average baselines;
- MAE, RMSE, MAPE, sMAPE, R², and residual analysis;
- recursive 1–48 hour forecasting;
- CSV upload and downloadable forecast output; and
- saved artifacts, tests, CI, and Streamlit deployment.

**Live application:** [Open Electricity Consumption Forecasting](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

[Open the complete project documentation](01-electricity-consumption-forecasting/)

---

### 2. Google Stock Price Prediction

A financial sequence-modeling project that uses recent Google (`GOOG`) daily return behavior to estimate the next trading-session closing price.

Key capabilities:

- return-based target formulation;
- previous ten trading-day return sequence;
- chronological 70% / 15% / 15% validation design;
- train-only feature and target scaling;
- TensorFlow/Keras Simple RNN with Huber loss;
- previous-close and five-day mean-return baselines;
- MAE, RMSE, MAPE, R², directional accuracy, and residual analysis;
- bundled sample, CSV upload, optional live-data mode, and downloadable outputs; and
- saved artifacts, tests, CI, and Streamlit deployment.

**Live application:** [Open Google Stock Price Prediction](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/)

[Open the complete project documentation](02-google-stock-price-prediction/)

> **Financial disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not financial advice and should not be used for investment decisions.

---

### 3. IMDb Movie Review Sentiment Analysis

An NLP sequence-classification project that uses an embedding layer and Simple RNN to classify IMDb movie reviews as positive or negative.

Key capabilities:

- deterministic text cleaning and vocabulary management;
- out-of-vocabulary handling and sequence padding;
- long-review processing through overlapping 80-token chunks;
- Embedding → SimpleRNN → Dense binary-classification architecture;
- validation-selected decision threshold;
- accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, and MCC;
- TF-IDF + Logistic Regression and majority-class baselines;
- confusion matrix, ROC, precision–recall, training, and error-analysis outputs;
- manual review, sample-review, and CSV batch-scoring workflows; and
- saved model and tokenizer artifacts, tests, CI, and Streamlit deployment.

**Live application:** [Open IMDb Movie Review Sentiment Analysis](https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/)

[Open the complete project documentation](03-imdb-data-analysis/)

---


### 4. SMS Spam Detection

A privacy-aware NLP sequence-classification project that uses an Embedding layer and Simple RNN to classify SMS messages as legitimate (`ham`) or `spam`.

Key capabilities:

- spam-aware normalization that preserves URL, phone, number, currency, and punctuation signals;
- normalized duplicate removal before splitting to prevent text leakage;
- stratified 70% / 15% / 15% training, validation, and test design;
- class-weighted Simple RNN training for the imbalanced spam class;
- 5,000-word training-only vocabulary and 50-token padded sequences;
- validation-selected spam threshold;
- accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, and MCC;
- majority, Multinomial Naive Bayes, and TF-IDF Logistic Regression baselines;
- manual message, privacy-safe sample, and CSV batch-scoring workflows; and
- saved model and tokenizer artifacts, tests, CI, privacy controls, and Streamlit deployment.

**Live application:** [Open SMS Spam Detection](https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/)

[Open the complete project documentation](04-sms-spam-detection/)

> **Privacy notice:** Do not upload private, sensitive, authentication, financial, health, employer, or customer SMS content to the public demo. The model is an educational portfolio prototype and should not be the sole basis for filtering real communications.

---

## What the Portfolio Covers

The six projects are intentionally varied so the repository demonstrates more than one type of recurrent-neural-network problem.

### Time-Series Forecasting

- **Electricity Consumption Forecasting** predicts future electricity demand from recent hourly consumption and calendar patterns.
- **Google Stock Price Prediction** estimates next-session closing price from recent daily return sequences.

These projects demonstrate chronological splitting, leakage prevention, sequence-window creation, scaling, regression evaluation, baseline comparison, residual analysis, and recursive or next-step forecasting.

### Natural Language Processing and Text Classification

- **IMDb Movie Review Sentiment Analysis** converts unstructured movie-review text into positive or negative sentiment predictions.
- **SMS Spam Detection** classifies messages as spam or legitimate while handling class imbalance, threshold trade-offs, and privacy constraints.
- **Text Generation** will model character- or word-level sequence continuation.
- **Word Embedding** will explore learned semantic representations and similarity.

These projects demonstrate text cleaning, tokenization, vocabulary control, embeddings, sequence padding, recurrent text modeling, classification metrics, probability interpretation, and error analysis.

### Representation Learning and Sequential Modeling

Across the portfolio, Simple RNN models learn representations from ordered observations rather than treating each row or token independently.

The completed projects cover:

- numeric time-series sequences;
- financial-return sequences;
- tokenized natural-language sequences;
- regression and binary classification;
- one-step and multi-step forecasting;
- probability-based sentiment and spam scoring;
- class-weighted learning for imbalanced message classification; and
- comparison with strong non-neural baselines.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every completed project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business-problem definition;
- reproducible data preparation;
- sequence and feature engineering;
- training, validation, and test separation;
- model evaluation;
- saved preprocessing and model artifacts;
- reusable prediction pipelines;
- manual and batch inference;
- downloadable outputs;
- local execution; and
- cloud deployment.

### Sequence Modeling with Correct Validation

Sequential data requires more care than randomly shuffled tabular data. The repository emphasizes:

- chronological splitting for forecasting projects;
- review-level splitting for text classification;
- training-only preprocessing;
- consistent sequence generation during training and inference;
- validation-based hyperparameter or threshold selection;
- untouched final test evaluation; and
- explicit documentation of leakage risks.

### Model Evaluation Based on the Actual Problem

The projects use metrics that match the task rather than relying on one headline score.

Examples include:

- MAE, RMSE, MAPE, sMAPE, R², and residual analysis for forecasting;
- precision, recall, F1, specificity, ROC-AUC, PR-AUC, and MCC for classification;
- directional accuracy for financial forecasting;
- confusion matrices and probability distributions;
- threshold analysis for sentiment and spam classification; and
- baseline comparisons to determine whether the Simple RNN adds measurable value.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- modular source files instead of notebook-only logic;
- saved scalers, tokenizers, metadata, and Keras models;
- consistent feature or token order between training and prediction;
- safe handling of missing values, unknown tokens, and invalid uploads;
- automated tests for important preprocessing and prediction paths;
- project-specific GitHub Actions workflows;
- Streamlit deployment from the repository’s `main` branch; and
- GitHub-safe data and artifact management.

### Business and Analytical Translation

The applications do not stop at raw model outputs. Depending on the project, they provide:

- future-demand forecasts;
- next-session price estimates;
- predicted sentiment;
- spam or ham classifications;
- probabilities and confidence levels;
- model and baseline comparisons;
- error interpretations;
- batch summaries; and
- downloadable scored datasets.

This demonstrates the ability to convert technical model outputs into information that can be understood by analysts, engineers, managers, and other business stakeholders.

### Responsible Model Communication

Each project clearly documents its intended scope and limitations. The repository avoids presenting portfolio models as production-ready financial, operational, or human-decision systems without additional validation, governance, and monitoring.

---

## Repository Convention

The repository is organized as a monorepo. Each completed project generally follows this structure:

```text
simple-rnn-projects/
├── .github/
│   └── workflows/
│       └── project-specific-ci.yml
│
├── project-folder/
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── data/
│   │   ├── sample_input.csv
│   │   └── README_data.md
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── src/
│   ├── tests/
│   ├── .gitignore
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements.txt
│   └── supporting project files
│
├── LICENSE
├── PROJECT_ROADMAP.md
└── README.md
```

The exact files vary by project, but the standards remain consistent:

- reproducible workflows;
- modular code;
- deployable inference;
- automated validation;
- clear documentation;
- safe repository practices; and
- transparent model limitations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Time-series regression | Electricity consumption forecasting |
| Financial forecasting | Google stock price prediction |
| NLP binary classification | IMDb sentiment analysis and SMS spam detection |
| Sequence-window generation | Electricity and Google stock projects |
| Tokenization and vocabulary control | IMDb sentiment analysis and SMS spam detection |
| Word embeddings | IMDb sentiment analysis, SMS spam detection, and planned embedding project |
| Chronological validation | Electricity and Google stock projects |
| Classification thresholding | IMDb sentiment analysis and SMS spam detection |
| Class-imbalance handling | Class weights, stratified splitting, PR-AUC, and spam-focused evaluation |
| Baseline forecasting | Electricity and Google stock projects |
| Classical NLP baselines | TF-IDF Logistic Regression and Naive Bayes for IMDb and SMS projects |
| Regression evaluation | MAE, RMSE, MAPE, sMAPE, R² |
| Classification evaluation | Precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC |
| Manual inference | Interactive Streamlit input workflows |
| Batch inference | CSV upload, sample scoring, downloadable outputs |
| Model deployment | Four Streamlit Community Cloud applications |
| Testing and CI/CD | pytest and project-specific GitHub Actions workflows |

---

## Core Skills Demonstrated

`Simple RNN` · `Recurrent Neural Networks` · `Sequence Modeling` · `Time-Series Forecasting` · `Financial Forecasting` · `Natural Language Processing` · `SMS Spam Detection` · `Text Cleaning` · `Pattern Tokenization` · `Vocabulary Management` · `Sequence Padding` · `Word Embeddings` · `Binary Classification` · `Class Weighting` · `Feature Engineering` · `Chronological Validation` · `Leakage Prevention` · `Threshold Selection` · `Precision–Recall Analysis` · `Baseline Comparison` · `Regression Evaluation` · `Classification Evaluation` · `Residual Analysis` · `Error Analysis` · `Privacy-Aware Deployment` · `TensorFlow` · `Keras` · `scikit-learn` · `pandas` · `Streamlit` · `Testing` · `GitHub Actions` · `CI/CD` · `Business Translation`

---

## Repository

**GitHub:** [unit-mole/simple-rnn-projects](https://github.com/unit-mole/simple-rnn-projects)

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics Engineering | Business Intelligence | Quality Analytics
