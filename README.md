# Simple RNN Projects

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-SimpleRNN-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Demos-red.svg)](#project-roadmap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Portfolio Projects](https://img.shields.io/badge/Portfolio%20Projects-6-informational.svg)](#project-roadmap)
[![Portfolio Ready](https://img.shields.io/badge/Portfolio%20Ready-2%20of%206-brightgreen.svg)](#project-roadmap)

A structured portfolio of **Simple Recurrent Neural Network projects** covering time-series forecasting, financial sequence modeling, natural-language processing, text classification, text generation, and representation learning.

## Portfolio Objective

This repository demonstrates how Simple RNN and recurrent neural-network methods can be applied to practical sequential-data problems. Each completed project is developed as a complete case study containing:

- A clearly defined business or analytical problem
- Reproducible data preparation and sequence generation
- Leakage-aware train, validation, and test design
- A trainable Simple RNN as the primary model
- Appropriate baseline comparison and model evaluation
- Saved model and preprocessing artifacts
- Interactive Streamlit deployment where appropriate
- Automated testing and GitHub Actions CI
- Local-run and hosting documentation
- Honest discussion of assumptions, limitations, and future improvements

## Project Roadmap

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Electricity Consumption Forecasting](01-electricity-consumption-forecasting/) | Time-series regression and demand forecasting | [Live Demo](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/) |
| 2 | [Google Stock Price Prediction](02-google-stock-price-prediction/) | Financial time-series forecasting | [Live Demo](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/) |
| 3 | [IMDb Data Analysis](03-imdb-data-analysis/) | Sentiment analysis and sequence classification | Planned |
| 4 | [SMS Spam Detection](04-sms-spam-detection/) | Binary text classification | Planned |
| 5 | [Text Generation](05-text-generation/) | Character- or word-level sequence generation | Planned |
| 6 | [Word Embedding](06-word-embedding/) | Representation learning and semantic analysis | Planned |

## Featured Projects

### 1. Electricity Consumption Forecasting

An end-to-end electricity-demand forecasting project that uses the previous **24 hourly observations** and calendar features to predict near-term consumption.

Key capabilities:

- Chronological 70% / 15% / 15% train, validation, and test split
- Training-only scaling and leakage prevention
- Trainable Keras `SimpleRNN`
- Naive, seasonal-naive, and moving-average baselines
- MAE, RMSE, MAPE, sMAPE, R², and residual analysis
- Recursive 1–48 hour forecasting
- CSV upload and downloadable forecast output
- Saved model artifacts, tests, CI, and Streamlit deployment

**Live application:** [Open Electricity Consumption Forecasting](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

[Open the complete project documentation](01-electricity-consumption-forecasting/)

---

### 2. Google Stock Price Prediction

A financial time-series project that uses recent Google (`GOOG`) daily return behavior to estimate the next trading-session closing price.

Key capabilities:

- Return-based target formulation
- Previous ten trading-day return sequence
- Chronological 70% / 15% / 15% validation design
- Train-only feature and target scaling
- TensorFlow/Keras Simple RNN with Huber loss
- Previous-close and five-day mean-return baselines
- MAE, RMSE, MAPE, R², directional accuracy, and residual analysis
- Bundled sample, CSV upload, optional live-data mode, and downloadable outputs
- Saved model artifacts, tests, CI, and Streamlit deployment

**Live application:** [Open Google Stock Price Prediction](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/)

[Open the complete project documentation](02-google-stock-price-prediction/)

> **Financial disclaimer:** The Google stock forecasting project is for educational and portfolio demonstration purposes only. It is not financial advice and should not be used for investment decisions.

## Repository Convention

The repository uses numbered folders so the development order remains clear on GitHub:

```text
simple-rnn-projects/
├── .github/
│   └── workflows/
├── 01-electricity-consumption-forecasting/
├── 02-google-stock-price-prediction/
├── 03-imdb-data-analysis/
├── 04-sms-spam-detection/
├── 05-text-generation/
├── 06-word-embedding/
├── .gitignore
├── LICENSE
├── PROJECT_ROADMAP.md
└── README.md
```

Each completed project follows this structure where appropriate:

```text
project-name/
├── README.md
├── README_HOSTING.md
├── app/
├── data/
├── images/
├── models/
├── notebooks/
├── outputs/
├── src/
├── tests/
├── requirements.txt
└── train_model.py
```

## Portfolio Positioning

These projects are designed to demonstrate more than notebook-based experimentation. The portfolio emphasizes:

- Translating business problems into machine-learning solutions
- Correct handling of chronological and sequential data
- Reproducible preprocessing, feature engineering, and sequence creation
- Simple RNN architecture design and model training
- Leakage prevention and train-only preprocessing
- Baseline benchmarking and business-relevant evaluation
- Clean inference and model-persistence workflows
- Interactive deployment through Streamlit
- Automated testing and CI/CD
- Responsible communication of assumptions, uncertainty, and limitations

## Skills Demonstrated

`Simple RNN` · `Recurrent Neural Networks` · `Time-Series Forecasting` · `Financial Forecasting` · `Sequence Modeling` · `TensorFlow` · `Keras` · `Feature Engineering` · `Chronological Validation` · `Leakage Prevention` · `Baseline Comparison` · `Model Evaluation` · `Residual Analysis` · `Streamlit` · `Testing` · `CI/CD`

## Repository

**GitHub:** [unit-mole/simple-rnn-projects](https://github.com/unit-mole/simple-rnn-projects)

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics Engineering | Business Intelligence | Quality Analytics
