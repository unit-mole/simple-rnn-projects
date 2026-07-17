# Simple RNN Projects

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-SimpleRNN-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Projects](https://img.shields.io/badge/Portfolio%20Projects-6-informational.svg)](#project-roadmap)

A structured portfolio of **Simple RNN and recurrent neural-network projects** for time-series forecasting, sequence classification, natural-language processing, text generation, and representation learning.

## Portfolio Objective

This repository demonstrates how recurrent neural networks can be applied to practical sequential-data problems. Each completed project is designed as a complete case study containing:

- A clearly defined business or analytical problem
- Reproducible data preparation and sequence generation
- Leakage-aware train, validation, and test design
- A trainable Simple RNN as the primary model
- Appropriate baseline comparison and model evaluation
- Saved model and preprocessing artifacts
- An interactive Streamlit demonstration where appropriate
- Testing, documentation, deployment guidance, and limitations
- Recruiter-friendly communication of technical and business value

## Project Roadmap

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Electricity Consumption Forecasting](01-electricity-consumption-forecasting/) | Time-series regression and demand forecasting | [Live Demo](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/) |
| 2 | Google Stock Price Prediction | Financial time-series forecasting | Planned |
| 3 | IMDb Data Analysis | Sentiment analysis and sequence classification | Planned |
| 4 | SMS Spam Detection | Binary text classification | Planned |
| 5 | Text Generation | Character- or word-level sequence generation | Planned |
| 6 | Word Embedding | Representation learning and semantic analysis | Planned |

## Repository Convention

The repository uses numbered project folders so that the portfolio order remains clear on GitHub:

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

Each completed project will use the following pattern where appropriate:

```text
project-name/
├── README.md
├── app/
├── assets/ or images/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── requirements.txt
├── README_HOSTING.md
└── Dockerfile
```

## Portfolio Positioning

These projects are intended to show more than notebook-based experimentation. The portfolio emphasizes:

- Translating business problems into machine-learning solutions
- Correct handling of chronological and sequential data
- Reproducible preprocessing, feature engineering, and sequence creation
- Simple RNN architecture design and model training
- Baseline benchmarking and business-relevant evaluation
- Clean inference and model-persistence workflows
- Interactive deployment through Streamlit
- Responsible communication of assumptions, limitations, and future improvements

## Current Featured Project

### Electricity Consumption Forecasting

The first project uses the previous **24 hourly observations** and calendar features to forecast the next electricity-consumption value. It includes:

- Chronological 70% / 15% / 15% splitting
- Training-only scaling
- A trainable Keras `SimpleRNN`
- Naive, seasonal-naive, and moving-average baselines
- MAE, RMSE, MAPE, sMAPE, and R² evaluation
- Residual and error analysis
- Recursive multi-hour forecasting
- CSV upload and downloadable forecast output
- A deployed Streamlit application

**Live application:** [Open Electricity Consumption Forecasting](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/)

[Open the complete project documentation](01-electricity-consumption-forecasting/)

## Repository

**GitHub:** [https://github.com/unit-mole/simple-rnn-projects](https://github.com/unit-mole/simple-rnn-projects)

## Author

**Anmol Tripathi**  
Quality Data Scientist building a portfolio for Data Science, Machine Learning, Applied AI, Analytics Engineering, Business Intelligence, and Quality Analytics roles.
