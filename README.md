# Simple RNN Projects

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-SimpleRNN-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Demos-red.svg)](https://streamlit.io/)
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
| 1 | [Electricity Consumption Forecasting](01-electricity-consumption-forecasting/) | Time-series regression and demand forecasting | Portfolio-ready Â· deployment pending |
| 2 | Google Stock Price Prediction | Financial time-series forecasting | Planned |
| 3 | IMDb Data Analysis | Sentiment analysis and sequence classification | Planned |
| 4 | SMS Spam Detection | Binary text classification | Planned |
| 5 | Text Generation | Character- or word-level sequence generation | Planned |
| 6 | Word Embedding | Representation learning and semantic analysis | Planned |

## Repository Convention

The repository uses numbered project folders so that the portfolio order remains clear on GitHub:

```text
simple-rnn-projects/
â”œâ”€â”€ .github/
â”‚   â””â”€â”€ workflows/
â”œâ”€â”€ 01-electricity-consumption-forecasting/
â”œâ”€â”€ 02-google-stock-price-prediction/
â”œâ”€â”€ 03-imdb-data-analysis/
â”œâ”€â”€ 04-sms-spam-detection/
â”œâ”€â”€ 05-text-generation/
â”œâ”€â”€ 06-word-embedding/
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE
â”œâ”€â”€ PROJECT_ROADMAP.md
â””â”€â”€ README.md
```

Each completed project will use the following pattern where appropriate:

```text
project-name/
â”œâ”€â”€ README.md
â”œâ”€â”€ app/
â”œâ”€â”€ assets/ or images/
â”œâ”€â”€ data/
â”œâ”€â”€ docs/
â”œâ”€â”€ models/
â”œâ”€â”€ notebooks/
â”œâ”€â”€ outputs/
â”œâ”€â”€ scripts/
â”œâ”€â”€ src/
â”œâ”€â”€ tests/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README_HOSTING.md
â””â”€â”€ Dockerfile
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
- MAE, RMSE, MAPE, sMAPE, and RÂ² evaluation
- Residual and error analysis
- Recursive multi-hour forecasting
- A Streamlit upload and forecast workflow

[Open the complete project documentation](01-electricity-consumption-forecasting/)

## Author

**Anmol Tripathi**  
Quality Data Scientist building a portfolio for Data Science, Machine Learning, Applied AI, Analytics Engineering, Business Intelligence, and Quality Analytics roles.


