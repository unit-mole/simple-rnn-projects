# Simple RNN Projects

A structured portfolio of six completed recurrent neural network projects covering time-series forecasting, financial sequence modeling, natural language processing, text classification, text generation, and neural representation learning.

**Portfolio status:** 6 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how Simple Recurrent Neural Networks and supporting neural representation techniques can be applied to practical sequential-data problems. Each project is developed as an end-to-end case study containing:

- a clearly defined business or analytical problem;
- reproducible data preparation, feature engineering, and sequence generation;
- leakage-aware training, validation, and test design;
- Simple RNN or neural representation model development;
- task-appropriate baseline comparison and evaluation;
- saved preprocessing and model artifacts;
- modular and reusable inference code;
- an interactive Streamlit demonstration;
- automated tests and GitHub Actions CI;
- local execution and deployment guidance;
- an honest discussion of assumptions, limitations, and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Data Analytics, Quality Analytics, Business Intelligence, and Analytics Engineering roles.

---

## Completed Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Electricity Consumption Forecasting](01-electricity-consumption-forecasting/) | Time-series regression and demand forecasting | [Live Demo](https://simple-rnn-projects-8mxgmrutejhv5mgxnddvra.streamlit.app/) |
| 2 | [Google Stock Price Prediction](02-google-stock-price-prediction/) | Financial time-series forecasting | [Live Demo](https://simple-rnn-projects-8ppkcyb6itqkquyzd32rsk.streamlit.app/) |
| 3 | [IMDb Movie Review Sentiment Analysis](03-imdb-data-analysis/) | NLP sentiment analysis and binary sequence classification | [Live Demo](https://simple-rnn-projects-ljp2wrybnrz4eheng2xsd8.streamlit.app/) |
| 4 | [SMS Spam Detection](04-sms-spam-detection/) | Imbalanced NLP binary classification and message filtering | [Live Demo](https://simple-rnn-projects-mb5hckxzin7hhatgfak2tm.streamlit.app/) |
| 5 | [Character-Level Text Generation](05-text-generation/) | Autoregressive next-character generation | [Live Demo](https://simple-rnn-projects-72u2s8vhngrexwwgbjpy6r.streamlit.app/) |
| 6 | [Word Embedding and NLP Representation Learning](06-word-embedding/) | Neural representation learning and semantic analysis | [Live Demo](https://simple-rnn-projects-kgg7njs6sltnwqqvmjirvm.streamlit.app/) |

---

## What the Portfolio Covers

The six projects are intentionally varied so that the repository demonstrates more than one type of recurrent-neural-network and sequential-data problem.

### Time-Series and Financial Forecasting

- **Electricity Consumption Forecasting** predicts near-term electricity demand from recent hourly consumption and calendar patterns.
- **Google Stock Price Prediction** estimates the next trading-session closing price from recent daily return sequences.

These projects demonstrate chronological splitting, training-only scaling, leakage prevention, sequence-window generation, regression evaluation, baseline comparison, residual analysis, and one-step or multi-step forecasting.

### Natural Language Processing and Text Classification

- **IMDb Movie Review Sentiment Analysis** classifies unstructured movie reviews as positive or negative.
- **SMS Spam Detection** classifies messages as spam or legitimate while handling class imbalance, threshold trade-offs, and privacy considerations.

These projects demonstrate text cleaning, tokenization, vocabulary control, sequence padding, embeddings, recurrent text modeling, class weighting, probability interpretation, threshold selection, and classification evaluation.

### Generative Sequence Modeling

- **Character-Level Text Generation** predicts the next character and generates text autoregressively from a seed prompt.

This project demonstrates character-level sequence creation, recurrent generation, temperature sampling, top-k sampling, perplexity, reproducible generation controls, baseline comparison, and qualitative output analysis.

### Representation Learning and Semantic Analysis

- **Word Embedding and NLP Representation Learning** trains dense word representations using a neural skip-gram objective and supports semantic-neighbor analysis, sentence vectors, PCA visualization, and lightweight semantic search.

This project demonstrates vocabulary management, context-window generation, neural embeddings, cosine similarity, dimensionality reduction, sentence representation, semantic retrieval, and comparison with non-neural representation baselines.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business-problem definition;
- reproducible data preparation;
- feature and sequence engineering;
- training, validation, and test separation;
- model development and evaluation;
- saved preprocessing and model artifacts;
- reusable prediction and generation pipelines;
- manual and batch inference;
- downloadable outputs;
- local execution;
- cloud deployment.

### Sequence Modeling with Correct Validation

Sequential data requires careful validation and preprocessing. The repository emphasizes:

- chronological splitting for forecasting and generative projects;
- review- or message-level splitting for text classification;
- training-only preprocessing and vocabulary fitting;
- consistent sequence generation during training and inference;
- validation-based threshold or model selection;
- untouched final test evaluation where applicable;
- explicit documentation of leakage risks.

### Model Evaluation Based on the Actual Problem

The projects use evaluation metrics that match the task rather than relying on one headline score.

Examples include:

- MAE, RMSE, MAPE, sMAPE, R², and residual analysis for forecasting;
- precision, recall, F1, specificity, ROC-AUC, PR-AUC, and MCC for classification;
- directional accuracy for financial forecasting;
- confusion matrices and probability distributions;
- threshold analysis for sentiment and spam classification;
- validation loss, next-character accuracy, perplexity, and qualitative review for text generation;
- context-prediction loss, perplexity, top-k accuracy, domain purity, and neighborhood analysis for word embeddings;
- baseline comparisons to determine whether the neural model adds measurable value.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- preprocessing fitted on training data only;
- consistent feature, token, and sequence order between training and prediction;
- saved scalers, tokenizers, vocabularies, metadata, and Keras or PyTorch models;
- safe handling of missing values, unknown tokens, and invalid uploads;
- modular source files rather than notebook-only logic;
- automated tests for important preprocessing and prediction paths;
- project-specific GitHub Actions workflows;
- Streamlit deployment from the main repository branch;
- GitHub-safe data and artifact management.

### Business and Analytical Translation

The applications do not stop at raw model outputs. Depending on the project, they provide:

- future-demand forecasts;
- next-session price estimates;
- predicted sentiment;
- spam or ham classifications;
- generated text with configurable sampling controls;
- nearest-word relationships;
- sentence representations;
- ranked semantic-search results;
- probabilities and confidence levels;
- model and baseline comparisons;
- error interpretations;
- batch summaries;
- downloadable scored datasets.

This demonstrates the ability to translate technical model outputs into information that can be understood by analysts, engineers, managers, and other business stakeholders.

### Responsible Model Communication

Each project documents its intended scope and limitations. The repository avoids presenting portfolio models as production-ready financial, operational, filtering, or language-generation systems without additional validation, governance, monitoring, and human oversight.

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this structure:

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
- safe repository practices;
- transparent model limitations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Time-series regression | Electricity consumption forecasting |
| Financial forecasting | Google stock price prediction |
| NLP binary classification | IMDb sentiment analysis and SMS spam detection |
| Character-level text generation | Autoregressive next-character prediction with a PyTorch Simple RNN |
| Generative sampling | Temperature, top-k, and reproducible random-seed controls |
| Sequence-window generation | Electricity forecasting, stock prediction, and text generation |
| Tokenization and vocabulary control | IMDb sentiment analysis, SMS spam detection, text generation, and word embeddings |
| Word embeddings | IMDb sentiment analysis, SMS spam detection, and neural skip-gram representation learning |
| Embedding similarity | Cosine-similarity nearest neighbors and semantic search |
| Dimensionality reduction | PCA visualization of learned word vectors |
| Representation baselines | One-hot, Bag-of-Words, TF-IDF, and TF-IDF with Truncated SVD |
| Chronological validation | Electricity forecasting, stock prediction, and text generation |
| Classification thresholding | IMDb sentiment analysis and SMS spam detection |
| Class-imbalance handling | Class weights, stratified splitting, PR-AUC, and spam-focused evaluation |
| Baseline forecasting | Electricity consumption and Google stock projects |
| Classical NLP baselines | TF-IDF Logistic Regression and Naive Bayes |
| Regression evaluation | MAE, RMSE, MAPE, sMAPE, R² |
| Classification evaluation | Precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC |
| Generative evaluation | Validation loss, next-character accuracy, perplexity, and sample analysis |
| Representation evaluation | Context accuracy, perplexity, domain purity, and semantic-neighbor analysis |
| Manual inference | Interactive Streamlit input workflows |
| Batch inference | CSV upload, sample scoring, and downloadable outputs |
| Model deployment | Six Streamlit Community Cloud applications |
| Testing and CI/CD | pytest and project-specific GitHub Actions workflows |

---

## Core Skills Demonstrated

`Simple RNN` · `Recurrent Neural Networks` · `Sequence Modeling` · `Time-Series Forecasting` · `Financial Forecasting` · `Natural Language Processing` · `Binary Classification` · `Character-Level Text Generation` · `Autoregressive Generation` · `Temperature Sampling` · `Top-k Sampling` · `Perplexity` · `Text Cleaning` · `Tokenization` · `Vocabulary Management` · `Sequence Padding` · `Word Embeddings` · `Neural Skip-gram` · `Dense Vector Representations` · `Context Windows` · `Cosine Similarity` · `PCA` · `Latent Semantic Analysis` · `Semantic Search` · `Class Weighting` · `Feature Engineering` · `Chronological Validation` · `Leakage Prevention` · `Threshold Selection` · `Precision–Recall Analysis` · `Baseline Comparison` · `Regression Evaluation` · `Classification Evaluation` · `Residual Analysis` · `Error Analysis` · `Responsible AI Communication` · `Privacy-Aware Deployment` · `TensorFlow` · `Keras` · `PyTorch` · `scikit-learn` · `pandas` · `Streamlit` · `Testing` · `GitHub Actions` · `CI/CD` · `Business Translation`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
