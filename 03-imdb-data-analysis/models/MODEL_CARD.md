# Model Card — IMDb Simple RNN Sentiment Classifier

## Intended use

Educational and portfolio demonstration of embedding-based recurrent text
classification.

## Model

```text
Embedding(80) → SimpleRNN(48) → Dense(32) → Dropout(0.30) → Sigmoid
```

Long reviews are split into overlapping 80-token chunks. The review-level
probability is the mean of the chunk probabilities.

## Training and evaluation scope

| Component | Reviews |
|---|---:|
| Training pool | 2,000 |
| Model-fitting partition | 1,600 |
| Validation partition | 400 |
| Untouched test set | 600 |

There is no post-selection full-data refit. The saved model corresponds to the
early-stopped model selected from the model-fitting and validation partitions.

## Simple RNN test results

| Metric | Result |
|---|---:|
| Accuracy | 74.33% |
| F1-score | 76.67% |
| ROC-AUC | 0.820 |
| PR-AUC | 0.799 |

## Fair baseline

The TF-IDF + Logistic Regression baseline is trained on the same 1,600-review
model-fitting partition.

| Metric | Result |
|---|---:|
| Accuracy | 84.50% |
| F1-score | 84.83% |
| ROC-AUC | 0.932 |
| PR-AUC | 0.934 |

The classical baseline is stronger for this evaluated configuration. The
Simple RNN remains the primary portfolio model because the project demonstrates
recurrent sequence modeling.

## Limitations

- Limited model-fitting data
- Lower negative-class specificity
- Weak long-range memory
- Sensitivity to sarcasm, mixed sentiment, and domain shift
- Probabilities are not formally calibrated

## Prohibited interpretation

The model must not be treated as a definitive assessment of a person, employee,
customer, or creative professional. Human review is required for consequential
use.
