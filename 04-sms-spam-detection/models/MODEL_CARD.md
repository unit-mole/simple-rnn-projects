# Model Card — SMS Spam Simple RNN

## Intended use

Educational and portfolio demonstration of recurrent SMS spam classification.

## Model

```text
Embedding(64) → SimpleRNN(48) → Dense(24) → Dropout(0.30) → Sigmoid
```

## Data and split

| Component | Messages |
|---|---:|
| Cleaned modeling corpus | 5,101 |
| Training | 3,570 |
| Validation | 765 |
| Test | 766 |

Normalized duplicates are removed before splitting. The test set is untouched
during training and threshold selection.

## Test results

| Metric | Result |
|---|---:|
| Accuracy | 98.04% |
| Spam precision | 90.43% |
| Spam recall | 93.41% |
| Spam F1 | 91.89% |
| ROC-AUC | 0.987 |
| PR-AUC | 0.961 |

## Baseline conclusion

TF-IDF + Logistic Regression achieved 98.83% accuracy and
95.03% spam F1. It is stronger on this split. The Simple RNN remains the
primary portfolio model because the project demonstrates recurrent sequence
modeling.

## Limitations

Small historical English corpus, uncalibrated probabilities, domain drift,
obfuscated spam, possible legitimate-message false positives, and a fixed
sequence length.

## Responsible use

Do not use this model as the sole basis for blocking or monitoring private
messages. Human review and organization-specific validation are required.
