# Model Card — IMDb Simple RNN Sentiment Classifier

## Intended use

Educational and portfolio demonstration of embedding-based recurrent text classification.

## Model

```text
Embedding(80) → SimpleRNN(48) → Dense(32) → Dropout(0.30) → Sigmoid
```

Long reviews are split into overlapping 80-token chunks. Review probability is the mean
of chunk probabilities.

## Training and evaluation

- Supplied real training subset: 2,000 reviews
- Untouched test set: 600 reviews
- Decision threshold: 0.43, selected from validation data
- Test accuracy: 74.33%
- Test F1: 76.67%
- ROC-AUC: 0.820
- PR-AUC: 0.799

## Baseline

TF-IDF + Logistic Regression achieved 86.67% accuracy and should be preferred for this
evaluated subset when predictive performance is the primary objective.

## Limitations

- Limited training data
- Lower negative-class specificity
- Weak long-range memory
- Sensitivity to sarcasm, mixed sentiment, and domain shift
- Probabilities are not formally calibrated

## Prohibited interpretation

The model must not be treated as a definitive assessment of a person, employee, customer,
or creative professional. Human review is required for consequential use.
