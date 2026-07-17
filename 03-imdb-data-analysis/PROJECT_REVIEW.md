# Original Project Review

## Identified task

The supplied files implement **IMDb movie-review sentiment classification** using a
Simple RNN. They are not an IMDb rating-prediction or general movie-metadata analysis
project.

## Strong elements in the supplied version

- Real Keras IMDb data loading
- Synthetic smoke-test stage
- Embedding and SimpleRNN layers
- Probability output
- Confusion matrix and classification report
- Downloadable CSV, Excel, and ZIP outputs
- Initial Streamlit interface

## Essential weaknesses corrected

1. Real test rows were used in the validation dataset and then evaluated again.
2. Only two final training epochs produced near-random test accuracy.
3. Synthetic reviews were mixed into the final real-data training stage.
4. The tokenizer and model were trained inside the Streamlit app.
5. TensorFlow failure silently changed the demonstrated model into a different classifier.
6. Long reviews were limited to one short sequence.
7. Metrics omitted ROC-AUC, PR-AUC, specificity, MCC, and baseline comparison.
8. Full review exports were not ideal for a public GitHub repository.
9. There was no modular training/inference separation, saved tokenizer, test suite, or CI.

## Portfolio version

The final package uses a saved Simple RNN, deterministic tokenizer, overlapping sequence
chunks, validation-only threshold selection, untouched testing, de-identified outputs,
a strong classical baseline, error analysis, tests, CI, and a multi-workflow Streamlit app.
