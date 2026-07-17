# Original Project Review

## Identified task

The supplied notebook and application implement binary SMS spam-versus-ham text
classification.

## Strong elements

- Real SMS Spam Collection loading
- Synthetic smoke-test stage
- Embedding and SimpleRNN architecture
- Spam probability output
- Confusion matrix and classification report
- Threshold sweep and error samples
- Manual and CSV Streamlit input

## Essential weaknesses corrected

1. The app trained on repeated synthetic templates instead of loading the real-data model.
2. The test partition was reused as validation data for early stopping.
3. The notebook reported 118 overlapping train/test texts.
4. Normalized duplicates were not removed before splitting.
5. The RNN did not use class weights despite severe imbalance.
6. Threshold tuning was performed on test predictions.
7. TensorFlow failure silently changed the demonstrated classifier.
8. Model and tokenizer artifacts were not saved.
9. Evaluation omitted ROC-AUC, PR-AUC, specificity, MCC, and strong baselines.
10. The app lacked a responsible-use and privacy warning.
11. There were no modular source files, tests, artifact validation, or CI.

## Portfolio solution

The final project uses real cleaned data, normalized duplicate removal, a
stratified 70/15/15 split, class-weighted Simple RNN training, validation-only
threshold selection, an untouched test set, saved artifacts, classical
baselines, privacy-safe samples, tests, CI, and a multi-workflow Streamlit app.
