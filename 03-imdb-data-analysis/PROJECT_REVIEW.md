# Essential GitHub Review and Corrections

## Changes completed

1. Corrected the training metadata so the saved model is documented as a
   1,600-review model-fitting partition with 400 validation reviews, not as a
   post-selection refit on all 2,000 reviews.
2. Recomputed the TF-IDF baseline on the same 1,600-review partition for a
   fair comparison.
3. Corrected the majority baseline so its predicted class is learned from the
   model-fitting partition.
4. Updated the retraining pipeline to regenerate every table and chart
   referenced by the README and Streamlit app.
5. Added repository artifact validation to CI.
6. Added safer CSV upload, batch-size, and review-length controls.
7. Added a downloadable CSV template.
8. Clarified that mixed sample reviews are binary-model stress tests.
9. Replaced fragile JSON loading in the app with standard `json.loads`.
10. Updated hosting documentation with the deployed application URL.

## Model conclusion

The Simple RNN remains the primary architecture for portfolio demonstration.
The fair TF-IDF baseline is stronger on the same model-fitting partition, which
is reported transparently.
