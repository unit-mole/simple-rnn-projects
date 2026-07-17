# Data Guide

## Included sample

`sample_input.csv` contains 120 days of **synthetic hourly electricity-consumption data**. It is intentionally safe to publish and lets the notebook, training pipeline, and Streamlit app run immediately.

Required columns:

```text
timestamp,consumption_kwh
```

Optional numeric/context columns may also be present. The current model learns from historical consumption plus calendar features derived from the timestamp.

## Recommended real public dataset

Use the UCI Machine Learning Repository dataset **Power Consumption of Tetouan City** (dataset ID 849). It contains 52,417 ten-minute observations, weather features, and power-consumption targets for three distribution zones.

Run:

```bash
python scripts/download_tetouan_data.py
```

Then train on one target, for example:

```bash
python train_model.py   --data "data/raw/Tetuan City power consumption.csv"   --timestamp-column "DateTime"   --target-column "Zone 1 Power Consumption"   --frequency "10min"   --lookback 144
```

A 144-step window represents the previous 24 hours at ten-minute frequency.

## Private or enterprise data

Do not commit confidential utility, customer, facility, or operational data. Place private files under `data/private/`; that folder is excluded by `.gitignore`.

Official dataset page: https://archive.ics.uci.edu/dataset/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity
