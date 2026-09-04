# Adaptive Data-Drift Detection and Mitigation

> 🥈 2nd Place — National AI Student Challenge 2026, Singtel Track

An automated machine-learning pipeline that detects data drift, identifies high-impact drifted features, and mitigates performance degradation under a fixed-LightGBM constraint.

## Results

| Metric | Result |
|---|---:|
| Baseline test AU-PRC | 0.699 |
| Final test AU-PRC | 0.846 |
| AU-PRC recovered | +0.147 |
| Mean cross-validation AU-PRC | 0.851 |
| Runtime on public data | ~4 seconds |

The pipeline uses a two-pass PSI × feature-importance strategy. It detects distribution shifts, distinguishes influential drift from harmless drift, and removes features that damage generalization before retraining the model.

## Technical approach

- Drift detection with Population Stability Index, Kolmogorov–Smirnov tests, and chi-square tests
- Importance-aware mitigation using LightGBM gain importance
- Automated feature engineering and data-quality guardrails
- Validation and ablation checks for mitigation decisions
- Interactive Streamlit dashboard for drift and performance analysis
- CPU-only execution with fixed LightGBM hyperparameters

## Repository structure

```text
.
├── data/
│   └── precomputed.json
├── src/
│   ├── dashboard.py
│   ├── drift_detector.py
│   ├── drift_mitigator.py
│   ├── feature_engineer.py
│   ├── guardrails.py
│   ├── main.py
│   └── validation.py
├── model.joblib
├── prediction.csv
├── report.pdf
└── requirements.txt
```

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python src/main.py \
  --train_data_filepath path/to/train.csv \
  --test_data_filepath path/to/test.csv
```

Run the validation suite:

```bash
python src/validation.py \
  --train_data_filepath path/to/train.csv \
  --test_data_filepath path/to/test.csv
```

Launch the dashboard:

```bash
streamlit run src/dashboard.py
```

## Demo and report

- [Live Streamlit dashboard](https://gen-i-singtel-naisc-2026-nenmri4bfkbuungxfrbhih.streamlit.app/)
- [Technical report](report.pdf)

## Notes

The original competition datasets are not included. Provide compatible train and test CSV files through the command-line arguments shown above.

## License

This project is released under the [MIT License](LICENSE).
