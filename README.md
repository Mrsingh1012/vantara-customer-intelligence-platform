# Vantara — Customer Intelligence Platform

Vantara is a customer analytics and churn-retention platform designed to analyze customer behavior, estimate churn risk, prioritize retention campaigns, and explore business-impact scenarios.

## Key Features

- Customer transaction analysis and RFM segmentation
- Churn prediction using Logistic Regression and Random Forest
- Time-based model evaluation
- Customer-level churn risk scoring and ranking
- Top-K retention campaign selection
- Business-impact scenario analysis
- Interactive Streamlit dashboard
- Automated report validation tests

## Technology Stack

- Python 3.10
- Pandas
- Scikit-learn
- Streamlit
- Unittest

## Project Structure

```text
customer-behavior-prediction/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── frontend/
│   └── vantara_dashboard.py
├── models_artifacts/
├── reports/
├── src/
│   ├── features/
│   ├── models/
│   └── segmentation/
├── tests/
└── README.md
```

## Setup

Run commands from the project root using the existing Python environment.

## Launch Dashboard

```powershell
.\.venv310\Scripts\python.exe -m streamlit run frontend/vantara_dashboard.py
```

Open http://localhost:8501 in your browser.

## Run Automated Tests

```powershell
.\.venv310\Scripts\python.exe -m unittest discover -s tests -v
```

The current report tests check campaign report availability, customer ID uniqueness, campaign capacity, probability ranges, and business-impact report structure.

## Generate Demo Customer Features

```powershell
.\.venv310\Scripts\python.exe src/models/build_demo_customer_features.py
```

## Generate Campaign Rankings

First inspect the available arguments:

```powershell
.\.venv310\Scripts\python.exe src/models/predict_churn_campaign.py --help
```

Example command for the current demo:

```powershell
.\.venv310\Scripts\python.exe src/models/predict_churn_campaign.py --input data/processed/churn_customer_features_demo.csv --output reports/churn_campaign_ranked_demo.csv --top-k 1500
```

## Current Demo Snapshot

- Customers scored: 1,993
- Campaign selection capacity: 1,500
- Customers not selected: 493

These figures describe the current demo output and may change when the data or model changes.

## Evaluation and Limitations

Historical model evaluation is exploratory because the test period has already been inspected during development. It should not be presented as an untouched final evaluation.

Business-impact figures are assumption-based scenarios, not measured ROI. A controlled experiment is needed to estimate the incremental effect of a retention campaign.

## Future Improvements

- Add tests for feature engineering and prediction logic
- Add data-quality and model-monitoring checks
- Evaluate a final untouched temporal holdout
- Measure real campaign outcomes
- Improve dashboard usability and deployment readiness
