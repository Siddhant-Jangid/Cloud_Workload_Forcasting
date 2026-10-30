# Cloud Workload Demand Forecasting & Capacity Planning

An end-to-end time-series forecasting pipeline and operational capacity planning solution built on a deterministic 10,000-VM subset of the Microsoft Azure Public Dataset V2. This project predicts system-wide virtual machine activity (`active_vm_count`) and mean fleet CPU utilization (`aggregate_cpu_utilization`) across a 2-hour (24-step) holdout horizon, translating predictions into an SLA-safe infrastructure provisioning policy.

## Executive Summary

Cloud infrastructure teams face an asymmetric operational trade-off: under-provisioning risks severe SLA breaches and workload throttling, while excessive over-provisioning wastes compute capital. This repository delivers:

1. **Workload Predictability Analysis**: Programmatic analysis across 1,340,000 raw readings (10,000 distinct VMs observed across 134 five-minute intervals). While individual VM CPU usage is heavily right-skewed (median 5.11%, 99th percentile 87.35%), macro-level aggregation stabilizes fleet utilization into a stationary, mean-reverting band (~9.14% to ~9.62%) with an invariant active fleet size of 10,000 VMs.
2. **Leakage-Free Multi-Step Forecasting**: Temporal validation using expanding-window (rolling-origin) cross-validation across the 110-step training window. We benchmarked persistence baselines, moving averages, regularized linear models (`Direct_Ridge`), gradient-boosted trees (`Direct_LightGBM`), and an exponentially damped mean-reverting forecaster.
3. **Operational Capacity Policy**: A dynamic headroom provisioning model (`Predicted CPU + 0.45% safety buffer`) that achieves **0% SLA breaches** across the 24-step holdout horizon while restricting average idle buffer overhead to **0.5374% CPU**.

## Repository Structure

```text
├── challenge_data/              # Raw dataset files (excluded from version control)
├── notebooks/
│   └── eda_visualizations.py    # Script to generate exploratory data analysis plots
├── reports/
│   ├── figures/                 # Exported high-resolution plots for reporting
│   └── final_report.md          # Full executive report, error analysis, and policy guide
├── src/
│   ├── __init__.py
│   ├── data.py                  # Data loading and SHA-256 checksum verification
│   ├── eda.py                   # Exploratory analysis and target time-series derivation
│   ├── baselines.py             # Temporal splitting, rolling-origin CV, and baselines
│   ├── models.py                # Lag/rolling feature engineering and direct ML models
│   ├── policy.py                # Production forecaster and capacity planning engine
│   └── predict.py               # CLI entry point to reproduce predictions.csv
├── tests/
│   ├── __init__.py
│   └── test_submission.py       # Automated pytest suite verifying output compliance
├── .gitignore                   # Git exclusion rules for data, caches, and environments
├── pyproject.toml               # Package build configuration and dependencies
├── predictions.csv              # Final 24-step holdout predictions
└── README.md                    # Project documentation and reproduction guide
```

## Setup & Installation

### 1. Prerequisites

* **Python 3.11+**
* Raw dataset archives (`vm_cpu_readings.csv.gz` and `vm_metadata.csv.gz`) placed inside the `challenge_data/` directory.

### 2. Virtual Environment & Dependencies

Create an isolated virtual environment and install the package in editable mode with development dependencies:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Reproducing Predictions & Running the Pipeline

### 1. One-Command Forecast Reproduction (`predictions.csv`)

To run the complete ingestion, target aggregation, model training, and holdout forecasting pipeline from scratch:

```bash
python -m src.predict --data challenge_data --output predictions.csv
```

### 2. Step-by-Step Module Execution

Each stage of the workflow can be executed independently to inspect intermediate outputs and validation metrics:

```bash
# 1. Verify raw data ingestion and shapes
python -m src.data

# 2. Run Exploratory Data Analysis & check physical/timestamp constraints
python -m src.eda

# 3. Generate EDA visualizations into reports/figures/
python -m notebooks.eda_visualizations

# 4. Run Rolling-Origin Cross-Validation for Baseline Models
python -m src.baselines

# 5. Run Rolling-Origin Cross-Validation for Direct ML Models (Ridge & LightGBM)
python -m src.models

# 6. Evaluate Holdout Performance & Operational Capacity Planning Policy
python -m src.policy
```

### 3. Automated Testing & Validation

Run the unit and integration test suite to verify that `predictions.csv` satisfies all schema, row count (24 rows), timestamp monotonicity (300s step increments), null-safety, and physical boundary (`[0, 100]`) constraints:

```bash
pytest -v
```

## Empirical Results & Model Selection

### Aggregate CPU Utilization (`aggregate_cpu_utilization`)

| Model Architecture                           | Validation CV MAE (%) | Validation CV RMSE (%) | Holdout MAE (%) | Holdout RMSE (%) | Max Peak Error (%) |
| :------------------------------------------- | :-------------------: | :--------------------: | :--------------: | :--------------: | :----------------: |
| **Last Value Naive**                   |        0.2802        |         0.3113         |      0.2319      |      0.2741      |       0.4421       |
| **Moving Average (Window=6)**          |        0.2309        |         0.2640         |        —        |        —        |         —         |
| **Moving Average (Window=3)**          |        0.2235        |         0.2557         |        —        |        —        |         —         |
| **Direct LightGBM**                    |        0.1891        |         0.2211         |      0.1984      |      0.2310      |       0.3805       |
| **Direct Ridge (L2)**                  |        0.1808        |         0.2105         |      0.1652      |      0.1920      |       0.3112       |
| **Historical Mean**                    |        0.1728        |         0.2000         |      0.1285      |      0.1542      |       0.2890       |
| **Damped Mean Reversion (Production)** |   **0.1691**   |    **0.1974**    | **0.1217** | **0.1472** |  **0.2736**  |

### Active VM Count (`active_vm_count`)

* Across all 110 training timestamps and 24 holdout timestamps, exactly 10,000 distinct VMs report metrics per interval (zero churn). The constant invariant predictor achieves **0.0000 MAE** and **0.0000 RMSE**.

### Operational Capacity Planning Summary

* **Policy Rule**: $\text{Provisioned Capacity}(t+h) = \min\left(100.0, \hat{y}_{t+h} + 0.45\%\right)$
* **Holdout SLA Breaches**: **0 / 24 intervals (0.0%)**
* **Mean Unutilized Safety Headroom**: **0.5374% CPU**

For an in-depth discussion of the methodology, feature engineering, residual error diagnostics, and infrastructure scaling recommendations, see [`reports/final_report.md`](reports/final_report.md).
