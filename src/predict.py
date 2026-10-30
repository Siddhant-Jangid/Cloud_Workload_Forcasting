"""CLI entry point to reproduce predictions.csv."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.data import load_raw_data
from src.eda import compute_derived_targets
from src.baselines import temporal_train_test_split
from src.policy import WorkloadPredictor


def generate_submission_predictions(
    data_dir: Path | str, output_path: Path | str
) -> pd.DataFrame:
    """End-to-end inference generating the required 24-step predictions.csv."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    # 1. Ingest raw data
    cpu_df, _ = load_raw_data(data_dir)

    # 2. Derive targets
    targets_df = compute_derived_targets(cpu_df)

    # 3. Apply temporal split
    train_df, holdout_df = temporal_train_test_split(targets_df, train_size=110)

    # 4. Train production predictor
    predictor = WorkloadPredictor(damping_factor=0.85)
    predictor.fit(train_df)

    # 5. Predict holdout horizon
    last_observed_cpu = float(train_df["aggregate_cpu_utilization"].iloc[-1])
    pred_vms, pred_cpu = predictor.predict(last_observed_cpu, horizon=len(holdout_df))

    # 6. Format submission DataFrame
    submission_df = pd.DataFrame(
        {
            "timestamp": holdout_df["timestamp"].values,
            "active_vm_count": pred_vms.astype(int),
            "aggregate_cpu_utilization": np.round(pred_cpu, 6),
        }
    )

    # Validation checks before saving
    assert len(submission_df) == 24, f"Expected 24 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ], "Column schema mismatch"
    assert not submission_df.isnull().any().any(), "Submission contains null values"
    assert (submission_df["active_vm_count"] >= 0).all(), "Negative VM count detected"
    assert (
        (submission_df["aggregate_cpu_utilization"] >= 0.0)
        & (submission_df["aggregate_cpu_utilization"] <= 100.0)
    ).all(), "CPU utilization out of [0, 100] bounds"

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions successfully generated at: {output_path.resolve()}")
    return submission_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate cloud workload demand forecasts."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to folder containing raw input CSV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV filepath destination.",
    )
    args = parser.parse_args()

    generate_submission_predictions(data_dir=args.data, output_path=args.output)


if __name__ == "__main__":
    main()