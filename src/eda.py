"""Exploratory data analysis utilities and metrics inspection."""

from pathlib import Path
import pandas as pd
import numpy as np


def inspect_timestamps(df: pd.DataFrame) -> dict:
    """Analyze timestamp continuity and intervals."""
    unique_timestamps = sorted(df["timestamp"].unique())
    diffs = np.diff(unique_timestamps)
    
    return {
        "num_unique_timestamps": len(unique_timestamps),
        "min_timestamp": unique_timestamps[0],
        "max_timestamp": unique_timestamps[-1],
        "step_intervals": pd.Series(diffs).value_counts().to_dict(),
        "is_strictly_regular": len(set(diffs)) == 1,
    }


def inspect_vm_lifecycles(cpu_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VM presence across timestamps and churn."""
    # Active VMs per timestamp
    active_vms = (
        cpu_df.groupby("timestamp")["vm_id"]
        .nunique()
        .reset_index(name="active_vm_count")
    )
    
    # Check VM lifespan in readings
    vm_lifespan = cpu_df.groupby("vm_id")["timestamp"].agg(["min", "max", "count"])
    vm_lifespan["span_duration"] = vm_lifespan["max"] - vm_lifespan["min"]
    
    print("\n--- VM Presence Summary ---")
    print(f"Total unique VMs in readings: {cpu_df['vm_id'].nunique()}")
    print(f"Total unique VMs in metadata: {meta_df['vm_id'].nunique()}")
    print(f"Mean active VMs per timestamp: {active_vms['active_vm_count'].mean():.2f}")
    print(f"Min active VMs: {active_vms['active_vm_count'].min()}")
    print(f"Max active VMs: {active_vms['active_vm_count'].max()}")
    
    return active_vms


def inspect_cpu_metrics(cpu_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze CPU summary statistics and look for anomalies/outliers."""
    stats = cpu_df[["min_cpu", "avg_cpu", "max_cpu"]].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    
    # Check physical constraints: min_cpu <= avg_cpu <= max_cpu
    violations = cpu_df[
        (cpu_df["min_cpu"] > cpu_df["avg_cpu"]) | (cpu_df["avg_cpu"] > cpu_df["max_cpu"])
    ]
    
    print("\n--- CPU Distribution Statistics ---")
    print(stats)
    print(f"\nConstraint violations (min > avg or avg > max): {len(violations)}")
    
    # Missing values check
    missing = cpu_df.isnull().sum()
    print("\n--- Missing Values Count ---")
    print(missing)
    
    return stats


def compute_derived_targets(cpu_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the two mandatory time-series target series:
    1. active_vm_count(t)
    2. aggregate_cpu_utilization(t)
    """
    targets = (
        cpu_df.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
    )
    return targets


if __name__ == "__main__":
    from src.data import load_raw_data

    DATA_DIR = Path("challenge_data")
    cpu_df, meta_df = load_raw_data(DATA_DIR)

    # 1. Timestamps
    ts_info = inspect_timestamps(cpu_df)
    print("\n--- Timestamp Analysis ---")
    for k, v in ts_info.items():
        print(f"{k}: {v}")

    # 2. VM Lifecycles
    active_vms_df = inspect_vm_lifecycles(cpu_df, meta_df)

    # 3. CPU Metrics
    cpu_stats = inspect_cpu_metrics(cpu_df)

    # 4. Target Generation Test
    targets_df = compute_derived_targets(cpu_df)
    print("\n--- Derived Targets Sample (First 5 timestamps) ---")
    print(targets_df.head())
    print("\n--- Derived Targets Sample (Last 5 timestamps) ---")
    print(targets_df.tail())