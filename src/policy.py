"""Evaluation diagnostics, error analysis, and operational capacity policy."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_raw_data
from src.eda import compute_derived_targets
from src.baselines import temporal_train_test_split, compute_metrics


class WorkloadPredictor:
    """Production forecaster combining exact static active VM prediction
    with a damped mean-reverting aggregate CPU model."""

    def __init__(self, damping_factor: float = 0.85):
        self.damping_factor = damping_factor
        self.fleet_mean: float = 0.0
        self.fleet_vm_count: int = 10000

    def fit(self, train_df: pd.DataFrame):
        self.fleet_vm_count = int(train_df["active_vm_count"].iloc[-1])
        self.fleet_mean = float(train_df["aggregate_cpu_utilization"].mean())

    def predict(self, last_cpu: float, horizon: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Generate forecasts for active_vm_count and aggregate_cpu_utilization."""
        # 1. Active VM count is invariant at 10,000
        pred_vm = np.full(horizon, self.fleet_vm_count, dtype=int)

        # 2. Damped transition from last observed CPU towards long-term fleet mean
        pred_cpu = []
        for h in range(1, horizon + 1):
            weight = self.damping_factor ** h
            val = weight * last_cpu + (1.0 - weight) * self.fleet_mean
            pred_cpu.append(val)

        return pred_vm, np.array(pred_cpu)


def run_capacity_planning_policy(
    forecast_cpu: np.ndarray,
    actual_cpu: np.ndarray | None = None,
    p95_residual_headroom: float = 0.45,
) -> pd.DataFrame:
    """Operational capacity planning policy:
    Calculates recommended provisioned capacity with dynamic safety buffer.
    """
    provisioned_cpu = forecast_cpu + p95_residual_headroom
    # Physical cap at 100%
    provisioned_cpu = np.clip(provisioned_cpu, 0.0, 100.0)

    policy_df = pd.DataFrame({
        "forecast_cpu": forecast_cpu,
        "provisioned_capacity_cpu": provisioned_cpu,
        "safety_headroom": p95_residual_headroom,
    })

    if actual_cpu is not None:
        policy_df["actual_cpu"] = actual_cpu
        # Sla breach occurs if actual demand exceeds provisioned capacity
        policy_df["sla_breach"] = actual_cpu > provisioned_cpu
        # Over-provisioning waste (idle headroom)
        policy_df["unutilized_headroom"] = provisioned_cpu - actual_cpu

    return policy_df


if __name__ == "__main__":
    cpu_df, _ = load_raw_data("challenge_data")
    targets_df = compute_derived_targets(cpu_df)
    train_df, holdout_df = temporal_train_test_split(targets_df, train_size=110)

    # Train production model
    predictor = WorkloadPredictor(damping_factor=0.85)
    predictor.fit(train_df)

    last_observed_cpu = train_df["aggregate_cpu_utilization"].iloc[-1]
    pred_vms, pred_cpu = predictor.predict(last_observed_cpu, horizon=len(holdout_df))

    # Evaluate against holdout
    y_true_cpu = holdout_df["aggregate_cpu_utilization"].to_numpy()
    metrics = compute_metrics(y_true_cpu, pred_cpu)

    print("\n=== Holdout Performance (24 Steps / 2 Hours) ===")
    print(f"Aggregate CPU -> MAE: {metrics['MAE']:.4f}%, RMSE: {metrics['RMSE']:.4f}%")
    print(f"Max Absolute Error (Peak Error): {np.max(np.abs(y_true_cpu - pred_cpu)):.4f}%")

    # Run operational policy
    policy_summary = run_capacity_planning_policy(pred_cpu, y_true_cpu)
    breaches = policy_summary["sla_breach"].sum()
    mean_waste = policy_summary["unutilized_headroom"].mean()

    print("\n=== Operational Capacity Policy Summary ===")
    print(f"Total timestamps: {len(policy_summary)}")
    print(f"SLA Breaches (Under-provisioning events): {breaches} ({(breaches/len(policy_summary))*100:.1f}%)")
    print(f"Average Safe Buffer Overhead: {mean_waste:.4f}% CPU")