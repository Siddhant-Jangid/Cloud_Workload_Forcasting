"""Baselines and time-series validation setup."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import load_raw_data
from src.eda import compute_derived_targets


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics MAE and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}


def temporal_train_test_split(
    df: pd.DataFrame, train_size: int = 110
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split targets strictly by time into train (first 110) and holdout (last 24)."""
    train_df = df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    return train_df, test_df


def rolling_origin_cv(
    train_df: pd.DataFrame,
    min_train_len: int = 50,
    horizon: int = 24,
    step: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window (rolling origin) train/validation index folds."""
    n_samples = len(train_df)
    splits = []
    
    for split_point in range(min_train_len, n_samples - horizon + 1, step):
        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, split_point + horizon)
        splits.append((train_idx, val_idx))
        
    return splits


class BaselineModels:
    """Standard time-series baseline forecasting methods for multi-step horizons."""

    @staticmethod
    def last_value_naive(history: np.ndarray, horizon: int) -> np.ndarray:
        """Repeat the last observed historical value across the horizon."""
        return np.full(shape=horizon, fill_value=history[-1])

    @staticmethod
    def moving_average(history: np.ndarray, horizon: int, window: int = 6) -> np.ndarray:
        """Repeat the mean of the last `window` historical points."""
        mean_val = np.mean(history[-window:])
        return np.full(shape=horizon, fill_value=mean_val)

    @staticmethod
    def historical_mean(history: np.ndarray, horizon: int) -> np.ndarray:
        """Repeat the mean of the entire observed history."""
        return np.full(shape=horizon, fill_value=np.mean(history))


def evaluate_baselines_on_validation(train_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate baseline strategies via Rolling-Origin Cross-Validation."""
    folds = rolling_origin_cv(train_df, min_train_len=60, horizon=24, step=6)
    cpu_series = train_df["aggregate_cpu_utilization"].to_numpy()
    vm_series = train_df["active_vm_count"].to_numpy()

    results = []

    for fold_i, (t_idx, v_idx) in enumerate(folds):
        train_cpu = cpu_series[t_idx]
        val_cpu = cpu_series[v_idx]
        
        train_vm = vm_series[t_idx]
        val_vm = vm_series[v_idx]

        # 1. Active VM: Constant 10000
        pred_vm = np.full(len(val_vm), fill_value=10000)
        vm_m = compute_metrics(val_vm, pred_vm)

        # 2. CPU: Last Value Naive
        pred_last = BaselineModels.last_value_naive(train_cpu, horizon=len(val_cpu))
        last_m = compute_metrics(val_cpu, pred_last)
        results.append({"fold": fold_i, "model": "CPU_LastValueNaive", **last_m})

        # 3. CPU: Moving Average (Window=3)
        pred_ma3 = BaselineModels.moving_average(train_cpu, horizon=len(val_cpu), window=3)
        ma3_m = compute_metrics(val_cpu, pred_ma3)
        results.append({"fold": fold_i, "model": "CPU_MA_Window3", **ma3_m})

        # 4. CPU: Moving Average (Window=6)
        pred_ma6 = BaselineModels.moving_average(train_cpu, horizon=len(val_cpu), window=6)
        ma6_m = compute_metrics(val_cpu, pred_ma6)
        results.append({"fold": fold_i, "model": "CPU_MA_Window6", **ma6_m})

        # 5. CPU: Historical Mean
        pred_hmean = BaselineModels.historical_mean(train_cpu, horizon=len(val_cpu))
        hmean_m = compute_metrics(val_cpu, pred_hmean)
        results.append({"fold": fold_i, "model": "CPU_HistoricalMean", **hmean_m})

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("model")[["MAE", "RMSE"]].mean().reset_index()
    print("\n=== Validation CV Baseline Benchmark (Aggregate CPU) ===")
    print(summary.to_string(index=False))
    print(f"\nActive VM Count baseline error: MAE={vm_m['MAE']:.4f}, RMSE={vm_m['RMSE']:.4f}")
    return summary


if __name__ == "__main__":
    cpu_df, _ = load_raw_data("challenge_data")
    targets_df = compute_derived_targets(cpu_df)
    
    train_df, holdout_df = temporal_train_test_split(targets_df, train_size=110)
    print(f"Train split shape: {train_df.shape} (timestamps 0 to {train_df['timestamp'].iloc[-1]})")
    print(f"Holdout split shape: {holdout_df.shape} (timestamps {holdout_df['timestamp'].iloc[0]} to {holdout_df['timestamp'].iloc[-1]})")

    evaluate_baselines_on_validation(train_df)