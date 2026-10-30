"""Feature engineering and multi-step forecasting models."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

from src.data import load_raw_data
from src.eda import compute_derived_targets
from src.baselines import temporal_train_test_split, compute_metrics, rolling_origin_cv


def build_lag_features(series: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """Construct autoregressive lag and rolling window features."""
    df_feat = pd.DataFrame(index=series.index)
    df_feat["y"] = series

    # Specific informative lags
    for lag in [1, 2, 3, 4, 6, 12]:
        if lag <= max_lag:
            df_feat[f"lag_{lag}"] = series.shift(lag)

    # Rolling statistics (computed strictly using past values via shift(1))
    for w in [3, 6, 12]:
        df_feat[f"rolling_mean_{w}"] = series.shift(1).rolling(w).mean()
        df_feat[f"rolling_std_{w}"] = series.shift(1).rolling(w).std()
        df_feat[f"rolling_min_{w}"] = series.shift(1).rolling(w).min()
        df_feat[f"rolling_max_{w}"] = series.shift(1).rolling(w).max()

    # Velocity / Momentum
    df_feat["diff_1"] = series.shift(1) - series.shift(2)
    df_feat["diff_3"] = series.shift(1) - series.shift(4)

    return df_feat


class DirectMultiStepForecaster:
    """Direct Multi-Step forecaster training a separate model per forecast horizon step."""

    def __init__(self, model_type: str = "ridge", **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.models: dict[int, object] = {}

    def _get_base_estimator(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.model_params.get("alpha", 1.0))
        elif self.model_type == "lgbm":
            return lgb.LGBMRegressor(
                n_estimators=self.model_params.get("n_estimators", 30),
                learning_rate=self.model_params.get("learning_rate", 0.05),
                max_depth=self.model_params.get("max_depth", 3),
                num_leaves=self.model_params.get("num_leaves", 7),
                min_child_samples=self.model_params.get("min_child_samples", 5),
                random_state=42,
                verbosity=-1,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit_and_predict(
        self,
        train_series: pd.Series,
        horizon: int = 24,
        max_lag: int = 12,
    ) -> np.ndarray:
        """Extract features, fit one estimator per forecast horizon h, and forecast."""
        feat_df = build_lag_features(train_series, max_lag=max_lag)
        feature_cols = [c for c in feat_df.columns if c != "y"]

        # Drop warm-up rows containing NaNs
        valid_df = feat_df.dropna()
        X_train_full = valid_df[feature_cols]
        y_train_full = valid_df["y"]

        # The feature vector representing the most recent available observation
        latest_X = feat_df.iloc[[-1]][feature_cols]

        predictions = []

        for h in range(1, horizon + 1):
            # Target for step h ahead: shift target backwards by h
            y_target = y_train_full.shift(-h).dropna()
            X_subset = X_train_full.loc[y_target.index]

            model = self._get_base_estimator()
            model.fit(X_subset, y_target)
            self.models[h] = model

            pred_h = model.predict(latest_X)[0]
            predictions.append(pred_h)

        return np.array(predictions)


def cross_validate_ml_models(train_df: pd.DataFrame) -> pd.DataFrame:
    """Run rolling-origin CV for Direct Ridge and LightGBM models."""
    folds = rolling_origin_cv(train_df, min_train_len=60, horizon=24, step=6)
    cpu_series = train_df["aggregate_cpu_utilization"]

    results = []

    for fold_i, (t_idx, v_idx) in enumerate(folds):
        train_s = cpu_series.iloc[t_idx]
        val_cpu = cpu_series.iloc[v_idx].to_numpy()

        # 1. Ridge Forecaster
        ridge_forecaster = DirectMultiStepForecaster(model_type="ridge", alpha=10.0)
        pred_ridge = ridge_forecaster.fit_and_predict(train_s, horizon=len(val_cpu))
        r_metrics = compute_metrics(val_cpu, pred_ridge)
        results.append({"fold": fold_i, "model": "Direct_Ridge", **r_metrics})

        # 2. LightGBM Forecaster
        lgbm_forecaster = DirectMultiStepForecaster(
            model_type="lgbm", n_estimators=25, learning_rate=0.03, max_depth=3
        )
        pred_lgbm = lgbm_forecaster.fit_and_predict(train_s, horizon=len(val_cpu))
        l_metrics = compute_metrics(val_cpu, pred_lgbm)
        results.append({"fold": fold_i, "model": "Direct_LightGBM", **l_metrics})

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("model")[["MAE", "RMSE"]].mean().reset_index()
    print("\n=== Validation CV ML Models Benchmark (Aggregate CPU) ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    cpu_df, _ = load_raw_data("challenge_data")
    targets_df = compute_derived_targets(cpu_df)
    train_df, holdout_df = temporal_train_test_split(targets_df, train_size=110)

    # Evaluate ML models under identical CV folds
    cross_validate_ml_models(train_df)