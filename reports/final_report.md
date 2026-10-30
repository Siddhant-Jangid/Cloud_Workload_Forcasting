# Cloud Workload Forecasting & Capacity Planning Report

## Executive Summary

This report analyzes a 10,000-VM cloud workload derived from the Microsoft Azure Public Dataset V2 to forecast near-term infrastructure demand across a 24-step (2-hour) holdout horizon. Fleet-wide target aggregation reveals that while individual VMs exhibit high variance, aggregate active VM count is invariant at 10,000, and mean fleet CPU demand fluctuates in a stationary, mean-reverting band (~9.1% to ~9.6%).

A damped mean-reversion forecaster outperformed both complex tree-based ensembles (LightGBM) and persistence baselines, achieving **0.1217% MAE** and **0.1472% RMSE** on the holdout set. Coupling this forecast with a dynamic safety buffer (+0.45% CPU headroom) delivered **0% SLA breaches** while limiting unutilized provisioning waste to 0.54% CPU.

---

## 1. Problem Formulation & Fleet Dynamics

Two system-level time series were extracted at 5-minute (300-second) intervals across 134 time steps:

1. **`active_vm_count(t)`**: Distinct VMs reporting metrics at step $t$.
2. **`aggregate_cpu_utilization(t)`**: Mean CPU usage across all active VMs.

### Workload Characteristics & Invariants

- **Static Fleet Footprint**: Throughout all 134 observation windows, exactly 10,000 VMs reported data without additions or deletions.
- **Skewed Micro-level vs. Gaussian Macro-level Behavior**: At the individual VM level, CPU usage is heavily right-skewed (median = 5.1%, 99th percentile = 87.4%). Under aggregation, the Central Limit Theorem stabilizes total fleet CPU demand into a narrow band centered at $\mu = 9.38\%$ with $\sigma = 0.18\%$.

---

## 2. Validation Architecture & Leakage Prevention

To reflect production conditions, data shuffling was strictly prohibited:

- **Training Window**: Steps 0 to 109 (~9.1 hours).
- **Holdout Horizon**: Steps 110 to 133 (24 steps / 2.0 hours).
- **Cross-Validation Scheme**: A 4-fold expanding-window (rolling-origin) CV was configured on the training set with a minimum training history of 60 steps and an evaluation horizon of 24 steps.

---

## 3. Model Progression & Evaluation

| Model Architecture                 | Cross-Validation MAE (%) | CV RMSE (%) | Holdout MAE (%) | Holdout RMSE (%) | Peak Error (%) |
| :--------------------------------- | :----------------------: | :---------: | :-------------: | :--------------: | :------------: |
| **Last Value Naive**               |          0.2802          |   0.3113    |     0.2319      |      0.2741      |     0.4421     |
| **Direct LightGBM**                |          0.1891          |   0.2211    |     0.1984      |      0.2310      |     0.3805     |
| **Direct Ridge**                   |          0.1808          |   0.2105    |     0.1652      |      0.1920      |     0.3112     |
| **Historical Mean**                |          0.1728          |   0.2000    |     0.1285      |      0.1542      |     0.2890     |
| **Damped Mean Reversion (Chosen)** |        **0.1691**        | **0.1974**  |   **0.1217**    |    **0.1472**    |   **0.2736**   |

### Why Simpler Outperformed Complex ML

Non-linear gradient boosted trees (LightGBM) overfit high-frequency variance within small window samples. Because aggregate fleet demand exhibits strong mean reversion toward fleet equilibrium, an exponentially damped transition model:

$$
\hat{y}_{t+h} = \alpha^h y_t + (1 - \alpha^h) \bar{y} \quad (\alpha = 0.85)
$$

optimally balanced immediate momentum with long-run stability.

---

## 4. Operational Capacity Planning Policy

Forecasting aggregate demand is only useful if it directly drives infrastructure decisions:

[Forecast: y_hat(t)] ──> [+0.45% Safety Buffer] ──> [Provisioning Target: C(t)] ──> [Physical Clamping: [0, 100%]]

### SLA Violation vs. Cost Trade-off

- **Asymmetric Loss Objective**: The operational cost of an under-provisioning event (SLA degradation, dropped queries) is substantially higher than the cost of modest idle capacity.
- **Safety Headroom**: Sizing headroom to the 95th percentile historical residual ($+0.45\%$ CPU) resulted in:
  - **SLA Breaches**: 0 out of 24 holdout intervals (100% reliability).
  - **Buffer Waste**: Average unutilized headroom of 0.537% CPU.

---

## 5. Infrastructure Recommendations

1. **Dynamic Right-Sizing**: Maintain provisioned infrastructure capacity at `Predicted CPU + 0.45%`. If aggregate demand approaches 12%, automated scaling alerts should trigger.
2. **Micro-Workload Colocation**: Because 50% of VMs run at $<5\%$ average CPU, the platform team should implement VM bin-packing and co-locate complementary bursty workloads on shared physical hosts to raise server efficiency without risking SLA breaches.
3. **Continuous Residual Monitoring**: If running residual errors consistently exceed $\pm 0.35\%$, trigger automated model parameter refitting.
