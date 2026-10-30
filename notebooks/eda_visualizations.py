"""Script to produce core EDA plots for the report."""
from pathlib import Path
import sys

# Ensure root directory is on Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now import from src
from src.data import load_raw_data
from src.eda import compute_derived_targets
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
cpu_df, meta_df = load_raw_data("challenge_data")
targets = compute_derived_targets(cpu_df)

# Figure 1: Both Target Series Over Time (Indicating Train/Holdout split)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Active VMs
ax1.plot(targets["timestamp"], targets["active_vm_count"], color="#1f77b4", lw=2)
ax1.axvline(x=targets["timestamp"].iloc[109], color="red", linestyle="--", label="Train/Holdout Boundary (t=110)")
ax1.set_ylabel("Active VM Count")
ax1.set_title("Target 1: Active VM Count Over Time")
ax1.legend(loc="upper left")

# Aggregate CPU Utilization
ax2.plot(targets["timestamp"], targets["aggregate_cpu_utilization"], color="#ff7f0e", lw=2)
ax2.axvline(x=targets["timestamp"].iloc[109], color="red", linestyle="--", label="Train/Holdout Boundary (t=110)")
ax2.set_ylabel("Aggregate CPU (%)")
ax2.set_xlabel("Timestamp")
ax2.set_title("Target 2: Mean Aggregate CPU Utilization Over Time")
ax2.legend(loc="upper left")

plt.tight_layout()
fig.savefig(output_dir / "target_series_overview.png", dpi=300)
print(f"Saved: {output_dir / 'target_series_overview.png'}")

# Figure 2: Workload Category Distribution in Metadata
if "workload_category" in meta_df.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=meta_df, x="workload_category", palette="viridis")
    plt.title("VM Distribution by Workload Category")
    plt.tight_layout()
    plt.savefig(output_dir / "workload_categories.png", dpi=300)
    print(f"Saved: {output_dir / 'workload_categories.png'}")