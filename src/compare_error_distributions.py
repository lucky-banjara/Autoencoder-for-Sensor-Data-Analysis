
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASELINE_PATH = (
    "results/reconstruction/validation_window_mse.npy"
)

DAMAGE_DIR = "results/damage_evaluation"
OUTPUT_DIR = "results/damage_evaluation"
PLOT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate reconstruction errors
# --------------------------------------------------

def load_errors(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    errors = np.asarray(
        np.load(path),
        dtype=np.float64
    )

    if errors.ndim != 1 or len(errors) == 0:
        raise ValueError(
            f"Expected a non-empty 1D error array: {path}"
        )

    if not np.isfinite(errors).all():
        raise ValueError(
            f"Non-finite errors found in: {path}"
        )

    if np.any(errors < 0):
        raise ValueError(
            f"Negative MSE found in: {path}"
        )

    return errors


baseline = load_errors(BASELINE_PATH)

states = {
    "UDS": baseline
}

for number in range(1, 9):
    state = f"DS{number}"
    path = os.path.join(
        DAMAGE_DIR,
        f"DS{number}_window_mse.npy"
    )

    states[state] = load_errors(path)


# --------------------------------------------------
# 3. Calculate descriptive statistics
# --------------------------------------------------

baseline_median = float(np.median(baseline))
rows = []

for state, errors in states.items():
    median = float(np.median(errors))

    rows.append({
        "state": state,
        "number_of_windows": len(errors),
        "mean_mse": float(np.mean(errors)),
        "median_mse": median,
        "p90_mse": float(np.percentile(errors, 90)),
        "p95_mse": float(np.percentile(errors, 95)),
        "maximum_mse": float(np.max(errors)),
        "median_ratio_vs_UDS": (
            median / baseline_median
            if baseline_median > 0
            else np.nan
        ),
    })

summary = pd.DataFrame(rows)

summary_path = os.path.join(
    OUTPUT_DIR,
    "error_distribution_comparison.csv"
)

summary.to_csv(summary_path, index=False)

print("=" * 75)
print("HBTA ERROR-DISTRIBUTION COMPARISON")
print("=" * 75)
print(f"UDS reference windows: {len(baseline)}")
print(f"UDS median MSE: {baseline_median:.6f}")
print("\nDescriptive comparison:")
print(
    summary.to_string(
        index=False,
        formatters={
            "mean_mse": "{:.6f}".format,
            "median_mse": "{:.6f}".format,
            "p90_mse": "{:.6f}".format,
            "p95_mse": "{:.6f}".format,
            "maximum_mse": "{:.6f}".format,
            "median_ratio_vs_UDS": "{:.2f}x".format,
        }
    )
)

print("\nSaved:", summary_path)


# --------------------------------------------------
# 4. Plot distributions
# --------------------------------------------------

labels = list(states.keys())
plot_data = [
    np.maximum(states[label], 1e-10)
    for label in labels
]

plt.figure(figsize=(12, 7))

plt.boxplot(
    plot_data,
    tick_labels=labels,
    showfliers=False
)

plt.yscale("log")
plt.ylabel("Reconstruction MSE (log scale)")
plt.xlabel("Configuration / damage state")
plt.title(
    "HBTA Reconstruction-Error Distributions"
)
plt.grid(True, axis="y", which="both", alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(
    PLOT_DIR,
    "error_distribution_comparison.png"
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved plot:", plot_path)
print("\nDistribution comparison complete.")