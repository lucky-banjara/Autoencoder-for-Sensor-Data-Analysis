
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASELINE_ERRORS_PATH = (
    "results/reconstruction/validation_window_mse.npy"
)

DAMAGE_RESULTS_DIR = "results/damage_evaluation"
OUTPUT_DIR = "results/damage_evaluation"
PLOT_DIR = "results/plots"

THRESHOLD_PERCENTILE = 95

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Validate and load baseline errors
# --------------------------------------------------

if not os.path.isfile(BASELINE_ERRORS_PATH):
    raise FileNotFoundError(
        f"Baseline errors not found: {BASELINE_ERRORS_PATH}\n"
        "Run src/reconstruct_signal.py first."
    )

baseline_errors = np.asarray(
    np.load(BASELINE_ERRORS_PATH),
    dtype=np.float64
)

if baseline_errors.ndim != 1 or len(baseline_errors) == 0:
    raise ValueError(
        "Baseline errors must be a non-empty 1D array."
    )

if not np.isfinite(baseline_errors).all():
    raise ValueError(
        "Baseline errors contain NaN or Inf values."
    )


# --------------------------------------------------
# 3. Calculate the preliminary baseline threshold
# --------------------------------------------------

threshold = float(
    np.percentile(
        baseline_errors,
        THRESHOLD_PERCENTILE
    )
)

baseline_flags = baseline_errors > threshold

baseline_flag_count = int(np.sum(baseline_flags))

baseline_flag_rate = (
    100.0 * baseline_flag_count / len(baseline_errors)
)


print("=" * 70)
print("HBTA RECONSTRUCTION-ERROR THRESHOLD EVALUATION")
print("=" * 70)

print("\nBaseline: Undamaged validation data (UDS)")
print(f"Number of windows: {len(baseline_errors)}")
print(f"Mean MSE: {np.mean(baseline_errors):.6f}")
print(f"Median MSE: {np.median(baseline_errors):.6f}")
print(
    f"{THRESHOLD_PERCENTILE}th percentile threshold: "
    f"{threshold:.6f}"
)
print(
    f"UDS windows above threshold: "
    f"{baseline_flag_count}/{len(baseline_errors)}"
)
print(f"Observed UDS exceedance rate: {baseline_flag_rate:.2f}%")


# --------------------------------------------------
# 4. Evaluate baseline and damage configurations
# --------------------------------------------------

results = []

# Include the baseline as a reference row.
results.append({
    "damage_state": "UDS",
    "configuration": "UDS_baseline_validation",
    "number_of_windows": len(baseline_errors),
    "threshold_mse": threshold,
    "mean_mse": float(np.mean(baseline_errors)),
    "median_mse": float(np.median(baseline_errors)),
    "p95_mse": float(np.percentile(baseline_errors, 95)),
    "max_mse": float(np.max(baseline_errors)),
    "windows_above_threshold": baseline_flag_count,
    "windows_at_or_below_threshold": (
        len(baseline_errors) - baseline_flag_count
    ),
    "exceedance_rate_percent": baseline_flag_rate,
})

for damage_number in range(1, 9):

    damage_state = f"DS{damage_number}"

    errors_path = os.path.join(
        DAMAGE_RESULTS_DIR,
        f"DS{damage_number}_window_mse.npy"
    )

    if not os.path.isfile(errors_path):
        print(
            f"\nWARNING: Missing results for {damage_state}: "
            f"{errors_path}"
        )
        continue

    errors = np.asarray(
        np.load(errors_path),
        dtype=np.float64
    )

    if errors.ndim != 1 or len(errors) == 0:
        print(
            f"\nWARNING: Empty or invalid error array for "
            f"{damage_state}; skipping."
        )
        continue

    if not np.isfinite(errors).all():
        print(
            f"\nWARNING: NaN or Inf values in {damage_state}; "
            "skipping."
        )
        continue

    # Flag a window when its MSE exceeds the UDS threshold.
    flags = errors > threshold

    flagged_count = int(np.sum(flags))
    unflagged_count = int(len(errors) - flagged_count)

    exceedance_rate = (
        100.0 * flagged_count / len(errors)
    )

    results.append({
        "damage_state": damage_state,
        "configuration": (
            f"MVS_P2_DS{damage_number}_SM_Y_01"
        ),
        "number_of_windows": len(errors),
        "threshold_mse": threshold,
        "mean_mse": float(np.mean(errors)),
        "median_mse": float(np.median(errors)),
        "p95_mse": float(np.percentile(errors, 95)),
        "max_mse": float(np.max(errors)),
        "windows_above_threshold": flagged_count,
        "windows_at_or_below_threshold": unflagged_count,
        "exceedance_rate_percent": exceedance_rate,
    })

    print(f"\n{damage_state}")
    print(f"  Windows evaluated: {len(errors)}")
    print(f"  Mean MSE: {np.mean(errors):.6f}")
    print(f"  Median MSE: {np.median(errors):.6f}")
    print(f"  Windows above threshold: {flagged_count}")
    print(f"  Windows at/below threshold: {unflagged_count}")
    print(f"  Exceedance rate: {exceedance_rate:.2f}%")


# --------------------------------------------------
# 5. Save the threshold evaluation report
# --------------------------------------------------

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "detection_threshold_summary.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print("\n" + "=" * 70)
print("THRESHOLD EVALUATION SUMMARY")
print("=" * 70)

columns_to_show = [
    "damage_state",
    "number_of_windows",
    "mean_mse",
    "median_mse",
    "windows_above_threshold",
    "exceedance_rate_percent",
]

print(
    results_df[columns_to_show].to_string(
        index=False,
        formatters={
            "mean_mse": "{:.6f}".format,
            "median_mse": "{:.6f}".format,
            "exceedance_rate_percent": "{:.2f}%".format,
        }
    )
)

print("\nSaved CSV report:")
print(csv_path)


# --------------------------------------------------
# 6. Plot threshold exceedance rates
# --------------------------------------------------

if len(results_df) > 0:

    labels = results_df["damage_state"].tolist()

    rates = results_df[
        "exceedance_rate_percent"
    ].to_numpy()

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        labels,
        rates
    )

    # Show the observed UDS exceedance rate as a reference.
    plt.axhline(
        baseline_flag_rate,
        linestyle="--",
        label=(
            f"Observed UDS exceedance rate "
            f"({baseline_flag_rate:.2f}%)"
        )
    )

    plt.ylabel("Windows above threshold (%)")
    plt.xlabel("Configuration / damage state")
    plt.title(
        "HBTA Autoencoder: Threshold Exceedance by State"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3
    )

    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        PLOT_DIR,
        "detection_threshold_exceedance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nSaved threshold exceedance plot:")
    print(plot_path)

print("\nThreshold evaluation complete.")
print(
    "\nInterpretation note: exceeding the threshold indicates "
    "unusual reconstruction error, not confirmed structural damage."
)