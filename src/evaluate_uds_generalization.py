
import os

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

H5_PATH = "data/data_100Hz.h5"
SCALER_PATH = "data/processed/scaler.pkl"
MODEL_PATH = "models/autoencoder.pth"

CHANNEL_PATH = "acceleration/AG01/y"
WINDOW_SIZE = 100

BASELINE_ERRORS_PATH = (
    "results/reconstruction/validation_window_mse.npy"
)

OUTPUT_DIR = "results/uds_generalization"
PLOT_DIR = "results/plots"

# Same experimental mode and sensor direction as the
# original baseline and DS1-DS8 evaluation.
UDS_CONFIGURATIONS = [
    "MVS_P1_UDS_SM_Y_01",
    "MVS_P1_UDS_SM_Y_02",
    "MVS_P2_UDS_SM_Y_02",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load model, scaler and baseline threshold
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = Autoencoder(input_dim=100, bottleneck_dim=32)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# The preprocessing script saves the scaler using joblib.
scaler = joblib.load(SCALER_PATH)

baseline_errors = np.load(BASELINE_ERRORS_PATH)

if not np.isfinite(baseline_errors).all():
    raise ValueError("Baseline errors contain NaN or Inf.")

threshold = float(np.percentile(baseline_errors, 95))

print("=" * 72)
print("HBTA UNSEEN-UDS GENERALISATION CHECK")
print("=" * 72)
print("Device:", device)
print(f"Baseline threshold: {threshold:.6f}")
print("Scaler and trained model loaded.")


# --------------------------------------------------
# 3. Windowing and reconstruction
# --------------------------------------------------

def make_windows(signal):
    """Create complete, non-overlapping 1-second windows."""
    count = len(signal) // WINDOW_SIZE

    if count == 0:
        raise ValueError("Signal is shorter than one window.")

    usable_length = count * WINDOW_SIZE

    return signal[:usable_length].reshape(count, WINDOW_SIZE)


def calculate_errors(windows, batch_size=128):
    """Apply the original scaler and calculate window MSE."""
    if not np.isfinite(windows).all():
        raise ValueError("Signal windows contain NaN or Inf.")

    scaled = scaler.transform(windows).astype(np.float32)

    error_parts = []

    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.from_numpy(
                scaled[start:start + batch_size]
            ).to(device)

            reconstructed = model(batch).cpu().numpy()

            batch_errors = np.mean(
                (scaled[start:start + batch_size] - reconstructed) ** 2,
                axis=1,
            )

            error_parts.append(batch_errors)

    errors = np.concatenate(error_parts)

    return errors


# --------------------------------------------------
# 4. Evaluate the unseen UDS recordings
# --------------------------------------------------

rows = []
error_arrays = {}

with h5py.File(H5_PATH, "r") as h5_file:

    for configuration in UDS_CONFIGURATIONS:
        dataset_path = f"{configuration}/{CHANNEL_PATH}"

        if dataset_path not in h5_file:
            print(f"WARNING: Missing dataset: {dataset_path}")
            continue

        signal = np.asarray(
            h5_file[dataset_path][:],
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(signal).all():
            print(f"WARNING: Invalid values in {configuration}; skipped.")
            continue

        windows = make_windows(signal)
        errors = calculate_errors(windows)

        exceedances = int(np.sum(errors > threshold))
        exceedance_rate = 100 * exceedances / len(errors)

        error_arrays[configuration] = errors

        rows.append({
            "configuration": configuration,
            "raw_samples": len(signal),
            "number_of_windows": len(errors),
            "discarded_samples": len(signal) % WINDOW_SIZE,
            "mean_mse": float(np.mean(errors)),
            "median_mse": float(np.median(errors)),
            "p95_mse": float(np.percentile(errors, 95)),
            "windows_above_threshold": exceedances,
            "exceedance_rate_percent": exceedance_rate,
        })

        safe_name = configuration.replace("/", "_")

        np.save(
            os.path.join(OUTPUT_DIR, f"{safe_name}_mse.npy"),
            errors,
        )

        print(f"\n{configuration}")
        print(f"  Windows: {len(errors)}")
        print(f"  Mean MSE: {np.mean(errors):.6f}")
        print(f"  Median MSE: {np.median(errors):.6f}")
        print(f"  95th percentile MSE: {np.percentile(errors, 95):.6f}")
        print(
            f"  Above threshold: {exceedances}/{len(errors)} "
            f"({exceedance_rate:.2f}%)"
        )


# --------------------------------------------------
# 5. Save summary
# --------------------------------------------------

if not rows:
    raise RuntimeError("No UDS configurations were successfully evaluated.")

summary = pd.DataFrame(rows)

csv_path = os.path.join(
    OUTPUT_DIR,
    "uds_generalization_summary.csv",
)

summary.to_csv(csv_path, index=False)

print("\n" + "=" * 72)
print("UDS GENERALISATION SUMMARY")
print("=" * 72)
print(summary.to_string(index=False, formatters={
    "mean_mse": "{:.6f}".format,
    "median_mse": "{:.6f}".format,
    "p95_mse": "{:.6f}".format,
    "exceedance_rate_percent": "{:.2f}%".format,
}))

print("\nSaved summary:", csv_path)


# --------------------------------------------------
# 6. Compare with the original UDS validation set
# --------------------------------------------------

original_label = "Original UDS validation"
plot_labels = [original_label] + list(error_arrays.keys())

plot_data = [baseline_errors] + [
    error_arrays[name] for name in error_arrays
]

plt.figure(figsize=(12, 7))

plt.boxplot(
    [np.maximum(errors, 1e-10) for errors in plot_data],
    tick_labels=plot_labels,
    showfliers=False,
)

plt.yscale("log")
plt.axhline(
    threshold,
    linestyle="--",
    label=f"95th percentile threshold = {threshold:.4f}",
)

plt.ylabel("Reconstruction MSE (log scale)")
plt.xlabel("Undamaged recording")
plt.title("HBTA Autoencoder: Unseen UDS Generalisation")
plt.xticks(rotation=15, ha="right")
plt.legend()
plt.grid(True, axis="y", which="both", alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(
    PLOT_DIR,
    "uds_generalization_comparison.png",
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved plot:", plot_path)
print("\nUDS generalisation check complete.")