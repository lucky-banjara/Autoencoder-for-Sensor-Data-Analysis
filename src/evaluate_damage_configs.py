
import os
import joblib
import h5py
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

BASELINE_ERRORS_PATH = (
    "results/reconstruction/validation_window_mse.npy"
)

CHANNEL_PATH = "acceleration/AG01/y"
WINDOW_SIZE = 100
INPUT_DIM = 100
BOTTLENECK_DIM = 32

OUTPUT_DIR = "results/damage_evaluation"
PLOT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

required_files = [
    H5_PATH,
    SCALER_PATH,
    MODEL_PATH,
    BASELINE_ERRORS_PATH,
]

for path in required_files:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Complete the relevant preprocessing or training "
            "steps before running damage evaluation."
        )


# --------------------------------------------------
# 3. Load the trained model and fitted scaler
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = Autoencoder(
    input_dim=INPUT_DIM,
    bottleneck_dim=BOTTLENECK_DIM
)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# IMPORTANT:
# The scaler was saved using joblib.dump() in preprocess_signal.py.
# Therefore, load it with joblib.load(), not pickle.load().
scaler = joblib.load(SCALER_PATH)

if not hasattr(scaler, "n_features_in_"):
    raise ValueError(
        "Loaded scaler does not report its fitted feature count."
    )

if scaler.n_features_in_ != INPUT_DIM:
    raise ValueError(
        f"Expected scaler with {INPUT_DIM} features, "
        f"but found {scaler.n_features_in_}."
    )

print("=" * 65)
print("HBTA DAMAGE-STATE EVALUATION")
print("=" * 65)
print("Device:", device)
print("Model loaded:", MODEL_PATH)
print("Scaler loaded:", SCALER_PATH)
print("Input dimension:", INPUT_DIM)
print("Window size:", WINDOW_SIZE, "samples")
print("Channel:", CHANNEL_PATH)


# --------------------------------------------------
# 4. Create non-overlapping signal windows
# --------------------------------------------------

def make_windows(signal, window_size=WINDOW_SIZE):
    """
    Split a signal into complete, non-overlapping windows.
    Incomplete trailing samples are discarded.
    """

    if signal.ndim != 1:
        signal = np.asarray(signal).reshape(-1)

    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    usable_length = (
        len(signal) // window_size
    ) * window_size

    if usable_length == 0:
        raise ValueError(
            "Signal is shorter than one complete window."
        )

    trimmed_signal = signal[:usable_length]

    return trimmed_signal.reshape(-1, window_size)


# --------------------------------------------------
# 5. Scale, reconstruct and calculate window errors
# --------------------------------------------------

def evaluate_windows(windows, batch_size=128):
    """
    Apply the original fitted scaler, reconstruct the windows,
    and calculate MSE for each window.
    """

    windows = np.asarray(windows, dtype=np.float64)

    if windows.ndim != 2 or windows.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Expected windows with shape (N, {INPUT_DIM}); "
            f"received {windows.shape}."
        )

    if not np.isfinite(windows).all():
        raise ValueError(
            "Input windows contain NaN or Inf values."
        )

    # Reuse the scaler fitted on undamaged training windows.
    scaled_windows = scaler.transform(windows).astype(np.float32)

    if not np.isfinite(scaled_windows).all():
        raise ValueError(
            "Scaled windows contain NaN or Inf values."
        )

    reconstructed_batches = []

    with torch.no_grad():

        for start in range(0, len(scaled_windows), batch_size):

            batch_array = scaled_windows[
                start:start + batch_size
            ]

            batch_tensor = torch.from_numpy(
                batch_array
            ).to(device)

            reconstructed_tensor = model(batch_tensor)

            reconstructed_batches.append(
                reconstructed_tensor.cpu().numpy()
            )

    reconstructed_windows = np.concatenate(
        reconstructed_batches,
        axis=0
    )

    # One mean squared reconstruction error per window.
    errors = np.mean(
        (scaled_windows - reconstructed_windows) ** 2,
        axis=1
    )

    return scaled_windows, reconstructed_windows, errors


# --------------------------------------------------
# 6. Load undamaged baseline validation errors
# --------------------------------------------------

baseline_errors = np.load(BASELINE_ERRORS_PATH)

if baseline_errors.ndim != 1 or len(baseline_errors) == 0:
    raise ValueError(
        "Baseline errors must be a non-empty 1D array."
    )

if not np.isfinite(baseline_errors).all():
    raise ValueError(
        "Baseline errors contain NaN or Inf values."
    )

baseline_summary = {
    "configuration": "UDS_baseline_validation",
    "damage_state": "UDS",
    "number_of_windows": len(baseline_errors),
    "mean_mse": float(np.mean(baseline_errors)),
    "median_mse": float(np.median(baseline_errors)),
    "std_mse": float(np.std(baseline_errors)),
    "p95_mse": float(np.percentile(baseline_errors, 95)),
    "max_mse": float(np.max(baseline_errors)),
}

summary_rows = [baseline_summary]
error_records = []

for index, error in enumerate(baseline_errors):

    error_records.append({
        "configuration": "UDS_baseline_validation",
        "damage_state": "UDS",
        "window_index": index,
        "reconstruction_mse": float(error),
    })

print("\nUndamaged baseline validation:")
print(f"  Windows: {len(baseline_errors)}")
print(f"  Mean MSE: {np.mean(baseline_errors):.6f}")
print(f"  Median MSE: {np.median(baseline_errors):.6f}")
print(
    f"  95th percentile MSE: "
    f"{np.percentile(baseline_errors, 95):.6f}"
)


# --------------------------------------------------
# 7. Evaluate damage states DS1 to DS8
# --------------------------------------------------

with h5py.File(H5_PATH, "r") as h5_file:

    for damage_number in range(1, 9):

        configuration = (
            f"MVS_P2_DS{damage_number}_SM_Y_01"
        )

        dataset_path = (
            f"{configuration}/{CHANNEL_PATH}"
        )

        print(f"\nEvaluating {configuration}...")

        if dataset_path not in h5_file:

            print(
                f"WARNING: Dataset not found: {dataset_path}"
            )
            continue

        signal = np.asarray(
            h5_file[dataset_path][:],
            dtype=np.float64
        ).reshape(-1)

        if not np.isfinite(signal).all():

            print(
                f"WARNING: Non-finite values in {configuration}. "
                "Skipping this configuration."
            )
            continue

        windows = make_windows(
            signal,
            WINDOW_SIZE
        )

        scaled, reconstructed, errors = evaluate_windows(
            windows
        )

        print(f"  Raw samples: {len(signal)}")
        print(f"  Complete windows: {len(windows)}")
        print(f"  Discarded trailing samples: "
              f"{len(signal) % WINDOW_SIZE}")
        print(f"  Mean MSE: {np.mean(errors):.6f}")
        print(f"  Median MSE: {np.median(errors):.6f}")
        print(f"  95th percentile MSE: "
              f"{np.percentile(errors, 95):.6f}")
        print(f"  Maximum MSE: {np.max(errors):.6f}")

        # Summary statistics for this configuration.
        summary_rows.append({
            "configuration": configuration,
            "damage_state": f"DS{damage_number}",
            "number_of_windows": len(errors),
            "mean_mse": float(np.mean(errors)),
            "median_mse": float(np.median(errors)),
            "std_mse": float(np.std(errors)),
            "p95_mse": float(np.percentile(errors, 95)),
            "max_mse": float(np.max(errors)),
        })

        # Per-window reconstruction errors.
        for index, error in enumerate(errors):

            error_records.append({
                "configuration": configuration,
                "damage_state": f"DS{damage_number}",
                "window_index": index,
                "reconstruction_mse": float(error),
            })

        # Save reconstruction results for this damage state.
        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"DS{damage_number}_window_mse.npy"
            ),
            errors
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"DS{damage_number}_scaled_windows.npy"
            ),
            scaled
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"DS{damage_number}_reconstructed.npy"
            ),
            reconstructed
        )


# --------------------------------------------------
# 8. Save summary and per-window CSV files
# --------------------------------------------------

summary_df = pd.DataFrame(summary_rows)
errors_df = pd.DataFrame(error_records)

summary_path = os.path.join(
    OUTPUT_DIR,
    "damage_state_summary.csv"
)

errors_path = os.path.join(
    OUTPUT_DIR,
    "all_window_reconstruction_errors.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

errors_df.to_csv(
    errors_path,
    index=False
)

print("\n" + "=" * 65)
print("SUMMARY OF RECONSTRUCTION ERRORS")
print("=" * 65)
print(summary_df.to_string(index=False))

print("\nSaved summary:", summary_path)
print("Saved per-window errors:", errors_path)


# --------------------------------------------------
# 9. Plot reconstruction-error distributions
# --------------------------------------------------

plot_labels = []
plot_data = []

# Keep UDS and DS1-DS8 in the summary's order.
for _, row in summary_df.iterrows():

    configuration = row["configuration"]

    matching_errors = errors_df.loc[
        errors_df["configuration"] == configuration,
        "reconstruction_mse"
    ].to_numpy()

    if len(matching_errors) > 0:

        plot_labels.append(
            row["damage_state"]
        )

        plot_data.append(
            matching_errors
        )

if plot_data:

    plt.figure(figsize=(12, 6))

    # "labels" supports older Matplotlib versions too.
    plt.boxplot(
        plot_data,
        tick_labels=plot_labels,
        showfliers=True
    )

    plt.ylabel("Reconstruction MSE")
    plt.xlabel("Configuration / damage state")
    plt.title(
        "HBTA Autoencoder Reconstruction Errors by Damage State"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    boxplot_path = os.path.join(
        PLOT_DIR,
        "damage_state_error_comparison.png"
    )

    plt.savefig(
        boxplot_path,
        dpi=300
    )

    plt.show()

    print("Saved comparison plot:", boxplot_path)

else:
    print("No error distributions available to plot.")

print("\nDamage-state evaluation complete.")