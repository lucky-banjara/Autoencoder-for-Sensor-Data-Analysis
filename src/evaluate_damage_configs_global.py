
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch

from autoencoder import Autoencoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

H5_PATH = Path("data/data_100Hz.h5")
SCALER_PATH = Path("data/processed_global/global_scaler.pkl")
MODEL_PATH = Path("models/autoencoder_global.pth")

OUTPUT_DIR = Path("results/damage_evaluation_global")
PLOT_DIR = Path("results/plots")

CHANNEL_PATH = "acceleration/AG01/y"
WINDOW_SIZE = 100
BOTTLENECK_DIM = 32

# Calibrated from the global model's UDS validation errors.
THRESHOLD = 0.463218

DAMAGE_CONFIGS = [
    f"MVS_P2_DS{i}_SM_Y_01"
    for i in range(1, 9)
]


def load_signal(h5_file, config_name):
    """Load the selected acceleration channel from one configuration."""
    dataset_path = f"{config_name}/{CHANNEL_PATH}"

    if dataset_path not in h5_file:
        raise KeyError(
            f"Dataset not found: {dataset_path}"
        )

    return np.asarray(h5_file[dataset_path][:], dtype=np.float64)


def make_windows(signal):
    """Create complete, non-overlapping 100-sample windows."""
    usable_length = (len(signal) // WINDOW_SIZE) * WINDOW_SIZE
    signal = signal[:usable_length]

    if usable_length == 0:
        raise ValueError("Signal is too short to create a window.")

    return signal.reshape(-1, WINDOW_SIZE)


def main():
    print("=" * 70)
    print("GLOBAL-SCALING DAMAGE CONFIGURATION EVALUATION")
    print("=" * 70)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load global scaling parameters.
    scaler_params = joblib.load(SCALER_PATH)
    global_mean = float(scaler_params["mean"])
    global_std = float(scaler_params["std"])

    if not np.isfinite(global_mean) or not np.isfinite(global_std):
        raise ValueError("Global scaling parameters are not finite.")

    if global_std <= 0:
        raise ValueError("Global scaling standard deviation must be positive.")

    print(f"Global mean: {global_mean:.8f}")
    print(f"Global standard deviation: {global_std:.8f}")
    print(f"Global threshold: {THRESHOLD:.6f}")

    # Load the trained global-scaling model.
    model = Autoencoder(
        input_dim=WINDOW_SIZE,
        bottleneck_dim=BOTTLENECK_DIM,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with h5py.File(H5_PATH, "r") as h5_file:
        for config_name in DAMAGE_CONFIGS:
            print(f"\nEvaluating {config_name}...")

            signal = load_signal(h5_file, config_name)
            windows = make_windows(signal)

            # Apply the same global scaler fitted on UDS training data.
            scaled_windows = (
                (windows - global_mean) / global_std
            ).astype(np.float32)

            if not np.isfinite(scaled_windows).all():
                raise ValueError(
                    f"Non-finite values in {config_name}"
                )

            # Reconstruct windows.
            with torch.no_grad():
                inputs = torch.from_numpy(scaled_windows).to(device)
                reconstructed = model(inputs).cpu().numpy()

            # Mean squared reconstruction error per window.
            errors = np.mean(
                (scaled_windows - reconstructed) ** 2,
                axis=1,
            )

            exceedances = errors > THRESHOLD
            exceedance_count = int(np.sum(exceedances))
            window_count = len(errors)

            summary = {
                "configuration": config_name,
                "window_count": window_count,
                "mean_mse": float(np.mean(errors)),
                "median_mse": float(np.median(errors)),
                "p95_mse": float(np.percentile(errors, 95)),
                "max_mse": float(np.max(errors)),
                "threshold": THRESHOLD,
                "exceedance_count": exceedance_count,
                "exceedance_percent": (
                    100.0 * exceedance_count / window_count
                ),
            }
            results.append(summary)

            # Save each configuration's individual errors.
            np.save(
                OUTPUT_DIR / f"{config_name}_window_mse.npy",
                errors,
            )

            print(f"  Windows: {window_count}")
            print(f"  Mean MSE: {summary['mean_mse']:.6f}")
            print(f"  Median MSE: {summary['median_mse']:.6f}")
            print(f"  95th percentile MSE: {summary['p95_mse']:.6f}")
            print(f"  Maximum MSE: {summary['max_mse']:.6f}")
            print(
                f"  Above threshold: {exceedance_count}/{window_count} "
                f"({summary['exceedance_percent']:.2f}%)"
            )

    # Save the summary table.
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "global_damage_state_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plot the percentage of windows above the global threshold.
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [
        name.replace("MVS_P2_", "").replace("_SM_Y_01", "")
        for name in summary_df["configuration"]
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, summary_df["exceedance_percent"])
    plt.axhline(
        5,
        linestyle="--",
        label="5% reference",
    )
    plt.xlabel("Configuration")
    plt.ylabel("Windows above global threshold (%)")
    plt.title("Global-Scaling Reconstruction Error Exceedance")
    plt.legend()
    plt.tight_layout()

    plot_path = PLOT_DIR / "global_damage_exceedance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("GLOBAL DAMAGE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Summary CSV: {summary_path}")
    print(f"Exceedance plot: {plot_path}")


if __name__ == "__main__":
    main()
