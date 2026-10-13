
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch

from autoencoder import Autoencoder


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("data/data_100Hz.h5")
SCALER_PATH = Path("data/processed/scaler.pkl")
MODEL_PATH = Path("models/autoencoder.pth")

OUTPUT_DIR = Path("results/uds_all_configurations")
PLOT_DIR = Path("results/plots")

CHANNEL_PATH = "acceleration/AG01/y"
WINDOW_SIZE = 100
THRESHOLD = 0.465729

# The original training configuration is already evaluated.
# The three additional SM-Y configurations are covered by
# evaluate_uds_generalization.py.
REMAINING_CONFIGS = [
    "MVS_P1_UDS_SM_Z_01",
    "MVS_P1_UDS_SM_Z_02",
    "MVS_P2_UDS_SM_Z_01",
    "MVS_P2_UDS_SM_Z_02",
    "MVS_P2_UDS_NM_Z_01",
    "MVS_P2_UDS_NM_Z_02",
]


def load_model(device):
    model = Autoencoder(input_dim=WINDOW_SIZE, bottleneck_dim=32)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def load_windows(dataset_path, config_name):
    with h5py.File(dataset_path, "r") as h5_file:
        dataset_key = f"{config_name}/{CHANNEL_PATH}"

        if dataset_key not in h5_file:
            raise KeyError(
                f"Dataset not found in HDF5 file: {dataset_key}"
            )

        signal = np.asarray(h5_file[dataset_key][:], dtype=np.float64)

    finite = np.isfinite(signal)

    if not finite.all():
        raise ValueError(
            f"{config_name} contains NaN or Inf values."
        )

    number_of_windows = len(signal) // WINDOW_SIZE
    usable_samples = number_of_windows * WINDOW_SIZE

    if number_of_windows == 0:
        raise ValueError(f"{config_name} has no complete windows.")

    windows = signal[:usable_samples].reshape(
        number_of_windows, WINDOW_SIZE
    )

    discarded_samples = len(signal) - usable_samples

    return windows, len(signal), discarded_samples


def evaluate_configuration(
    config_name, scaler, model, device
):
    windows, raw_samples, discarded_samples = load_windows(
        DATASET_PATH, config_name
    )

    # Use the existing baseline scaler without fitting it again.
    scaled_windows = scaler.transform(windows).astype(np.float32)

    with torch.no_grad():
        inputs = torch.from_numpy(scaled_windows).to(device)
        reconstructed = model(inputs)

        reconstructed_windows = (
            reconstructed.cpu().numpy()
        )

    # Calculate one MSE per one-second window.
    window_mse = np.mean(
        (scaled_windows - reconstructed_windows) ** 2,
        axis=1,
    )

    above_threshold = int(np.sum(window_mse > THRESHOLD))

    if "_NM_" in config_name:
        condition_group = "NM-Z"
    else:
        condition_group = "SM-Z"

    summary = {
        "configuration": config_name,
        "condition_group": condition_group,
        "raw_samples": raw_samples,
        "number_of_windows": len(window_mse),
        "discarded_samples": discarded_samples,
        "mean_mse": float(np.mean(window_mse)),
        "median_mse": float(np.median(window_mse)),
        "p95_mse": float(np.percentile(window_mse, 95)),
        "windows_above_threshold": above_threshold,
        "exceedance_rate_percent": (
            100.0 * above_threshold / len(window_mse)
        ),
    }

    return summary, window_mse


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("ALL REMAINING UNDAMAGED CONFIGURATIONS")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Threshold: {THRESHOLD:.6f}")
    print("Loading existing scaler and trained model...")

    scaler = joblib.load(SCALER_PATH)
    model = load_model(device)

    summaries = []
    all_errors = {}

    for config_name in REMAINING_CONFIGS:
        print(f"\nEvaluating {config_name}...")

        summary, window_mse = evaluate_configuration(
            config_name, scaler, model, device
        )

        summaries.append(summary)
        all_errors[config_name] = window_mse

        print(f"  Condition: {summary['condition_group']}")
        print(f"  Windows: {summary['number_of_windows']}")
        print(f"  Mean MSE: {summary['mean_mse']:.6f}")
        print(f"  Median MSE: {summary['median_mse']:.6f}")
        print(f"  95th percentile MSE: {summary['p95_mse']:.6f}")
        print(
            "  Above threshold: "
            f"{summary['windows_above_threshold']}/"
            f"{summary['number_of_windows']} "
            f"({summary['exceedance_rate_percent']:.2f}%)"
        )

        np.save(
            OUTPUT_DIR / f"{config_name}_window_mse.npy",
            window_mse,
        )

    summary_df = pd.DataFrame(summaries)

    summary_path = OUTPUT_DIR / "uds_all_configurations_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    errors_df = pd.DataFrame([
        {
            "configuration": config_name,
            "window_index": window_index,
            "reconstruction_mse": float(error),
            "above_threshold": bool(error > THRESHOLD),
        }
        for config_name, errors in all_errors.items()
        for window_index, error in enumerate(errors)
    ])

    errors_path = OUTPUT_DIR / "uds_all_window_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    # Plot the exceedance rates, keeping condition groups visible.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))

    labels = summary_df["configuration"].str.replace(
        "MVS_", "", regex=False
    )

    bars = ax.bar(
        labels,
        summary_df["exceedance_rate_percent"],
    )

    ax.axhline(
        5.0,
        linestyle="--",
        label="5% reference (not a calibrated threshold target)",
    )

    ax.set_title(
        "Reconstruction-error threshold exceedance "
        "in remaining undamaged configurations"
    )
    ax.set_xlabel("Undamaged configuration")
    ax.set_ylabel("Windows above baseline threshold (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(
        bars, summary_df["exceedance_rate_percent"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    plot_path = PLOT_DIR / "uds_remaining_configuration_exceedance.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved window errors: {errors_path}")
    print(f"Saved plot: {plot_path}")
    print("\nUDS diagnostic evaluation complete.")
    print(
        "Note: SM-Z/NM-Z results are cross-condition diagnostics; "
        "they are not direct measures of damage-detection accuracy."
    )


if __name__ == "__main__":
    main()