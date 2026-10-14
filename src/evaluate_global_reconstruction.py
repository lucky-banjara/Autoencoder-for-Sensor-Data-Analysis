
from pathlib import Path

import numpy as np
import torch

from autoencoder import Autoencoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/processed_global")
MODEL_PATH = Path("models/autoencoder_global.pth")
OUTPUT_DIR = Path("results/global_reconstruction")

WINDOW_SIZE = 100
BOTTLENECK_DIM = 32


def main():
    print("=" * 65)
    print("GLOBAL-SCALING RECONSTRUCTION EVALUATION")
    print("=" * 65)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)

    if not np.isfinite(X_val).all():
        raise ValueError("Validation data contains NaN or Inf.")

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

    with torch.no_grad():
        inputs = torch.from_numpy(X_val).to(device)
        reconstructed = model(inputs).cpu().numpy()

    # One mean-squared reconstruction error per window.
    window_mse = np.mean(
        (X_val - reconstructed) ** 2,
        axis=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_val_original.npy", X_val)
    np.save(OUTPUT_DIR / "X_val_reconstructed.npy", reconstructed)
    np.save(OUTPUT_DIR / "validation_window_mse.npy", window_mse)

    mean_mse = float(np.mean(window_mse))
    median_mse = float(np.median(window_mse))
    p95_mse = float(np.percentile(window_mse, 95))
    max_mse = float(np.max(window_mse))
    mean_absolute_error = float(
        np.mean(np.abs(X_val - reconstructed))
    )

    print("\nEvaluation results:")
    print(f"  Validation windows: {len(X_val)}")
    print(f"  Mean window MSE: {mean_mse:.6f}")
    print(f"  Median window MSE: {median_mse:.6f}")
    print(f"  95th percentile MSE: {p95_mse:.6f}")
    print(f"  Maximum window MSE: {max_mse:.6f}")
    print(f"  Overall mean absolute error: {mean_absolute_error:.6f}")

    print(f"\nSaved results to: {OUTPUT_DIR}")
    print("GLOBAL RECONSTRUCTION EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
