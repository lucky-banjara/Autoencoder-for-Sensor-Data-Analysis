
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

MODEL_PATH = "models/autoencoder.pth"
VALIDATION_PATH = "data/processed/X_val.npy"
OUTPUT_DIR = "results/reconstruction"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Select the available device
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 3. Load the validation windows
# --------------------------------------------------

X_val = np.load(VALIDATION_PATH).astype(np.float32)

print("Validation data shape:", X_val.shape)

if X_val.ndim != 2 or X_val.shape[1] != 100:
    raise ValueError("Expected validation data with shape (N, 100).")

if not np.isfinite(X_val).all():
    raise ValueError("Validation data contains NaN or Inf values.")


# --------------------------------------------------
# 4. Load the trained autoencoder
# --------------------------------------------------

model = Autoencoder(input_dim=100, bottleneck_dim=32)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Trained autoencoder loaded successfully.")


# --------------------------------------------------
# 5. Reconstruct validation windows
# --------------------------------------------------

with torch.no_grad():
    input_tensor = torch.from_numpy(X_val).to(device)
    reconstructed_tensor = model(input_tensor)
    X_reconstructed = reconstructed_tensor.cpu().numpy()

print("Reconstructed data shape:", X_reconstructed.shape)


# --------------------------------------------------
# 6. Calculate reconstruction errors
# --------------------------------------------------

# One MSE value for each 100-sample window
window_errors = np.mean(
    (X_val - X_reconstructed) ** 2,
    axis=1
)

overall_mse = float(np.mean(window_errors))
overall_mae = float(np.mean(np.abs(X_val - X_reconstructed)))

print("\nReconstruction results")
print("----------------------")
print(f"Number of windows: {len(window_errors)}")
print(f"Mean window MSE: {overall_mse:.6f}")
print(f"Mean absolute error: {overall_mae:.6f}")
print(f"Minimum window MSE: {window_errors.min():.6f}")
print(f"Maximum window MSE: {window_errors.max():.6f}")
print(f"Median window MSE: {np.median(window_errors):.6f}")


# --------------------------------------------------
# 7. Save the reconstruction data
# --------------------------------------------------

np.save(
    os.path.join(OUTPUT_DIR, "X_val_reconstructed.npy"),
    X_reconstructed
)

np.save(
    os.path.join(OUTPUT_DIR, "validation_window_mse.npy"),
    window_errors
)

np.save(
    os.path.join(OUTPUT_DIR, "X_val_original.npy"),
    X_val
)

error_df = pd.DataFrame({
    "window_index": np.arange(len(window_errors)),
    "start_time_seconds": np.arange(len(window_errors)),
    "reconstruction_mse": window_errors
})

error_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation_reconstruction_errors.csv"),
    index=False
)


# --------------------------------------------------
# 8. Plot one original and reconstructed window
# --------------------------------------------------

window_index = 0
sample_points = np.arange(100)

plt.figure(figsize=(12, 5))

plt.plot(
    sample_points,
    X_val[window_index],
    label="Original validation window"
)

plt.plot(
    sample_points,
    X_reconstructed[window_index],
    label="Reconstructed window"
)

plt.xlabel("Sample within 1-second window")
plt.ylabel("Scaled acceleration")
plt.title(
    f"Original vs Reconstructed Signal — Window {window_index}"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "original_vs_reconstructed.png"
)

plt.savefig(plot_path, dpi=300)
plt.show()

print("\nFiles saved in:", OUTPUT_DIR)
print(" - X_val_original.npy")
print(" - X_val_reconstructed.npy")
print(" - validation_window_mse.npy")
print(" - validation_reconstruction_errors.csv")
print(" - original_vs_reconstructed.png")