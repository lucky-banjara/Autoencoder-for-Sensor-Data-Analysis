
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "results/reconstruction"
OUTPUT_DIR = "results/plots"

X_original = np.load(
    os.path.join(DATA_DIR, "X_val_original.npy")
)
X_reconstructed = np.load(
    os.path.join(DATA_DIR, "X_val_reconstructed.npy")
)
errors = np.load(
    os.path.join(DATA_DIR, "validation_window_mse.npy")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select one low-error window and two high-error windows.
low_error_index = int(np.argmin(errors))
high_error_indices = [
    int(np.argmax(errors)),
    88,
]

selected_indices = [low_error_index] + high_error_indices

print("Selected windows for inspection:")
for idx in selected_indices:
    print(f"Window {idx}: MSE = {errors[idx]:.6f}")

fig, axes = plt.subplots(
    len(selected_indices), 1, figsize=(12, 10)
)

for ax, idx in zip(axes, selected_indices):
    samples = np.arange(X_original.shape[1])

    ax.plot(
        samples,
        X_original[idx],
        label="Original"
    )
    ax.plot(
        samples,
        X_reconstructed[idx],
        label="Reconstructed"
    )

    ax.set_title(
        f"Window {idx} — Reconstruction MSE: {errors[idx]:.6f}"
    )
    ax.set_xlabel("Sample within 1-second window")
    ax.set_ylabel("Scaled acceleration")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR, "low_vs_high_error_windows.png"
)
plt.savefig(output_path, dpi=300)
plt.show()

print("\nPlot saved to:", output_path)