
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ERROR_FILE = "results/reconstruction/validation_window_mse.npy"
OUTPUT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load reconstruction error for each validation window
errors = np.load(ERROR_FILE)

if errors.ndim != 1 or len(errors) == 0:
    raise ValueError("Expected a non-empty 1D array of window errors.")

if not np.isfinite(errors).all():
    raise ValueError("Reconstruction errors contain NaN or Inf.")

window_indices = np.arange(len(errors))

# Calculate descriptive statistics
print("Undamaged validation reconstruction-error analysis")
print("--------------------------------------------------")
print(f"Number of windows: {len(errors)}")
print(f"Mean MSE:          {np.mean(errors):.6f}")
print(f"Median MSE:        {np.median(errors):.6f}")
print(f"Standard deviation:{np.std(errors):.6f}")
print(f"Minimum MSE:       {np.min(errors):.6f}")
print(f"75th percentile:   {np.percentile(errors, 75):.6f}")
print(f"90th percentile:   {np.percentile(errors, 90):.6f}")
print(f"95th percentile:   {np.percentile(errors, 95):.6f}")
print(f"Maximum MSE:       {np.max(errors):.6f}")

# Rank windows from highest to lowest error
ranking = pd.DataFrame({
    "window_index": window_indices,
    "start_time_in_validation_seconds": window_indices,
    "reconstruction_mse": errors
})

ranking = ranking.sort_values(
    "reconstruction_mse",
    ascending=False
)

ranking.to_csv(
    "results/reconstruction/ranked_validation_errors.csv",
    index=False
)

print("\n10 windows with the highest reconstruction errors:")
print(ranking.head(10).to_string(index=False))

# Plot 1: Distribution of reconstruction errors
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=25, edgecolor="black")
plt.xlabel("Reconstruction MSE")
plt.ylabel("Number of windows")
plt.title("Distribution of Undamaged Validation Reconstruction Errors")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "validation_error_distribution.png"),
    dpi=300
)
plt.show()

# Plot 2: Reconstruction error over validation-window order
plt.figure(figsize=(12, 5))
plt.plot(window_indices, errors, marker="o", markersize=3)

p95 = np.percentile(errors, 95)
plt.axhline(
    p95,
    linestyle="--",
    label=f"95th percentile = {p95:.4f}"
)

plt.xlabel("Validation window index")
plt.ylabel("Reconstruction MSE")
plt.title("Reconstruction Error Across Undamaged Validation Windows")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "validation_error_over_time.png"),
    dpi=300
)
plt.show()

print("\nSaved:")
print(" - results/reconstruction/ranked_validation_errors.csv")
print(" - results/plots/validation_error_distribution.png")
print(" - results/plots/validation_error_over_time.png")