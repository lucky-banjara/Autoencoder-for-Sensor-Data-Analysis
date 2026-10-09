
import os
import numpy as np
import matplotlib.pyplot as plt

HISTORY_FILE = "models/training_history.npz"
OUTPUT_DIR = "results/plots"

# Load saved training history
history = np.load(HISTORY_FILE)

train_loss = history["train_loss"]
val_loss = history["val_loss"]

epochs = np.arange(1, len(train_loss) + 1)

# Identify best validation epoch
best_epoch = int(np.argmin(val_loss)) + 1
best_val_loss = float(np.min(val_loss))

# Plot losses
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best validation epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("HBTA Autoencoder Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)

plot_path = os.path.join(
    OUTPUT_DIR,
    "training_history.png"
)

plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Epochs completed: {len(train_loss)}")
print(f"Best validation epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Plot saved to: {plot_path}")