
import os
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_FILE = "data/processed/X_train.npy"
VAL_FILE = "data/processed/X_val.npy"
OUTPUT_DIR = "models"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10
SEED = 42


# --------------------------------------------------
# Reproducibility and device
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# --------------------------------------------------
# Load processed data
# --------------------------------------------------

X_train = np.load(TRAIN_FILE).astype(np.float32)
X_val = np.load(VAL_FILE).astype(np.float32)

if not np.isfinite(X_train).all():
    raise ValueError("Training data contains NaN or Inf.")

if not np.isfinite(X_val).all():
    raise ValueError("Validation data contains NaN or Inf.")

if X_train.ndim != 2 or X_train.shape[1] != 100:
    raise ValueError(f"Unexpected training shape: {X_train.shape}")

if X_val.ndim != 2 or X_val.shape[1] != 100:
    raise ValueError(f"Unexpected validation shape: {X_val.shape}")

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")


# --------------------------------------------------
# Convert arrays to PyTorch tensors
# --------------------------------------------------

train_tensor = torch.from_numpy(X_train)
val_tensor = torch.from_numpy(X_val)

train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# --------------------------------------------------
# Initialise model and optimiser
# --------------------------------------------------

model = Autoencoder(input_dim=100, bottleneck_dim=32).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": []
}


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()
    running_train_loss = 0.0

    for (batch,) in train_loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch.size(0)

    train_loss = running_train_loss / len(train_dataset)

    # Evaluate on validation data without updating weights
    model.eval()

    with torch.no_grad():
        val_input = val_tensor.to(DEVICE)
        val_reconstructed = model(val_input)
        val_loss = criterion(val_reconstructed, val_input).item()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train loss: {train_loss:.6f} | "
        f"Validation loss: {val_loss:.6f}"
    )

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch + 1}.")
        break


# --------------------------------------------------
# Save model and training history
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

if best_state is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_state)

torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "autoencoder.pth")
)

np.savez(
    os.path.join(OUTPUT_DIR, "training_history.npz"),
    train_loss=np.array(history["train_loss"]),
    val_loss=np.array(history["val_loss"])
)

print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Saved model: {OUTPUT_DIR}/autoencoder.pth")
print(f"Saved history: {OUTPUT_DIR}/training_history.npz")