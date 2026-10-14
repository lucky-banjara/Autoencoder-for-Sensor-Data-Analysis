
import os
import random
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import Autoencoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_PATH = "data/processed_global/X_train.npy"
VAL_PATH = "data/processed_global/X_val.npy"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_global.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history_global.npz")

INPUT_DIM = 100
BOTTLENECK_DIM = 32

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    """Load the globally scaled training and validation arrays."""

    X_train = np.load(TRAIN_PATH).astype(np.float32)
    X_val = np.load(VAL_PATH).astype(np.float32)

    if X_train.ndim != 2 or X_train.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Unexpected training shape: {X_train.shape}"
        )

    if X_val.ndim != 2 or X_val.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Unexpected validation shape: {X_val.shape}"
        )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training or validation data is empty.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains NaN or Inf.")

    if not np.isfinite(X_val).all():
        raise ValueError("Validation data contains NaN or Inf.")

    print("\nData loaded:")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Validation shape: {X_val.shape}")
    print(f"  Training mean: {X_train.mean():.6f}")
    print(f"  Training std: {X_train.std():.6f}")
    print(f"  Validation mean: {X_val.mean():.6f}")
    print(f"  Validation std: {X_val.std():.6f}")

    return X_train, X_val


# --------------------------------------------------
# Train and validate
# --------------------------------------------------

def main():

    print("=" * 65)
    print("HBTA AUTOENCODER TRAINING - GLOBAL SCALING EXPERIMENT")
    print("=" * 65)

    set_seed(SEED)

    # Select available hardware.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nDevice: {device}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Early-stopping patience: {PATIENCE}")

    # Load globally scaled data.
    X_train, X_val = load_data()

    train_tensor = torch.from_numpy(X_train)
    val_tensor = torch.from_numpy(X_val)

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    # Match the original baseline configuration.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Create the same architecture as the baseline.
    model = Autoencoder(
        input_dim=INPUT_DIM,
        bottleneck_dim=BOTTLENECK_DIM,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nModel:")
    print(f"  Architecture: {INPUT_DIM} -> 64 -> "
          f"{BOTTLENECK_DIM} -> 64 -> {INPUT_DIM}")
    print(f"  Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Store training history.
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        # Training phase.
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            batch_size_actual = batch.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual

        epoch_train_loss = train_loss_sum / train_count

        # Validation phase.
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)

                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)

                batch_size_actual = batch.size(0)
                val_loss_sum += loss.item() * batch_size_actual
                val_count += batch_size_actual

        epoch_val_loss = val_loss_sum / val_count

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch:03d}/{MAX_EPOCHS}] "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}"
        )

        # Save the best model using validation loss.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch

            torch.save(model.state_dict(), MODEL_PATH)

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping.
        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch}."
            )
            break

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    np.savez(
        HISTORY_PATH,
        train_losses=np.asarray(train_losses),
        val_losses=np.asarray(val_losses),
        best_epoch=np.asarray(best_epoch),
        best_val_loss=np.asarray(best_val_loss),
    )

    print("\n" + "=" * 65)
    print("GLOBAL-SCALING TRAINING COMPLETE")
    print("=" * 65)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {MODEL_PATH}")
    print(f"Training history saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
