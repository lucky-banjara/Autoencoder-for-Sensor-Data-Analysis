
import os
import h5py
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_FILE = "data/data_100Hz.h5"

CONFIGURATION = "MVS_P2_UDS_SM_Y_01"
CHANNEL = "acceleration/AG01/y"

SAMPLE_RATE = 100
WINDOW_SIZE = 100          # 100 samples = 1 second
TRAIN_RATIO = 0.80

OUTPUT_DIR = "data/processed"


# --------------------------------------------------
# Load signal
# --------------------------------------------------

def load_signal():
    """Load the selected acceleration signal from the HDF5 file."""

    dataset_path = f"{CONFIGURATION}/{CHANNEL}"

    with h5py.File(DATA_FILE, "r") as h5_file:
        if dataset_path not in h5_file:
            raise KeyError(
                f"Dataset not found in HDF5 file: {dataset_path}"
            )

        signal = np.asarray(
            h5_file[dataset_path][:],
            dtype=np.float64
        )

    return signal


# --------------------------------------------------
# Create fixed-length windows
# --------------------------------------------------

def create_windows(signal, window_size):
    """Create non-overlapping windows and discard incomplete remainder."""

    if window_size <= 0:
        raise ValueError("Window size must be greater than zero.")

    number_of_windows = len(signal) // window_size

    if number_of_windows == 0:
        raise ValueError(
            "Signal is too short to create a complete window."
        )

    usable_samples = number_of_windows * window_size
    signal = signal[:usable_samples]

    windows = signal.reshape(number_of_windows, window_size)

    return windows


# --------------------------------------------------
# Main preprocessing
# --------------------------------------------------

def main():

    print("=" * 60)
    print("HBTA SIGNAL PREPROCESSING")
    print("=" * 60)

    # 1. Load the raw signal
    signal = load_signal()

    if not np.isfinite(signal).all():
        raise ValueError(
            "Raw signal contains NaN or Inf values. "
            "Inspect the data before preprocessing."
        )

    print(f"\nConfiguration: {CONFIGURATION}")
    print(f"Channel: {CHANNEL}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")

    print("\nRaw signal:")
    print(f"  Samples: {len(signal)}")
    print(f"  Duration: {len(signal) / SAMPLE_RATE:.2f} seconds")
    print(f"  Mean: {np.mean(signal):.8f}")
    print(f"  Std: {np.std(signal):.8f}")
    print(f"  Min: {np.min(signal):.8f}")
    print(f"  Max: {np.max(signal):.8f}")
    print(f"  NaN: {np.isnan(signal).sum()}")
    print(f"  Inf: {np.isinf(signal).sum()}")

    # 2. Create non-overlapping windows
    windows = create_windows(signal, WINDOW_SIZE)

    print("\nWindowing:")
    print(f"  Window size: {WINDOW_SIZE} samples")
    print(
        f"  Window duration: "
        f"{WINDOW_SIZE / SAMPLE_RATE:.2f} seconds"
    )
    print(f"  Number of windows: {len(windows)}")
    print(f"  Window shape: {windows.shape}")

    # 3. Split chronologically into training and validation sets
    split_index = int(len(windows) * TRAIN_RATIO)

    X_train = windows[:split_index]
    X_val = windows[split_index:]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Training or validation set is empty. "
            "Check the train/validation ratio."
        )

    print("\nTrain/validation split:")
    print(f"  Training windows: {len(X_train)}")
    print(f"  Validation windows: {len(X_val)}")

    # 4. Fit the scaler on training data only.
    # StandardScaler scales each of the 100 sample positions
    # across the training windows.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\nScaling:")
    print("  Scaler: StandardScaler")
    print("  Fitted on: training windows only")
    print("  Features: sample positions within each window")

    print("\nScaled training data:")
    print(f"  Mean: {np.mean(X_train_scaled):.8f}")
    print(f"  Std: {np.std(X_train_scaled):.8f}")
    print(f"  Min: {np.min(X_train_scaled):.8f}")
    print(f"  Max: {np.max(X_train_scaled):.8f}")

    print("\nScaled validation data:")
    print(f"  Mean: {np.mean(X_val_scaled):.8f}")
    print(f"  Std: {np.std(X_val_scaled):.8f}")
    print(f"  Min: {np.min(X_val_scaled):.8f}")
    print(f"  Max: {np.max(X_val_scaled):.8f}")

    # 5. Verify the processed arrays
    print("\nFinal quality check:")

    print(f"  Training NaN: {np.isnan(X_train_scaled).sum()}")
    print(f"  Training Inf: {np.isinf(X_train_scaled).sum()}")
    print(f"  Validation NaN: {np.isnan(X_val_scaled).sum()}")
    print(f"  Validation Inf: {np.isinf(X_val_scaled).sum()}")

    if not np.isfinite(X_train_scaled).all():
        raise ValueError("Scaled training data contains invalid values.")

    if not np.isfinite(X_val_scaled).all():
        raise ValueError("Scaled validation data contains invalid values.")

    # 6. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 7. Save processed arrays
    train_path = os.path.join(OUTPUT_DIR, "X_train.npy")
    val_path = os.path.join(OUTPUT_DIR, "X_val.npy")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")

    np.save(train_path, X_train_scaled)
    np.save(val_path, X_val_scaled)

    # Save the fitted scaler using Joblib
    joblib.dump(scaler, scaler_path)

    print("\nSaved files:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {scaler_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()