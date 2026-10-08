import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


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
    dataset_path = f"{CONFIGURATION}/{CHANNEL}"

    with h5py.File(DATA_FILE, "r") as file:
        signal = np.array(file[dataset_path], dtype=np.float64)

    return signal


# --------------------------------------------------
# Create fixed-length windows
# --------------------------------------------------

def create_windows(signal, window_size):
    number_of_windows = len(signal) // window_size

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

    # Load signal
    signal = load_signal()

    print(f"\nConfiguration: {CONFIGURATION}")
    print(f"Channel: {CHANNEL}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")

    print(f"\nRaw signal:")
    print(f"  Samples: {len(signal)}")
    print(f"  Duration: {len(signal) / SAMPLE_RATE:.2f} seconds")
    print(f"  Mean: {np.mean(signal):.8f}")
    print(f"  Std: {np.std(signal):.8f}")
    print(f"  Min: {np.min(signal):.8f}")
    print(f"  Max: {np.max(signal):.8f}")
    print(f"  NaN: {np.isnan(signal).sum()}")
    print(f"  Inf: {np.isinf(signal).sum()}")

    # Create windows
    windows = create_windows(signal, WINDOW_SIZE)

    print(f"\nWindowing:")
    print(f"  Window size: {WINDOW_SIZE} samples")
    print(f"  Window duration: {WINDOW_SIZE / SAMPLE_RATE:.2f} seconds")
    print(f"  Number of windows: {len(windows)}")
    print(f"  Window shape: {windows.shape}")

    # Chronological train/validation split
    split_index = int(len(windows) * TRAIN_RATIO)

    X_train = windows[:split_index]
    X_val = windows[split_index:]

    print(f"\nTrain/validation split:")
    print(f"  Training windows: {len(X_train)}")
    print(f"  Validation windows: {len(X_val)}")

    # Fit scaler ONLY on training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler for validation data
    X_val_scaled = scaler.transform(X_val)

    print(f"\nScaling:")
    print("  Scaler: StandardScaler")
    print("  Fitted on: training data only")

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

    # Check for invalid values
    print("\nFinal quality check:")

    print(
        f"  Training NaN: "
        f"{np.isnan(X_train_scaled).sum()}"
    )

    print(
        f"  Training Inf: "
        f"{np.isinf(X_train_scaled).sum()}"
    )

    print(
        f"  Validation NaN: "
        f"{np.isnan(X_val_scaled).sum()}"
    )

    print(
        f"  Validation Inf: "
        f"{np.isinf(X_val_scaled).sum()}"
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save processed data
    np.save(
        os.path.join(OUTPUT_DIR, "X_train.npy"),
        X_train_scaled
    )

    np.save(
        os.path.join(OUTPUT_DIR, "X_val.npy"),
        X_val_scaled
    )

    # Save scaler
    joblib.dump(
        scaler,
        os.path.join(OUTPUT_DIR, "scaler.pkl")
    )

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR}/X_train.npy")
    print(f"  {OUTPUT_DIR}/X_val.npy")
    print(f"  {OUTPUT_DIR}/scaler.pkl")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()