
import os
import h5py
import numpy as np
import joblib


DATA_FILE = "data/data_100Hz.h5"
CONFIGURATION = "MVS_P2_UDS_SM_Y_01"
CHANNEL = "acceleration/AG01/y"

SAMPLE_RATE = 100
WINDOW_SIZE = 100
TRAIN_RATIO = 0.80

OUTPUT_DIR = "data/processed_global"


def main():
    print("=" * 60)
    print("HBTA GLOBAL-SCALING EXPERIMENT")
    print("=" * 60)

    dataset_path = f"{CONFIGURATION}/{CHANNEL}"

    with h5py.File(DATA_FILE, "r") as h5_file:
        if dataset_path not in h5_file:
            raise KeyError(f"Dataset not found: {dataset_path}")
        signal = np.asarray(h5_file[dataset_path][:], dtype=np.float64)

    if not np.isfinite(signal).all():
        raise ValueError("Raw signal contains NaN or Inf.")

    # Keep complete one-second windows only.
    number_of_windows = len(signal) // WINDOW_SIZE
    usable_samples = number_of_windows * WINDOW_SIZE
    signal = signal[:usable_samples]
    windows = signal.reshape(number_of_windows, WINDOW_SIZE)

    # Same chronological split as the baseline.
    split_index = int(len(windows) * TRAIN_RATIO)
    train_raw = windows[:split_index]
    val_raw = windows[split_index:]

    # Fit global mean/std using training samples only.
    train_samples = train_raw.reshape(-1)
    mean = float(np.mean(train_samples))
    std = float(np.std(train_samples))

    if std == 0:
        raise ValueError("Training signal standard deviation is zero.")

    # Use the same global transformation for every sample.
    X_train = ((train_raw - mean) / std).astype(np.float32)
    X_val = ((val_raw - mean) / std).astype(np.float32)

    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Scaled data contains NaN or Inf.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)

    # Save the exact parameters for inference on other recordings.
    joblib.dump(
        {"mean": mean, "std": std},
        os.path.join(OUTPUT_DIR, "global_scaler.pkl"),
    )

    print(f"Configuration: {CONFIGURATION}")
    print(f"Channel: {CHANNEL}")
    print(f"Raw samples used: {usable_samples}")
    print(f"Window count: {len(windows)}")
    print(f"Training windows: {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")
    print(f"Global training mean: {mean:.8f}")
    print(f"Global training std: {std:.8f}")
    print(f"Training scaled mean/std: {X_train.mean():.6f}/{X_train.std():.6f}")
    print(f"Validation scaled mean/std: {X_val.mean():.6f}/{X_val.std():.6f}")
    print(f"Saved experiment data to: {OUTPUT_DIR}")
    print("GLOBAL-SCALING PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()