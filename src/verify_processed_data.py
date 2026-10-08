import os
import numpy as np
import joblib


# --------------------------------------------------
# File paths
# --------------------------------------------------

TRAIN_FILE = "data/processed/X_train.npy"
VAL_FILE = "data/processed/X_val.npy"
SCALER_FILE = "data/processed/scaler.pkl"


# --------------------------------------------------
# Check files exist
# --------------------------------------------------

print("=" * 60)
print("VERIFYING PROCESSED HBTA DATA")
print("=" * 60)

print("\nFile existence:")

for file_path in [TRAIN_FILE, VAL_FILE, SCALER_FILE]:
    print(f"  {file_path}: {os.path.exists(file_path)}")


# --------------------------------------------------
# Load processed arrays
# --------------------------------------------------

X_train = np.load(TRAIN_FILE)
X_val = np.load(VAL_FILE)

# Load scaler
scaler = joblib.load(SCALER_FILE)


# --------------------------------------------------
# Training data
# --------------------------------------------------

print("\n" + "-" * 60)
print("TRAINING DATA")
print("-" * 60)

print(f"Shape: {X_train.shape}")
print(f"Data type: {X_train.dtype}")
print(f"Mean: {np.mean(X_train):.8f}")
print(f"Std: {np.std(X_train):.8f}")
print(f"Min: {np.min(X_train):.8f}")
print(f"Max: {np.max(X_train):.8f}")
print(f"NaN: {np.isnan(X_train).sum()}")
print(f"Inf: {np.isinf(X_train).sum()}")


# --------------------------------------------------
# Validation data
# --------------------------------------------------

print("\n" + "-" * 60)
print("VALIDATION DATA")
print("-" * 60)

print(f"Shape: {X_val.shape}")
print(f"Data type: {X_val.dtype}")
print(f"Mean: {np.mean(X_val):.8f}")
print(f"Std: {np.std(X_val):.8f}")
print(f"Min: {np.min(X_val):.8f}")
print(f"Max: {np.max(X_val):.8f}")
print(f"NaN: {np.isnan(X_val).sum()}")
print(f"Inf: {np.isinf(X_val).sum()}")


# --------------------------------------------------
# Scaler
# --------------------------------------------------

print("\n" + "-" * 60)
print("SCALER")
print("-" * 60)

print(f"Scaler type: {type(scaler).__name__}")
print(f"Number of features: {scaler.n_features_in_}")

print("\nFirst 10 scaler means:")
print(scaler.mean_[:10])

print("\nFirst 10 scaler scales:")
print(scaler.scale_[:10])


# --------------------------------------------------
# Sample windows
# --------------------------------------------------

print("\n" + "-" * 60)
print("SAMPLE WINDOWS")
print("-" * 60)

print("\nFirst training window:")
print(X_train[0])

print("\nFirst validation window:")
print(X_val[0])


# --------------------------------------------------
# Final checks
# --------------------------------------------------

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)