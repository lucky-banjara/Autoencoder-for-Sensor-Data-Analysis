
import h5py
import numpy as np
import pandas as pd

H5_PATH = "data/data_100Hz.h5"
CHANNEL_PATH = "acceleration/AG01/y"

rows = []

with h5py.File(H5_PATH, "r") as h5_file:
    for config in h5_file.keys():
        if "UDS" not in config:
            continue

        dataset_path = f"{config}/{CHANNEL_PATH}"

        if dataset_path not in h5_file:
            print(f"Channel missing: {config}")
            continue

        signal = np.asarray(
            h5_file[dataset_path][:],
            dtype=np.float64
        ).reshape(-1)

        rows.append({
            "configuration": config,
            "samples": len(signal),
            "duration_seconds": len(signal) / 100,
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "minimum": float(np.min(signal)),
            "maximum": float(np.max(signal)),
            "finite": bool(np.isfinite(signal).all()),
        })

df = pd.DataFrame(rows)

if df.empty:
    print("No UDS configurations found for this channel.")
else:
    print("\nUNDAMAGED CONFIGURATION INVENTORY")
    print("=" * 90)
    print(df.to_string(index=False))

    output_path = "data/uds_configuration_inventory.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")