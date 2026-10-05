import h5py
import numpy as np


FILE_PATH = "data/data_100Hz.h5"


CONFIGURATIONS = [
    "MVS_P2_UDS_SM_Y_01",
    "MVS_P2_DS1_SM_Y_01",
    "MVS_P2_DS2_SM_Y_01",
    "MVS_P2_DS3_SM_Y_01",
    "MVS_P2_DS4_SM_Y_01",
    "MVS_P2_DS5_SM_Y_01",
    "MVS_P2_DS6_SM_Y_01",
    "MVS_P2_DS7_SM_Y_01",
    "MVS_P2_DS8_SM_Y_01",
]


CHANNEL = "acceleration/AG01/y"


print("=" * 70)
print("AUTOENCODER EXPERIMENT SELECTION CHECK")
print("=" * 70)

print(f"\nChannel: {CHANNEL}")
print(f"Configurations: {len(CONFIGURATIONS)}")


with h5py.File(FILE_PATH, "r") as file:

    for configuration in CONFIGURATIONS:

        dataset_path = f"{configuration}/{CHANNEL}"

        print("\n" + configuration)

        if dataset_path not in file:
            print("  ERROR: Channel not found")
            continue

        data = file[dataset_path][:]

        print(f"  Samples: {len(data)}")
        print(f"  Duration: {len(data) / 100:.2f} seconds")
        print(f"  Mean: {np.mean(data):.8f}")
        print(f"  Std: {np.std(data):.8f}")
        print(f"  Min: {np.min(data):.8f}")
        print(f"  Max: {np.max(data):.8f}")
        print(f"  NaN: {np.isnan(data).sum()}")
        print(f"  Inf: {np.isinf(data).sum()}")


print("\n" + "=" * 70)
print("SELECTION CHECK COMPLETE")
print("=" * 70)