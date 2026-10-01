import h5py
import numpy as np

FILE_PATH = "data/data_100Hz.h5"


def inspect_dataset(file, path):
    data = file[path][:]

    print(f"\nDataset: {path}")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")

    print(f"  First 10 values:")
    print(f"    {data[:10]}")

    print(f"  Minimum: {np.min(data)}")
    print(f"  Maximum: {np.max(data)}")
    print(f"  Mean:    {np.mean(data)}")
    print(f"  Std:     {np.std(data)}")

    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()

    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")


def main():
    channels = [
        "MVS_P1_DS1_SM_Y_01/acceleration/AG01/y",
        "MVS_P1_DS1_SM_Y_01/acceleration/AG01/z",
        "MVS_P1_DS1_SM_Y_01/acceleration/AL01/z",
        "MVS_P1_DS1_SM_Y_01/acceleration/AS/z",  # Double check this name
        "MVS_P1_DS1_SM_Y_01/strain/SB01/x",
        "MVS_P1_DS1_SM_Y_01/strain/SC01/y",
    ]
    
    with h5py.File(FILE_PATH, "r") as file:
        print("=" * 70)
        print("HBTA SENSOR VALUE INSPECTION")
        print("=" * 70)
        
        for channel in channels:
            # Check if the path exists before attempting to read it
            if channel in file:
                inspect_dataset(file, channel)
            else:
                print(f"\n⚠️ Warning: Channel '{channel}' not found in file.")
                
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()