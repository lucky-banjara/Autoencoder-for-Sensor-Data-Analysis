import h5py
import numpy as np

FILE_PATH = "data/data_100Hz.h5"

CONFIGURATION = "MVS_P1_DS1_SM_Y_01"

CHANNELS = [
    f"{CONFIGURATION}/acceleration/AG01/y",
    f"{CONFIGURATION}/acceleration/AG01/z",
    f"{CONFIGURATION}/acceleration/AL01/z",
    f"{CONFIGURATION}/acceleration/AS/y",
    f"{CONFIGURATION}/strain/SB01/x",
    f"{CONFIGURATION}/strain/SC01/y",
]


def print_attributes(name, obj):
    print(f"\n{name or 'ROOT'}")

    if len(obj.attrs) == 0:
        print("  No attributes")
        return

    for key, value in obj.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.ndarray):
            value = value.tolist()

        print(f"  {key}: {value}")


def main():
    with h5py.File(FILE_PATH, "r") as file:

        print("=" * 70)
        print("HBTA METADATA INSPECTION")
        print("=" * 70)

        print_attributes("ROOT", file)

        if CONFIGURATION in file:
            group = file[CONFIGURATION]
            print_attributes(CONFIGURATION, group)

        print("\nSELECTED CHANNELS")
        print("-" * 70)

        for path in CHANNELS:
            if path not in file:
                print(f"\nNot found: {path}")
                continue

            dataset = file[path]

            print(f"\n{path}")
            print(f"  Shape: {dataset.shape}")
            print(f"  Data type: {dataset.dtype}")
            print_attributes(path, dataset)

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()