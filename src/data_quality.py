import h5py
import numpy as np

FILE_PATH = "data/data_100Hz.h5"


def main():

    total_datasets = 0
    datasets_with_nan = 0
    datasets_with_inf = 0

    total_nan = 0
    total_inf = 0

    with h5py.File(FILE_PATH, "r") as file:

        print("=" * 70)
        print("HBTA DATA QUALITY CHECK")
        print("=" * 70)

        def check_dataset(name, obj):

            nonlocal total_datasets
            nonlocal datasets_with_nan
            nonlocal datasets_with_inf
            nonlocal total_nan
            nonlocal total_inf

            if not isinstance(obj, h5py.Dataset):
                return

            total_datasets += 1

            data = obj[:]

            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()

            total_nan += nan_count
            total_inf += inf_count

            if nan_count > 0:
                datasets_with_nan += 1
                print(f"\nNaN found:")
                print(f"  {name}")
                print(f"  NaN count: {nan_count}")

            if inf_count > 0:
                datasets_with_inf += 1
                print(f"\nInf found:")
                print(f"  {name}")
                print(f"  Inf count: {inf_count}")

        file.visititems(check_dataset)

    print("\n" + "-" * 70)
    print("QUALITY CHECK SUMMARY")
    print("-" * 70)

    print(f"Total datasets checked:       {total_datasets}")
    print(f"Datasets containing NaN:      {datasets_with_nan}")
    print(f"Datasets containing Inf:      {datasets_with_inf}")
    print(f"Total NaN values:             {total_nan}")
    print(f"Total Inf values:             {total_inf}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()