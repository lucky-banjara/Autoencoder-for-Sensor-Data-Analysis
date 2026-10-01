import h5py
from collections import Counter


FILE_PATH = "data/data_100Hz.h5"


def main():
    with h5py.File(FILE_PATH, "r") as file:

        datasets = []

        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "path": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype)
                })

        file.visititems(collect_datasets)

        print("=" * 70)
        print("HBTA DATASET INVENTORY")
        print("=" * 70)

        print(f"\nTotal datasets: {len(datasets)}")

        # ---------------------------------------------------------
        # Top-level configurations
        # ---------------------------------------------------------

        configurations = set()

        for item in datasets:
            parts = item["path"].split("/")
            if len(parts) >= 1:
                configurations.add(parts[0])

        print("\nConfigurations:")
        for configuration in sorted(configurations):
            print(f"  - {configuration}")

        # ---------------------------------------------------------
        # Sensor types
        # ---------------------------------------------------------

        sensor_types = Counter()

        for item in datasets:
            parts = item["path"].split("/")

            if len(parts) >= 2:
                sensor_types[parts[1]] += 1

        print("\nSensor types:")
        for sensor_type, count in sorted(sensor_types.items()):
            print(f"  - {sensor_type}: {count} datasets")

        # ---------------------------------------------------------
        # Data types
        # ---------------------------------------------------------

        dtypes = Counter(item["dtype"] for item in datasets)

        print("\nData types:")
        for dtype, count in dtypes.items():
            print(f"  - {dtype}: {count}")

        # ---------------------------------------------------------
        # Shapes
        # ---------------------------------------------------------

        shapes = Counter(item["shape"] for item in datasets)

        print("\nDataset shapes:")
        for shape, count in shapes.items():
            print(f"  - {shape}: {count} datasets")

        # ---------------------------------------------------------
        # First 20 datasets
        # ---------------------------------------------------------

        print("\nFirst 20 datasets:")
        print("-" * 70)

        for item in datasets[:20]:
            print(f"{item['path']}")
            print(f"    Shape: {item['shape']}")
            print(f"    Type:  {item['dtype']}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()