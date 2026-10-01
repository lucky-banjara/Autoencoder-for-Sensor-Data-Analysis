import h5py
import numpy as np
import pandas as pd

FILE_PATH = "data/data_100Hz.h5"
OUTPUT_FILE = "data/hbta_channel_statistics.csv"


def main():

    statistics = []

    with h5py.File(FILE_PATH, "r") as file:

        def analyse_dataset(name, obj):

            if not isinstance(obj, h5py.Dataset):
                return

            parts = name.split("/")

            if len(parts) != 4:
                return

            configuration = parts[0]
            sensor_type = parts[1]
            sensor = parts[2]
            axis = parts[3]

            data = obj[:]

            statistics.append({
                "configuration": configuration,
                "sensor_type": sensor_type,
                "sensor": sensor,
                "axis": axis,
                "samples": len(data),
                "minimum": np.min(data),
                "maximum": np.max(data),
                "mean": np.mean(data),
                "std": np.std(data)
            })

        file.visititems(analyse_dataset)

    # Convert results into a DataFrame
    df = pd.DataFrame(statistics)

    # Save results
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 70)
    print("HBTA CHANNEL STATISTICS")
    print("=" * 70)

    print(f"\nTotal channels analysed: {len(df)}")

    print("\nOverall statistics:")
    print("-" * 70)

    print(f"Minimum value across dataset: {df['minimum'].min()}")
    print(f"Maximum value across dataset: {df['maximum'].max()}")
    print(f"Smallest channel std:          {df['std'].min()}")
    print(f"Largest channel std:           {df['std'].max()}")

    print("\nMean of channel means:")
    print(f"  {df['mean'].mean()}")

    print("\nAverage standard deviation:")
    print(f"  {df['std'].mean()}")

    print("\nStatistics by sensor type:")
    print("-" * 70)

    summary = df.groupby("sensor_type").agg(
        channels=("sensor", "count"),
        average_mean=("mean", "mean"),
        average_std=("std", "mean"),
        minimum=("minimum", "min"),
        maximum=("maximum", "max")
    )

    print(summary)

    print("\nTop 10 channels by standard deviation:")
    print("-" * 70)

    top_std = df.sort_values("std", ascending=False).head(10)

    print(
        top_std[
            [
                "configuration",
                "sensor_type",
                "sensor",
                "axis",
                "std"
            ]
        ].to_string(index=False)
    )

    print("\nBottom 10 channels by standard deviation:")
    print("-" * 70)

    bottom_std = df.sort_values("std").head(10)

    print(
        bottom_std[
            [
                "configuration",
                "sensor_type",
                "sensor",
                "axis",
                "std"
            ]
        ].to_string(index=False)
    )

    print("\nStatistics saved to:")
    print(f"  {OUTPUT_FILE}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()