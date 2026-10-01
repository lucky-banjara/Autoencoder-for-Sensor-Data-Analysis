import h5py
from collections import defaultdict

FILE_PATH = "data/data_100Hz.h5"


def main():
    with h5py.File(FILE_PATH, "r") as file:

        # ---------------------------------------------------------
        # Store sensor information
        # ---------------------------------------------------------
        sensors = defaultdict(lambda: defaultdict(set))

        for path in file:
            pass

        def collect_channels(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            parts = name.split("/")

            # Expected structure:
            # configuration / sensor_type / sensor / axis
            if len(parts) != 4:
                return

            configuration = parts[0]
            sensor_type = parts[1]
            sensor = parts[2]
            axis = parts[3]

            sensors[sensor_type][sensor].add(axis)

        file.visititems(collect_channels)

        # ---------------------------------------------------------
        # Acceleration channels
        # ---------------------------------------------------------
        print("=" * 70)
        print("HBTA SENSOR CHANNEL INVENTORY")
        print("=" * 70)

        print("\nACCELERATION SENSORS")
        print("-" * 70)

        acceleration = sensors["acceleration"]

        print(f"Number of acceleration sensor IDs: {len(acceleration)}")

        for sensor in sorted(acceleration):
            axes = sorted(acceleration[sensor])
            print(f"  {sensor}: {', '.join(axes)}")

        # ---------------------------------------------------------
        # Strain channels
        # ---------------------------------------------------------
        print("\nSTRAIN SENSORS")
        print("-" * 70)

        strain = sensors["strain"]

        print(f"Number of strain sensor IDs: {len(strain)}")

        for sensor in sorted(strain):
            axes = sorted(strain[sensor])
            print(f"  {sensor}: {', '.join(axes)}")

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------
        total_acceleration_channels = sum(
            len(axes) for axes in acceleration.values()
        )

        total_strain_channels = sum(
            len(axes) for axes in strain.values()
        )

        print("\nSUMMARY")
        print("-" * 70)
        print(f"Acceleration sensor IDs: {len(acceleration)}")
        print(f"Acceleration channels:   {total_acceleration_channels}")
        print(f"Strain sensor IDs:       {len(strain)}")
        print(f"Strain channels:         {total_strain_channels}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()