import h5py
import csv

FILE_PATH = "data/data_100Hz.h5"
OUTPUT_PATH = "data/hbta_configuration_metadata.csv"


def decode_value(value):
    """Convert HDF5 attribute values into readable Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        return str(value)

    return value


with h5py.File(FILE_PATH, "r") as file:

    configurations = []

    for name in file.keys():

        obj = file[name]

        if not isinstance(obj, h5py.Group):
            continue

        attributes = {
            key: decode_value(value)
            for key, value in obj.attrs.items()
        }

        configurations.append({
            "configuration": name,
            "damage_state": attributes.get("damage_state"),
            "date": attributes.get("date"),
            "duration": attributes.get("duration"),
            "mvs_direction": attributes.get("mvs_direction"),
            "mvs_mode": attributes.get("mvs_mode"),
            "mvs_position": attributes.get("mvs_position"),
            "mvs_position_coordinates": attributes.get(
                "mvs_position_coordinates"
            ),
            "sample_rate": attributes.get("sample_rate"),
            "temperature": attributes.get("temperature"),
            "train_passing": attributes.get("train_passing"),
            "wind_range": attributes.get("wind_range"),
        })


fieldnames = [
    "configuration",
    "damage_state",
    "date",
    "duration",
    "mvs_direction",
    "mvs_mode",
    "mvs_position",
    "mvs_position_coordinates",
    "sample_rate",
    "temperature",
    "train_passing",
    "wind_range",
]


with open(OUTPUT_PATH, "w", newline="") as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for configuration in configurations:
        writer.writerow(configuration)


print("=" * 70)
print("HBTA CONFIGURATION METADATA")
print("=" * 70)

print(f"\nTotal configurations: {len(configurations)}")

for configuration in configurations:

    print("\n" + configuration["configuration"])

    print(
        f"  damage_state: "
        f"{configuration['damage_state']}"
    )

    print(
        f"  mvs_direction: "
        f"{configuration['mvs_direction']}"
    )

    print(
        f"  mvs_mode: "
        f"{configuration['mvs_mode']}"
    )

    print(
        f"  mvs_position: "
        f"{configuration['mvs_position']}"
    )

    print(
        f"  sample_rate: "
        f"{configuration['sample_rate']}"
    )

    print(
        f"  train_passing: "
        f"{configuration['train_passing']}"
    )

    print(
        f"  wind_range: "
        f"{configuration['wind_range']}"
    )

print("\n" + "=" * 70)
print(f"Saved metadata to: {OUTPUT_PATH}")
print("=" * 70)