import h5py

file_path = "data/data_100Hz.h5"

with h5py.File(file_path, "r") as file:

    print("HDF5 file opened successfully.")
    print("\nContents of the file:")

    def show_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name}")
            print(f"          Shape: {obj.shape}")
            print(f"          Data type: {obj.dtype}")

    file.visititems(show_structure)
    