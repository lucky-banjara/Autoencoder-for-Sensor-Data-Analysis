import h5py
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "data/data_100Hz.h5"

SAMPLE_RATE = 100
SECONDS_TO_PLOT = 10

CHANNELS = [
    "MVS_P1_DS1_SM_Y_01/acceleration/AG01/y",
    "MVS_P1_DS1_SM_Y_01/acceleration/AG01/z",
    "MVS_P1_DS1_SM_Y_01/acceleration/AL01/z",
    "MVS_P1_DS1_SM_Y_01/acceleration/AS/y",
    "MVS_P1_DS1_SM_Y_01/strain/SB01/x",
    "MVS_P1_DS1_SM_Y_01/strain/SC01/y",
]


def main():

    samples_to_plot = SAMPLE_RATE * SECONDS_TO_PLOT

    with h5py.File(FILE_PATH, "r") as file:

        for channel in CHANNELS:

            if channel not in file:
                print(f"Channel not found: {channel}")
                continue

            data = file[channel][:samples_to_plot]

            time = np.arange(len(data)) / SAMPLE_RATE

            plt.figure(figsize=(12, 5))

            plt.plot(time, data)

            plt.title(channel)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Sensor value")
            plt.grid(True)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()