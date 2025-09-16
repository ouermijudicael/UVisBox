import numpy as np
import os


def load_dataset():
    """
    Load sea surface temperature data from a CSV file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "sea_surface_temperature/sst.dat")
    data = np.genfromtxt(file_path, skip_header=1)
    data = data[:, 1:]  # Exclude the first column (time)

    return data
