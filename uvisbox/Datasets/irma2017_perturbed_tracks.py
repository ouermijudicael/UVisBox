import numpy as np
import os
def load_dataset():
    """
    Load perturbed IRMA 2017 hurricane track data from a npy file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "irma2017_perturbed_tracks/irma2017_perturbed_tracks.npy")
    data = np.load(file_path)
    return data