import numpy as np
import os
import scipy.io as sio

def load_data():
    """
    Load the Darcy flow dataset from a .mat file.
    The .mat file should be located in the same directory as this script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'neuraluq_darcy_flow/output_Darcy_NN.mat')
    data = sio.loadmat(data_path)

    return data