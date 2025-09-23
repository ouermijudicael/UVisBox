import numpy as np
import os
def load_data():
    """
    Load the ensemble UV data from text files.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    uv_data = np.zeros((68, 68, 15, 2))
    for i in range(1, 16):
        data = np.loadtxt(os.path.join(current_dir, 'txtmembersSeparate', 'uv_ens'+str(i)+'.txt'), delimiter=',',skiprows=1)
        u1 = data[:,0]
        v1 = data[:,1]
        uv_data[:, :, i-1, 0] = np.reshape(u1,[68,68])
        uv_data[:, :, i-1, 1] = np.reshape(v1,[68,68])

    return uv_data