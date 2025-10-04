
from uvisbox.Core.BandDepths.vector_depths import compute_vector_depths_2D
import numpy as np

def calculate_spread_2D(vectors, depths, percentil):
    """
    Calculate the spread in 2D polar coordinates.
    
    Parameters:
    -----------
        vectors : numpy.ndarray
            Array of shape (n, 2) in polar coordinates (magnitude, angle)
        depths : numpy.ndarray
            Array of shape (n,) representing the depth of each vector
        percentil : float
            The percentile for depth filtering
    
    Returns:
    --------
        tuple
            Indices of vectors with min/max magnitude, angle among those with depth > 1.0-percentil
    """

    first_quadrant = False # Set to True if you want to restrict to first quadrant 0 to pi/2
    second_quadrant = False # Set to True if you want to restrict to second quadrant pi/2 to pi
    third_quadrant = False # Set to True if you want to restrict to third quadrant -pi to -pi/2
    fourth_quadrant = False # Set to True if you want to restrict to fourth quadrant -pi/2 to 0
    indices = np.where(depths >= 1.0-percentil)[0]
    median_idx = np.argmax(depths)
    if indices.size > 0:
        filtered_vectors = vectors[indices]
        # Check if any vector is in the first quadrant
        first_quadrant = np.any((filtered_vectors[:, 1] >= 0) & (filtered_vectors[:, 1] <= np.pi/2)) 
        second_quadrant = np.any((filtered_vectors[:, 1] > np.pi/2) & (filtered_vectors[:, 1] <= np.pi))
        third_quadrant = np.any((filtered_vectors[:, 1] < -np.pi/2) & (filtered_vectors[:, 1] >= -np.pi))
        fourth_quadrant = np.any((filtered_vectors[:, 1] < 0) & (filtered_vectors[:, 1] >= -np.pi/2))
        if ((not first_quadrant and second_quadrant and third_quadrant and not fourth_quadrant ) or  
           (not first_quadrant and second_quadrant and third_quadrant and fourth_quadrant ) or  
              (first_quadrant and second_quadrant and third_quadrant and not fourth_quadrant )):
            # find smallest positive angle and smallest negative angle
            pos_angles = filtered_vectors[filtered_vectors[:, 1] >= 0]
            neg_angles = filtered_vectors[filtered_vectors[:, 1] < 0]
            min_angle = np.min(pos_angles[:, 1])
            max_angle = np.max(neg_angles[:, 1])
            min_mag = np.min(filtered_vectors[:, 0])
            max_mag = np.max(filtered_vectors[:, 0])
        else:           
            min_angle = np.min(filtered_vectors[:, 1])
            max_angle = np.max(filtered_vectors[:, 1])
            min_mag = np.min(filtered_vectors[:, 0])
            max_mag = np.max(filtered_vectors[:, 0])
    else:
        min_angle, max_angle = 0.0, 0.0
        min_mag, max_mag = 0.0, 0.0

    return median_idx, min_mag, max_mag, min_angle, max_angle
