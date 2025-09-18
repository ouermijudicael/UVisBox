import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from sklearn.decomposition import PCA

def cartesian_to_polar(vectors):
    """
    Convert 2D Cartesian vectors to polar coordinates (magnitude, angle).

    Parameters
    ----------
    vectors : numpy.ndarray
        Array of shape (n, 2) representing 2D Cartesian vectors.

    Returns
    -------
    polar_coords : numpy.ndarray
        Array of shape (n, 2) with columns [magnitude, angle in radians].
    """
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid_indices = magnitudes > 1.0e-8  # Filter vectors with norm greater than 1.0e-8
    angles = np.zeros_like(magnitudes)
    angles[valid_indices] = np.arctan2(vectors[valid_indices, 1], vectors[valid_indices, 0])
    angles = np.unwrap(angles)  # Unwrap to ensure continuity in angles
    angles = (angles + np.pi) % (2 * np.pi) - np.pi  # Ensure angles are in the range [-pi, pi]
    polar_coords = np.column_stack((magnitudes, angles))
    return polar_coords


def cartesian_to_spherical(vectors):
    """
    Convert 3D Cartesian vectors to spherical coordinates (magnitude, theta, phi).
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 3) representing 3D Cartesian vectors.
    
    Returns
    -------
        spherical_coords : numpy.ndarray
            Array of shape (n, 3) with columns [magnitude, theta, phi].
    """
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid_indices = magnitudes > 1.0e-8  # Filter vectors with norm greater than 1.0e-8
    theta = np.zeros_like(magnitudes)
    phi = np.zeros_like(magnitudes)
    theta[valid_indices] = np.arccos(np.clip(vectors[valid_indices, 2] / magnitudes[valid_indices], -1.0, 1.0))  # Clip to handle numerical precision issues
    phi[valid_indices] = np.arctan2(vectors[valid_indices, 1], vectors[valid_indices, 0])
    phi = np.unwrap(phi)  # Unwrap to ensure continuity in angles
    spherical_coords = np.column_stack((magnitudes, theta, phi))
    # if np.any(np.isnan(spherical_coords)):
    #     print("Invalid vectors detected:", vectors)
    #     print("Spherical coordinates:", spherical_coords)
    #     raise ValueError("NaN values found in spherical coordinates conversion.")
    return spherical_coords


def angular_spread(vectors):
    """
    Calculate the angular spread and magnitude spread of a set of vectors.

    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.

    Returns
    -------
        angular_spread_deg : float
            The angular spread in degrees.
        magnitude_spread : float
            The magnitude spread.
        min_idx : int
            Index of the vector with the minimum angle.
        max_idx : int
            Index of the vector with the maximum angle.
    """
    # Calculate the angular spread (max-min angle from origin) and return indices
    angles_from_origin = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles_unwrapped = np.unwrap(angles_from_origin)
    min_idx = np.argmin(angles_unwrapped)
    max_idx = np.argmax(angles_unwrapped)
    angular_spread = angles_unwrapped[max_idx] - angles_unwrapped[min_idx]
    angular_spread_deg = np.degrees(angular_spread)
    return angular_spread_deg, min_idx, max_idx


def magnitude_spread(vectors):
    """
    Calculate the magnitude spread of a set of vectors.
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.
    Returns
    -------
        magnitude_spread : float
            The magnitude spread.
        min_mag_idx : int
            Index of the vector with the minimum magnitude.
        max_mag_idx : int
            Index of the vector with the maximum magnitude.
    """
    # Calculate the magnitude spread (max-min magnitude) and return indices
    magnitudes = np.linalg.norm(vectors, axis=1)
    min_mag_idx = np.argmin(magnitudes)
    max_mag_idx = np.argmax(magnitudes)
    magnitude_spread = magnitudes[max_mag_idx] - magnitudes[min_mag_idx]
    return magnitude_spread, min_mag_idx, max_mag_idx


def compute_vector_depths_2D(vectors):
    """
    Compute the depth of each vector in the ensemble.
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.
    Returns
    -------
        depths : numpy.ndarray
            Array of shape (n,) with the depth of each vector.
    """
    n = vectors.shape[0]
    depths = np.zeros(n)
    # Calculate n choose 2 combinations of indices
    n = vectors.shape[0]
    pairs = list(combinations(np.arange(n), 2))

    # Compute depth for each vector
    for i in range(n):
        depth = 0
        for j, k in pairs:
            angle_i = np.arctan2(vectors[i, 1], vectors[i, 0])  
            angle_j = np.arctan2(vectors[j, 1], vectors[j, 0])
            angle_k = np.arctan2(vectors[k, 1], vectors[k, 0])
            mag_i = np.linalg.norm(vectors[i])
            mag_j = np.linalg.norm(vectors[j])
            mag_k = np.linalg.norm(vectors[k])
            # print(f"angle_i: {angle_i}, angle_j: {angle_j}, angle_k: {angle_k}")

            # Check if vector i is between vectors j and k and magnitude is also between
            if (angle_j <= angle_i <= angle_k or angle_k <= angle_i <= angle_j) and \
               (min(mag_j, mag_k) <= mag_i <= max(mag_j, mag_k)):
                depth += 1

        depths[i] = depth

    # scale depths between 0 and 1
    if depths.max() > 0.0:
        depths = (depths - depths.min()) / (depths.max() - depths.min())
    return depths


def compute_vector_depths_3D(vectors):
    """
    Compute the depth of each vector in the ensemble in 3D. Assumes vectors are in
       spherical coordinates (magnitude, theta, phi).
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 3) representing 3D spherical coordinates.
    Returns
    -------
        depths : numpy.ndarray
            Array of shape (n,) with the depth of each vector.
    """
    n = vectors.shape[0]
    depths = np.zeros(n)
    # Calculate n choose 2 combinations of indices
    n = vectors.shape[0]
    pairs = list(combinations(np.arange(n), 2))

    # Compute depth for each vector
    for i in range(n):
        depth = 0
        for j, k in pairs:
            mag_i, phi_i, theta_i = vectors[i][0], vectors[i][1], vectors[i][2]
            mag_j, phi_j, theta_j = vectors[j][0], vectors[j][1], vectors[j][2]
            mag_k, phi_k, theta_k = vectors[k][0], vectors[k][1], vectors[k][2]

            # Check if vector i is between vectors j and k in spherical coordinates
            if (theta_j < theta_i < theta_k or theta_k < theta_i < theta_j) and \
               (phi_j < phi_i < phi_k or phi_k < phi_i < phi_j) and \
               (min(mag_j, mag_k) < mag_i < max(mag_j, mag_k)):
                depth += 1
        depths[i] = depth

        
    # scale depths between 0 and 1
    if depths.max() > 0.0:
        depths = (depths - depths.min()) / (depths.max() - depths.min())
    

    return depths

