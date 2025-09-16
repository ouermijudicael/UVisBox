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



def calculate_spread_3D(vectors, depths, percentil):
    """
    Calculate the spread in 3D spherical coordinates.
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 3) in spherical coordinates (magnitude, theta, phi)
        depths : numpy.ndarray
            Array of shape (n,) representing the depth of each vector
        percentil : float
            The percentile for depth filtering
    Returns
    -------
        tuple
            Indices of vectors with min/max magnitude, theta, phi among those with depth > 1.0-percentil
    """

    indices = np.where(depths > 1.0-percentil)[0]
    median_idx = np.argmax(depths)
    if indices.size > 0:
        filtered_vectors = vectors[indices]
        min_phi_idx = np.argmin(filtered_vectors[:, 2])
        max_phi_idx = np.argmax(filtered_vectors[:, 2])
        min_theta_idx = np.argmin(filtered_vectors[:, 1])
        max_theta_idx = np.argmax(filtered_vectors[:, 1])
        min_mag_idx = np.argmin(filtered_vectors[:, 0])
        max_mag_idx = np.argmax(filtered_vectors[:, 0])
    else:
        min_phi_idx, max_phi_idx = None, None
        min_theta_idx, max_theta_idx = None, None
        min_mag_idx, max_mag_idx = None, None

    return median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx


def calculate_spread_2D(vectors, depths, percentil):
    """
    Calculate the spread in 2D polar coordinates.
    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) in polar coordinates (magnitude, angle)
        depths : numpy.ndarray
            Array of shape (n,) representing the depth of each vector
        percentil : float
            The percentile for depth filtering
    Returns
    -------
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


def getDirectionalVariations(vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors):
    """
    Compute the directional variation of the vectors.
    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the x, y, and z coordinates of the positions.
    vectors : numpy.ndarray
        Array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components.
    median_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the median vector components.
    max_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the max vector components.
    min_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the min vector components.
    domain : numpy.ndarray
        Array of shape (3, 2) where the first dimension contains the x, y, and z domain limits.
    Returns
    -------
    directional_variation : numpy.ndarray
        Array of shape (num_points, 4, 2) where the 4 represents the
        (pca variance, pca first component, pca second component, pca mean) and the 2 represents
        the x and y components of the pca components.
    """

    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    local_median_vector = np.zeros(3)
    local_XX = np.zeros([num_ensemble_members, 2])
    directional_variation = np.zeros([num_points, 3, 2])

    for i_p in range(num_points):
        r = median_vectors[i_p][0]
        theta = median_vectors[i_p][1]
        phi = median_vectors[i_p][2]
        local_median_vector[0] = r*np.sin(theta)*np.cos(phi)
        local_median_vector[1] = r*np.sin(theta)*np.sin(phi)
        local_median_vector[2] = r*np.cos(theta)
        phi = -phi
        theta = -theta

        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        ii_m = 0
        for i_m in range(num_ensemble_members):
            if(depths[i_p][i_m] >= depth_threshold):
                tmp = vectors[i_p][i_m]
                vec = np.zeros(3)
                vec[0] = tmp[0]*np.sin(tmp[1])*np.cos(tmp[2])
                vec[1] = tmp[0]*np.sin(tmp[1])*np.sin(tmp[2])
                vec[2] = tmp[0]*np.cos(tmp[1])
                vec = np.dot(Ry, np.dot(Rz, vec))
                vec =  vec * (max_vectors[i_p][0]/(np.absolute(vec[2]) + 1.0e-10))
                local_XX[ii_m][0] = vec[0]
                local_XX[ii_m][1] = vec[1]
                ii_m = ii_m + 1
        local_X = local_XX[0:ii_m]
        n_local_points, n_local_features = local_X.shape
        pca = PCA(n_components=2)

        if n_local_points > 2 and n_local_features > 2: # ensures that we have enough points and features for PCA
            pca.fit(local_X)
            pca_components = pca.components_
            pca_mean = pca.mean_
            pca_variance = pca.explained_variance_
            v0_scale = pca_variance[0]
            v1_scale = pca_variance[1]
        else:
            pca_components = np.zeros((2, 2))
            v0_scale = 1.0
            v1_scale = 1.0

        if(np.absolute(v0_scale) < 1.0e-20): # ensure non-zero scale
            v0_scale = 1.0
            v1_scale = 1.0

        directional_variation[i_p][0][0] = v0_scale
        directional_variation[i_p][0][1] = np.maximum(v1_scale, 0.1*v0_scale)
        directional_variation[i_p][1][0] = pca_components[0][0]
        directional_variation[i_p][1][1] = pca_components[0][1]
        directional_variation[i_p][2][0] = pca_components[1][0]
        directional_variation[i_p][2][1] = pca_components[1][1]


    return directional_variation

