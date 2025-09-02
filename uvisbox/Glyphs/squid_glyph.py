import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from sklearn.decomposition import PCA
import pyvista as pv

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
    angles_from_origin = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles =np.unwrap(angles_from_origin)
    polar_coords = np.column_stack((magnitudes, angles))
    # if np.any(np.isnan(polar_coords)):
    #     print("Invalid vectors detected:", vectors)
    #     raise ValueError("NaN values found in polar coordinates conversion.")
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


def compute_vector_depths(vectors):
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
            if (angle_j < angle_i < angle_k or angle_k < angle_i < angle_j) and \
               (min(mag_j, mag_k) < mag_i < max(mag_j, mag_k)):
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


def get_squid_glyph_info(vectors, vector_depths, percentile1=0.5, percentile2=0.9):
    """
    Get information about the squid glyph based on vector depths.

    Parameters
    ----------
        vectors : numpy.ndarray
            Array of shape (n, 2) representing 2D Cartesian vectors.
        vector_depths : numpy.ndarray
            Array of shape (n,) with the depth of each vector.
        percentile1 : float
            First percentile to consider for high depth vectors.
        percentile2 : float
            Second percentile to consider for low depth vectors.

    Returns
    -------
        theta_high : numpy.ndarray
            Array of shape (2, 1) with the angular range of high depth vectors.
        r_high : float
            Magnitude of the minimum high depth vector.
        min_angle_high : float
            Angle of the minimum high depth vector.
        max_angle_high : float
            Angle of the maximum high depth vector.
        theta_low : numpy.ndarray
            Array of shape (2, 1) with the angular range of low depth vectors.
        r_low : float
            Magnitude of the maximum low depth vector.
        min_angle_low : float
            Angle of the minimum low depth vector.
        max_angle_low : float
            Angle of the maximum low depth vector.
        median_mag : float
            Magnitude of the vector with maximum depth.
        median_angle : float
            Angle of the vector with maximum depth.

    """
    # Find index of vector with maximum depth
    max_depth_idx = np.argmax(vector_depths)
    # Find all indices where vector_depth > 1.0-percentile1
    high_depth_indices = np.where(vector_depths > 1.0-percentile1)[0]
    # Find all indices where vector_depth > 1.0-percentile2
    low_depth_indices = np.where(vector_depths > 1.0-percentile2)[0]
    mag_spread, min_mag_idx, max_mag_idx = magnitude_spread(vectors)
    median_mag = np.linalg.norm(vectors[max_depth_idx])
    median_angle = np.arctan2(vectors[max_depth_idx, 1], vectors[max_depth_idx, 0])

    if high_depth_indices.size > 0:
        high_depth_vectors = vectors[high_depth_indices]
        ang_spread_high, min_ang_idx_high, max_ang_idx_high = angular_spread(vectors[high_depth_indices])
        # print(f"High depth indices: {high_depth_indices}")
        theta_high = np.array([[np.arctan2(vectors[min_ang_idx_high, 1], vectors[min_ang_idx_high, 0])], [np.arctan2(vectors[max_ang_idx_high, 1], vectors[max_ang_idx_high, 0])]])
        r_high = np.linalg.norm(vectors[min_mag_idx])
        min_angle_high = np.arctan2(vectors[min_ang_idx_high, 1], vectors[min_ang_idx_high, 0])
        max_angle_high = np.arctan2(vectors[max_ang_idx_high, 1], vectors[max_ang_idx_high, 0])
        # print(f"Angular spread (deg) among high depth vectors: {ang_spread_high:.2f}")
    else:
        theta_high, r_high, min_angle_high, max_angle_high = (None, None, None, None)
        # print("No vectors found in high_depth_indices.")

    if low_depth_indices.size > 0:
        low_depth_vectors = vectors[low_depth_indices]
        ang_spread_low, min_ang_idx_low, max_ang_idx_low = angular_spread(vectors[low_depth_indices])
        # print(f"Low depth indices: {low_depth_indices}")
        theta_low = np.array([[np.arctan2(vectors[min_ang_idx_low, 1], vectors[min_ang_idx_low, 0])], [np.arctan2(vectors[max_ang_idx_low, 1], vectors[max_ang_idx_low, 0])]])
        r_low = np.linalg.norm(vectors[max_mag_idx])
        min_angle_low = np.arctan2(vectors[min_ang_idx_low, 1], vectors[min_ang_idx_low, 0])
        max_angle_low = np.arctan2(vectors[max_ang_idx_low, 1], vectors[max_ang_idx_low, 0])
        # print(f"Angular spread (deg) among low depth vectors: {ang_spread_low:.2f}")
    else:
        theta_low, r_low, min_angle_low, max_angle_low = (None, None, None, None)
        # print("No vectors found in low_depth_indices.")

    return theta_high, r_high, min_angle_high, max_angle_high, theta_low, r_low, min_angle_low, max_angle_low, median_mag, median_angle


def draw_wedges_with_arrow(ax, centers, theta1, theta2, mid_angle, r1, r2):
    """
    Draws multiple wedges with arrows.
    Parameters
    ----------
    centers : numpy.ndarray
        Array of shape (n, 2) representing the center positions of the wedges.
    theta1 : numpy.ndarray
        Array of shape (n, 2), each row [theta1_start, theta1_end] for the first wedge.
    theta2 : numpy.ndarray
        Array of shape (n, 2), each row [theta2_start, theta2_end] for the second wedge (arrowhead).
    mid_angle : iterable
        Iterable of length n, mid angle for arrow direction.
    r1 : float or iterable
        Radius for the first wedge.
    r2 : float or iterable of length n
        Radius for the second wedge.
    """
    n = centers.shape[0]
    # Support scalar or iterable for r1 and r2
    r1s = np.full(n, r1) if np.isscalar(r1) else np.asarray(r1)
    r2s = np.full(n, r2) if np.isscalar(r2) else np.asarray(r2)
    for i in range(n):
        center = centers[i]
        t1_start, t1_end = theta1[i]
        t2_start, t2_end = theta2[i]
        wedge = Wedge(center=center, r=r1s[i], theta1=t1_start, theta2=t1_end, facecolor='skyblue', edgecolor='skyblue', alpha=0.3)
        wedge2 = Wedge(center=center, r=r2s[i], theta1=t2_start, theta2=t2_end, facecolor='skyblue', edgecolor='skyblue', alpha=1.0)
        ax.annotate(
            '', 
            xy=(center[0] + r2s[i] * np.cos(np.deg2rad(mid_angle[i])), center[1] + r2s[i] * np.sin(np.deg2rad(mid_angle[i]))),
            xytext=center,
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=3, mutation_scale=20, alpha=0.8)
        )
        ax.add_patch(wedge)
        ax.add_patch(wedge2)


def uncertainty_squid_glyphs(positions, ensemble_vectors, percentil1, percentil2, scale=0.2, ax=None):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 2) representing the ensemble vectors for each position.
    percentil1 : float
        The first percentile for depth filtering.
    percentil2 : float
        The second percentile for depth filtering.
    scale : float
        The scale factor for the glyphs.
    ax : matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.


    Returns
    -------
    ax : matplotlib axis
        The axis with the drawn squid glyphs.
    """

    if ax is None:
        fig, ax = plt.subplots()
    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    theta1 = np.zeros((num_positions, 2))
    theta2 = np.zeros((num_positions, 2))
    mid_angle = np.zeros(num_positions)
    r1 = np.zeros(num_positions)
    r2 = np.zeros(num_positions)

    for i_pos in range(num_positions):
        # computer vector depths for ensemble_vectors[i_pos]
        depths = compute_vector_depths(ensemble_vectors[i_pos])
        # print(f"Depths for position {i_pos}: {depths}")
        theta_high, r_high, min_angle_high, max_angle_high, theta_low, r_low, min_angle_low, max_angle_low, median_mag, median_angle = get_squid_glyph_info(ensemble_vectors[i_pos], depths, percentil1, percentil2)
        theta1[i_pos] = np.degrees([min_angle_high, max_angle_high]) if min_angle_high is not None else [0, 0]
        theta2[i_pos] = np.degrees([min_angle_low, max_angle_low]) if min_angle_low is not None else [0, 0]
        mid_angle[i_pos] = np.degrees(median_angle)
        r1[i_pos] = r_high * scale if r_high is not None else 0
        r2[i_pos] = r_low * scale if r_low is not None else 0

    draw_wedges_with_arrow(ax, positions, theta1, theta2, mid_angle, r1, r2)

    return ax

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


def buildSuperElipticalSquidNP(directional_variations, positions, vectors, min_vectors, 
                                    median_vectors, max_vectors, glyph_markers, scale, resolution, num_of_glyphs):
    """
    Build superelliptical squid glyphs for 3D visualization. Assumes vectors are in spherical coordinates (magnitude, theta, phi).

    Parameters
    ----------
    directional_variations : numpy.ndarray
        Array of shape (n, 4, 2) where the 4 represents the
        (pca variance, pca first component, pca second component, pca mean) and the 2 represents
        the x and y components of the pca components.
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
    min_vectors : numpy.ndarray
        Array of shape (n, 3) The minimum vectors in spherical coordinates.
    median_vectors : numpy.ndarray
        Array of shape (n, 3) The median vectors in spherical coordinates.
    max_vectors : numpy.ndarray
        Array of shape (n, 3) The maximum vectors in spherical coordinates.
    glyph_markers : numpy.ndarray
        Array of shape (n,) with values 0 (no glyph), 1 (full glyph), 2 (arrow only)
    scale : float
        The scale factor for the glyphs.
    resolution : int
        The resolution of the base circle of the squid glyph.
    num_of_glyphs : int
        The number of glyphs to be created.

    Returns
    -------
    points : numpy.ndarray
        Array of shape (m, 3) The points of the squid glyphs.
    polygons : numpy.ndarray
        Array of shape (k,) The polygon connectivity for the glyphs.
    """
    
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*((resolution + 1)*3 + resolution*2 + 1), 3))
    polygons = np.zeros((num_of_glyphs*((resolution)*8 ),3), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    for i_p in range(num_points):
        if(glyph_markers[i_p] == 1):
            position = positions[i_p]
            v0_scale = directional_variations[i_p][0][0]
            v1_scale = directional_variations[i_p][0][1]
            # # elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
            if(np.absolute(v0_scale) < 1.e-20):
                elipse_scale = 1.0
                angle = 0.001 # min angle paramter
            else:

                elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
                v0 = vectors[i_p][1]
                v0 = directional_variations[i_p][1]
                angle = np.arctan2( v0[1], v0[0])
                # mean_vals = directional_variations[ii_t][i_p][3]
            min_vector = min_vectors[i_p]
            median_vector = median_vectors[i_p]
            max_vector = max_vectors[i_p]

            
            r0 = max_vector[0]*np.tan(max_vector[1]*0.5)
            r0 = np.maximum(r0, 0.01*max_vector[0])
            r1 = r0 * elipse_scale
            phi_vals = np.linspace(0, 2*np.pi, resolution)
            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
            phi = median_vector[2]
            theta = median_vector[1]
            
            Ry = np.zeros((3, 3))
            Ry[0][0] = np.cos(theta)
            Ry[0][2] = np.sin(theta)
            Ry[1][1] = 1
            Ry[2][0] = -np.sin(theta)
            Ry[2][2] = np.cos(theta)
            Rz = np.zeros((3, 3))
            Rz[0][0] = np.cos(phi)
            Rz[0][1] = -np.sin(phi)
            Rz[1][0] = np.sin(phi)
            Rz[1][1] = np.cos(phi)
            Rz[2][2] = 1

            ## mappoints to the position
            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], 0.0]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            pt = position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]

            center_id = points_id + resolution 
            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1
            old_points_id = points_id
            points_id = points_id + resolution + 1


            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], max_vector[0]-min_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            pt = np.dot(Ry, np.array([0, 0, max_vector[0]-min_vector[0]])) 
            pt = np.dot(Rz, pt)
            pt = pt*scale + position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]

            center_id = points_id + resolution

            for i in range(resolution):
                polygons[polygons_id, 0] = old_points_id + i
                polygons[polygons_id, 1] = points_id + i
                polygons[polygons_id, 2] = points_id + (i+1)%resolution
                polygons_id += 1
                polygons[polygons_id, 0] = points_id + (i+1)%resolution
                polygons[polygons_id, 1] = old_points_id + (i+1)%resolution
                polygons[polygons_id, 2] = old_points_id + i
                polygons_id += 1

            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1
            
            old_points_id = points_id
            points_id = points_id + resolution + 1



            # # # angle0 = np.arctan2(v0_scale, max_vector[0])
            # shaft_base_v0_scale = v0_scale/max_vector[0]*min_vector[0]
            # shaft_base_v1_scale = v1_scale/max_vector[0]*min_vector[0]
            shaft_base_r0 = min_vector[0]*np.tan(np.absolute(max_vector[1])*0.5)
            shaft_base_r0 = np.maximum(shaft_base_r0, 0.01*min_vector[0])
            shaft_base_r1 = shaft_base_r0 * elipse_scale

            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* shaft_base_r0
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* shaft_base_r1
            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], max_vector[0] - min_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]
            old_points_id = points_id
            points_id += resolution
            # shaft_top_v0_scale = v0_scale/max_vector[0]*(0.2*max_vector[0])
            # shaft_top_v1_scale = v1_scale/max_vector[0]*(0.2*max_vector[0])
            shaft_top_r0 = 0.2*max_vector[0]*np.tan(np.absolute(max_vector[1])*0.5)
            shaft_top_r0 = np.maximum(shaft_top_r0, 0.01*0.2*max_vector[0])
            shaft_top_r1 = shaft_top_r0 * elipse_scale

            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* shaft_top_r0
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* shaft_top_r1
            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], 0.8*max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            for i in range(resolution):
                polygons[polygons_id, 0] = old_points_id + i
                polygons[polygons_id, 1] = points_id + i
                polygons[polygons_id, 2] = points_id + (i+1)%resolution
                polygons_id += 1
                polygons[polygons_id, 0] = points_id + (i+1)%resolution
                polygons[polygons_id, 1] = old_points_id + (i+1)%resolution
                polygons[polygons_id, 2] = old_points_id + i
                polygons_id += 1

            old_points_id += points_id
            points_id += resolution

            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0 
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]

            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], 0.8*max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            pt = np.dot(Ry, np.array([0.0, 0.0, 0.8*max_vector[0]]))
            pt = np.dot(Rz, pt)
            pt = pt*scale + position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]
            center_id = points_id + resolution
            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1
            
            

            # cone tip
            # pt = np.dot(Ry, np.array([mean_vals[0], mean_vals[1], max_vector[0]]))
            pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
            pt = np.dot(Rz, pt)
            pt = pt*scale + position
            points[points_id + resolution+1, 0] = pt[0]
            points[points_id + resolution+1, 1] = pt[1]
            points[points_id + resolution+1, 2] = pt[2]
            tip_id = points_id + resolution + 1

            for i in range(resolution):
                polygons[polygons_id, 0] = points_id + i
                polygons[polygons_id, 1] = tip_id
                polygons[polygons_id, 2] = points_id + (i + 1)%resolution
                polygons_id += 1
            old_points_id = points_id
            points_id = points_id + resolution + 2

    return points, polygons



def uncertainty_squid_glyphs_3D_2(positions, ensemble_vectors, percentil, scale=0.5, ax=None):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 3D.
    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
        The ensemble vectors for each position in Cartesian coordinates.
    percentil : float
        The first percentile for depth filtering.
    scale : float
        The scale factor for the glyphs.
    ax : matplotlib 3D axis
        The axis to draw on. If None, a new figure and axis will be created.


    Returns
    -------
    ax : matplotlib 3D axis
        The axis with the drawn squid glyphs.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_spherical_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_spherical_vectors[i] = cartesian_to_spherical(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_3D(ensemble_spherical_vectors[i])

    min_vectors = np.zeros((num_positions, 3))
    max_vectors = np.zeros((num_positions, 3))
    median_vectors = np.zeros((num_positions, 3))
    glyph_markers = np.zeros((num_positions,), dtype=int)
    numb_of_glyphs = 0
    num_arrows = 0
    for i_pos in range(num_positions):
        # find indices where depth > 1.0-percentil
        median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx = calculate_spread_3D(ensemble_spherical_vectors[i_pos], depths[i_pos], percentil)
        median_vectors[i_pos] = ensemble_spherical_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0,0])
        min_mag_temp = ensemble_spherical_vectors[i_pos][min_mag_idx] if min_mag_idx is not None else np.array([0,0,0])
        max_mag_temp = ensemble_spherical_vectors[i_pos][max_mag_idx] if max_mag_idx is not None else np.array([0,0,0])
        min_phi = ensemble_spherical_vectors[i_pos][min_phi_idx][2] if min_phi_idx is not None else 0
        max_phi = ensemble_spherical_vectors[i_pos][max_phi_idx][2] if max_phi_idx is not None else 0
        min_theta = ensemble_spherical_vectors[i_pos][min_theta_idx][1] if min_theta_idx is not None else 0
        max_theta = ensemble_spherical_vectors[i_pos][max_theta_idx][1] if max_theta_idx is not None else 0
        min_vectors[i_pos] = np.array([min_mag_temp[0], min_theta, min_phi])
        max_vectors[i_pos] = np.array([max_mag_temp[0], max_theta, max_phi])
        if (np.absolute(max_vectors[i_pos][0]-min_vectors[i_pos][0]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][1]-min_vectors[i_pos][1]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][2]-min_vectors[i_pos][2]) > 1e-5):
            glyph_markers[i_pos] = 1
            numb_of_glyphs += 1
        elif(max_vectors[i_pos][0] > 1e-3):
            glyph_markers[i_pos] = 2
            num_arrows += 1


    # compute directional variations
    depth_threshold = 1.0 - percentil
    directional_variations = getDirectionalVariations(ensemble_vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors)
    
    # build squid glyphs
    points, polygons = buildSuperElipticalSquidNP(directional_variations, positions, ensemble_spherical_vectors, min_vectors, median_vectors, max_vectors, glyph_markers, scale, resolution=10, num_of_glyphs=numb_of_glyphs)
    
    # Draw the glyphs using trisurf
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=polygons, color='skyblue', alpha=0.75, edgecolor='grey')

    return ax, points, polygons



def uncertainty_squid_glyphs_3D(positions, ensemble_vectors, percentil, scale=0.5, 
                                show_edges=True, glyph_color='lightblue'):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 3D.
    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
        The ensemble vectors for each position in Cartesian coordinates.
    percentil : float
        The first percentile for depth filtering.
    scale : float, optional
        The scale factor for the glyphs. Default is 0.5.
    show_edges : bool, optional
        Whether to show edges of the glyphs. Default is True.
    glyph_color : str, optional
        The color of the glyphs. Default is 'lightblue'.

    Returns
    -------
    plotter : pyvista.Plotter
        The pyvista plotter with the drawn squid glyphs.
    points : numpy.ndarray
        The points of the squid glyphs.
    polygons : numpy.ndarray
        The polygon connectivity for the glyphs.
    """

    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_spherical_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_spherical_vectors[i] = cartesian_to_spherical(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_3D(ensemble_spherical_vectors[i])

    min_vectors = np.zeros((num_positions, 3))
    max_vectors = np.zeros((num_positions, 3))
    median_vectors = np.zeros((num_positions, 3))
    glyph_markers = np.zeros((num_positions,), dtype=int)
    numb_of_glyphs = 0
    num_arrows = 0
    for i_pos in range(num_positions):
        # find indices where depth > 1.0-percentil
        median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx = calculate_spread_3D(ensemble_spherical_vectors[i_pos], depths[i_pos], percentil)
        median_vectors[i_pos] = ensemble_spherical_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0,0])
        min_mag_temp = ensemble_spherical_vectors[i_pos][min_mag_idx] if min_mag_idx is not None else np.array([0,0,0])
        max_mag_temp = ensemble_spherical_vectors[i_pos][max_mag_idx] if max_mag_idx is not None else np.array([0,0,0])
        min_phi = ensemble_spherical_vectors[i_pos][min_phi_idx][2] if min_phi_idx is not None else 0
        max_phi = ensemble_spherical_vectors[i_pos][max_phi_idx][2] if max_phi_idx is not None else 0
        min_theta = ensemble_spherical_vectors[i_pos][min_theta_idx][1] if min_theta_idx is not None else 0
        max_theta = ensemble_spherical_vectors[i_pos][max_theta_idx][1] if max_theta_idx is not None else 0
        min_vectors[i_pos] = np.array([min_mag_temp[0], min_theta, min_phi])
        max_vectors[i_pos] = np.array([max_mag_temp[0], max_theta, max_phi])
        if (np.absolute(max_vectors[i_pos][0]-min_vectors[i_pos][0]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][1]-min_vectors[i_pos][1]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][2]-min_vectors[i_pos][2]) > 1e-5):
            glyph_markers[i_pos] = 1
            numb_of_glyphs += 1
        elif(max_vectors[i_pos][0] > 1e-3):
            glyph_markers[i_pos] = 2
            num_arrows += 1


    # compute directional variations
    depth_threshold = 1.0 - percentil
    directional_variations = getDirectionalVariations(ensemble_vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors)
    
    # build squid glyphs
    points, polygons = buildSuperElipticalSquidNP(directional_variations, positions, ensemble_spherical_vectors, min_vectors, median_vectors, max_vectors, glyph_markers, scale, resolution=10, num_of_glyphs=numb_of_glyphs)
    
    triangles = np.hstack([np.full((polygons.shape[0], 1), 3), polygons])
    triangles_flat = triangles.reshape(-1)
    mesh = pv.PolyData(points, triangles_flat) # 
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=glyph_color, show_edges=show_edges)
    plotter.add_axes()
    plotter.set_background('white')
    plotter.show_grid()
    plotter.show_axes()
    plotter.show(title="3D Uncertainty Squid Glyphs")

    return plotter, points, polygons
