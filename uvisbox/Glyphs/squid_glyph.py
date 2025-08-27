import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


def angular_spread(vectors):
    """Calculate the angular spread and magnitude spread of a set of vectors.
    Input
        vectors: numpy array of shape (n, 2)
    Output
        angular_spread_deg: float, the angular spread in degrees
        magnitude_spread: float, the magnitude spread
        min_idx: int, index of the vector with the minimum angle
        max_idx: int, index of the vector with the maximum angle
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
    """Calculate the magnitude spread of a set of vectors.
    Input
        vectors: numpy array of shape (n, 2)
    Output
        magnitude_spread: float, the magnitude spread
        min_mag_idx: int, index of the vector with the minimum magnitude
        max_mag_idx: int, index of the vector with the maximum magnitude
    """
    # Calculate the magnitude spread (max-min magnitude) and return indices
    magnitudes = np.linalg.norm(vectors, axis=1)
    min_mag_idx = np.argmin(magnitudes)
    max_mag_idx = np.argmax(magnitudes)
    magnitude_spread = magnitudes[max_mag_idx] - magnitudes[min_mag_idx]
    return magnitude_spread, min_mag_idx, max_mag_idx


def compute_vector_depths(vectors):
    """Compute the depth of each vector in the ensemble.
    Input
        vectors: numpy array of shape (n, 2)
    Output
        depths: numpy array of shape (n,), depth of each vector
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
    depths = (depths - depths.min()) / (depths.max() - depths.min())
    return depths


def get_squid_glyph_info(vectors, vector_depths, percentile1=0.5, percentile2=0.9):

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
    centers: numpy array of shape (n, 2)
    theta1: numpy array of shape (n, 2), each row [theta1_start, theta1_end] for the first wedge
    theta2: numpy array of shape (n, 2), each row [theta2_start, theta2_end] for the second wedge (arrowhead)
    mid_angle: iterable of length n, mid angle for arrow direction
    r1: float or iterable of length n, radius for the first wedge
    r2: float or iterable of length n, radius for the second wedge
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
    """Draws uncertainty squid glyphs for the given positions and ensemble vectors.
    Input
    -----
    positions: numpy array of shape (n, 2)
        The positions of the squid glyphs.
    ensemble_vectors: numpy array of shape (n, m, 2)
        The ensemble vectors for each position.
    percentil1: float
        The first percentile for depth filtering.
    percentil2: float
        The second percentile for depth filtering.
    scale: float
        The scale factor for the glyphs.
    ax: matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.


    Output
    ------
    ax: matplotlib axis
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
