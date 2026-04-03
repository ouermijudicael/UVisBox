import numpy as np


def probabilistic_marching_squares_mesh(summary_statistics, x_coords=None, y_coords=None):
    """
    Process summary statistics and spatial coordinates for visualization.

    This function exists to maintain consistency with the stats->mesh->vis pipeline
    architecture used in other modules, even though no mesh transformation is needed
    for probabilistic marching squares.

    Parameters:
    -----------
        summary_statistics : dict
            Dictionary containing:
            - 'level_crossing_probability': np.ndarray
                2D array of shape (y_dim-1, x_dim-1) representing the probability
                of contour presence in each cell.
        x_coords : np.ndarray, optional
            1D array of x-axis coordinates defining the spatial domain.
            Length must match x_dim (the number of grid vertices, which is
            one more than the probability image width). If None, pixel indices are used.
        y_coords : np.ndarray, optional
            1D array of y-axis coordinates defining the spatial domain.
            Length must match y_dim (the number of grid vertices, which is
            one more than the probability image height). If None, pixel indices are used.

    Returns:
    --------
        dict
            Dictionary containing:
            - 'level_crossing_probability': np.ndarray
            - 'extent': tuple (x_min, x_max, y_min, y_max) or None
            - 'x_coords': np.ndarray or None
            - 'y_coords': np.ndarray or None
    """
    prob = summary_statistics['level_crossing_probability']
    height, width = prob.shape  # (y_dim-1, x_dim-1)

    if x_coords is not None:
        x_coords = np.asarray(x_coords)
        if x_coords.ndim != 1 or len(x_coords) != width + 1:
            raise ValueError(f"x_coords must be 1D with length {width + 1} (grid vertices), got shape {x_coords.shape}")
    if y_coords is not None:
        y_coords = np.asarray(y_coords)
        if y_coords.ndim != 1 or len(y_coords) != height + 1:
            raise ValueError(f"y_coords must be 1D with length {height + 1} (grid vertices), got shape {y_coords.shape}")

    if x_coords is not None and y_coords is not None:
        extent = (x_coords[0], x_coords[-1], y_coords[0], y_coords[-1])
    else:
        extent = None

    return {
        'level_crossing_probability': prob,
        'extent': extent,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }
