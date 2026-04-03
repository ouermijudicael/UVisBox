import numpy as np


def contour_boxplot_mesh(summary_statistics, x_coords=None, y_coords=None):
    """
    Process summary statistics to create mesh data for visualization.

    This function aggregates percentile bands by overwriting values in descending percentile order,
    creating a single image that shows all bands with appropriate color values.

    Parameters:
    -----------
    summary_statistics : dict
        Dictionary from contour_boxplot_summary_statistics containing:
        - 'median': Binary image for median contour
        - 'percentile_bands': List of (percentile, band_image) tuples
        - 'outliers': List of outlier binary images
    x_coords : np.ndarray, optional
        1D array of x-axis coordinates defining the spatial domain.
        Length must match the image width. If None, pixel indices are used.
    y_coords : np.ndarray, optional
        1D array of y-axis coordinates defining the spatial domain.
        Length must match the image height. If None, pixel indices are used.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'percentile_bands_image': 2D array where each pixel value represents the percentile/100
                                   it belongs to. Lower percentiles overwrite higher ones.
        - 'median': Binary image (unchanged from input)
        - 'outliers': List of binary images (unchanged from input)
        - 'extent': Tuple (x_min, x_max, y_min, y_max) defining the spatial domain,
                    or None if no coordinates were provided.
        - 'x_coords': 1D array of x-axis coordinates, or None.
        - 'y_coords': 1D array of y-axis coordinates, or None.

    Notes:
    ------
    The aggregation works by:
    1. Starting with highest percentile band (e.g., 90th)
    2. Painting pixels with value 0.90 where band is non-zero
    3. Overwriting with lower percentile bands (e.g., 75th → 0.75)
    4. Continuing until lowest percentile
    This creates a layered visualization where inner bands overwrite outer bands.

    Examples:
    ---------
    >>> stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.5)
    >>> mesh_data = contour_boxplot_mesh(stats)
    >>> print(mesh_data['percentile_bands_image'].shape)
    (100, 100)
    """
    # Get image dimensions from median
    median = summary_statistics['median']
    height, width = median.shape

    # Validate coordinate arrays if provided
    if x_coords is not None:
        x_coords = np.asarray(x_coords)
        if x_coords.ndim != 1 or len(x_coords) != width:
            raise ValueError(f"x_coords must be 1D with length {width}, got shape {x_coords.shape}")
    if y_coords is not None:
        y_coords = np.asarray(y_coords)
        if y_coords.ndim != 1 or len(y_coords) != height:
            raise ValueError(f"y_coords must be 1D with length {height}, got shape {y_coords.shape}")

    # Compute extent from coordinates
    if x_coords is not None and y_coords is not None:
        extent = (x_coords[0], x_coords[-1], y_coords[0], y_coords[-1])
    else:
        extent = None

    # Initialize aggregated image with zeros
    percentile_bands_image = np.zeros((height, width), dtype=np.float32)
    
    # Get percentile bands and sort by percentile in descending order
    percentile_bands = summary_statistics['percentile_bands']
    sorted_bands = sorted(percentile_bands, key=lambda x: x[0], reverse=True)
    
    # Apply bands in descending order (highest to lowest)
    # Lower percentile values will overwrite higher ones
    for percentile, band_image in sorted_bands:
        # Where band is non-zero, overwrite with the band value
        mask = band_image > 0
        percentile_bands_image[mask] = band_image[mask]
    
    return {
        'percentile_bands_image': percentile_bands_image,
        'median': median,
        'outliers': summary_statistics['outliers'],
        'extent': extent,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }