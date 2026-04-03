import numpy as np
from uvisbox.Core.BandDepths.contour_banddepth import contour_banddepth
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def contour_boxplot_summary_statistics(ensemble_images, isovalue, boxplot_style=None, eps=0, workers=11):
    """
    Compute summary statistics for contour boxplot from ensemble scalar fields.
    
    This function extracts binary contours at the given isovalue, computes their band depths,
    and generates percentile bands, median, and outlier information.
    
    Parameters:
    -----------
    ensemble_images : np.ndarray
        3D array of shape (n_ensemble, height, width) containing ensemble scalar fields.
    isovalue : float
        Threshold value for creating binary images. Pixels with values < isovalue are set to 1 (inside),
        values >= isovalue are set to 0 (outside).
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for percentiles and outlier detection. If None, uses default configuration.
    eps : float, optional
        Epsilon tolerance for subset checks in band depth computation. With eps=0 (default),
        exact pixel-level containment is required, which can be too strict for continuously-varying
        rasterized contours. Values like 0.01-0.05 allow approximate containment.
    workers : int, optional
        Number of parallel workers for band depth computation. Default is 11.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'median': Binary image (np.ndarray) representing the median contour sublevel set.
                   1 = inside contour (value < isovalue), 0 = outside contour (value >= isovalue).
        - 'percentile_bands': List of tuples (percentile, band_image) where band_image is a 2D array
                             with values in [0, percentile/100]. percentile/100 means inside the band,
                             0 means outside the band.
        - 'outliers': List of binary images representing outlier contours (same format as median).
        - 'sorted_contours': 3D array (n_ensemble, height, width) of binary images sorted by depth.
        - 'sorted_indices': Array of indices showing the original order of sorted contours.
        - 'depths': Array of computed band depth values for each contour.
    
    Examples:
    ---------
    >>> ensemble = np.random.randn(50, 100, 100)
    >>> stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.5)
    >>> print(stats.keys())
    dict_keys(['median', 'percentile_bands', 'outliers', 'sorted_contours', 'sorted_indices', 'depths'])
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()
    
    # Validate input
    if not isinstance(ensemble_images, np.ndarray):
        ensemble_images = np.array(ensemble_images)
    
    if ensemble_images.ndim != 3:
        raise ValueError(f"ensemble_images must be 3D array of shape (n_ensemble, height, width). Got shape {ensemble_images.shape}")
    
    n_ensemble, height, width = ensemble_images.shape
    
    # Create binary images: 1 if value < isovalue (inside), 0 otherwise (outside)
    binary_images = (ensemble_images < isovalue).astype(np.bool_)
    
    # Compute band depths
    depths = contour_banddepth(binary_images, eps=eps, workers=workers)
    
    # Sort by depth in descending order (highest depth first = median)
    sorted_indices = np.argsort(depths)[::-1]
    sorted_contours = binary_images[sorted_indices]
    sorted_depths = depths[sorted_indices]
    
    # Get median (highest depth contour)
    median = sorted_contours[0].astype(np.uint8)
    
    # Compute percentile bands
    percentile_bands = []
    for percentile in sorted(boxplot_style.percentiles, reverse=True):
        # Calculate how many contours to include
        n_include = int(np.ceil(n_ensemble * (percentile / 100.0)))
        n_include = max(1, min(n_include, n_ensemble))
        
        # Get selected contours
        selected_contours = sorted_contours[:n_include]
        
        # Compute band envelope: union minus intersection
        union = np.any(selected_contours, axis=0)
        intersection = np.all(selected_contours, axis=0)
        band_envelope = (union & ~intersection)
        
        # Create band image: percentile/100 inside band, 0 outside
        band_image = np.zeros((height, width), dtype=np.float32)
        band_image[band_envelope] = percentile / 100.0
        
        percentile_bands.append((percentile, band_image))
    
    # Identify outliers (beyond the largest percentile)
    outliers = []
    if boxplot_style.show_outliers and len(boxplot_style.percentiles) > 0:
        max_percentile = max(boxplot_style.percentiles)
        outlier_start_idx = int(np.ceil(n_ensemble * (max_percentile / 100.0)))
        
        for idx in range(outlier_start_idx, n_ensemble):
            outliers.append(sorted_contours[idx].astype(np.uint8))
    
    return {
        'median': median,
        'percentile_bands': percentile_bands,
        'outliers': outliers,
        'sorted_contours': sorted_contours,
        'sorted_indices': sorted_indices,
        'depths': sorted_depths
    }