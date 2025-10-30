import numpy as np
from uvisbox.Core.BandDepths.functional_banddepth import functional_banddepth, modified_functional_banddepth

def get_band(data, percentile, method='fbd'):
    """
    Compute the band envelope for a given percentile using functional band depth.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number of points per curve.
    percentile : float
        Percentile value (0-100) for the band envelope.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fbd': functional band depth (default)
        - 'mfbd': modified functional band depth
    
    Returns:
    --------
    bottom_curve : np.ndarray
        1D array representing the bottom envelope of the band.
    top_curve : np.ndarray
        1D array representing the top envelope of the band.
    
    Examples:
    ---------
    >>> data = np.random.randn(100, 50)  # 100 curves, 50 points each
    >>> bottom, top = get_band(data, 50, method='fbd')  # 50th percentile band
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (N, D).")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    
    # Compute band depths based on method
    if method == 'fbd':
        depths = functional_banddepth(data)
    elif method == 'mfbd':
        depths = modified_functional_banddepth(data)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fbd' or 'mfbd'.")

    # Sort curves by depth (descending order - highest depth first)
    sorted_indices = np.argsort(depths)[::-1]
    sorted_curves = data[sorted_indices]
    
    # Calculate the number of curves to include
    n_curves = sorted_curves.shape[0]
    index = int(np.ceil(n_curves * (percentile / 100)))
    
    # Get the top `index` curves with highest depth
    selected_curves = sorted_curves[:index]
    
    # Compute envelope
    top_curve = np.max(selected_curves, axis=0)
    bottom_curve = np.min(selected_curves, axis=0)
    
    return bottom_curve, top_curve

def functional_depth_mesh(top_curve, bottom_curve, scale=1.0):
    """
    Create a 2D triangular mesh representing the functional depth area between the top and bottom curves.
    
    Parameters:
    -----------
    top_curve : np.ndarray
        1D array representing the top curve.
    bottom_curve : np.ndarray
        1D array representing the bottom curve.
    scale : float, optional
        Scale factor for the depth area. Default is 1.0.
    
    Returns:
    -----------
    vertices : np.ndarray
        2D array of shape (n_vertices, 2) representing the vertices of the mesh.
    """
    if top_curve.shape != bottom_curve.shape:
        raise ValueError("All input curves must have the same shape.")
    
    # Create a grid of points between the curves
    x = np.linspace(0, 1, top_curve.shape[0])
    y_top = top_curve * scale
    y_bottom = bottom_curve * scale

    points = np.zeros((top_curve.shape[0] * 2, 2))
    triangles = np.zeros(( (top_curve.shape[0]-1) * 2, 3), dtype=int)
    i_pt = 0
    i_tr = 0
    for i in range(top_curve.shape[0]):
        points[i_pt] = [x[i], y_top[i]]
        points[i_pt + 1] = [x[i], y_bottom[i]]
        i_pt += 2
        if i > 0:
            triangles[i_tr] = [i_pt - 4, i_pt - 3, i_pt - 2]
            triangles[i_tr + 1] = [i_pt - 3, i_pt - 1, i_pt - 2]
            i_tr += 2
            
    return points, triangles