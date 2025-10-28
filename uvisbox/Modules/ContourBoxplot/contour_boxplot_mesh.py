import numpy as np

def get_band(ordered_binary_images, percentile):
    """
    Compute the contour band envelope for a given percentile using contour band depth.
    
    Parameters:
    -----------
    ordered_binary_images : np.ndarray
        3D array of shape (N, H, W) where N is the number of binary images and H, W are the height and width of each image. 
        The images should be ordered by their contour band depths in descending order (highest depth first).
    percentile : float
        Percentile value (0-100) for the band envelope. 
    Returns:
    --------
    a binary image representing the band envelope.
    Examples:
    ---------
    1. Get the 50th percentile band envelope:
       band_envelope = get_band(ordered_images, 50)
    2. Get the 75th percentile band envelope:
       band_envelope = get_band(ordered_images, 75)
    """

    # Validate input
    if not isinstance(ordered_binary_images, np.ndarray):
        ordered_binary_images = np.array(ordered_binary_images)
    if ordered_binary_images.ndim != 3:
        raise ValueError("Input ordered_images must be a 3D array of shape (N, H, W).")
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    
    # ordered_binary_images are supposed to be np.bool type
    if ordered_binary_images.dtype != np.bool_:
        ordered_binary_images = ordered_binary_images.astype(np.bool_)

    n_images = ordered_binary_images.shape[0]
    index = int(np.ceil(n_images * (percentile / 100)))
    
    # Get the top `index` images with highest depth
    selected_images = ordered_binary_images[:index]
    
    # Compute envelope by taking pixel-wise maximum
    # handle case where no images are selected
    if index == 0:
        band_envelope = np.zeros(ordered_binary_images.shape[1:], dtype=ordered_binary_images.dtype)
    else:
        # union: pixel is True if any selected image has it
        union = np.any(selected_images, axis=0)
        # intersection: pixel is True if all selected images have it
        intersection = np.all(selected_images, axis=0)
        # envelope = union minus intersection (pixels present in some but not all)
        band_envelope = (union & ~intersection).astype(ordered_binary_images.dtype)
    
    return band_envelope