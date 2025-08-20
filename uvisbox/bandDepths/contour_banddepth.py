import numpy as np

def choose2(x):
    ### helper function returns x choose 2
    return int(1/8 * (2 * x -1 ) ** 2 - 1/8)
    
def get_combinations(n):
    ### for n elements, get all 2-subsets
    # returns (n_combination, 2) as list of pair of indices
    n_combinations = choose2(n)
    combinations = np.zeros((n_combinations,2), np.int32)

    idx = 0
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            combinations[idx] = (i, j)
            idx += 1
    return combinations

def _epsilon_subset(A, B, eps):
    ### determine if two sets are epsilon-close (Order matters!)
    cardA = np.sum(A)
    return cardA == 0 or np.sum(np.bitwise_and(A, np.logical_not(B))) / cardA < eps

def _portion_subset(A, B):
    ### determine if two sets are partial-overlapping (Order matters!)
    cardA = np.sum(A)
    if cardA == 0:
        return 0
    return np.sum(np.bitwise_and(A,np.logical_not(B)))/ cardA

def contour_banddepth(data, combination = None, allow_portion=False, eps = 0):
    """
    Calculates the band depth of binary contour data using pairwise combinations.
    R. T. Whitaker, M. Mirzargar and R. M. Kirby, "Contour Boxplots: A Method for Characterizing Uncertainty in Feature Sets from Simulation Ensembles," in IEEE Transactions on Visualization and Computer Graphics, vol. 19, no. 12, pp. 2713-2722, Dec. 2013, doi: 10.1109/TVCG.2013.143
    Parameters
    ----------
    data : array-like or np.ndarray
        Input data representing binary contours. Should be convertible to a boolean NumPy array.
    combination : list of tuple, optional
        List of index pairs (xdx, ydx) specifying which images to combine for band depth calculation.
        If None, combinations are generated automatically.
    allow_portion : bool, default False
        If True, uses a portion-based subset calculation for depth; otherwise, uses epsilon-based subset.
    eps : float, default 0
        Epsilon tolerance for subset checks when `allow_portion` is False.
    Returns
    -------
    depths : np.ndarray
        Array of band depth values for each image in the input data.
    Raises
    ------
    ValueError
        If the input data cannot be converted to a boolean array.
    Notes
    -----
    - The function expects binary contour data, where each image is represented as a boolean array.
    - Depth is computed by evaluating how each image fits within the intersection and union of pairs of images.
    - Helper functions `_portion_subset` and `_epsilon_subset` are used for subset checks.
    """
   
    
    ### data validation:
    if isinstance(data, np.ndarray):
        if data.dtype != np.bool_:
            binary_data = data.astype(np.bool_)
        else:
            binary_data = data
    else:
        try:
            binary_data = np.array(data, dtype=np.bool_)
        except Exception as e:
            raise ValueError("Input data could not be converted to a boolean array.") from e
        
    n_images = binary_data.shape[0]
    if combination is None:
        combination = get_combinations(n_images)
    
    depths = np.zeros([n_images])
    for tdx in np.arange(n_images):
        target = binary_data[tdx]
        for xdx, ydx in combination:
            intersection = np.bitwise_and(binary_data[xdx], binary_data[ydx])
            union = np.bitwise_or(binary_data[xdx], binary_data[ydx])
            if allow_portion:
                depths[tdx] += _portion_subset(intersection, target) + _portion_subset(target, union)
            else:
                if _epsilon_subset(intersection, target, eps) and _epsilon_subset(target, union, eps):
                    depths[tdx] += 1
    return depths