import numpy as np
from multiprocessing import get_context


def choose2(x):
    """
    Helper function to compute x choose 2

    Parameters:
    -----------
    x : int
        number of elements
    
    Returns:
    --------
    [] : int
        x choose 2  
    """
    return int(1/8 * (2 * x -1 ) ** 2 - 1/8)
    

def get_combinations(n):
    """
    returns (n_combination, 2) as list of pair of indices for n elements, get all 2-subsets

    Parameters:
    -----------
    n : int
        number of elements
    
    Returns:
    -----------
    combinations : np.ndarray
        array of shape (n_choose_2, 2)
    """
    n_combinations = choose2(n)
    combinations = np.zeros((n_combinations,2), np.int32)

    idx = 0
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            combinations[idx] = (i, j)
            idx += 1
    return combinations

def _epsilon_subset(A, B, eps, cardA=None):
    """ 
    determine if two sets are epsilon-close (Order matters!)

    Parameters:
    -----------
    A : np.ndarray
        binary array representing set A
    B : np.ndarray
        binary array representing set B
    eps : float
        tolerance level
    cardA : int, optional
        Pre-computed cardinality of A (for optimization)

    Returns:
    -----------
    bool : True if A is an epsilon-subset of B, False otherwise
    """
    if cardA is None:
        cardA = np.count_nonzero(A)
    
    if cardA == 0:
        return True
    
    # Fast path for eps=0 (exact subset check)
    if eps == 0:
        # A is subset of B if A & ~B is empty
        return np.count_nonzero(A & ~B) == 0
    
    # General case
    return np.count_nonzero(A & ~B) / cardA <= eps

def _portion_subset(A, B, cardA=None):
    """
    determine if two sets are partial-overlapping (Order matters!)

    Parameters:
    -----------
    A : np.ndarray
        binary array representing set A
    B : np.ndarray
        binary array representing set B
    cardA : int, optional
        Pre-computed cardinality of A (for optimization)
    
    Returns:
    -----------
    float : portion of A not in B
    """
    if cardA is None:
        cardA = np.count_nonzero(A)
    
    if cardA == 0:
        return 0
    
    return np.count_nonzero(A & ~B) / cardA

def _compute_depth_for_image_optimized(args):
    """
    Helper function to compute depth for a single image (for multiprocessing).
    Optimized to avoid passing large arrays - uses indices instead.
    
    Parameters:
    -----------
    args : tuple
        (tdx, target, cardA_target, binary_data, combination, allow_portion, eps)
    
    Returns:
    --------
    tuple : (tdx, depth_value)
    """
    tdx, target, cardA_target, binary_data, combination, allow_portion, eps = args
    depth = 0
    
    for xdx, ydx in combination:
        intersection = binary_data[xdx] & binary_data[ydx]
        union = binary_data[xdx] | binary_data[ydx]
        
        if allow_portion:
            # Early termination: skip if intersection is empty
            cardA_intersection = np.count_nonzero(intersection)
            if cardA_intersection == 0:
                depth += _portion_subset(target, union, cardA_target)
            else:
                depth += _portion_subset(intersection, target, cardA_intersection) + _portion_subset(target, union, cardA_target)
        else:
            # Early termination: skip if intersection is empty
            if not np.any(intersection):
                continue
            
            if _epsilon_subset(intersection, target, eps) and _epsilon_subset(target, union, eps, cardA_target):
                depth += 1
    
    return (tdx, depth)

def _compute_depth_for_image(args):
    """
    Helper function to compute depth for a single image (for multiprocessing).
    
    Parameters:
    -----------
    args : tuple
        (tdx, target, cardA_target, intersections, unions, combination, allow_portion, eps)
    
    Returns:
    --------
    tuple : (tdx, depth_value)
    """
    tdx, target, cardA_target, intersections, unions, combination, allow_portion, eps = args
    depth = 0
    
    for idx, (xdx, ydx) in enumerate(combination):
        intersection = intersections[idx]
        union = unions[idx]
        
        if allow_portion:
            # Early termination: skip if intersection is empty
            cardA_intersection = np.count_nonzero(intersection)
            if cardA_intersection == 0:
                depth += _portion_subset(target, union, cardA_target)
            else:
                depth += _portion_subset(intersection, target, cardA_intersection) + _portion_subset(target, union, cardA_target)
        else:
            # Early termination: skip if intersection is empty
            if not np.any(intersection):
                continue
            
            if _epsilon_subset(intersection, target, eps) and _epsilon_subset(target, union, eps, cardA_target):
                depth += 1
    
    return (tdx, depth)

def contour_banddepth(data, combination = None, allow_portion=False, eps = 0, workers=12):
    """
    Calculates the band depth of binary contour data using pairwise combinations. 
    R. T. Whitaker, M. Mirzargar and R. M. Kirby, "Contour Boxplots: A Method for 
    Characterizing Uncertainty in Feature Sets from Simulation Ensembles," in IEEE Transactions 
    on Visualization and Computer Graphics, vol. 19, no. 12, pp. 2713-2722, Dec. 2013, doi: 10.1109/TVCG.2013.143
    
    Parameters:
    -----------
    data : np.ndarray
        Input data representing binary contours. Should be convertible to a boolean NumPy array.
    combination : list of tuples, optional
        List of index pairs (xdx, ydx) specifying which images to combine for band depth calculation. If None, combinations are generated automatically.
    allow_portion : bool, default False
        If True, uses a portion-based subset calculation for depth; otherwise, uses epsilon-based subset.
    eps : float, default 0
        Epsilon tolerance for subset checks when `allow_portion` is False.
    workers : int, default 12
        Number of parallel workers for multiprocessing. Set to 1 to disable parallel processing.

    Returns:
    --------
    depths : np.ndarray
        Array of band depth values for each image in the input data.

    Raises
    ------
    ValueError:
        If the input data cannot be converted to a boolean array.
        
    Notes
    -----
    - The function expects binary contour data, where each image is represented as a boolean array.
    - Depth is computed by evaluating how each image fits within the intersection and union of pairs of images.
    - Helper functions `_portion_subset` and `_epsilon_subset` are used for subset checks.
    - Uses fork context for multiprocessing to ensure compatibility with macOS.
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
    
    # Pre-compute cardinalities for all target images (lightweight optimization)
    cardinalities = np.array([np.count_nonzero(img) for img in binary_data])
    
    depths = np.zeros([n_images])
    
    if workers == 1:
        # Sequential processing with pre-computed intersections/unions
        # Pre-compute all intersections and unions (major optimization)
        n_combinations = len(combination)
        intersections = np.empty((n_combinations,) + binary_data.shape[1:], dtype=np.bool_)
        unions = np.empty((n_combinations,) + binary_data.shape[1:], dtype=np.bool_)
        
        for idx, (xdx, ydx) in enumerate(combination):
            intersections[idx] = binary_data[xdx] & binary_data[ydx]
            unions[idx] = binary_data[xdx] | binary_data[ydx]
        
        for tdx in range(n_images):
            target = binary_data[tdx]
            cardA_target = cardinalities[tdx]
            
            for idx in range(n_combinations):
                intersection = intersections[idx]
                union = unions[idx]
                
                if allow_portion:
                    # Early termination: skip if intersection is empty
                    cardA_intersection = np.count_nonzero(intersection)
                    if cardA_intersection == 0:
                        depths[tdx] += _portion_subset(target, union, cardA_target)
                    else:
                        depths[tdx] += _portion_subset(intersection, target, cardA_intersection) + _portion_subset(target, union, cardA_target)
                else:
                    # Early termination: skip if intersection is empty
                    if not np.any(intersection):
                        continue
                    
                    if _epsilon_subset(intersection, target, eps) and _epsilon_subset(target, union, eps, cardA_target):
                        depths[tdx] += 1
    else:
        # Parallel processing - compute intersections on-the-fly in each worker to avoid memory overhead
        # This trades some redundant computation for better parallelization
        ctx = get_context('fork')
        with ctx.Pool(processes=workers) as pool:
            # Prepare arguments for each image
            args_list = [
                (tdx, binary_data[tdx], cardinalities[tdx], binary_data, combination, allow_portion, eps)
                for tdx in range(n_images)
            ]
            # Compute depths in parallel
            results = pool.map(_compute_depth_for_image_optimized, args_list)
            # Collect results
            for tdx, depth in results:
                depths[tdx] = depth
    
    return depths

__all__ = ["contour_banddepth", "choose2", "get_combinations"]