from uvisbox.Core.BandDepths.functional_banddepth import *

# available functions:
#   - calculate_band(sorted_curves, percentile)
#   - functional_banddepth(data, dtype=np.float64):
#   - modified_functional_banddepth(data, dtype=np.float64):
#   - fbd = functional_banddepth
#   - mfbd = modified_functional_banddepth
#

def band_depths(data, method='fdb'):
    """
    Compute band depths for a set of functional curves.

    Parameters:
    -----------
    curves : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number of points per curve.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fdb': Functional band depth (default)
        - 'mfbd': Modified functional band depth
    """
    if method == 'fdb':
        return functional_banddepth(data)
    elif method == 'mfbd':
        return modified_functional_banddepth(data)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fdb' or 'mfbd'.")