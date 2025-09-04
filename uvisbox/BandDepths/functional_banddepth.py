import numpy as np

def functional_banddepth(data, dtype=np.float64):
    # Compute the functional band depth of the input data
    # The band is assumed to be formed by 2-subsets
    # This implements Sun, Y., Genton, M.G. and Nychka, D.W. (2012), Exact fast computation of band depth for large functional datasets: How quickly can one million curves be ranked?. Stat, 1: 68-74. https://doi.org/10.1002/sta4.8
    # Data is a 2D array of shape (N, D)

    ### input validation
    _data = _data_validation(data, dtype=dtype)
    _n, _ = _data.shape

    ### functional band depth computation
    _data = np.argsort(_data, axis=0)     # compute the order
    _data = np.argsort(_data, axis=0) + 1 # compute
    _down = np.min(_data, axis=1) - 1     # counting number smaller than the row
    _up = _n - np.max(_data, axis=1)      # counting number larger than the row

    ### return the band depths
    return _up * _down + _n - 1

def modified_functional_banddepth(data, dtype=np.float64):
    # Compute the modified functional band depth of the input data
    # The band is assumed to be formed by 2-subsets
    # This implements Sun, Y., Genton, M.G. and Nychka, D.W. (2012), Exact fast computation of band depth for large functional datasets: How quickly can one million curves be ranked?. Stat, 1: 68-74. https://doi.org/10.1002/sta4.8
    # Data is a 2D array of shape (N, D)

    ### input validation
    _data = _data_validation(data, dtype=dtype)
    _n, _d = _data.shape

    ### functional band depth computation
    _data = np.argsort(_data, axis=0)     # compute the order
    _data = np.argsort(_data, axis=0) + 1 # compute the rank
    _data = (_n - _data) * (_data - 1)    # following Sun et al. (2012)

    ### return the band depths
    return (np.sum(_data, axis=1)/_d) + _n - 1 # band depth as probability

def _data_validation(data, dtype=np.float64):
    if isinstance(data, np.ndarray):
        _data = data
    elif isinstance(data, list):
        _data = np.array(data, copy=True)
    else:
        raise TypeError("Input data must be a numpy ndarray or a 2D list.")
    if _data.ndim != 2:
        raise ValueError("Input data must be a 2D array of shape (N, D).")
    _n, _d = _data.shape
    if _n < 1 or _d < 1:
        raise ValueError("Input data must have at least one sample and one feature.")
    _data = _data.astype(dtype)  # cast data to the specified dtype
    return _data

fdb = functional_banddepth
mdb = modified_functional_banddepth