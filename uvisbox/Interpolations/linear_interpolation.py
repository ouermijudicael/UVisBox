import numpy as np

def linear_interpolate(in_value, in_min, in_max, out_min, out_max):
    """
    Performs linear interpolation to estimate an output value (`out`) corresponding to an input value (`in`).

    Parameters:
        in_value (float): The input value for which to interpolate the output.
        in_min (float): The minimum value of the input range.
        in_max (float): The maximum value of the input range.
        out_min (float): The minimum value of the output range.
        out_max (float): The maximum value of the output range.

    Returns:
        float: The interpolated output value for the given input.

    Notes:
        If `in_max` equals `in_min`, returns the average of `out_min` and `out_max` to avoid division by zero.
    """
    if (in_max - in_min) == 0:
        return (out_min + out_max) / 2
    return out_min + (out_max - out_min) * (in_value - in_min) / (in_max - in_min)
