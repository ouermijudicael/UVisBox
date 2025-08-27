import numpy as np

def linear_interpolate(x, x_min, x_max, y_min, y_max):
    """
    Performs linear interpolation to estimate an output value (`out`) corresponding to an input value (`in`).

    Parameters:
        in (float): The input value for which to interpolate the output.
        x_min (float): The minimum value of the input range.
        x_max (float): The maximum value of the input range.
        y_min (float): The minimum value of the output range.
        y_max (float): The maximum value of the output range.

    Returns:
        float: The interpolated output value for the given input.

    Notes:
        If `x_max` equals `x_min`, returns the average of `y_min` and `y_max` to avoid division by zero.
    """
    if (x_max - x_min) == 0:
        return (y_min + y_max) / 2
    return y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)
