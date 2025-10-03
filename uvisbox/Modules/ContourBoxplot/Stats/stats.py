from uvisbox.Core.BandDepths.contour_banddepth import *

import numpy as np

def find_percentile(sorted_images, percentile):
    n_images = sorted_images.shape[0]
    index = int(np.ceil(n_images * (percentile / 100)))

    before = sorted_images[:index]

    # Find union and intersection
    union = np.any(before, axis=0)
    intersection = np.all(before, axis=0)
    # Pixels in union but not in intersection
    union_minus_intersection = union & (~intersection)

    return union_minus_intersection