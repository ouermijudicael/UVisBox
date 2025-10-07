import numpy as np
from matplotlib.patches import Wedge


def matplotlib_uncertainty_lobes_vis(ax, centers, theta1, theta2, mid_angle, r1, r2, r_arrow, show_median):
    """
    Draws multiple wedges with arrows.
    Parameters:
    -----------
    centers : numpy.ndarray
        Array of shape (n, 2) representing the center positions of the wedges.
    theta1 : numpy.ndarray
        Array of shape (n, 2), each row [theta1_start, theta1_end] for the first wedge.
    theta2 : numpy.ndarray
        Array of shape (n, 2), each row [theta2_start, theta2_end] for the second wedge (arrowhead).
    mid_angle : iterable
        Iterable of length n, mid angle for arrow direction.
    r1 : float or iterable
        Radius for the first wedge.
    r2 : float or iterable of length n
        Radius for the second wedge.
    r_arrow : float or iterable of length n
        Length of the arrow.
    show_median : bool
        Whether to show the median arrow.
    """
    n = centers.shape[0]
    # Support scalar or iterable for r1 and r2
    for i in range(n):
        center = centers[i]
        t1_start, t1_end = theta1[i]
        wedge = Wedge(center=center, r=r1[i], theta1=t1_start, theta2=t1_end, facecolor='skyblue', edgecolor='skyblue', alpha=0.3)
        if r2[i] > 0.0:
            t2_start, t2_end = theta2[i]
            wedge2 = Wedge(center=center, r=r2[i], theta1=t2_start, theta2=t2_end, facecolor='skyblue', edgecolor='skyblue', alpha=1.0)
        if show_median:
            ax.annotate(
                '',
                xy=(center[0] + r_arrow[i] * np.cos(np.deg2rad(mid_angle[i])), center[1] + r_arrow[i] * np.sin(np.deg2rad(mid_angle[i]))),
                xytext=center,
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=3, mutation_scale=20, alpha=1.0)
            )
        ax.add_patch(wedge)
        if r2[i] > 0.0:
            ax.add_patch(wedge2)

    return ax