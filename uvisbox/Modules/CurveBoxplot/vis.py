import matplotlib.pyplot as plt
import numpy as np

def vis(points, triangles, median_curve, curve_dim, ax=None, color_map='viridis', median_color='red', alpha=1.0):
    if curve_dim == 2:
        if ax is None:  
            fig, ax = plt.subplots()
        # ax.triplot(points[:, 0], points[:, 1], triangles, color='gray', alpha=0.5)
        ax.tripcolor(points[:, 0], points[:, 1], triangles, facecolors=np.ones(triangles.shape[0]), cmap=color_map, alpha=alpha)
        ax.plot(median_curve[:, 0], median_curve[:, 1], color=median_color, linewidth=2)
    elif curve_dim == 3:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap=color_map, alpha=alpha)
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color=median_color, linewidth=2)
        else:
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, cmap=color_map, alpha=alpha)
            ax.plot(median_curve[:, 0], median_curve[:, 1], median_curve[:, 2], color=median_color, linewidth=2)
    else:
        raise ValueError("curve_dim must be 2 or 3 for plotting.")

    return ax

