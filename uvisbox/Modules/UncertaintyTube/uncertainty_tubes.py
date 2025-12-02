from scipy.__config__ import show
from .uncertainty_tubes_stats import uncertainty_tubes_summary_statistics
from .uncertainty_tubes_mesh import uncertainty_tubes_mesh
from .uncertainty_tubes_vis import visualize_uncertainty_tubes
import numpy as np


def uncertainty_tubes(trajectories, colormap="viridis", plotter=None, resolution=20, e_proj=1, sym=False, clim=None, n_jobs=1):
    """
    Generate and visualize 3D uncertainty tubes from trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Array of shape (n_steps, n_starting_locations, n_ensemble_members, 3) representing the 3D trajectories.
    colormap (str, optional):
        Colormap to use for rendering the tube. Defaults to "viridis".
    plotter (matplotlib.axes.Axes or pyvista.Plotter, optional):
        The plotting object to use. If None, a new Matplotlib figure/axis is created.
    resolution (int, optional):
        Number of points to sample on each cross-section boundary. Defaults to 20.
    e_proj (float, optional):
        Exponent controlling the superellipse shape. e_proj=1 creates a standard ellipse.
        Defaults to 1.
    sym (bool, optional):
        If True, forces the superellipse to be symmetric. Defaults to False.
    n_jobs (int, optional):
        Number of parallel jobs to use. If n_jobs=1, uses sequential processing.
        Defaults to 1.

    Returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter: The plotting object with the visualization.
    """
    # Stage 1: Statistics
    summary_statistics = uncertainty_tubes_summary_statistics(trajectories, n_jobs=n_jobs)

    # Stage 2: Mesh Generation
    mesh_data = uncertainty_tubes_mesh(
        summary_statistics,
        resolution=resolution,
        e_proj=e_proj,
        sym=sym,
        n_jobs=n_jobs
    )

    # Stage 3: Visualization
    plotter = visualize_uncertainty_tubes(mesh_data, colormap=colormap, clim=clim, plotter=plotter)

    return plotter