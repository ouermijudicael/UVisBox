import numpy as np
import matplotlib.pyplot as plt
from .functional_boxplot_stats import functional_boxplot_summary_statistics
from .functional_boxplot_mesh import functional_boxplot_mesh
from .functional_boxplot_vis import visualize_functional_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


def functional_boxplot(data, method='fbd', vmin=0, vmax=1, boxplot_style=None, ax=None):
    """
    Create a functional band depth boxplot with multiple percentile bands.

    This function computes functional band depths, plots bands in descending percentile 
    order (largest to smallest for proper layering), and highlights the median curve.

    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (N, D) where N is the number of curves and D is the number 
        of points per curve.
    method : str, optional
        Method for computing band depth. Options are:
        - 'fbd': functional band depth (default)
        - 'mfbd': modified functional band depth
    boxplot_style : BoxplotStyleConfig, optional
        Configuration for the boxplot visualization including percentiles, colormap,
        and median/outlier styling. If None, uses default configuration.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib axes object with the plot.

    Raises:
    -------
    ValueError
        If data is not 2D or if method is invalid.

    Notes:
    ------
    - Input data is not modified (computation happens on a copy)
    - Bands are plotted from largest to smallest percentile for proper visual layering
    - The median curve is the curve with the highest band depth value
    - Outliers are curves beyond the largest percentile
    - Curve depths are always computed internally

    Examples:
    ---------
    >>> import numpy as np
    >>> from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
    >>> from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    >>> 
    >>> # Generate synthetic functional data
    >>> t = np.linspace(0, 1, 100)
    >>> data = np.array([np.sin(2*np.pi*t) + 0.2*np.random.randn(100) for _ in range(50)])
    >>> 
    >>> # Basic usage with default settings
    >>> ax = functional_boxplot(data)
    >>> 
    >>> # Custom styling
    >>> style = BoxplotStyleConfig(
    ...     percentiles=[10, 50, 90],
    ...     percentile_colormap='plasma',
    ...     show_median=True,
    ...     show_outliers=True
    ... )
    >>> ax = functional_boxplot(data, boxplot_style=style)
    >>> 
    >>> # Plot on existing axes
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(12, 6))
    >>> functional_boxplot(data, ax=ax)
    >>> plt.show()
    """
    # Use default config if none provided
    if boxplot_style is None:
        boxplot_style = BoxplotStyleConfig()

    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 2:
        raise ValueError(
            f"Input data must be a 2D array of shape (N, D). Got {data.ndim}D array.")

    # Pipeline: Stats -> Mesh -> Vis
    # Step 1: Compute summary statistics
    stats = functional_boxplot_summary_statistics(
        data, method=method, boxplot_style=boxplot_style)

    # Step 2: Process through mesh (identity function for functional boxplot)
    mesh_data = functional_boxplot_mesh(stats)

    # Step 3: Visualize
    ax = visualize_functional_boxplot(
        mesh_data, vmin=vmin, vmax=vmax, boxplot_style=boxplot_style, ax=ax)

    return ax