"""
FunctionalBoxplot Module

This module provides functional boxplot functionality for uncertainty visualization.
"""

# Import meshing and visualization functionality
try:
    from .functional_boxplot_mesh import functional_boxplot_mesh
    from .functional_boxplot_stats import (
        band_depths,
        get_band,
        summary_statistics
    )
    from .functional_boxplot_vis import (
        plot_band,
        visualize_functional_boxplot
    )
    from .functional_boxplot import functional_boxplot
except ImportError:
    pass

__all__ = [
    'functional_boxplot_mesh',
    'band_depths',
    'get_band',
    'summary_statistics',
    'plot_band',
    'visualize_functional_boxplot',
    'functional_boxplot'
]

# Stats subdirectory is available for future expansion