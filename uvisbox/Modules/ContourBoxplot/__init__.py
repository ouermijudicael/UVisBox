"""
ContourBoxplot Module

This module provides contour-based boxplot functionality for uncertainty visualization.
"""

# Import visualization functionality
try:
    from .contour_boxplot_vis import *
    from .contour_boxplot_mesh import *
    from .contour_boxplot_stats import *
    from uvisbox.Modules.ContourBoxplot.contour_boxplot import *
except ImportError:
    pass

# Mesh and Stats subdirectories are available for future expansion