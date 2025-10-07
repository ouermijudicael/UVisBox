"""
Curve_Boxplot Module

This module provides curve-based boxplot functionality for uncertainty visualization.
"""

# Import functionality
try:
    from .cuve_boxplot_vis import *
    from .curve_boxplot_mesh import *
    from .curve_boxplot_stats import *
    from uvisbox.Modules.CurveBoxplot.curve_boxplot import *
except ImportError:
    pass

# Stats subdirectory is available for future expansion