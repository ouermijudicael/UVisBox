"""
FunctionalBoxplot Module

This module provides functional boxplot functionality for uncertainty visualization.
"""

# Import meshing and visualization functionality
try:
    from .functional_boxplot_mesh import *
    from .functional_boxplot_stats import *
    from .functional_boxplot_vis import *
    from uvisbox.Modules.FunctionalBoxplot.functional_boxplot import *
except ImportError:
    pass

# Stats subdirectory is available for future expansion