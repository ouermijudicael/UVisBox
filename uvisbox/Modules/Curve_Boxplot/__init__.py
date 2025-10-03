"""
Curve_Boxplot Module

This module provides curve-based boxplot functionality for uncertainty visualization.
"""

# Import meshing and visualization functionality
try:
    from .Mesh.curve_banddepth_meshing import *
    from .Vis.curve_banddepth_plot import *
except ImportError:
    pass

# Stats subdirectory is available for future expansion