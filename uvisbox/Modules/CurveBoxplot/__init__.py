"""
Curve_Boxplot Module

This module provides curve-based boxplot functionality for uncertainty visualization.
"""

# Import functionality
try:
    from .Vis.vis import *
    from .Mesh.mesh import *
    from .Stats.stats import *
    from plot import *
except ImportError:
    pass

# Stats subdirectory is available for future expansion