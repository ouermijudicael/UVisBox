"""
FunctionalBoxplot Module

This module provides functional boxplot functionality for uncertainty visualization.
"""

# Import meshing and visualization functionality
try:
    from .Mesh.functional_depth_mesh import *
    from .Vis.vis import *
    from .Stats.stats import *
    from plot import *
except ImportError:
    pass

# Stats subdirectory is available for future expansion