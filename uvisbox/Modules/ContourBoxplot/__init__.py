"""
ContourBoxplot Module

This module provides contour-based boxplot functionality for uncertainty visualization.
"""

# Import visualization functionality
try:
    from .Vis.vis import *
    from .Mesh.mesh import *
    from .Stats.stats import *
    from plot import *
except ImportError:
    pass

# Mesh and Stats subdirectories are available for future expansion