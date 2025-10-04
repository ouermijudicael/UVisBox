"""
SquidGlyphs3D module for UVisBox.
This module provides functionality for visualizing uncertainty in 3D vector fields using squid glyphs.
"""
# Import meshing, statistics, and visualization functionality
try:
    from .Mesh.squid_glyphs_meshing_3D import *
    from .Stats.vector_stats import *
    from .Vis.pyvista_plot import *
    from plot import *
except ImportError:
    pass
