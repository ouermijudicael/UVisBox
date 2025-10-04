"""
SquidGlyphs2D Module

This module provides functionality for visualizing uncertainty in 2D vector fields using squid glyphs.
"""
# Import meshing, statistics, and visualization functionality
try:
    from .Mesh.squid_glyphs_meshing_2D import *
    from .Stats.vector_stats import *
    from .Vis.matplotlib_vis import *
    from plot import *
except ImportError:
    pass    
# Stats subdirectory is available for future expansion