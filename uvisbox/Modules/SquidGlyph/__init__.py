"""
SquidGlyph Module

This module provides squid glyph visualization functionality for uncertainty visualization.
"""

# Import mesh, stats, and visualization functionality
try:
    from .Mesh.squid_glyphs_meshing import *
    from .Stats.squid_glyphs import *
    from .Vis.squid_glyphs_plot import *
except ImportError:
    pass