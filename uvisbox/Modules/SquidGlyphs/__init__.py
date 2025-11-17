"""
Squid Glyph Uncertainty Visualization Module.

High-level API (for most users):
    - squid_glyph_2D: Visualize 2D vector field uncertainty
    - squid_glyph_3D: Visualize 3D vector field uncertainty

Low-level API (for advanced customization):
    Stats Stage:
        - squid_glyphs_2d_summary_statistics: Compute 2D vector depths and spreads
        - squid_glyphs_3d_summary_statistics: Compute 3D vector depths, spreads, and PCA
    
    Mesh Stage:
        - squid_glyphs_2d_mesh: Build 2D glyph geometry
        - squid_glyphs_3d_mesh: Build 3D glyph geometry
    
    Visualization Stage:
        - visualize_squid_glyphs_2d: Render 2D glyphs with matplotlib
        - visualize_squid_glyphs_3d: Render 3D glyphs with pyvista
"""

# High-level API
from .squid_glyphs import squid_glyph_2D, squid_glyph_3D

# Low-level API - Stats
from .squid_glyphs_stats import (
    squid_glyphs_2d_summary_statistics,
    squid_glyphs_3d_summary_statistics,
)

# Low-level API - Mesh
from .squid_glyphs_mesh import (
    squid_glyphs_2d_mesh,
    squid_glyphs_3d_mesh,
)

# Low-level API - Visualization
from .squid_glyphs_vis import (
    visualize_squid_glyphs_2d,
    visualize_squid_glyphs_3d,
)

__all__ = [
    # High-level API
    'squid_glyph_2D',
    'squid_glyph_3D',
    
    # Low-level API - Stats
    'squid_glyphs_2d_summary_statistics',
    'squid_glyphs_3d_summary_statistics',
    
    # Low-level API - Mesh
    'squid_glyphs_2d_mesh',
    'squid_glyphs_3d_mesh',
    
    # Low-level API - Visualization
    'visualize_squid_glyphs_2d',
    'visualize_squid_glyphs_3d',
]