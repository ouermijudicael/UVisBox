"""
Squid Glyph Uncertainty Visualization Module.

High-level API (for most users):
    - squid_glyph_2D: Visualize 2D vector field uncertainty
    - squid_glyph_3D: Visualize 3D vector field uncertainty

Low-level API (for advanced customization):
    Stats Stage:
        - compute_squid_glyph_stats_2d: Compute 2D vector depths and spreads
        - compute_squid_glyph_stats_3d: Compute 3D vector depths, spreads, and PCA
    
    Mesh Stage:
        - build_squid_glyph_mesh_2d: Build 2D glyph geometry
        - build_squid_glyph_mesh_3d: Build 3D glyph geometry
    
    Visualization Stage:
        - render_squid_glyph_2d: Render 2D glyphs with matplotlib
        - render_squid_glyph_3d: Render 3D glyphs with pyvista
"""

# High-level API
from .squid_glyphs import squid_glyph_2D, squid_glyph_3D

# Low-level API - Stats
from .squid_glyphs_stats import (
    compute_squid_glyph_stats_2d,
    compute_squid_glyph_stats_3d,
)

# Low-level API - Mesh
from .squid_glyphs_mesh import (
    build_squid_glyph_mesh_2d,
    build_squid_glyph_mesh_3d,
)

# Low-level API - Visualization
from .squid_glyphs_vis import (
    render_squid_glyph_2d,
    render_squid_glyph_3d,
)

__all__ = [
    # High-level API
    'squid_glyph_2D',
    'squid_glyph_3D',
    
    # Low-level API - Stats
    'compute_squid_glyph_stats_2d',
    'compute_squid_glyph_stats_3d',
    
    # Low-level API - Mesh
    'build_squid_glyph_mesh_2d',
    'build_squid_glyph_mesh_3d',
    
    # Low-level API - Visualization
    'render_squid_glyph_2d',
    'render_squid_glyph_3d',
]