"""
UncertaintyTube Module

This module provides uncertainty tube functionality for uncertainty visualization.
"""

# Import mesh functionality
try:
    from .uncertainty_tubes_mesh import (
        generate_uncertainty_tube_mesh_2D,
        compute_alignment_scores,
        apply_circular_alignment,
        circular_align_min_twist,
        generate_tube_mesh,
        calculate_mesh_dimensions,
        align_cross_sections,
        add_seed_vertices,
        generate_seed_faces,
        add_segment_triangles,
    )
except ImportError:
    pass

# Import stats functionality
try:
    from .uncertainty_tubes_stats import (
        expcos,
        expsin,
        project_points_to_plane,
        compute_eigen_2d,
        build_2d_superellipse,
        uncertainty_cross_section,
        generate_cross_sections,
        project_points_onto_line,
        generate_cross_sections_2D,
    )
except ImportError:
    pass

# Import visualization functionality
try:
    from .uncertainty_tubes_vis import (
        matplotlib_uncertainty_tube_vis,
        matplotlib_uncertainty_tube_2D_vis,
    )
except ImportError:
    pass

# Import high-level API
try:
    from .uncertainty_tubes import (
        uncertainty_tubes_2D,
        uncertainty_tubes_3D,
    )
except ImportError:
    pass