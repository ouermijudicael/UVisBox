"""
Mesh generation for uncertainty lobes.

This module builds wedge-shaped mesh geometry for uncertainty lobe visualization:
    - Create wedge vertices for each lobe
    - Triangulate wedge faces
    - Support dual-lobe rendering (inner + outer)
"""

import numpy as np


def _create_wedge_vertices(center, r, theta_start, theta_end, median_angle, num_points=20):
    """
    Create vertices for a wedge (circular sector).
    
    The arc is always drawn in the direction that includes the median angle,
    ensuring the wedge represents the correct angular spread.
    
    Parameters:
    -----------
    center : numpy.ndarray
        Shape (2,) - center position [x, y]
    r : float
        Radius of the wedge
    theta_start : float
        Starting angle in radians (minimum angle)
    theta_end : float
        Ending angle in radians (maximum angle)
    median_angle : float
        Median direction angle in radians (used to determine arc direction)
    num_points : int
        Number of points along the arc (default: 20)
    
    Returns:
    --------
    vertices : numpy.ndarray
        Shape (num_points + 1, 2) - vertices [center, arc_point_1, ..., arc_point_n]
    """
    # Normalize angles to [0, 2π)
    theta_start_norm = theta_start % (2 * np.pi)
    theta_end_norm = theta_end % (2 * np.pi)
    median_angle_norm = median_angle % (2 * np.pi)
    
    # Calculate the angular span in both directions
    # Direction 1: counterclockwise from start to end
    if theta_end_norm >= theta_start_norm:
        span_ccw = theta_end_norm - theta_start_norm
    else:
        span_ccw = (2 * np.pi - theta_start_norm) + theta_end_norm
    
    # Direction 2: clockwise from start to end (or ccw from end to start)
    span_cw = 2 * np.pi - span_ccw
    
    # Check which direction contains the median angle
    # Test if median is in the counterclockwise arc from start to end
    if theta_end_norm >= theta_start_norm:
        median_in_ccw = theta_start_norm <= median_angle_norm <= theta_end_norm
    else:
        # Arc wraps around 0
        median_in_ccw = median_angle_norm >= theta_start_norm or median_angle_norm <= theta_end_norm
    
    # Generate arc points in the correct direction
    if median_in_ccw:
        # Go counterclockwise from start to end
        if theta_end_norm >= theta_start_norm:
            theta_range = np.linspace(theta_start_norm, theta_end_norm, num_points)
        else:
            # Wrap around: go from start to 2π, then from 0 to end
            theta_range = np.linspace(theta_start_norm, theta_start_norm + span_ccw, num_points)
            theta_range = theta_range % (2 * np.pi)
    else:
        # Go clockwise from start to end (same as ccw from end to start)
        if theta_start_norm >= theta_end_norm:
            theta_range = np.linspace(theta_start_norm, theta_start_norm + span_cw, num_points)
            theta_range = theta_range % (2 * np.pi)
        else:
            # Wrap around the other way
            theta_range = np.linspace(theta_start_norm, theta_start_norm - span_cw, num_points)
            theta_range = theta_range % (2 * np.pi)
    
    arc_points = np.column_stack([
        center[0] + r * np.cos(theta_range),
        center[1] + r * np.sin(theta_range)
    ])
    
    # Prepend center point
    vertices = np.vstack([center, arc_points])
    
    return vertices


def _triangulate_wedge(num_points):
    """
    Create triangle indices for a wedge.
    
    Parameters:
    -----------
    num_points : int
        Number of points along the arc
    
    Returns:
    --------
    triangles : numpy.ndarray
        Shape (num_points - 1, 3) - triangle indices
    """
    triangles = []
    for i in range(num_points - 1):
        # Triangle: [center, arc_point_i, arc_point_i+1]
        triangles.append([0, i + 1, i + 2])
    
    return np.array(triangles, dtype=int)


def build_uncertainty_lobes_mesh_2d(positions, stats_2d, scale=0.2, arc_resolution=20):
    """
    Build 2D uncertainty lobe mesh from statistics.
    
    Creates wedge-shaped glyphs with:
    1. Outer lobe (percentile1 - larger angular and magnitude spread)
    2. Inner lobe (percentile2 - smaller spread, optional)
    3. Median arrow direction
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Shape (n, 2) - lobe center positions
    stats_2d : dict
        From compute_uncertainty_lobes_stats_2d()
    scale : float
        Glyph scale factor (default: 0.2)
    arc_resolution : int
        Number of points per wedge arc (default: 20)
    
    Returns:
    --------
    mesh_2d : dict
        {
            'wedges': list of dicts - each containing 'vertices' and 'triangles' for outer lobe,
            'inner_wedges': list of dicts - each containing 'vertices' and 'triangles' for inner lobe (if percentile2 != None),
            'arrows': dict - {'positions': (n, 2), 'directions': (n, 2), 'lengths': (n,)}
        }
    """
    num_positions = positions.shape[0]
    
    theta1 = stats_2d['theta1']  # (n, 2) - [min_angle, max_angle] for outer lobe
    theta2 = stats_2d['theta2']  # (n, 2) or None - for inner lobe
    mid_angle = stats_2d['mid_angle']  # (n,) - median angles
    r1 = stats_2d['r1']  # (n,) - minimum magnitude (outer lobe)
    r2 = stats_2d['r2']  # (n,) - maximum magnitude (inner lobe)
    r_arrow = stats_2d['r_arrow']  # (n,) - median magnitudes
    
    # Apply scale
    r1_scaled = r1 * scale
    r2_scaled = r2 * scale
    r_arrow_scaled = r_arrow * scale
    
    # Build wedges
    wedges = []  # Outer lobes
    inner_wedges = []  # Inner lobes
    
    for i in range(num_positions):
        # Outer lobe (percentile1)
        theta_start = theta1[i, 0]  # min_angle
        theta_end = theta1[i, 1]    # max_angle
        median = mid_angle[i]        # median angle
        
        vertices = _create_wedge_vertices(
            positions[i], r1_scaled[i], theta_start, theta_end, median, arc_resolution
        )
        triangles = _triangulate_wedge(arc_resolution)
        
        wedges.append({
            'vertices': vertices,
            'triangles': triangles,
            'position_idx': i
        })
        
        # Inner lobe (percentile2, if provided)
        if theta2 is not None and r2_scaled[i] > 0.0:
            theta_start2 = theta2[i, 0]
            theta_end2 = theta2[i, 1]
            
            vertices2 = _create_wedge_vertices(
                positions[i], r2_scaled[i], theta_start2, theta_end2, median, arc_resolution
            )
            triangles2 = _triangulate_wedge(arc_resolution)
            
            inner_wedges.append({
                'vertices': vertices2,
                'triangles': triangles2,
                'position_idx': i
            })
    
    # Build arrow data
    arrow_directions = np.column_stack([
        np.cos(mid_angle),
        np.sin(mid_angle)
    ])
    
    arrows = {
        'positions': positions,
        'directions': arrow_directions,
        'lengths': r_arrow_scaled
    }
    
    return {
        'wedges': wedges,
        'inner_wedges': inner_wedges if len(inner_wedges) > 0 else None,
        'arrows': arrows
    }
