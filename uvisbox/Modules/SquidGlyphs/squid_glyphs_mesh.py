

import numpy as np


def _rotate_point_2d(pt, angle):
    """Rotate a 2D point by given angle."""
    return np.array([
        pt[0] * np.cos(angle) - pt[1] * np.sin(angle),
        pt[0] * np.sin(angle) + pt[1] * np.cos(angle)
    ])


def _add_quad_2d(points, polygons, point_idx, tri_idx, corners, position, rot_angle, scale):
    """Add a quadrilateral as 2 triangles to the 2D mesh."""
    # Rotate and position all 4 corners
    pts = []
    for corner in corners:
        pt = corner * scale
        pt = _rotate_point_2d(pt, rot_angle)
        pt = pt + position
        pts.append(pt)
    
    # Add points
    for i, pt in enumerate(pts):
        points[point_idx + i] = pt
    
    # Add triangles (counter-clockwise winding)
    polygons[tri_idx] = [point_idx, point_idx + 1, point_idx + 2]
    polygons[tri_idx + 1] = [point_idx + 1, point_idx + 3, point_idx + 2]
    
    return point_idx + 4, tri_idx + 2


def _add_triangle_2d(points, polygons, point_idx, tri_idx, corners, position, rot_angle, scale):
    """Add a triangle to the 2D mesh."""
    # Rotate and position all 3 corners
    pts = []
    for corner in corners:
        pt = corner * scale
        pt = _rotate_point_2d(pt, rot_angle)
        pt = pt + position
        pts.append(pt)
    
    # Add points
    for i, pt in enumerate(pts):
        points[point_idx + i] = pt
    
    # Add triangle
    polygons[tri_idx] = [point_idx, point_idx + 1, point_idx + 2]
    
    return point_idx + 3, tri_idx + 1


def build_squid_glyph_mesh_2d(positions, stats_2d, scale=0.2):
    """
    Build 2D squid glyph mesh from statistics.
    
    Creates arrow-shaped glyphs with three components:
    1. Base rectangle (uncertainty in magnitude)
    2. Shaft rectangle (tapered middle section)
    3. Head triangle (arrow tip)
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Shape (n, 2) - glyph positions
    stats_2d : dict
        From compute_squid_glyph_stats_2d()
    scale : float
        Glyph scale factor (default: 0.2)
    
    Returns:
    --------
    mesh_2d : dict
        {
            'points': (k, 2) - vertex positions,
            'polygons': (m, 3) - triangle connectivity
        }
    """
    median_vectors = stats_2d['median_vectors']
    min_mags = stats_2d['min_mag']
    max_mags = stats_2d['max_mag']
    min_angles = stats_2d['min_angle']
    max_angles = stats_2d['max_angle']
    
    num_positions = positions.shape[0]
    glyphs_points = np.zeros((num_positions * 11, 2))
    glyphs_polygons = np.zeros((num_positions * 5, 3), dtype=int)
    
    tri_idx = 0
    point_idx = 0
    
    for i_pos in range(num_positions):
        # Extract glyph parameters
        median_vector = median_vectors[i_pos]
        min_mag = min_mags[i_pos]
        max_mag = max_mags[i_pos]
        min_angle = min_angles[i_pos]
        max_angle = max_angles[i_pos]
        position = positions[i_pos]
        
        # Compute glyph dimensions
        delta_h = max_mag - min_mag  # Height of base (magnitude uncertainty)
        h = median_vector[0]  # Median magnitude
        rad_angle = (max_angle - min_angle) * 0.5  # Half angular spread
        rot_angle = median_vector[1] - np.pi * 0.5  # Rotation to align with median direction
        base = 2 * np.arctan(rad_angle) * max_mag  # Width of base
        
        # Skip degenerate glyphs
        if base <= 1e-5 or delta_h <= 1e-5:
            continue
        
        # 1. BASE RECTANGLE (magnitude uncertainty)
        base_corners = np.array([
            [-base * 0.5, 0],          # Bottom left
            [base * 0.5, 0],           # Bottom right
            [-base * 0.5, delta_h],    # Top left
            [base * 0.5, delta_h]      # Top right
        ])
        point_idx, tri_idx = _add_quad_2d(glyphs_points, glyphs_polygons, point_idx, tri_idx,
                                          base_corners, position, rot_angle, scale)
        
        # 2. SHAFT RECTANGLE (tapered middle section)
        shaft_width = np.arctan(rad_angle) * h
        shaft_corners = np.array([
            [-shaft_width, delta_h],              # Bottom left
            [shaft_width, delta_h],               # Bottom right
            [-np.arctan(rad_angle) * max_mag * 0.2, max_mag * 0.8],  # Top left
            [np.arctan(rad_angle) * max_mag * 0.2, max_mag * 0.8]    # Top right
        ])
        point_idx, tri_idx = _add_quad_2d(glyphs_points, glyphs_polygons, point_idx, tri_idx,
                                          shaft_corners, position, rot_angle, scale)
        
        # 3. HEAD TRIANGLE (arrow tip)
        head_corners = np.array([
            [-base * 0.5, max_mag * 0.8],  # Left base
            [base * 0.5, max_mag * 0.8],   # Right base
            [0, max_mag]                    # Tip
        ])
        point_idx, tri_idx = _add_triangle_2d(glyphs_points, glyphs_polygons, point_idx, tri_idx,
                                              head_corners, position, rot_angle, scale)
    
    return {
        'points': glyphs_points[:point_idx],
        'polygons': glyphs_polygons[:tri_idx]
    }


def build_squid_glyph_mesh_3d(positions, stats_3d, point_values=None, scale=0.5, resolution=10):
    """
    Build 3D squid glyph mesh from statistics.
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Shape (n, 3) - glyph positions
    stats_3d : dict
        From compute_squid_glyph_stats_3d()
    point_values : numpy.ndarray, optional
        Shape (n,) - scalar values for coloring
    scale : float
        Glyph scale factor (default: 0.5)
    resolution : int
        Circle resolution (default: 10)
    
    Returns:
    --------
    mesh_3d : dict
        {
            'points': (k, 3) - vertex positions,
            'polygons': (m, 3) - triangle connectivity,
            'point_values': (k,) - scalar values for coloring
        }
    """
    directional_variations = stats_3d['directional_variations']
    vectors = stats_3d['ensemble_spherical_vectors']
    min_vectors = stats_3d['min_vectors']
    median_vectors = stats_3d['median_vectors']
    max_vectors = stats_3d['max_vectors']
    glyph_markers = stats_3d['glyph_markers']
    num_of_glyphs = stats_3d['num_glyphs']
    
    num_positions = positions.shape[0]
    
    if point_values is None:
        scaler_values = np.zeros(num_positions)
    else:
        scaler_values = point_values
    
    # Call existing implementation
    points, polygons, points_values = squid_glyphs_meshing_3D(
        directional_variations, positions, vectors, min_vectors, median_vectors,
        max_vectors, scaler_values, glyph_markers, scale, resolution, num_of_glyphs
    )
    
    return {
        'points': points,
        'polygons': polygons,
        'point_values': points_values
    }


def _compute_ellipse_scale_and_angle(directional_variations, i_p):
    """Compute ellipse scale factor and rotation angle from PCA components."""
    v0_scale = directional_variations[i_p][0][0]
    v1_scale = directional_variations[i_p][0][1]
    
    if np.absolute(v0_scale) < 1.e-20:
        return 1.0, 0.001
    
    elipse_scale = np.maximum(v1_scale / v0_scale, 0.01)
    v0 = directional_variations[i_p][1]
    
    if np.absolute(v0[0]) < 1.e-16 and np.absolute(v0[1]) < 1.e-16:
        angle = 0.0
    else:
        angle = np.arctan2(v0[1], v0[0])
    
    return elipse_scale, angle


def _compute_superellipse_profile(r0, r1, angle, resolution):
    """Compute superellipse profile in 2D (x, y coordinates)."""
    phi_vals = np.linspace(0, 2 * np.pi, resolution)
    x0 = np.abs(np.cos(phi_vals))**(2/4) * np.sign(np.cos(phi_vals)) * r0
    y0 = np.abs(np.sin(phi_vals))**(2/4) * np.sign(np.sin(phi_vals)) * r1
    
    # Rotate by angle
    x = x0 * np.cos(angle) - y0 * np.sin(angle)
    y = x0 * np.sin(angle) + y0 * np.cos(angle)
    
    return x, y


def _create_rotation_matrix(median_vector):
    """Create 3D rotation matrix from spherical coordinates."""
    phi = median_vector[2]
    theta = median_vector[1]
    
    # Rotation around y-axis by theta (polar angle from z-axis)
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Rotation around z-axis by phi (azimuthal angle)
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry


def _add_circle_ring(points, points_values, points_id, x, y, z_offset, R, position, scale, scaler_value, resolution):
    """Add a circular ring of vertices to the mesh."""
    for i in range(resolution):
        pt = R @ np.array([x[i], y[i], z_offset])
        pt = pt * scale + position
        points[points_id + i] = pt
        points_values[points_id + i] = scaler_value
    return points_id + resolution


def _add_center_point(points, points_values, points_id, z_offset, R, position, scale, scaler_value):
    """Add a center point for a circle."""
    if z_offset == 0:
        pt = position
    else:
        pt = R @ np.array([0, 0, z_offset])
        pt = pt * scale + position
    
    points[points_id] = pt
    points_values[points_id] = scaler_value
    return points_id + 1


def _triangulate_disk(polygons, polygons_id, center_id, ring_start_id, resolution):
    """Triangulate a disk (center + outer ring)."""
    for i in range(resolution):
        polygons[polygons_id] = [ring_start_id + i, center_id, ring_start_id + (i + 1) % resolution]
        polygons_id += 1
    return polygons_id


def _triangulate_cylinder(polygons, polygons_id, bottom_ring_id, top_ring_id, resolution):
    """Triangulate a cylindrical surface between two rings."""
    for i in range(resolution):
        # Triangle 1
        polygons[polygons_id] = [bottom_ring_id + i, top_ring_id + i, top_ring_id + (i + 1) % resolution]
        polygons_id += 1
        # Triangle 2
        polygons[polygons_id] = [top_ring_id + (i + 1) % resolution, bottom_ring_id + (i + 1) % resolution, bottom_ring_id + i]
        polygons_id += 1
    return polygons_id


def squid_glyphs_meshing_3D(directional_variations, positions, vectors, min_vectors, 
                                    median_vectors, max_vectors, scaler_values, glyph_markers, scale, resolution, num_of_glyphs):
    """
    Build superelliptical squid glyphs for 3D visualization.
    
    Creates glyphs with these components:
    1. Base disk (at position)
    2. Body cylinder (from base to shoulder)
    3. Shaft (tapered section)
    4. Head cone (arrow tip)
    
    Parameters:
    -----------
    directional_variations : numpy.ndarray
        PCA components for cross-section shape
    positions : numpy.ndarray
        Glyph center positions (n, 3)
    vectors : numpy.ndarray
        Ensemble vectors (unused, for compatibility)
    min_vectors, median_vectors, max_vectors : numpy.ndarray
        Vector statistics in spherical coordinates
    scaler_values : numpy.ndarray
        Scalar values for coloring
    glyph_markers : numpy.ndarray
        Glyph type flags (0=none, 1=full, 2=arrow only)
    scale : float
        Overall glyph scale
    resolution : int
        Number of vertices per circle
    num_of_glyphs : int
        Number of full glyphs to create
    
    Returns:
    --------
    points, polygons, points_values : tuple
        Mesh geometry and scalar values
    """
    num_points = positions.shape[0]
    
    # Pre-allocate arrays
    points = np.zeros((num_of_glyphs * ((resolution + 1) * 3 + resolution * 2 + 1), 3))
    polygons = np.zeros((num_of_glyphs * (resolution * 8), 3), dtype=np.int32)
    points_values = np.zeros(num_of_glyphs * ((resolution + 1) * 3 + resolution * 2 + 1))
    
    points_id = 0
    polygons_id = 0
    
    for i_p in range(num_points):
        if glyph_markers[i_p] != 1:  # Skip if not a full glyph
            continue
        
        # Extract parameters
        position = positions[i_p]
        min_vector = min_vectors[i_p]
        median_vector = median_vectors[i_p]
        max_vector = max_vectors[i_p]
        scaler_value = scaler_values[i_p]
        
        # Compute ellipse parameters
        elipse_scale, angle = _compute_ellipse_scale_and_angle(directional_variations, i_p)
        
        # Compute rotation matrix
        R = _create_rotation_matrix(median_vector)
        
        # Base radius (at max_vector magnitude)
        r0_base = np.maximum(max_vector[0] * np.tan(max_vector[1] * 0.5), 0.01 * max_vector[0])
        r1_base = r0_base * elipse_scale
        x_base, y_base = _compute_superellipse_profile(r0_base, r1_base, angle, resolution)
        
        # 1. BASE DISK (bottom at position)
        base_ring_id = points_id
        points_id = _add_circle_ring(points, points_values, points_id, x_base, y_base, 0.0, 
                                     R, position, scale, scaler_value, resolution)
        base_center_id = points_id
        points_id = _add_center_point(points, points_values, points_id, 0.0, 
                                       R, position, scale, scaler_value)
        polygons_id = _triangulate_disk(polygons, polygons_id, base_center_id, base_ring_id, resolution)
        
        # 2. BODY TOP (at shoulder height = max_mag - min_mag)
        body_height = max_vector[0] - min_vector[0]
        body_top_ring_id = points_id
        points_id = _add_circle_ring(points, points_values, points_id, x_base, y_base, body_height,
                                     R, position, scale, scaler_value, resolution)
        body_top_center_id = points_id
        points_id = _add_center_point(points, points_values, points_id, body_height,
                                       R, position, scale, scaler_value)
        
        # Triangulate body cylinder
        polygons_id = _triangulate_cylinder(polygons, polygons_id, base_ring_id, body_top_ring_id, resolution)
        polygons_id = _triangulate_disk(polygons, polygons_id, body_top_center_id, body_top_ring_id, resolution)
        
        # 3. SHAFT BOTTOM (narrow at shoulder)
        shaft_base_r0 = np.maximum(min_vector[0] * np.tan(np.absolute(max_vector[1]) * 0.5), 0.01 * min_vector[0])
        shaft_base_r1 = shaft_base_r0 * elipse_scale
        x_shaft_base, y_shaft_base = _compute_superellipse_profile(shaft_base_r0, shaft_base_r1, angle, resolution)
        
        shaft_bottom_ring_id = points_id
        points_id = _add_circle_ring(points, points_values, points_id, x_shaft_base, y_shaft_base, body_height,
                                     R, position, scale, scaler_value, resolution)
        
        # 4. SHAFT TOP (at 0.8 * max_mag)
        shaft_top_r0 = np.maximum(0.2 * max_vector[0] * np.tan(np.absolute(max_vector[1]) * 0.5), 0.01 * 0.2 * max_vector[0])
        shaft_top_r1 = shaft_top_r0 * elipse_scale
        x_shaft_top, y_shaft_top = _compute_superellipse_profile(shaft_top_r0, shaft_top_r1, angle, resolution)
        
        shaft_top_ring_id = points_id
        points_id = _add_circle_ring(points, points_values, points_id, x_shaft_top, y_shaft_top, 0.8 * max_vector[0],
                                     R, position, scale, scaler_value, resolution)
        
        # Triangulate shaft
        polygons_id = _triangulate_cylinder(polygons, polygons_id, shaft_bottom_ring_id, shaft_top_ring_id, resolution)
        
        # 5. HEAD BASE (wide ring at 0.8 * max_mag)
        head_base_ring_id = points_id
        points_id = _add_circle_ring(points, points_values, points_id, x_base, y_base, 0.8 * max_vector[0],
                                     R, position, scale, scaler_value, resolution)
        head_base_center_id = points_id
        points_id = _add_center_point(points, points_values, points_id, 0.8 * max_vector[0],
                                       R, position, scale, scaler_value)
        polygons_id = _triangulate_disk(polygons, polygons_id, head_base_center_id, head_base_ring_id, resolution)
        
        # 6. HEAD TIP (cone point at max_mag)
        tip_id = points_id
        points_id = _add_center_point(points, points_values, points_id, max_vector[0],
                                       R, position, scale, scaler_value)
        
        # Triangulate cone
        for i in range(resolution):
            polygons[polygons_id] = [head_base_ring_id + i, tip_id, head_base_ring_id + (i + 1) % resolution]
            polygons_id += 1
    
    return points[:points_id], polygons[:polygons_id], points_values[:points_id]


