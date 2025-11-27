import numpy as np

def expcos(x, e): return np.sign(np.cos(x)) * np.abs(np.cos(x))**e

def expsin(x, e): return np.sign(np.sin(x)) * np.abs(np.sin(x))**e

def compute_alignment_scores(reference, target, shifts):
    """
    Compute alignment scores for all possible shifts of target relative to reference.
    
    Args:
        reference (np.ndarray): Reference point set (shape: n_points × num_dims)
        target (np.ndarray): Target point set to align (shape: n_points × num_dims)
        shifts (np.ndarray): Array of shift values to try
        
    Returns:
        np.ndarray: Scores for each shift (lower is better)
    """
    # Create all shifted versions at once (shape: n_shifts, n_points, num_dims)
    all_shifted = np.array([np.roll(target, -shift, axis=0) for shift in shifts])
    
    # Vectorized distance computation
    distances = np.linalg.norm(reference[np.newaxis, :, :] - all_shifted, axis=2)
    return np.sum(distances, axis=1)  # Sum over points for each shift

def apply_circular_alignment(points, best_shift, best_is_reversed):
    """
    Apply circular alignment transformation to a set of points based on the calculated 
    shift and orientation.
    
    Parameters:
    ----------- 
    points (np.ndarray): 
        Points to transform (shape: n_points × num_dims)
    best_shift (int): 
        Number of positions to shift
    best_is_reversed (bool): 
        Whether to reverse point order
    
    Returns:
    --------
    aligned (np.ndarray): 
        Transformed points
    point_correspondence_map (np.ndarray): 
        Index mapping from original to transformed points
    """
    m = len(points)
    
    # Apply best alignment to points array
    points_oriented = points[::-1] if best_is_reversed else points.copy()
    aligned = np.roll(points_oriented, -best_shift, axis=0)
    
    # Create index mapping for correspondence
    indices_map = np.arange(m)
    if best_is_reversed:
        indices_map = indices_map[::-1]
    point_correspondence_map = np.roll(indices_map, -best_shift)

    return aligned, point_correspondence_map

def circular_align_min_twist(points_ref, points_target, stride=1):
    """
    Aligns two sets of points forming closed loops with minimal twist.
    
    Parameters:
    ----------- 
    points_ref (np.ndarray): 
        Reference point set (shape: n_points × num_dims)
    points_target (np.ndarray): 
        Target point set to align (shape: n_points × num_dims)
    stride (int): 
        Stride for sampling subset of points (for efficiency)
        
    Returns:
    ------- 
    aligned_points (np.ndarray): 
        Aligned target points
    correspondence_map (np.ndarray): 
        Index mapping from original to aligned points
    """
    # Check if stride is larger than array length
    if stride >= len(points_ref):
        stride = 1  # Fall back to using all points
        
    # Create subset of points based on stride
    indices = np.arange(0, len(points_ref), stride)
    ref_subset = points_ref[indices]
    target_subset = points_target[indices]
    n = len(ref_subset)
    
    # Edge case: empty subset
    if n == 0:
        return points_target.copy(), np.arange(len(points_target))

    # Create all possible shifts for both orientations
    possible_shifts = np.arange(n)

    # Find best orientation and shift
    scores_forward = compute_alignment_scores(ref_subset, target_subset, possible_shifts)
    scores_reverse = compute_alignment_scores(ref_subset, target_subset[::-1], possible_shifts)

    # Stack scores and find global minimum
    all_scores = np.vstack([scores_forward, scores_reverse])
    best_orientation_idx, best_shift_idx = np.unravel_index(np.argmin(all_scores), all_scores.shape)

    best_is_reversed = bool(best_orientation_idx)
    best_shift = possible_shifts[best_shift_idx]

    # Scale shift for full array - using modular arithmetic
    m = len(points_ref)
    best_shift_full = (best_shift * stride) % m
    
    # Apply the alignment using the helper function
    return apply_circular_alignment(points_target, best_shift_full, best_is_reversed)

def calculate_mesh_dimensions(n_steps, n_seeds, resolution):
    """Calculate dimensions needed for mesh arrays."""
    vertices_per_seed = n_steps * resolution
    total_vertices = n_seeds * vertices_per_seed
    triangles_per_seed = (n_steps - 1) * 2 * resolution
    return total_vertices, triangles_per_seed

def align_cross_sections(ellipsoids, seed_idx, n_steps, resolution):
    """Align cross-sections to minimize twisting for a given seed."""
    seed_ellipsoids = np.empty((n_steps, resolution, 3))
    
    seed_ellipsoids[0] = ellipsoids[0, seed_idx, :, :]

    for j in range(1, n_steps):
        points0 = seed_ellipsoids[j-1]
        points1 = ellipsoids[j, seed_idx, :, :]
        points1_aligned, _ = circular_align_min_twist(points0, points1)
        seed_ellipsoids[j] = points1_aligned
    
    return seed_ellipsoids

def _build_cross_section_3d(mean_point, eigvals, eigvecs, resolution=20, e_proj=1, sym=False):
    """
    Helper function to build the 3D superellipse for a single cross-section.
    
    Parameters:
    ----------- 
    mean_point (np.ndarray): The 3D center point for the cross-section.
    eigvals (np.ndarray): The two largest eigenvalues (2,).
    eigvecs (np.ndarray): The corresponding two eigenvectors (3, 2).
    resolution (int): Number of points for the superellipse.
    e_proj (float): Superellipse exponent.
    sym (bool): Symmetric flag for superellipse.
    
    Returns:
    --------
    np.ndarray: The 3D points of the superellipse cross-section (resolution, 3).
    """
    # Build 2D superellipse in its local coordinate system
    radii = 2.0 * np.sqrt(eigvals[:2])  # 2-sigma ellipse
    if sym:
        radii[1] = radii[0]  # Force symmetric shape if requested

    theta = np.linspace(0, 2 * np.pi, resolution+1)
    x = radii[0] * expcos(theta, e_proj)
    y = radii[1] * expsin(theta, e_proj)
    superellipse_2d = np.stack([x[:-1], y[:-1]], axis=1) # (resolution, 2)

    # Transform 2D superellipse to 3D space using eigenvectors and mean point
    primary_axis = eigvecs[:, 0]
    secondary_axis = eigvecs[:, 1]
    
    cross_section_3d = mean_point + superellipse_2d[:, 0, np.newaxis] * primary_axis + \
                       superellipse_2d[:, 1, np.newaxis] * secondary_axis
                       
    return cross_section_3d

def _add_seed_vertices_and_uvs(seed_ellipsoids, eigen_values_for_seed, n_steps, resolution, global_max_eigval):
    """
    Adds vertices and UVs for a single seed to the main arrays and returns them.
    U coordinate is the normalized magnitude of the max eigen value.
    V coordinate is the ratio of small eigen values to larger eigen values.
    """
    current_seed_vertices = seed_ellipsoids.reshape(-1, 3)
    current_seed_uvs = np.empty((n_steps * resolution, 2))

    for step_idx in range(n_steps):
        v_start = step_idx * resolution
        v_end = (step_idx + 1) * resolution

        max_eigval = np.max(eigen_values_for_seed[step_idx])
        min_eigval = np.min(eigen_values_for_seed[step_idx])
        
        u_coord = max_eigval / global_max_eigval if global_max_eigval > 0 else 0
        v_coord = min_eigval / max_eigval if max_eigval > 0 else 0

        current_seed_uvs[v_start:v_end, 0] = u_coord
        current_seed_uvs[v_start:v_end, 1] = v_coord

    return current_seed_vertices, current_seed_uvs

def _generate_seed_faces_logic(n_steps, resolution):
    """
    Generates triangle faces for a single seed, with local indexing.
    Adapted from original generate_seed_faces and add_segment_triangles.
    """
    total_segment_triangles = (n_steps - 1) * 2 * resolution
    seed_faces = np.empty((total_segment_triangles, 3), dtype=np.int32)
    face_offset = 0

    for step_idx in range(1, n_steps):
        ellipsoid0_start = (step_idx-1) * resolution
        ellipsoid1_start = step_idx * resolution
        
        res_indices = np.arange(resolution)
        
        tri1_v0 = ellipsoid0_start + res_indices
        tri1_v1 = ellipsoid0_start + ((res_indices + 1) % resolution)
        tri1_v2 = ellipsoid1_start + ((res_indices + 1) % resolution)
        
        tri2_v0 = ellipsoid0_start + res_indices
        tri2_v1 = ellipsoid1_start + ((res_indices + 1) % resolution)
        tri2_v2 = ellipsoid1_start + res_indices
        
        seed_faces[face_offset:face_offset + resolution, 0] = tri1_v0
        seed_faces[face_offset:face_offset + resolution, 1] = tri1_v1
        seed_faces[face_offset:face_offset + resolution, 2] = tri1_v2
        face_offset += resolution
        
        seed_faces[face_offset:face_offset + resolution, 0] = tri2_v0
        seed_faces[face_offset:face_offset + resolution, 1] = tri2_v1
        seed_faces[face_offset:face_offset + resolution, 2] = tri2_v2
        face_offset += resolution
            
    return seed_faces

def _process_single_seed_mesh(cross_sections_3d_all_seeds, eigen_values_all_seeds, seed_idx, n_steps, resolution, global_max_eigval):
    """
    Helper function for parallel mesh generation for a single seed.
    Returns vertices, faces, and uv_coords for this seed, with local indexing for faces.
    """
    seed_ellipsoids = align_cross_sections(cross_sections_3d_all_seeds, seed_idx, n_steps, resolution)
    eigen_values_for_seed = eigen_values_all_seeds[:, seed_idx, :]
    
    seed_vertices, seed_uv_coords = _add_seed_vertices_and_uvs(seed_ellipsoids, eigen_values_for_seed, n_steps, resolution, global_max_eigval)

    seed_faces = _generate_seed_faces_logic(n_steps, resolution)
    
    return seed_vertices, seed_faces, seed_uv_coords, seed_vertices.shape[0]

def uncertainty_tube_mesh(summary_statistics, resolution=20, e_proj=1, sym=False, n_jobs=1):
    """
    Generate the 3D mesh for uncertainty tubes.
    
    Parameters:
    ----------- 
    summary_statistics (dict): 
        Dictionary containing "mean_trajectory", "eigen_values", and "eigen_vectors".
    resolution (int, optional): 
        Number of points to sample on each cross-section boundary. Defaults to 20.
    e_proj (float, optional): 
        Exponent controlling the superellipse shape. e_proj=1 creates a standard ellipse.
        Defaults to 1.
    sym (bool, optional): 
        If True, forces the superellipse to be symmetric. Defaults to False.
    n_jobs (int, optional): 
        Number of parallel jobs to use. If n_jobs=1, uses sequential processing.
        Defaults to 1.
        
    Returns:
    --------
    dict:
        - "vertices" (np.ndarray): Tube mesh vertices.
        - "faces" (np.ndarray): Triangle indices.
        - "uv_coords" (np.ndarray): UV coordinates for texture mapping.
    """
    mean_trajectory = summary_statistics["mean_trajectory"]
    eigen_values = summary_statistics["eigen_values"]
    eigen_vectors = summary_statistics["eigen_vectors"]

    n_steps, n_starting_locations, _ = mean_trajectory.shape
    num_dims = mean_trajectory.shape[-1]

    cross_sections_3d = np.zeros((n_steps, n_starting_locations, resolution, num_dims))

    for location_index in range(n_starting_locations):
        cross_sections_3d[0, location_index, :, :] = mean_trajectory[0, location_index, :]

    for step_index in range(1, n_steps):
        for location_index in range(n_starting_locations):
            cross_sections_3d[step_index, location_index, :, :] = _build_cross_section_3d(
                mean_trajectory[step_index, location_index, :],
                eigen_values[step_index, location_index, :],
                eigen_vectors[step_index, location_index, :, :],
                resolution=resolution,
                e_proj=e_proj,
                sym=sym
            )
            
    vertices, faces, uv_coords = _generate_tube_mesh_from_cross_sections(cross_sections_3d, eigen_values, n_jobs)

    return {
        "vertices": vertices,
        "faces": faces,
        "uv_coords": uv_coords
    }

def _generate_tube_mesh_from_cross_sections(cross_sections_3d, eigen_values, n_jobs):
    """
    Generates the actual tube mesh (vertices, faces, uv_coords) from the 3D cross-sections.
    """
    n_steps, n_seeds, resolution, num_dims = cross_sections_3d.shape
    
    total_vertices, triangles_per_seed = calculate_mesh_dimensions(n_steps, n_seeds, resolution)
    total_faces = n_seeds * triangles_per_seed
    
    vertices = np.empty((total_vertices, num_dims))
    faces_combined = np.empty((total_faces, 3), dtype=np.int32)
    uv_coords = np.empty((total_vertices, 2))
    
    global_max_eigval = np.max(eigen_values) if eigen_values.size > 0 else 1

    use_parallel = n_jobs != 1
    if use_parallel:
        try:
            import multiprocessing as mp
            ctx = mp.get_context('fork')
            Pool = ctx.Pool
        except (ImportError, ValueError, NotImplementedError):
            use_parallel = False
            print("Warning: multiprocessing with 'fork' context not available. Using sequential processing instead.")

    if use_parallel:
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                _process_single_seed_mesh,
                [(cross_sections_3d, eigen_values, seed_idx, n_steps, resolution, global_max_eigval) for seed_idx in range(n_seeds)]
            )
        
        vertex_idx = 0
        face_idx = 0
        for seed_idx, (seed_vertices, seed_faces, seed_uv_coords, seed_vertex_count) in enumerate(results):
            seed_vertex_start = vertex_idx
            
            vertices[vertex_idx:vertex_idx + seed_vertex_count] = seed_vertices
            uv_coords[vertex_idx:vertex_idx + seed_vertex_count] = seed_uv_coords
            
            faces_combined[face_idx:face_idx + seed_faces.shape[0]] = seed_faces + seed_vertex_start
            
            vertex_idx += seed_vertex_count
            face_idx += seed_faces.shape[0]

    else: # Sequential processing
        vertex_idx = 0
        face_idx = 0
        for seed_idx in range(n_seeds):
            seed_vertex_start = vertex_idx
            
            seed_ellipsoids = align_cross_sections(cross_sections_3d, seed_idx, n_steps, resolution)
            eigen_values_for_seed = eigen_values[:, seed_idx, :]

            current_seed_vertices, current_seed_uvs = _add_seed_vertices_and_uvs(seed_ellipsoids, eigen_values_for_seed, n_steps, resolution, global_max_eigval)
            
            current_seed_faces = _generate_seed_faces_logic(n_steps, resolution) + seed_vertex_start

            vertices[vertex_idx : vertex_idx + current_seed_vertices.shape[0]] = current_seed_vertices
            uv_coords[vertex_idx : vertex_idx + current_seed_uvs.shape[0]] = current_seed_uvs
            faces_combined[face_idx : face_idx + current_seed_faces.shape[0]] = current_seed_faces

            vertex_idx += current_seed_vertices.shape[0]
            face_idx += current_seed_faces.shape[0]
            
    return vertices, faces_combined, uv_coords