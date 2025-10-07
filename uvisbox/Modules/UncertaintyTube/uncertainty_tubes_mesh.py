import numpy as np



def generate_uncertainty_tube_mesh_2D(mean_trajectories, cross_sections):
    """
    Generate uncertainty tube mesh from mean trajectories and cross-sections.

    Parameters
    ----------
    mean_trajectories : np.ndarray
        Array of shape (n_trajectories, n_time_steps, 2) representing the mean trajectory.
    cross_sections : np.ndarray
        Array of shape (n_trajectories, n_time_steps, 2, 2) representing the cross-sections.

    Returns
    -------
    points: np.ndarray
        Array of shape (n_trajectories*n_time_steps*2, 2) representing the tube mesh vertices.
    tube_mesh : np.ndarray
        Array of shape (n_trajectories*n_time_steps*2, 3) representing the tube mesh faces.
    """
    n_trajectories, n_time_steps, _ = mean_trajectories.shape
    points = np.zeros((n_trajectories * n_time_steps * 2, 2))
    tube_mesh = np.zeros((n_trajectories * n_time_steps * 2, 3), dtype=int)
    i_point = 0
    i_face = 0
    for i_traj in range(n_trajectories):    
        for i_t in range(n_time_steps):
            if i_t == 0:
                points[i_point] = mean_trajectories[i_traj, i_t] + cross_sections[i_traj, i_t, 0]
                points[i_point + 1] = mean_trajectories[i_traj, i_t] - cross_sections[i_traj, i_t, 0]
                i_point += 2
            else:
                line_dir = mean_trajectories[i_traj, i_t] - mean_trajectories[i_traj, i_t - 1]
                line_dir = line_dir / np.linalg.norm(line_dir)  # Normalize direction
                perp_line_dir = np.array([-line_dir[1], line_dir[0]])  # Perpendicular direction
                # add point onto perp_line_dir direction passing through mean_trajectories[i_traj, i_t] 
                # with distance cross_sections[i_traj, i_t, 0] from mean_trajectories[i_traj, i_t]
                points[i_point] = mean_trajectories[i_traj, i_t] + cross_sections[i_traj, i_t, 0] * perp_line_dir
                points[i_point + 1] = mean_trajectories[i_traj, i_t] - cross_sections[i_traj, i_t, 0] * perp_line_dir
                # create faces
                tube_mesh[i_face] = [i_point - 2, i_point - 1, i_point]
                tube_mesh[i_face + 1] = [i_point - 1, i_point + 1, i_point]
                i_point += 2
                i_face += 2

    return points, tube_mesh


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
    
    Args:
        points (np.ndarray): Points to transform (shape: n_points × num_dims)
        best_shift (int): Number of positions to shift
        best_is_reversed (bool): Whether to reverse point order
        
    Returns:
        tuple:
            - aligned (np.ndarray): Transformed points
            - point_correspondence_map (np.ndarray): Index mapping from original to transformed points
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
    
    Args:
        points_ref (np.ndarray): Reference point set (shape: n_points × num_dims)
        points_target (np.ndarray): Target point set to align (shape: n_points × num_dims)
        stride (int): Stride for sampling subset of points (for efficiency)
        
    Returns:
        tuple: (aligned_points, correspondence_map)
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


def generate_tube_mesh(trajectories, ellipsoids, n_jobs=1):
    """
    Generate triangle mesh for uncertainty tubes with sequential or parallel alignment.
    
    Args:
        trajectories (np.ndarray): Ensemble trajectories with shape (n_steps, n_seeds, n_samples, num_dims)
        ellipsoids (np.ndarray): Cross-section boundary points with shape (n_steps, n_seeds, resolution, num_dims)
        n_jobs (int, optional): Number of parallel jobs to use. If n_jobs=1, uses sequential processing.
            If joblib is not available, falls back to sequential processing. Defaults to 1.
        
    Returns:
        tuple:
            - vertices (np.ndarray): Mesh vertices with shape (total_vertices, 3)
            - faces (np.ndarray): Triangle indices with shape (n_seeds, triangles_per_seed, 3)
            - mean_trajectories (np.ndarray): Mean trajectory paths
    """
    mean_trajectories = np.mean(trajectories, axis=2)
    n_steps, n_seeds, resolution, num_dims = ellipsoids.shape
    
    # Pre-allocate arrays based on mesh dimensions
    total_vertices, triangles_per_seed = calculate_mesh_dimensions(n_steps, n_seeds, resolution)
    vertices = np.empty((total_vertices, num_dims))
    faces = np.empty((n_seeds, triangles_per_seed, 3), dtype=np.int32)
    
    # Check if parallel processing is requested and available
    use_parallel = n_jobs != 1
    if use_parallel:
        try:
            import multiprocessing as mp
            # Ensure 'fork' context is available
            mp.get_context('fork')
        except (ImportError, ValueError):
            use_parallel = False
            print("Warning: multiprocessing with 'fork' context not available. Using sequential processing instead.")
    
    if use_parallel:
        # Process tubes in parallel using multiprocessing with 'fork' context
        pool_context = mp.get_context('fork')
        with pool_context.Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                _process_single_seed,
                [(ellipsoids, seed_idx, n_steps, resolution) for seed_idx in range(n_seeds)]
            )
        
        # Combine results
        vertex_idx = 0
        for seed_idx, (seed_ellipsoids, seed_vertex_count) in enumerate(results):
            seed_vertex_start = vertex_idx
            
            # Add vertices
            vertices[vertex_idx:vertex_idx + seed_vertex_count] = seed_ellipsoids.reshape(-1, num_dims)
            
            # Generate faces
            generate_seed_faces(faces, seed_idx, seed_vertex_start, n_steps, resolution)
            
            # Update vertex index
            vertex_idx += seed_vertex_count
    else:
        # Sequential processing (original code)
        vertex_idx = 0
        for seed_idx in range(n_seeds):
            seed_vertex_start = vertex_idx
            
            # Align and store all cross-sections for current seed
            seed_ellipsoids = align_cross_sections(ellipsoids, seed_idx, n_steps, resolution)
            
            # Add vertices for this seed
            vertex_idx = add_seed_vertices(vertices, seed_ellipsoids, vertex_idx, n_steps, resolution)
            
            # Generate faces between consecutive cross-sections
            generate_seed_faces(faces, seed_idx, seed_vertex_start, n_steps, resolution)
    
    return vertices, faces, mean_trajectories



def _process_single_seed(ellipsoids, seed_idx, n_steps, resolution):
    """
    Process a single seed's tube (used for parallel processing).
    
    Args:
        ellipsoids (np.ndarray): All cross-sections
        seed_idx (int): Current seed index
        n_steps (int): Number of time steps
        resolution (int): Number of points per cross-section
        
    Returns:
        tuple:
            - seed_ellipsoids (np.ndarray): Aligned cross-sections for this seed
            - vertex_count (int): Number of vertices for this seed
    """
    # Align cross-sections
    seed_ellipsoids = align_cross_sections(ellipsoids, seed_idx, n_steps, resolution)
    vertex_count = n_steps * resolution
    
    return seed_ellipsoids, vertex_count


def calculate_mesh_dimensions(n_steps, n_seeds, resolution):
    """Calculate dimensions needed for mesh arrays."""
    vertices_per_seed = n_steps * resolution
    total_vertices = n_seeds * vertices_per_seed
    triangles_per_seed = (n_steps - 1) * 2 * resolution
    return total_vertices, triangles_per_seed


def align_cross_sections(ellipsoids, seed_idx, n_steps, resolution):
    """Align cross-sections to minimize twisting for a given seed."""
    seed_ellipsoids = np.empty((n_steps, resolution, 3))
    seed_ellipsoids[0] = ellipsoids[0, seed_idx, :, :]  # First ellipsoid (no alignment needed)
    
    # Align subsequent cross-sections
    for j in range(1, n_steps):
        points0 = seed_ellipsoids[j-1]  # Use previously aligned ellipsoid as reference
        points1 = ellipsoids[j, seed_idx, :, :]
        points1_aligned, _ = circular_align_min_twist(points0, points1)
        seed_ellipsoids[j] = points1_aligned
    
    return seed_ellipsoids


def add_seed_vertices(vertices, seed_ellipsoids, vertex_idx, n_steps, resolution):
    """Add all vertices for a seed to the vertices array."""
    for step_idx in range(n_steps):
        vertices[vertex_idx:vertex_idx + resolution] = seed_ellipsoids[step_idx]
        vertex_idx += resolution
    return vertex_idx


def generate_seed_faces(faces, seed_idx, seed_vertex_start, n_steps, resolution):
    """Generate triangle faces between consecutive cross-sections."""
    # Use a clearer variable name
    face_offset = 0  # Offset within this seed's section of the faces array
    
    for step_idx in range(1, n_steps):
        # Vertex indices for ellipsoids at j-1 and j
        ellipsoid0_start = seed_vertex_start + (step_idx-1) * resolution
        ellipsoid1_start = seed_vertex_start + step_idx * resolution
        
        # Generate triangle indices
        res_indices = np.arange(resolution)
        
        # Generate and store triangles
        face_offset = add_segment_triangles(
            faces, seed_idx, face_offset,
            ellipsoid0_start, ellipsoid1_start,
            res_indices, resolution
        )


def add_segment_triangles(faces, seed_idx, face_offset, first_cross_section_start, second_cross_section_start, res_indices, resolution):
    """Add triangles connecting two consecutive cross-sections."""
    # Triangle 1 indices (connecting current point to next point and corresponding point on next cross-section)
    tri1_v0 = first_cross_section_start + res_indices
    tri1_v1 = first_cross_section_start + ((res_indices + 1) % resolution)
    tri1_v2 = second_cross_section_start + ((res_indices + 1) % resolution)
    
    # Triangle 2 indices (connecting current point to corresponding point on next cross-section and next point on next cross-section)
    tri2_v0 = first_cross_section_start + res_indices
    tri2_v1 = second_cross_section_start + ((res_indices + 1) % resolution)
    tri2_v2 = second_cross_section_start + res_indices
    
    # Store triangles in faces array
    n_triangles_segment = 2 * resolution
    
    faces[seed_idx, face_offset:face_offset + resolution, 0] = tri1_v0
    faces[seed_idx, face_offset:face_offset + resolution, 1] = tri1_v1
    faces[seed_idx, face_offset:face_offset + resolution, 2] = tri1_v2
    
    faces[seed_idx, face_offset + resolution:face_offset + n_triangles_segment, 0] = tri2_v0
    faces[seed_idx, face_offset + resolution:face_offset + n_triangles_segment, 1] = tri2_v1
    faces[seed_idx, face_offset + resolution:face_offset + n_triangles_segment, 2] = tri2_v2
    
    return face_offset + n_triangles_segment

