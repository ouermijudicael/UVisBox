import numpy as np


def cone_glyphs_mesh(positions, stats_3d, point_values=None, scale=0.5, resolution=10):
    """
    Build 3D cone glyph mesh from statistics.
    
    Parameters:
    -----------
    positions : numpy.ndarray
        Shape (n, 3) - glyph positions
    stats_3d : dict
        From cone_glyphs_summary_statistics()
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
    if point_values is None:
        point_values = np.zeros(positions.shape[0])
    num_points = positions.shape[0]
    points = np.zeros((stats_3d['num_glyphs'] * ((resolution + 1) + 2), 3))
    polygons = np.zeros((stats_3d['num_glyphs'] * ((resolution) * 2), 3), dtype=np.int32)
    scalar_values = np.zeros((stats_3d['num_glyphs'] * ((resolution + 1) + 2)))

    glyph_types = stats_3d['glyph_types']
    min_vectors = stats_3d['spread_min_vectors']
    median_vectors = stats_3d['median_vectors']
    max_vectors = stats_3d['spread_max_vectors']

    points_id = 0
    polygons_id = 0
    old_points_id = 0
    for i_p in range(num_points):
        if(glyph_types[i_p] == 1):
            position = positions[i_p]
            min_vector = min_vectors[i_p]
            median_vector = median_vectors[i_p]
            max_vector = max_vectors[i_p]

            
            r0 = median_vector[0]*np.tan(max_vector[1]*0.5)
            r0 = np.maximum(r0, 0.01*max_vector[0])
            phi_vals = np.linspace(0, 2*np.pi, resolution)
            x = np.cos(phi_vals)* r0
            y = np.sin(phi_vals)* r0

            phi = median_vector[2]
            theta = median_vector[1]
           
            Rx = np.zeros((3, 3))
            Rx[0][0] = 1
            Rx[1][1] = np.cos(theta)
            Rx[1][2] = -np.sin(theta)
            Rx[2][1] = np.sin(theta)
            Rx[2][2] = np.cos(theta)

            Ry = np.zeros((3, 3))
            Ry[0][0] = np.cos(theta)
            Ry[0][2] = np.sin(theta)
            Ry[1][1] = 1
            Ry[2][0] = -np.sin(theta)
            Ry[2][2] = np.cos(theta)

            Rz = np.zeros((3, 3))
            Rz[0][0] = np.cos(phi)
            Rz[0][1] = -np.sin(phi)
            Rz[1][0] = np.sin(phi)
            Rz[1][1] = np.cos(phi)
            Rz[2][2] = 1
            R = Rz @ Ry 
            ## mappoints to the position
            for i in range(resolution):
                pt = np.dot(R, np.array([x[i], y[i], 0.0]))
                # pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]
                scalar_values[points_id + i] = point_values[i_p]

            pt = position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]
            scalar_values[points_id + resolution] = point_values[i_p]

            center_id = points_id + resolution 
            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1

            tip_id = points_id + resolution + 1
            pt = np.dot(R, np.array([0.0, 0.0, median_vector[0]]))
            pt = pt*scale + position
            points[tip_id, 0] = pt[0]
            points[tip_id, 1] = pt[1]
            points[tip_id, 2] = pt[2]
            scalar_values[tip_id] = point_values[i_p]

            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = tip_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1

            points_id = points_id + resolution + 2


    return {
        'points': points,
        'polygons': polygons,
        'point_values': scalar_values
    }


def cone_glyphs_meshing(positions, min_vectors, 
                                    median_vectors, max_vectors, scaler_values, glyph_markers, scale, resolution, num_of_glyphs):
    """
    Build superelliptical squid glyphs for 3D visualization. Assumes vectors are in spherical coordinates (magnitude, theta, phi).

    Parameters:
    -----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    min_vectors : numpy.ndarray
        Array of shape (n, 3) The minimum vectors in spherical coordinates.
    median_vectors : numpy.ndarray
        Array of shape (n, 3) The median vectors in spherical coordinates.
    max_vectors : numpy.ndarray
        Array of shape (n, 3) The maximum vectors in spherical coordinates.
    scaler_values : numpy.ndarray
        Array of shape (n,) The scaler values for each glyph.
    glyph_markers : numpy.ndarray
        Array of shape (n,) with values 0 (no glyph), 1 (full glyph), 2 (arrow only)
    scale : float
        The scale factor for the glyphs.
    resolution : int
        The resolution of the base circle of the squid glyph.
    num_of_glyphs : int
        The number of glyphs to be created.

    Returns:
    --------
    points : numpy.ndarray
        Array of shape (m, 3) The points of the squid glyphs.
    polygons : numpy.ndarray
        Array of shape (k,) The polygon connectivity for the glyphs.
    """
    
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*((resolution + 1)+ 2), 3))
    polygons = np.zeros((num_of_glyphs*((resolution)*2 ),3), dtype=np.int32)
    points_values = np.zeros((num_of_glyphs*((resolution + 1)+ 2)))
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    for i_p in range(num_points):
        if(glyph_markers[i_p] == 1):
            position = positions[i_p]
            # v0_scale = directional_variations[i_p][0][0]
            # v1_scale = directional_variations[i_p][0][1]
            # # # elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
            # if(np.absolute(v0_scale) < 1.e-20):
            #     elipse_scale = 1.0
            #     angle = 0.001 # min angle paramter
            # else:
            #     elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
            #     v0 = vectors[i_p][1]
            #     v0 = directional_variations[i_p][1]
            #     if (np.absolute(v0[0]) < 1.e-16 and np.absolute(v0[1]) < 1.e-16):
            #         angle = 0.0
            #     else:
            #         angle = np.arctan2( v0[1], v0[0])
            #     # mean_vals = directional_variations[ii_t][i_p][3]
            min_vector = min_vectors[i_p]
            median_vector = median_vectors[i_p]
            max_vector = max_vectors[i_p]

            
            r0 = max_vector[0]*np.tan(max_vector[1]*0.5)
            r0 = np.maximum(r0, 0.01*max_vector[0])
            phi_vals = np.linspace(0, 2*np.pi, resolution)
            x = np.cos(phi_vals)* r0
            y = np.sin(phi_vals)* r0

            phi = median_vector[2]
            theta = median_vector[1]
           
            Rx = np.zeros((3, 3))
            Rx[0][0] = 1
            Rx[1][1] = np.cos(theta)
            Rx[1][2] = -np.sin(theta)
            Rx[2][1] = np.sin(theta)
            Rx[2][2] = np.cos(theta)

            Ry = np.zeros((3, 3))
            Ry[0][0] = np.cos(theta)
            Ry[0][2] = np.sin(theta)
            Ry[1][1] = 1
            Ry[2][0] = -np.sin(theta)
            Ry[2][2] = np.cos(theta)

            Rz = np.zeros((3, 3))
            Rz[0][0] = np.cos(phi)
            Rz[0][1] = -np.sin(phi)
            Rz[1][0] = np.sin(phi)
            Rz[1][1] = np.cos(phi)
            Rz[2][2] = 1
            R = Rz @ Rx 
            ## mappoints to the position
            for i in range(resolution):
                pt = np.dot(R, np.array([x[i], y[i], 0.0]))
                # pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]
                points_values[points_id + i] = scaler_values[i_p]

            pt = position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]
            points_values[points_id + resolution] = scaler_values[i_p]

            center_id = points_id + resolution 
            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1

            tip_id = points_id + resolution + 1
            pt = np.dot(R, np.array([0.0, 0.0, median_vector[0]]))
            pt = pt*scale + position
            points[tip_id, 0] = pt[0]
            points[tip_id, 1] = pt[1]
            points[tip_id, 2] = pt[2]
            points_values[tip_id] = scaler_values[i_p]

            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = tip_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1

            points_id = points_id + resolution + 2

    return points, polygons, points_values


