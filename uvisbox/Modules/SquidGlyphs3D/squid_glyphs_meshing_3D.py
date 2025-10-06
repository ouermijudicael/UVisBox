
import numpy as np

def squid_glyphs_meshing_3D(directional_variations, positions, vectors, min_vectors, 
                                    median_vectors, max_vectors, glyph_markers, scale, resolution, num_of_glyphs):
    """
    Build superelliptical squid glyphs for 3D visualization. Assumes vectors are in spherical coordinates (magnitude, theta, phi).

    Parameters:
    ----------
    directional_variations : numpy.ndarray
        Array of shape (n, 4, 2) where the 4 represents the
        (pca variance, pca first component, pca second component, pca mean) and the 2 represents
        the x and y components of the pca components.
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
    min_vectors : numpy.ndarray
        Array of shape (n, 3) The minimum vectors in spherical coordinates.
    median_vectors : numpy.ndarray
        Array of shape (n, 3) The median vectors in spherical coordinates.
    max_vectors : numpy.ndarray
        Array of shape (n, 3) The maximum vectors in spherical coordinates.
    glyph_markers : numpy.ndarray
        Array of shape (n,) with values 0 (no glyph), 1 (full glyph), 2 (arrow only)
    scale : float
        The scale factor for the glyphs.
    resolution : int
        The resolution of the base circle of the squid glyph.
    num_of_glyphs : int
        The number of glyphs to be created.

    Returns:
    -------
    points : numpy.ndarray
        Array of shape (m, 3) The points of the squid glyphs.
    polygons : numpy.ndarray
        Array of shape (k,) The polygon connectivity for the glyphs.
    """
    
    num_points = positions.shape[0]
    points = np.zeros((num_of_glyphs*((resolution + 1)*3 + resolution*2 + 1), 3))
    polygons = np.zeros((num_of_glyphs*((resolution)*8 ),3), dtype=np.int32)
    points_id = 0
    polygons_id = 0
    old_points_id = 0
    for i_p in range(num_points):
        if(glyph_markers[i_p] == 1):
            position = positions[i_p]
            v0_scale = directional_variations[i_p][0][0]
            v1_scale = directional_variations[i_p][0][1]
            # # elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
            if(np.absolute(v0_scale) < 1.e-20):
                elipse_scale = 1.0
                angle = 0.001 # min angle paramter
            else:

                elipse_scale = np.maximum(v1_scale/v0_scale, 0.01)
                v0 = vectors[i_p][1]
                v0 = directional_variations[i_p][1]
                angle = np.arctan2( v0[1], v0[0])
                # mean_vals = directional_variations[ii_t][i_p][3]
            min_vector = min_vectors[i_p]
            median_vector = median_vectors[i_p]
            max_vector = max_vectors[i_p]

            
            r0 = max_vector[0]*np.tan(max_vector[1]*0.5)
            r0 = np.maximum(r0, 0.01*max_vector[0])
            r1 = r0 * elipse_scale
            phi_vals = np.linspace(0, 2*np.pi, resolution)
            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
            phi = median_vector[2]
            theta = median_vector[1]
            
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

            ## mappoints to the position
            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], 0.0]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            pt = position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]

            center_id = points_id + resolution 
            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1
            old_points_id = points_id
            points_id = points_id + resolution + 1


            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], max_vector[0]-min_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            pt = np.dot(Ry, np.array([0, 0, max_vector[0]-min_vector[0]])) 
            pt = np.dot(Rz, pt)
            pt = pt*scale + position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]

            center_id = points_id + resolution

            for i in range(resolution):
                polygons[polygons_id, 0] = old_points_id + i
                polygons[polygons_id, 1] = points_id + i
                polygons[polygons_id, 2] = points_id + (i+1)%resolution
                polygons_id += 1
                polygons[polygons_id, 0] = points_id + (i+1)%resolution
                polygons[polygons_id, 1] = old_points_id + (i+1)%resolution
                polygons[polygons_id, 2] = old_points_id + i
                polygons_id += 1

            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1
            
            old_points_id = points_id
            points_id = points_id + resolution + 1



            # # # angle0 = np.arctan2(v0_scale, max_vector[0])
            # shaft_base_v0_scale = v0_scale/max_vector[0]*min_vector[0]
            # shaft_base_v1_scale = v1_scale/max_vector[0]*min_vector[0]
            shaft_base_r0 = min_vector[0]*np.tan(np.absolute(max_vector[1])*0.5)
            shaft_base_r0 = np.maximum(shaft_base_r0, 0.01*min_vector[0])
            shaft_base_r1 = shaft_base_r0 * elipse_scale

            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* shaft_base_r0
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* shaft_base_r1
            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], max_vector[0] - min_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]
            old_points_id = points_id
            points_id += resolution
            # shaft_top_v0_scale = v0_scale/max_vector[0]*(0.2*max_vector[0])
            # shaft_top_v1_scale = v1_scale/max_vector[0]*(0.2*max_vector[0])
            shaft_top_r0 = 0.2*max_vector[0]*np.tan(np.absolute(max_vector[1])*0.5)
            shaft_top_r0 = np.maximum(shaft_top_r0, 0.01*0.2*max_vector[0])
            shaft_top_r1 = shaft_top_r0 * elipse_scale

            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* shaft_top_r0
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* shaft_top_r1
            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]
            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], 0.8*max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            for i in range(resolution):
                polygons[polygons_id, 0] = old_points_id + i
                polygons[polygons_id, 1] = points_id + i
                polygons[polygons_id, 2] = points_id + (i+1)%resolution
                polygons_id += 1
                polygons[polygons_id, 0] = points_id + (i+1)%resolution
                polygons[polygons_id, 1] = old_points_id + (i+1)%resolution
                polygons[polygons_id, 2] = old_points_id + i
                polygons_id += 1

            old_points_id += points_id
            points_id += resolution

            x0 = np.abs(np.cos(phi_vals))**(2/4)*np.sign(np.cos(phi_vals))* r0 
            y0 = np.abs(np.sin(phi_vals))**(2/4)*np.sign(np.sin(phi_vals))* r1

            x = x0*np.cos(angle) - y0*np.sin(angle) #+ mean_vals[0]
            y = x0*np.sin(angle) + y0*np.cos(angle) #+ mean_vals[1]

            for i in range(resolution):
                pt = np.dot(Ry, np.array([x[i], y[i], 0.8*max_vector[0]]))
                pt = np.dot(Rz, pt)
                pt = pt*scale + position
                points[points_id + i, 0] = pt[0]
                points[points_id + i, 1] = pt[1]
                points[points_id + i, 2] = pt[2]

            pt = np.dot(Ry, np.array([0.0, 0.0, 0.8*max_vector[0]]))
            pt = np.dot(Rz, pt)
            pt = pt*scale + position
            points[points_id + resolution, 0] = pt[0]
            points[points_id + resolution, 1] = pt[1]
            points[points_id + resolution, 2] = pt[2]
            center_id = points_id + resolution
            for i in range(resolution):
                polygons[polygons_id,0] = points_id + i
                polygons[polygons_id,1] = center_id
                polygons[polygons_id,2] = points_id + (i+1)%resolution
                polygons_id += 1
            
            

            # cone tip
            # pt = np.dot(Ry, np.array([mean_vals[0], mean_vals[1], max_vector[0]]))
            pt = np.dot(Ry, np.array([0, 0, max_vector[0]]))
            pt = np.dot(Rz, pt)
            pt = pt*scale + position
            points[points_id + resolution+1, 0] = pt[0]
            points[points_id + resolution+1, 1] = pt[1]
            points[points_id + resolution+1, 2] = pt[2]
            tip_id = points_id + resolution + 1

            for i in range(resolution):
                polygons[polygons_id, 0] = points_id + i
                polygons[polygons_id, 1] = tip_id
                polygons[polygons_id, 2] = points_id + (i + 1)%resolution
                polygons_id += 1
            old_points_id = points_id
            points_id = points_id + resolution + 2

    return points, polygons


