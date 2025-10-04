

import numpy as np

def squid_glyphs_meshing_2D(positions, ensemble_polar_vectors, vector_depths, percentil1, scale=0.2):
    """
    Build squid glyphs for 2D visualization. Assumes vectors are in polar coordinates (magnitude, angle).
    
    Parameters:
    ----------
    positions : numpy.ndarray
        Array of shape (n, 2) The positions of the squid glyphs.
    ensemble_polar_vectors : numpy.ndarray
        Array of shape (n, m, 2) The ensemble polar vectors for each position.
    vector_depths : numpy.ndarray
        Array of shape (n, m) The vector depths for each position.
    percentil1 : float
        The first percentile for depth filtering.
    scale : float
        The scale factor for the glyphs.
    
    Returns:
    -------
    glyphs_points : numpy.ndarray
        Array of shape (k, 2) The points of the squid glyphs.
    glyphs_polygons : numpy.ndarray
        Array of shape (k, 3) The polygons of the squid glyphs.
    """
    num_positions = ensemble_polar_vectors.shape[0]

    glyphs_points = np.zeros((num_positions*11, 2))
    glyphs_polygons = np.zeros((num_positions*5, 3), dtype=int)
    tri_idx = 0
    point_idx = 0
    for i_pos in range(num_positions):

        median_idx = np.argmax(vector_depths[i_pos])
        median_vector = ensemble_polar_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0])
        indices = np.where(vector_depths[i_pos] >= 1.0 - percentil1)[0]
        min_mag = np.min(ensemble_polar_vectors[i_pos][indices][:, 0]) if indices.size > 0 else 0
        max_mag = np.max(ensemble_polar_vectors[i_pos][indices][:, 0]) if indices.size > 0 else 0
        min_angle = np.min(ensemble_polar_vectors[i_pos][indices][:, 1]) if indices.size > 0 else 0
        max_angle = np.max(ensemble_polar_vectors[i_pos][indices][:, 1]) if indices.size > 0 else 0
        
        # rotate all angles by 90-mid_angle so the median vector aligns with the y-axis
        #  and project all vectors onto the x-axis
        # x_projection = ensemble_polar_vectors[i_pos][indices, 0] * np.cos(ensemble_polar_vectors[i_pos][indices, 1] - np.radians(mid_angle[i_pos]))
    
        delta_h = max_mag -min_mag
        h = median_vector[0]

        rad_angle = (max_angle - min_angle)*0.5
        rot_angle = median_vector[1]
        base = 2* np.arctan(rad_angle)*max_mag
        # build 2D squid glyph triangulation
        if (base > 1e-5) and (delta_h > 1e-5):
            # base bottom left
            pt = np.array([- base * 0.5, 0]) * scale
            pt = np.array([pt[0]*np.cos(rot_angle) - pt[1]*np.sin(rot_angle),
                           pt[0]*np.sin(rot_angle) + pt[1]*np.cos(rot_angle)])
            pt = pt + positions[i_pos]
            
            # base bottom right
            pt1 = np.array([base * 0.5, 0]) * scale
            pt1 = np.array([pt1[0]*np.cos(rot_angle) - pt1[1]*np.sin(rot_angle),
                            pt1[0]*np.sin(rot_angle) + pt1[1]*np.cos(rot_angle)])
            pt1 = pt1 + positions[i_pos]

            # base top left
            pt2 = np.array([- base * 0.5, delta_h]) * scale
            pt2 = np.array([pt2[0]*np.cos(rot_angle) - pt2[1]*np.sin(rot_angle),
                            pt2[0]*np.sin(rot_angle) + pt2[1]*np.cos(rot_angle)])
            pt2 = pt2 + positions[i_pos]
            # base top right
            pt3 = np.array([base * 0.5, delta_h]) * scale
            pt3 = np.array([pt3[0]*np.cos(rot_angle) - pt3[1]*np.sin(rot_angle),
                            pt3[0]*np.sin(rot_angle) + pt3[1]*np.cos(rot_angle)])
            pt3 = pt3 + positions[i_pos]
            glyphs_points[point_idx] = pt
            glyphs_points[point_idx+1] = pt1
            glyphs_points[point_idx+2] = pt2
            glyphs_points[point_idx+3] = pt3
            glyphs_polygons[tri_idx] = [point_idx, point_idx+1, point_idx+2]
            glyphs_polygons[tri_idx+1] = [point_idx+1, point_idx+3, point_idx+2]
            tri_idx += 2
            point_idx += 4

            # shaft bottom left
            shaft_pt = np.array([- np.arctan(rad_angle)*h, delta_h]) * scale
            shaft_pt = np.array([shaft_pt[0]*np.cos(rot_angle) - shaft_pt[1]*np.sin(rot_angle),
                                 shaft_pt[0]*np.sin(rot_angle) + shaft_pt[1]*np.cos(rot_angle)])
            shaft_pt = shaft_pt + positions[i_pos]
            # shaft bottom right
            shaft_pt1 = np.array([np.arctan(rad_angle)*h, delta_h]) * scale
            shaft_pt1 = np.array([shaft_pt1[0]*np.cos(rot_angle) - shaft_pt1[1]*np.sin(rot_angle),
                                  shaft_pt1[0]*np.sin(rot_angle) + shaft_pt1[1]*np.cos(rot_angle)])
            shaft_pt1 = shaft_pt1 + positions[i_pos]
            # shaft top left
            shaft_pt2 = np.array([- np.arctan(rad_angle)*max_mag*0.2, max_mag*0.8]) * scale
            shaft_pt2 = np.array([shaft_pt2[0]*np.cos(rot_angle) - shaft_pt2[1]*np.sin(rot_angle),
                                  shaft_pt2[0]*np.sin(rot_angle) + shaft_pt2[1]*np.cos(rot_angle)])
            shaft_pt2 = shaft_pt2 + positions[i_pos]
            # shaft top right
            shaft_pt3 = np.array([np.arctan(rad_angle)*max_mag*0.2, max_mag*0.8]) * scale
            shaft_pt3 = np.array([shaft_pt3[0]*np.cos(rot_angle) - shaft_pt3[1]*np.sin(rot_angle),
                                  shaft_pt3[0]*np.sin(rot_angle) + shaft_pt3[1]*np.cos(rot_angle)])
            shaft_pt3 = shaft_pt3 + positions[i_pos]
            glyphs_points[point_idx] = shaft_pt
            glyphs_points[point_idx+1] = shaft_pt1
            glyphs_points[point_idx+2] = shaft_pt2
            glyphs_points[point_idx+3] = shaft_pt3      
            glyphs_polygons[tri_idx] = [point_idx, point_idx+1, point_idx+2]
            glyphs_polygons[tri_idx+1] = [point_idx+1, point_idx+3, point_idx+2]
            tri_idx += 2
            point_idx += 4

            # head bottom left
            head_pt = np.array([- base *0.5, max_mag*0.8]) * scale
            head_pt = np.array([head_pt[0]*np.cos(rot_angle) - head_pt[1]*np.sin(rot_angle),
                               head_pt[0]*np.sin(rot_angle) + head_pt[1]*np.cos(rot_angle)])
            head_pt = head_pt + positions[i_pos]
            # head bottom right
            head_pt1 = np.array([base *0.5, max_mag*0.8]) * scale
            head_pt1 = np.array([head_pt1[0]*np.cos(rot_angle) - head_pt1[1]*np.sin(rot_angle),
                                 head_pt1[0]*np.sin(rot_angle) + head_pt1[1]*np.cos(rot_angle)])
            head_pt1 = head_pt1 + positions[i_pos]
            # head top (tip)
            head_pt2 = np.array([0, max_mag]) * scale
            head_pt2 = np.array([head_pt2[0]*np.cos(rot_angle) - head_pt2[1]*np.sin(rot_angle),
                                 head_pt2[0]*np.sin(rot_angle) + head_pt2[1]*np.cos(rot_angle)])
            head_pt2 = head_pt2 + positions[i_pos]
            glyphs_points[point_idx] = head_pt
            glyphs_points[point_idx+1] = head_pt1
            glyphs_points[point_idx+2] = head_pt2
            glyphs_polygons[tri_idx] = [point_idx, point_idx+1, point_idx+2]
            tri_idx += 1
            point_idx += 3  

    return glyphs_points[:point_idx], glyphs_polygons[:tri_idx]

