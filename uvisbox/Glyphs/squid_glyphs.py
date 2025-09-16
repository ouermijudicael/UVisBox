import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pyvista as pv
from .vector_depths import compute_vector_depths_3D, cartesian_to_spherical, calculate_spread_3D
from .vector_depths import getDirectionalVariations, compute_vector_depths_2D, cartesian_to_polar


def build_squid_glyphs_mesh_3D(directional_variations, positions, vectors, min_vectors, 
                                    median_vectors, max_vectors, glyph_markers, scale, resolution, num_of_glyphs):
    """
    Build superelliptical squid glyphs for 3D visualization. Assumes vectors are in spherical coordinates (magnitude, theta, phi).

    Parameters
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

    Returns
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

def plot_uncertainty_squid_glyphs_3D(points, triangles, ax=None, show_edges=True, glyph_color='lightblue'):
    """
    Plots the 3D squid glyphs using pyvista.
    Parameters:
    ----------
    points : numpy.ndarray
        Array of shape (m, 3) The points of the squid glyphs.
    triangles : numpy.ndarray
        Array of shape (n, 3) The triangle connectivity of the squid glyphs.
    ax : pyvista.Plotter, optional
        The pyvista plotter to use. If None, a new plotter will be created.
    show_edges : bool, optional
        Whether to show edges of the glyphs. Default is True.
    glyph_color : str, optional
        The color of the glyphs. Default is 'lightblue'.

    Returns:
    -------
    ax : pyvista.Plotter
        The pyvista plotter with the drawn squid glyphs.
    """
    triangles = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
    triangles_flat = triangles.reshape(-1)
    mesh = pv.PolyData(points, triangles_flat) # 
    if ax is None:
        ax = pv.Plotter()
    ax.add_mesh(mesh, color=glyph_color, show_edges=show_edges)
    ax.add_axes()
    ax.set_background('white')
    ax.show_grid()
    ax.show_axes()
    ax.show(title="3D Uncertainty Squid Glyphs")

    return ax


def uncertainty_squid_glyphs_3D(positions, ensemble_vectors, percentil, scale=0.5, 
                                show_edges=True, glyph_color='lightblue'):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 3D.
    Parameters:
    ----------
    positions : numpy.ndarray
        Array of shape (n, 3) The positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
        The ensemble vectors for each position in Cartesian coordinates.
    percentil : float
        The first percentile for depth filtering.
    scale : float, optional
        The scale factor for the glyphs. Default is 0.5.
    show_edges : bool, optional
        Whether to show edges of the glyphs. Default is True.
    glyph_color : str, optional
        The color of the glyphs. Default is 'lightblue'.

    Returns:
    -------
    plotter : pyvista.Plotter
        The pyvista plotter with the drawn squid glyphs.
    points : numpy.ndarray
        The points of the squid glyphs.
    polygons : numpy.ndarray
        The polygon connectivity for the glyphs.
    """

    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_spherical_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_spherical_vectors[i] = cartesian_to_spherical(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_3D(ensemble_spherical_vectors[i])

    min_vectors = np.zeros((num_positions, 3))
    max_vectors = np.zeros((num_positions, 3))
    median_vectors = np.zeros((num_positions, 3))
    glyph_markers = np.zeros((num_positions,), dtype=int)
    numb_of_glyphs = 0
    num_arrows = 0
    for i_pos in range(num_positions):
        # find indices where depth > 1.0-percentil
        median_idx, min_mag_idx, max_mag_idx, min_theta_idx, max_theta_idx, min_phi_idx, max_phi_idx = calculate_spread_3D(ensemble_spherical_vectors[i_pos], depths[i_pos], percentil)
        median_vectors[i_pos] = ensemble_spherical_vectors[i_pos][median_idx] if median_idx is not None else np.array([0,0,0])
        min_mag_temp = ensemble_spherical_vectors[i_pos][min_mag_idx] if min_mag_idx is not None else np.array([0,0,0])
        max_mag_temp = ensemble_spherical_vectors[i_pos][max_mag_idx] if max_mag_idx is not None else np.array([0,0,0])
        min_phi = ensemble_spherical_vectors[i_pos][min_phi_idx][2] if min_phi_idx is not None else 0
        max_phi = ensemble_spherical_vectors[i_pos][max_phi_idx][2] if max_phi_idx is not None else 0
        min_theta = ensemble_spherical_vectors[i_pos][min_theta_idx][1] if min_theta_idx is not None else 0
        max_theta = ensemble_spherical_vectors[i_pos][max_theta_idx][1] if max_theta_idx is not None else 0
        min_vectors[i_pos] = np.array([min_mag_temp[0], min_theta, min_phi])
        max_vectors[i_pos] = np.array([max_mag_temp[0], max_theta, max_phi])
        if (np.absolute(max_vectors[i_pos][0]-min_vectors[i_pos][0]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][1]-min_vectors[i_pos][1]) > 1e-5) and \
           (np.absolute(max_vectors[i_pos][2]-min_vectors[i_pos][2]) > 1e-5):
            glyph_markers[i_pos] = 1
            numb_of_glyphs += 1
        elif(max_vectors[i_pos][0] > 1e-3):
            glyph_markers[i_pos] = 2
            num_arrows += 1


    # compute directional variations
    depth_threshold = 1.0 - percentil
    directional_variations = getDirectionalVariations(ensemble_vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors)
    
    # build squid glyphs
    points, polygons = build_squid_glyphs_mesh_3D(directional_variations, positions, ensemble_spherical_vectors, min_vectors, median_vectors, max_vectors, glyph_markers, scale, resolution=10, num_of_glyphs=numb_of_glyphs)
    
   
    return plotter, points, polygons


def build_squid_glyphs_mesh_2D(positions, ensemble_polar_vectors, vector_depths,percentil1, scale=0.2):
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


def plot_uncertainty_squid_glyphs_2D(glyphs_points, glyphs_polygons, ax=None):
    """
    Plots the squid glyphs in 2D using matplotlib.
    Parameters
    ----------
    glyphs_points : numpy.ndarray
        Array of shape (k, 2) The points of the squid glyphs.
    glyphs_polygons : numpy.ndarray
        Array of shape (m, 3) The polygons of the squid glyphs.
    ax : matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.
    Returns
    -------
    ax : matplotlib axis
        The axis with the drawn squid glyphs.
    """
   
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    tri_colors = np.ones((glyphs_polygons.shape[0]))*0.8
    ax.tripcolor(glyphs_points[:, 0], glyphs_points[:, 1], glyphs_polygons, facecolors=tri_colors, cmap='RdBu_r',)
   
    return ax


def uncertainty_squid_glyphs_2D(positions, ensemble_vectors, percentil1, scale=0.2, ax=None):
    """
    Draws uncertainty squid glyphs for the given positions and ensemble vectors in 2D.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the squid glyphs.
    ensemble_vectors : numpy.ndarray
        Array of shape (n, m, 2) representing the ensemble vectors for each position.
    percentil1 : float
        The first percentile for depth filtering.
    scale : float
        The scale factor for the glyphs.
    ax : matplotlib axis
        The axis to draw on. If None, a new figure and axis will be created.

    Returns
    -------
    ax : matplotlib axis
        The axis with the drawn squid glyphs.
    """
    num_positions, num_ens_members =ensemble_vectors.shape[0], ensemble_vectors.shape[1]
    
    # Convert ensemble_vectors to spherical coordinates
    ensemble_polar_vectors = np.zeros_like(ensemble_vectors)
    for i in range(num_positions):
        ensemble_polar_vectors[i] = cartesian_to_polar(ensemble_vectors[i])

    # Ccalculate vector depths in spherical coordinates
    depths = np.zeros((num_positions, num_ens_members))
    for i in range(num_positions):
        depths[i] = compute_vector_depths_2D(ensemble_polar_vectors[i])

    # build squid glyphs
    glyphs_points, glyphs_polygons = build_squid_glyphs_mesh_2D(positions, ensemble_polar_vectors, depths, percentil1, scale)
    
    # plot squid glyphs
    ax = plot_uncertainty_squid_glyphs_2D(glyphs_points, glyphs_polygons, ax)

    return ax

