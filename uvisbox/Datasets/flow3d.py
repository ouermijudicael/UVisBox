import numpy as np
def abc_flow(t, x, a=1., b=1., c=1.):
    """
    Arnold Beltrami Childress flow

    """
    u = a * np.sin(x[:, 2]) + b * np.cos(x[:, 1])
    v = c * np.sin(x[:, 0]) + a * np.cos(x[:, 2])
    w = b * np.sin(x[:, 1]) + c * np.cos(x[:, 0])
    return u, v, w

def flowmap_3d(seed, t0, t1, n_steps, scale=1.0, xy_scale=1.0):
    number_of_seeds, num_dims = seed.shape
    n_samples = 100
    trajectories = np.zeros((n_steps+1, number_of_seeds, n_samples, num_dims))
    mean_trajectories = np.zeros((n_steps+1, number_of_seeds, num_dims))
    for j in range(n_samples):
        trajectories[0, :, j, :] = seed
    mean_trajectories[0] = seed

    t = np.linspace(t0, t1, n_steps+1)
    for i in range(1, n_steps+1):
        dt = t[i] - t[i-1]
        u, v, w = abc_flow(t[i-1], mean_trajectories[i-1, :, :])
        for j in range(n_samples):
            trajectories[i, :, j, 0] = trajectories[i-1,
                                                    :, j, 0] + dt * u + 0.005*np.random.randn() * scale * xy_scale
            trajectories[i, :, j, 1] = trajectories[i-1,
                                                    :, j, 1] + dt * v + 0.005*np.random.randn() * scale * xy_scale
            trajectories[i, :, j, 2] = trajectories[i-1,
                                                    :, j, 2] + dt * w + 0.005*np.random.randn() * scale
        mean_trajectories[i] = np.mean(trajectories[i], axis=1)

    return trajectories

def create_swirl_ensemble(center_line_start, center_line_end, num_curves=10, num_points=100, 
                         major_axis_length=1.0, minor_axis_length=0.3, swirl_frequency=2.0, 
                         distribution='uniform', noise_level=0.0, interior_fraction=0.5, 
                         expansion_factor=1.0):
    """
    Create an ensemble of curves that form elliptical cross-sections that swirl around a straight line in 3D.
    
    Parameters:
    -----------
    center_line_start : array-like, shape (3,)
        Starting point of the central straight line [x, y, z]
    center_line_end : array-like, shape (3,)
        Ending point of the central straight line [x, y, z]
    num_curves : int
        Number of curves to generate
    num_points : int
        Number of points per curve
    major_axis_length : float
        Length of the major axis of the ellipse
    minor_axis_length : float
        Length of the minor axis of the ellipse
    swirl_frequency : float
        How many times the ellipse rotates along the straight line
    distribution : str
        'uniform' for uniform distribution along ellipse perimeter, 
        'interior' for uniform distribution within ellipse interior,
        'mixed' for combination of both
    noise_level : float
        Amount of random noise to add to positions
    interior_fraction : float
        For 'mixed' distribution, fraction of curves in interior (0-1)
    expansion_factor : float
        Controls how much the ellipse expands. 0=no expansion, 1=linear expansion
    
    Returns:
    --------
    curves : list
        List of curves, each curve is an array of shape (num_points, 3)
    center_line : array, shape (num_points, 3)
        The central straight line
    """
    
    # Convert to numpy arrays
    start = np.array(center_line_start)
    end = np.array(center_line_end)
    
    # Create parameter t from 0 to 1
    t = np.linspace(0, 1, num_points)
    
    # Central straight line
    center_line = start[np.newaxis, :] + t[:, np.newaxis] * (end - start)[np.newaxis, :]
    
    # Direction vector of the central line
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    
    # Create two perpendicular vectors to the direction
    # Find a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])
    
    # Create orthonormal basis
    perp1 = np.cross(direction, temp)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    curves = []
    
    # Generate angles for all curves
    angles = np.random.uniform(0, 2*np.pi, num_curves)
    
    for i, base_angle in enumerate(angles):
        curve_points = []
        
        # Determine if this curve is on boundary or interior
        if distribution == 'uniform':
            # All curves on ellipse perimeter
            ellipse_radius = 1.0
        elif distribution == 'interior':
            # All curves randomly distributed inside ellipse
            ellipse_radius = np.sqrt(np.random.uniform(0, 1))
        elif distribution == 'mixed':
            # Mix of boundary and interior curves
            if i < num_curves * (1 - interior_fraction):
                ellipse_radius = 1.0  # On boundary
            else:
                ellipse_radius = np.sqrt(np.random.uniform(0, 1))  # Inside
        else:  # gaussian
            # Gaussian distribution within ellipse
            ellipse_radius = min(abs(np.random.normal(0, 0.3)), 1.0)
        
        for j, t_val in enumerate(t):
            # Calculate swirl angle at this t position
            swirl_angle = 2 * np.pi * swirl_frequency * t_val
            
            # Scale the ellipse size based on t (variation increases with t)
            scale_factor = 0.1 + expansion_factor * t_val
            
            # Ellipse parameters in local 2D coordinates with scaling
            ellipse_x = ellipse_radius * major_axis_length * np.cos(base_angle) * scale_factor
            ellipse_y = ellipse_radius * minor_axis_length * np.sin(base_angle) * scale_factor
            
            # Rotate the ellipse coordinates by the swirl angle
            rotated_x = ellipse_x * np.cos(swirl_angle) - ellipse_y * np.sin(swirl_angle)
            rotated_y = ellipse_x * np.sin(swirl_angle) + ellipse_y * np.cos(swirl_angle)
            
            # Add noise if specified
            if noise_level > 0:
                rotated_x += np.random.normal(0, noise_level)
                rotated_y += np.random.normal(0, noise_level)
            
            # Transform to 3D coordinates
            point_3d = (center_line[j] + 
                       rotated_x * perp1 + 
                       rotated_y * perp2)
            
            curve_points.append(point_3d)
        
        curve = np.array(curve_points)
        curves.append(curve)
    
    return curves, center_line