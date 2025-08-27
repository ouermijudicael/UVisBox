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