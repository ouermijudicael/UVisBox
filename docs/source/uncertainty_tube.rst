.. _uncertainty_tube_example:

Uncertainty Tube Example
------------------------

This example demonstrates how to generate and visualize uncertainty tubes for 3D trajectories using the ``uvisbox`` library.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from uvisbox.Datasets import flowmap_3d
    from uvisbox.Interpolations import linear_interpolate
    from uvisbox.UncertaintyTube import (
        generate_uncertainty_tube,
        generate_tube_mesh,
        plot_uncertainty_tube_from_mesh,
    )
    from uvisbox.Colors.colortree import ColorTree

    t0 = 0
    t1 = 5
    n_steps = 30
    number_of_seeds = 4  # Number of seed points

    # Scale factors for each seed
    scale = np.arange(number_of_seeds)
    scale = linear_interpolate(scale, 0, number_of_seeds-1, 1.5, 2.0)
    xy_scale = np.ones(number_of_seeds)
    xy_scale[1::2] = 0.1  # Set every second element to 0.1

    # Generate random seed points in 3D in [-1,1]^3
    seeds = np.random.uniform(-1, 1, (number_of_seeds, 3))

    # Generate trajectories
    trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)

    # Generate uncertainty tube cross-sections and eigenvalues
    cross_sections, eigen_values = generate_uncertainty_tube(trajectories, None, 16, e_proj=0.5, n_jobs=2)

    # Prepare eigenvalues for color mapping
    eigen_values = np.transpose(eigen_values, (1, 0, 2, 3))
    eigen_values = eigen_values.reshape((-1, 2)).astype(np.float32)
    max_eigen_values = eigen_values.max(axis=1)
    rescaled_max_eigen_values = linear_interpolate(
        max_eigen_values, max_eigen_values.min(), max_eigen_values.max(), 0.0, 1.0
    )
    eigen_values_ratio = np.nan_to_num(
        eigen_values[:, 1] / eigen_values[:, 0], nan=0.0, posinf=1.0, neginf=0.0
    )
    uv_coords = np.stack([rescaled_max_eigen_values, eigen_values_ratio], axis=1)

    # Generate tube mesh
    vertices, faces, mean_trajectories = generate_tube_mesh(trajectories, cross_sections, n_jobs=12)

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121, projection='3d')
    plot_uncertainty_tube_from_mesh(vertices, faces, mean_trajectories, uv_coords, axis=ax)

    # Show color map
    ax = fig.add_subplot(122)
    height, width = 100, 100
    value_grid = np.linspace(0, 1, width)[None, :]
    uncertainty_grid = np.linspace(0, 1, height)[:, None]
    image = np.stack(
        [uncertainty_grid * np.ones((height, width)), value_grid * np.ones((height, width))], axis=-1
    )
    colormap = ColorTree(depth=4, cmap="viridis", invert_u=True)
    colors = colormap(image, discrete=True)
    ax.imshow(colors, origin='lower', extent=(0, 1, 0, 1))
    ax.margins(50)
    ax.set_title("Color Map")
    ax.set_ylabel("Uncertainty")
    ax.set_xlabel("Symmetry")
    plt.tight_layout()
    plt.show()

**Notes:**

- The left subplot shows the uncertainty tube in 3D, colored by uncertainty and symmetry.
- The right subplot shows the color map used for visualization.