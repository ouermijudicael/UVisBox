.. _squid_glyphs_3d_example:

Squid Glyph 3D Example
----------------------

This example demonstrates how to visualize 3D vector fields and their uncertainty using squid glyphs with the ``uvisbox`` library.

.. code-block:: python

    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from uvisbox.Glyphs.squid_glyph import (
        cartesian_to_spherical,
        compute_vector_depths_3D,
        uncertainty_squid_glyphs_3D,
    )
    import matplotlib.pyplot as plt

    def generate_3d_grid_vectors():
        grid_size = 10
        vectors = []
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    vectors.append((x, y, z))
        return vectors

    if __name__ == "__main__":
        vectors = generate_3d_grid_vectors()

        # 3D visualization of the vectors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_coords, y_coords, z_coords = zip(*vectors)
        ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
        for x, y, z in vectors:
            ax.quiver(x, y, z, x + x, y + y, z + z, color='r', arrow_length_ratio=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        # Convert to spherical coordinates
        spherical_vectors = cartesian_to_spherical(np.array(vectors))

        # Create an ensemble with Gaussian noise
        ensemble_size = 20
        noise_std_dev = 0.1
        ensemble_vectors = np.zeros((len(spherical_vectors), ensemble_size, 3))
        for i, vec in enumerate(spherical_vectors):
            for j in range(ensemble_size):
                noise = np.random.normal(0, noise_std_dev, 3)
                ensemble_vectors[i, j] = vec + noise

        # Compute vector depths
        vector_depths = np.zeros((len(spherical_vectors), ensemble_size))
        for i in range(len(spherical_vectors)):
            vector_depths[i] = compute_vector_depths_3D(ensemble_vectors[i])

        # Plot ensemble vectors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, vec in enumerate(spherical_vectors):
            for j in range(ensemble_size):
                r, theta, phi = ensemble_vectors[i, j]
                vx = r * np.sin(theta) * np.cos(phi) + x_coords[i]
                vy = r * np.sin(theta) * np.sin(phi) + y_coords[i]
                vz = r * np.cos(theta) + z_coords[i]
                ax.quiver(x_coords[i], y_coords[i], z_coords[i], vx, vy, vz, color='g', alpha=0.3, arrow_length_ratio=0.05)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        # Visualize uncertainty using squid glyphs
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2 = uncertainty_squid_glyphs_3D(np.array(vectors), ensemble_vectors, 0.5, 0.25, ax=ax2)
        plt.title("Uncertainty Squid Glyphs in 3D")
        plt.show()

**Explanation:**

- Generates a 3D grid of vectors.
- Visualizes the vectors as points and arrows.
- Converts vectors to spherical coordinates and creates an ensemble with Gaussian noise.
- Computes vector depths for uncertainty quantification.
- Visualizes the ensemble and uncertainty using squid glyphs.

For more details, see the ``uvisbox.Glyphs.squid_glyph`` module documentation.