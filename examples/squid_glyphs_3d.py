import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from uvisbox.Glyphs.squid_glyph import cartesian_to_spherical, compute_vector_depths_3D, uncertainty_squid_glyphs_3D


def generate_3d_grid_vectors():
    """
    Generate a 3D grid of vectors.
    """
    grid_size = 10
    vectors = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                vectors.append((x, y, z))
    return vectors

if __name__ == "__main__":
    vectors = generate_3d_grid_vectors()
    # print("Generated 3D grid vectors:")
    # for vector in vectors:
    #     print(vector)
    # 3D visualization of the vectors using matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from vectors
    x_coords, y_coords, z_coords = zip(*vectors)
    

    # Plot the vectors as points
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
    # Add arrows to represent vectors
    for x, y, z in vectors:
        ax.quiver(x, y, z, x+x, y+y, z+z, color='r', arrow_length_ratio=0.1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

    # Convert to spherical coordinates
    spherical_vectors = cartesian_to_spherical(np.array(vectors))
    # print("Cartesian and Spherical Coordinates:")
    # for cartesian, spherical in zip(vectors, spherical_vectors):
    #     print(f"Cartesian: {cartesian}, Spherical: (r={spherical[0]:.2f}, theta={spherical[1]:.2f}, phi={spherical[2]:.2f})")

    # Add Gaussian noise to spherical vectors to create an ensemble
    ensemble_size = 20
    noise_std_dev = 0.1  # Standard deviation for Gaussian noise

    ensemble_vectors = np.zeros((len(spherical_vectors), ensemble_size, 3))
    for i, vec in enumerate(spherical_vectors):
        for j in range(ensemble_size):
            noise = np.random.normal(0, noise_std_dev, 3)
            noisy_vec = vec + noise
            ensemble_vectors[i, j] = noisy_vec
        
    # Compute vector depths for the ensemble
    vector_depths = np.zeros((len(spherical_vectors), ensemble_size))
    for i in range(len(spherical_vectors)):
        vector_depths[i] = compute_vector_depths_3D(ensemble_vectors[i])

    # print("Vector Depths for each ensemble member at each point:")
    # for i, depths in enumerate(vector_depths):
    #     print(f"Point {i} depths: {depths}")
    # # Plot ensemble vectors using arrows
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, vec in enumerate(spherical_vectors):
        for j in range(ensemble_size):
            noisy_vec = ensemble_vectors[i, j]
            # Convert back to Cartesian coordinates for plotting
            r, theta, phi = noisy_vec
            vx = r * np.sin(theta) * np.cos(phi) + x_coords[i]  # Adjust x to start from original point
            vy = r * np.sin(theta) * np.sin(phi) + y_coords[i]  # Adjust y to start from original point
            vz = r * np.cos(theta) + z_coords[i]  # Adjust z to start from original point
            ax.quiver(x_coords[i], y_coords[i], z_coords[i], vx, vy, vz, color='g', alpha=0.3, arrow_length_ratio=0.05)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

    # Visualize uncertainty using squid glyphs
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2 = uncertainty_squid_glyphs_3D(np.array(vectors), ensemble_vectors, 0.5, 0.25, ax=ax2)
    plt.title("Uncertainty Squid Glyphs in 3D")
    plt.show()
