import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from uvisbox.Glyphs.squid_glyph import cartesian_to_spherical, compute_vector_depths_3D, uncertainty_squid_glyphs_3D
import matplotlib.pyplot as plt

# Generate a 3D grid of vectors. [vx,vy,vz]=[x,y,z]
grid_size = 3
vectors = []
for x in range(grid_size):
    for y in range(grid_size):
        for z in range(grid_size):
            vectors.append((x, y, z))

grid_points = np.array(vectors)


spherical_vectors = cartesian_to_spherical(np.array(vectors)) # Convert to spherical coordinates

# Add Gaussian noise to spherical vectors to create an ensemble
ensemble_size = 20 # Number of ensemble members
noise_std_dev = 0.1  # Standard deviation for Gaussian noise
ensemble_vectors = np.zeros((len(spherical_vectors), ensemble_size, 3)) # initialize ensemble vectors
for i, vec in enumerate(spherical_vectors):
    for j in range(ensemble_size):
        noise = np.random.normal(0, noise_std_dev, 3)
        noisy_vec = vec + noise
        ensemble_vectors[i, j] = noisy_vec
    


# Plot ensemble vectors using arrows
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for i, vec in enumerate(spherical_vectors):
    for j in range(ensemble_size):
        noisy_vec = ensemble_vectors[i, j]
        # Convert back to Cartesian coordinates for plotting
        r, theta, phi = noisy_vec
        vx = r * np.sin(theta) * np.cos(phi) + grid_points[i, 0]  # Adjust x to start from original point
        vy = r * np.sin(theta) * np.sin(phi) + grid_points[i, 1]  # Adjust y to start from original point
        vz = r * np.cos(theta) + grid_points[i, 2]  # Adjust z to start from original point
        ax.quiver(grid_points[i, 0], grid_points[i, 1], grid_points[i, 2], vx, vy, vz, color='g', alpha=0.3, arrow_length_ratio=0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ensemble Vectors in 3D')

# The line below computes the uncertainty squid glyphs and plots the uncertainty glyphs.
#  it takes in 
#  positions : numpy.ndarray
#         Array of shape (n, 3) The positions of the squid glyphs.
#     ensemble_vectors : numpy.ndarray
#         Array of shape (n, m, 3) The ensemble vectors in spherical coordinates.
#         The ensemble vectors for each position in Cartesian coordinates.
#     percentil1 : float
#         The first percentile for depth filtering.
#     percentil2 : float
#         The second percentile for depth filtering.
#     scale : float
#         The scale factor for the glyphs.
#     ax : matplotlib 3D axis
#         The axis to draw on. If None, a new figure and axis will be created.
# 
ax2 = uncertainty_squid_glyphs_3D(grid_points, ensemble_vectors, 0.5, 0.25, ax=ax2)

ax2.set_title('Uncertainty Squid Glyphs in 3D')
plt.savefig("3d_vector_field_with_squid_glyphs.png")
plt.show()