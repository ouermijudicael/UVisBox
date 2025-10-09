"""
This example demonstrates how to visualize 3D vector fields and their uncertainty using squid glyphs with the ``uvisbox`` library.
The example generates a 3D grid of vectors, creates an ensemble by adding Gaussian noise, and then visualizes the ensemble using squid glyphs to represent uncertainty.

Import necessary libraries

.. code-block:: python

    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from uvisbox.Core.BandDepths.vector_depths import cartesian_to_spherical
    from uvisbox.Modules.SquidGlyphs import squid_glyph_3D
    import pyvista as pv

.. code-block:: python

    # Generate a 3D grid of vectors. [vx,vy,vz]=[x,y,z]
    grid_size = 3
    vectors = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                vectors.append((x, y, z))

    grid_points = np.array(vectors)
    # Convert to spherical coordinates
    spherical_vectors = cartesian_to_spherical(np.array(vectors)) 

    # Add Gaussian noise to spherical vectors to create an ensemble
    ensemble_size = 20 # Number of ensemble members
    noise_std_dev = 0.1  # Standard deviation for Gaussian noise
    ensemble_vectors = np.zeros((len(spherical_vectors), ensemble_size, 3)) # initialize ensemble vectors
    for i, vec in enumerate(spherical_vectors):
        for j in range(ensemble_size):
            noise = np.random.normal(0, noise_std_dev, 3)
            noisy_vec = vec + noise
            ensemble_vectors[i, j] = noisy_vec
            # Convert back to Cartesian coordinates for plotting
            r, theta, phi = noisy_vec
            vx = r * np.sin(theta) * np.cos(phi)
            vy = r * np.sin(theta) * np.sin(phi)
            vz = r * np.cos(theta)
            ensemble_vectors_cartesian[i, j] = np.array([vx, vy, vz])

Convert back to cartesian vectors for plotting with pyvista

.. code-block:: python

    plot_points = []
    plot_directions = []
    for i, vec in enumerate(spherical_vectors):
        for j in range(ensemble_size):
            plot_points.append(grid_points[i])
            plot_directions.append((ensemble_vectors_cartesian[i, j][0], ensemble_vectors_cartesian[i, j][1], ensemble_vectors_cartesian[i, j][2]))
        # conver t to numpy arrays
        plot_points = np.array(plot_points)
        plot_directions = np.array(plot_directions)
        # Set up a pyvista plotter with two subplots
        plotter = pv.Plotter(shape=(1, 2))
        # Plot ensemble vectors using arrows in the first subplot
        plotter.subplot(0, 0)
        plotter.add_arrows(plot_points, plot_directions, color='green', mag=0.1, opacity=0.5)
        plotter.add_axes()
        plotter.add_text('Ensemble Vectors in 3D', font_size=12)

Calculate and plot squid glyphs in the second subplot with 50th percentile 
filtering and scale vector lengths by 0.1

.. code-block:: python

    plotter.subplot(0, 1)
    plotter, points, triangles = squid_glyph_3D(grid_points, ensemble_vectors, 0.5, 0.1, ax=plotter)
    plotter.add_text('Uncertainty Squid Glyphs in 3D', font_size=12)
    plotter.show()

.. image:: _static/squid_glyphs_example_3D.png
   :alt: Squid Glyphs Example 3D
   :align: center

"""
# import necessary libraries
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from uvisbox.Core.BandDepths.vector_depths import cartesian_to_spherical
from uvisbox.Modules.SquidGlyphs import squid_glyph_3D
import pyvista as pv


# Generate a 3D grid of vectors. [vx,vy,vz]=[x,y,z]
grid_size = 3
vectors = []
for x in range(grid_size):
    for y in range(grid_size):
        for z in range(grid_size):
            vectors.append((x, y, z))

grid_points = np.array(vectors)
# Convert to spherical coordinates
spherical_vectors = cartesian_to_spherical(np.array(vectors)) 

# Add Gaussian noise to spherical vectors to create an ensemble
ensemble_size = 20 # Number of ensemble members
noise_std_dev = 0.1  # Standard deviation for Gaussian noise
ensemble_vectors = np.zeros((len(spherical_vectors), ensemble_size, 3)) # initialize ensemble vectors
ensemble_vectors_cartesian = np.zeros((len(spherical_vectors), ensemble_size, 3)) # initialize ensemble vectors in cartesian coordinates
for i, vec in enumerate(spherical_vectors):
    for j in range(ensemble_size):
        noise = np.random.normal(0, noise_std_dev, 3)
        noisy_vec = vec + noise
        ensemble_vectors[i, j] = noisy_vec
        # Convert back to Cartesian coordinates for plotting
        r, theta, phi = noisy_vec
        vx = r * np.sin(theta) * np.cos(phi)
        vy = r * np.sin(theta) * np.sin(phi)
        vz = r * np.cos(theta)
        ensemble_vectors_cartesian[i, j] = np.array([vx, vy, vz])

# plotting with pyvista

plot_points = []
plot_directions = []
for i, vec in enumerate(spherical_vectors):
    for j in range(ensemble_size):
        plot_points.append(grid_points[i])
        plot_directions.append((ensemble_vectors_cartesian[i, j][0], ensemble_vectors_cartesian[i, j][1], ensemble_vectors_cartesian[i, j][2]))
# convert to numpy arrays
plot_points = np.array(plot_points)
plot_directions = np.array(plot_directions)
# Set up a pyvista plotter with two subplots
plotter = pv.Plotter(shape=(1, 2))
# Plot ensemble vectors using arrows in the first subplot
plotter.subplot(0, 0)
plotter.add_arrows(plot_points, plot_directions, color='green', mag=0.1, opacity=0.5)
plotter.add_axes()
plotter.add_text('Ensemble Vectors in 3D', font_size=12)

# Calculate and plot squid glyphs in the second subplot with 50th percentile 
# filtering and scale vector lengths by 0.1

plotter.subplot(0, 1)
plotter, points, triangles = squid_glyph_3D(grid_points, ensemble_vectors_cartesian, 0.5, 0.1, ax=plotter)
plotter.add_text('Uncertainty Squid Glyphs in 3D', font_size=12)
# plotter.show()
# plotter.screenshot("squid_glyphs_example_3D.png")