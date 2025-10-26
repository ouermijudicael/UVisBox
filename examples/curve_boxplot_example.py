"""
This example demonstrates how to create a curve band depth plot using randomly generated curves.
It generates synthetic curves by:
1. Creating a base smooth curve (sine wave)
2. Adding Y-direction shifts sampled from a Gaussian distribution
3. Adding slight random noise at each time step

This approach creates realistic curve ensembles where curves follow a similar pattern but with
controlled variations, making the band depth analysis more meaningful.

Generate random curves with structured variation:

.. code-block:: python

    from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curves_band_mesh
    from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_banddepths
    import matplotlib.pyplot as plt
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    n_curves = 50
    n_steps = 150
    
    # Generate base curve (smooth sine wave)
    t = np.linspace(0, 4 * np.pi, n_steps)
    base_curve = np.zeros((n_steps, 2))
    base_curve[:, 0] = t
    base_curve[:, 1] = np.sin(t) * 2
    
    # Generate variations
    curves = np.zeros((n_curves, n_steps, 2))
    for curve_idx in range(n_curves):
        curve = base_curve.copy()
        
        # Add global Y-shift from Gaussian
        y_shift = np.random.normal(0, 0.5)
        curve[:, 1] += y_shift
        
        # Add local noise at each step
        for step in range(n_steps):
            step_noise = np.random.normal(0, 0.15, 2)
            curve[step, :] += step_noise
        
        curves[curve_idx] = curve


Create figure with 2 subplots. Plot all curves in light gray on the left subplot.

.. code-block:: python

    # Create a figure with 2 subplots  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot all curves in light gray
    for curve in curves:
        ax1.plot(curve[:, 0], curve[:, 1], color='lightgray', alpha=0.5, linewidth=0.8)
    ax1.set_title('All Generated Curves (50 curves)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)


Calculate curve band depths, sort the curves in descending order,
generate meshes for multiple percentile bands (100%, 90%, 50%, 25%), 
and plot them with different colors along with the median curve.

.. code-block:: python

    # Calculate curve band depths for all curves
    cur_depths = curve_banddepths(curves)

    # Sort curves by depth in descending order (highest depth = median)
    sorted_indices = np.argsort(-cur_depths)  
    sorted_curves = curves[sorted_indices]

    # Define percentiles and colors for visualization
    percentiles = [100, 90, 50, 25]
    colors = ['#e0e0e0', '#a0c4e8', '#5a8dc4', '#2e5f8a']  # Light to dark blue
    labels = ['100% (All data)', '90%', '50%', '25%']

    # Plot bands from largest to smallest so smaller bands appear on top
    for percentile, color, label in zip(percentiles, colors, labels):
        points, triangles = curves_band_mesh(sorted_curves, percentile=percentile)
        ax2.tripcolor(points[:, 0], points[:, 1], triangles, 
                     facecolors=[color] * len(triangles), 
                     edgecolors='none', alpha=0.7, label=label)

    # Plot median curve (highest depth curve)
    median_curve = sorted_curves[0]
    ax2.plot(median_curve[:, 0], median_curve[:, 1], 
            color='red', label='Median Curve', linewidth=2.5, zorder=10)

    ax2.set_title('Curve Band Depth Plot with Percentile Bands')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

.. image:: _static/curve_boxplot_example.png
   :alt: Curve Boxplot Example
   :align: center

"""

# Import necessary libraries
from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curves_band_mesh
from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_banddepths
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for curve generation
n_curves = 50        # Number of curves (samples)
n_steps = 150        # Number of time steps
n_dims = 2           # 2D curves (x, y)

# Generate base curve (the "true" curve that all samples are based on)
# This creates a smooth trajectory
t = np.linspace(0, 4 * np.pi, n_steps)
base_curve = np.zeros((n_steps, n_dims))
base_curve[:, 0] = t  # X increases linearly with time
base_curve[:, 1] = np.sin(t) * 2  # Y follows a sine wave

# Generate random curves by adding variations to the base curve
curves = np.zeros((n_curves, n_steps, n_dims))

for curve_idx in range(n_curves):
    # Start with the base curve
    curve = base_curve.copy()
    
    # Add a global Y-direction shift sampled from Gaussian
    # This creates different "lanes" for each curve
    y_shift = np.random.normal(0, 0.5)
    curve[:, 1] += y_shift
    
    # Add slight randomness at each step
    # This creates local perturbations along the curve
    for step in range(n_steps):
        step_noise = np.random.normal(0, 0.15, n_dims)
        curve[step, :] += step_noise
    
    curves[curve_idx] = curve

print(f"Generated {n_curves} curves with {n_steps} time steps")
print(f"Base curve: smooth sine wave with random Y-shifts and local noise")
print(f"Curves shape: {curves.shape}")

# Create a figure with 2 subplots  
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot base curve in black
ax1.plot(base_curve[:, 0], base_curve[:, 1], color='black', linewidth=2, label='Base Curve', zorder=10)

# Plot all curves in light gray
for curve in curves:
    ax1.plot(curve[:, 0], curve[:, 1], color='lightgray', alpha=0.5, linewidth=0.8)
ax1.set_title('All Generated Curves (50 curves)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Calculate curve band depths for all curves
print("Computing curve band depths...")
cur_depths = curve_banddepths(curves)
print(f"Band depths computed. Min: {cur_depths.min():.4f}, Max: {cur_depths.max():.4f}")

# Sort curves by depth in descending order (highest depth = median)
sorted_indices = np.argsort(-cur_depths)  
sorted_curves = curves[sorted_indices]

print(f"Median curve has depth: {cur_depths[sorted_indices[0]]:.4f}")

# Define percentiles and colors for visualization
percentiles = [100, 90, 50, 25]
colors = ['#e0e0e0', '#a0c4e8', '#5a8dc4', '#2e5f8a']  # Light to dark blue
labels = ['100% (All data)', '90%', '50%', '25%']

# Plot bands from largest to smallest so smaller bands appear on top
for percentile, color, label in zip(percentiles, colors, labels):
    print(f"Generating mesh for {percentile}% band...")
    points, triangles = curves_band_mesh(sorted_curves, percentile=percentile)
    ax2.triplot(points[:, 0], points[:, 1], triangles, 
                color=color, linewidth=0.5, alpha=0.3)
    # Fill the triangles with the specified color
    from matplotlib.patches import Polygon
    for tri in triangles:
        poly = Polygon(points[tri], facecolor=color, edgecolor='none', alpha=0.7)
        ax2.add_patch(poly)
    # Add a dummy plot for the legend
    ax2.fill([], [], color=color, alpha=0.7, label=label)

# Plot median curve (highest depth curve)
median_curve = sorted_curves[0]
ax2.plot(median_curve[:, 0], median_curve[:, 1], 
        color='red', label='Median Curve', linewidth=2.5, zorder=10)

ax2.set_title('Curve Band Depth Plot with Percentile Bands')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("curve_boxplot_example.png", dpi=150)
print("Displaying plot...")
plt.show()