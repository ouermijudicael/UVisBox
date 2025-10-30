"""
This example demonstrates how to create a curve band depth plot using randomly generated curves.
It generates synthetic curves by:
1. Creating a base smooth curve (sine wave)
2. Adding Y-direction shifts sampled from a Gaussian distribution
3. Adding slight random noise at each time step

This approach creates realistic curve ensembles where curves follow a similar pattern but with
controlled variations, making the band depth analysis more meaningful.

Import necessary libraries

.. code-block:: python

    from scipy.__config__ import show
    from uvisbox.Modules.CurveBoxplot import curve_boxplot
    from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    import matplotlib.pyplot as plt
    import numpy as np

.. code-block:: python

    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for curve generation
    n_curves = 50        # Number of curves (samples)
    n_steps = 150        # Number of time steps
    n_dims = 2           # 2D curves (x, y)

    print("Curve Boxplot Example - Random Gaussian Curves")

    # Generate base curve (the "true" curve that all samples are based on)
    # This creates a smooth trajectory
    t = np.linspace(0, 4 * np.pi, n_steps)
    base_curve = np.zeros((n_steps, n_dims))
    base_curve[:, 0] = t  # X increases linearly with time
    base_curve[:, 1] = np.sin(t) * 2  # Y follows a sine wave

    # Generate random curves by adding variations to the base curve
    curves = np.zeros((n_curves, n_steps, n_dims))

    print("=" * 70)
    print("Generating random curves...")
    print("=" * 70)

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

.. code-block:: python

    # Create a figure with 2 subplots  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: Plot all curves with base curve
    print("Plotting all curves...")
    ax1.plot(base_curve[:, 0], base_curve[:, 1], color='black', 
            linewidth=2, label='Base Curve', zorder=10)

    for curve in curves:
        ax1.plot(curve[:, 0], curve[:, 1], color='lightgray', alpha=0.5, linewidth=0.8)

    ax1.set_title('All Generated Curves (50 curves)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

Right subplot: Use curve_boxplot to plot multiple percentile bands

.. code-block:: python

    print("Creating curve boxplot with multiple percentile bands...")

    # Create custom styling configuration
    style = BoxplotStyleConfig(
        percentiles=[90, 50, 25],
        percentile_colormap='viridis',  # Use viridis colormap for bands
        show_median=True,
        median_color='red',
        show_outliers=True
    )

    # Call the curve_boxplot function with the style configuration
    curve_boxplot(curves, boxplot_style=style, ax=ax2)

    ax2.set_title('Curve Band Depth Plot with Percentile Bands')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

Add colorbar to show percentile mapping

.. code-block:: python

    import matplotlib.cm as cm
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    # Adjust layout to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = Normalize(vmin=min(style.percentiles), vmax=max(style.percentiles))
    cbar = ColorbarBase(cbar_ax, cmap=cm.get_cmap(style.percentile_colormap), norm=norm, orientation='vertical')
    cbar.set_label('Percentile (%)', rotation=270, labelpad=20)

    print("=" * 70)
    print("Displaying plot...")
    print("=" * 70)
    plt.show()

.. image:: _static/curve_boxplot_example.png
   :alt: Curve Boxplot Example
   :align: center

"""

# Import necessary libraries
from scipy.__config__ import show
from uvisbox.Modules.CurveBoxplot import curve_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for curve generation
n_curves = 50        # Number of curves (samples)
n_steps = 150        # Number of time steps
n_dims = 2           # 2D curves (x, y)

print("=" * 70)
print("Curve Boxplot Example - Random Gaussian Curves")
print("=" * 70)

# Generate base curve (the "true" curve that all samples are based on)
# This creates a smooth trajectory
t = np.linspace(0, 4 * np.pi, n_steps)
base_curve = np.zeros((n_steps, n_dims))
base_curve[:, 0] = t  # X increases linearly with time
base_curve[:, 1] = np.sin(t) * 2  # Y follows a sine wave

# Generate random curves by adding variations to the base curve
curves = np.zeros((n_curves, n_steps, n_dims))

print("Generating random curves...")
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

# Left subplot: Plot all curves with base curve
print("Plotting all curves...")
ax1.plot(base_curve[:, 0], base_curve[:, 1], color='black', 
         linewidth=2, label='Base Curve', zorder=10)

for curve in curves:
    ax1.plot(curve[:, 0], curve[:, 1], color='lightgray', alpha=0.5, linewidth=0.8)

ax1.set_title('All Generated Curves (50 curves)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Right subplot: Use curve_boxplot to plot multiple percentile bands
print("Creating curve boxplot with multiple percentile bands...")

# Create custom styling configuration
style = BoxplotStyleConfig(
    percentiles=[90, 50, 25],
    percentile_colormap='viridis',  # Use viridis colormap for bands
    show_median=True,
    median_color='red',
    show_outliers=True
)

# Call the curve_boxplot function with the style configuration
curve_boxplot(curves, boxplot_style=style, ax=ax2)

ax2.set_title('Curve Band Depth Plot with Percentile Bands')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Add colorbar to show percentile mapping
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Adjust layout to make room for colorbar
plt.tight_layout(rect=[0, 0, 0.9, 1])

# Create a new axis for the colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
norm = Normalize(vmin=min(style.percentiles), vmax=max(style.percentiles))
cbar = ColorbarBase(cbar_ax, cmap=cm.get_cmap(style.percentile_colormap), norm=norm, orientation='vertical')
cbar.set_label('Percentile (%)', rotation=270, labelpad=20)

print("=" * 70)
print("Displaying plot...")
print("=" * 70)
# plt.savefig("curve_boxplot_example.png", dpi=150)
plt.show()