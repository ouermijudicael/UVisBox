"""
3D Curve Boxplot Example using PyVista

This example demonstrates how to create a 3D curve band depth plot using PyVista
for interactive 3D visualization. It generates synthetic 3D curves by:
1. Creating a base smooth 3D curve (spiral/helix)
2. Adding random shifts in all directions sampled from Gaussian distributions
3. Adding slight random noise at each time step

The visualization uses PyVista for superior 3D interactivity:
- Smooth rotation and zooming
- Better lighting and rendering
- Side-by-side comparison of all curves vs. boxplot
"""

import numpy as np
import pyvista as pv
from uvisbox.Modules.CurveBoxplot import curve_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for curve generation
n_curves = 50        # Number of curves (samples)
n_steps = 150        # Number of time steps
n_dims = 3           # 3D curves (x, y, z)

print("=" * 70)
print("3D Curve Boxplot Example - PyVista Backend")
print("=" * 70)

# Generate base curve (the "true" curve that all samples are based on)
# This creates a smooth 3D spiral/helix trajectory
t = np.linspace(0, 4 * np.pi, n_steps)
base_curve = np.zeros((n_steps, n_dims))
base_curve[:, 0] = np.cos(t) * 2         # X: circular motion
base_curve[:, 1] = np.sin(t) * 2         # Y: circular motion
base_curve[:, 2] = t / (2 * np.pi)       # Z: linear ascent

# Generate random curves by adding variations to the base curve
curves = np.zeros((n_curves, n_steps, n_dims))

print("Generating random 3D curves...")
for curve_idx in range(n_curves):
    # Start with the base curve
    curve = base_curve.copy()
    
    # Add global shifts in all directions sampled from Gaussian
    x_shift = np.random.normal(0, 0.3)
    y_shift = np.random.normal(0, 0.3)
    z_shift = np.random.normal(0, 0.1)
    
    curve[:, 0] += x_shift
    curve[:, 1] += y_shift
    curve[:, 2] += z_shift
    
    # Add slight randomness at each step
    for step in range(n_steps):
        step_noise = np.random.normal(0, 0.1, n_dims)
        curve[step, :] += step_noise
    
    curves[curve_idx] = curve

print(f"Generated {n_curves} curves with {n_steps} time steps")
print(f"Base curve: 3D spiral/helix with random shifts and local noise")
print(f"Curves shape: {curves.shape}")
print()

# Configure boxplot style
style = BoxplotStyleConfig(
    percentiles=[90, 50],
    percentile_colormap='viridis',
    show_median=True,
    median_color='red',
    median_width=3,
    show_outliers=True,
    outliers_color='gray',
    outliers_alpha=0.4
)

# Create PyVista plotter with side-by-side subplots
print("Creating PyVista visualization...")
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 700))

# Left subplot: All curves
plotter.subplot(0, 0)
plotter.add_text("All Generated 3D Curves (50 curves)", font_size=12, position='upper_edge')

# Add base curve
base_line = pv.Spline(base_curve[:, 0:3], n_points=base_curve.shape[0])
plotter.add_mesh(base_line, color='black', line_width=3, label='Base Curve')

# Add all generated curves
for curve in curves:
    curve_line = pv.Spline(curve[:, 0:3], n_points=curve.shape[0])
    plotter.add_mesh(curve_line, color='lightgray', opacity=0.4, line_width=1)

plotter.add_axes()
plotter.show_grid()

# Right subplot: Curve boxplot using the pipeline
plotter.subplot(0, 1)
plotter.add_text("3D Curve Band Depth Plot", font_size=12, position='upper_edge')

# Use curve_boxplot to generate the visualization directly
# This automatically uses PyVista backend for 3D curves
# Pass the plotter as 'ax' parameter
print("Computing curve boxplot and rendering...")
curve_boxplot(curves, boxplot_style=style, ax=plotter)

plotter.add_axes()
plotter.show_grid()
plotter.add_legend()

# Link camera views between subplots
plotter.link_views()

print()
print("=" * 70)
print("Interactive 3D Visualization (PyVista)")
print("=" * 70)
print("Controls:")
print("  - Left mouse button: Rotate")
print("  - Middle mouse button: Pan")
print("  - Mouse wheel: Zoom")
print("  - Right mouse button: Zoom (drag)")
print("  - 'r': Reset camera")
print("  - 'q': Quit")
print()
print("Camera views are linked between both plots!")
print("=" * 70)

# Show the plotter
plotter.show()

