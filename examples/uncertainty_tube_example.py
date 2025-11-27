import numpy as np
import pyvista as pv

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes import uncertainty_tube # New import

# Generate random seed points and compute their trajectories in a 3D flow field
t0 = 0
t1 = 5
n_steps = 30
number_of_seeds = 10

scale = np.arange(number_of_seeds)
scale = linear_interpolate(scale, 0, number_of_seeds - 1, 1.5, 2.0)
xy_scale = np.ones(number_of_seeds)
xy_scale[1::2] = 0.1

seeds = np.random.uniform(-1, 1, (number_of_seeds, 3))
trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)

# Create PyVista plotter
plotter = pv.Plotter()

# Generate and visualize uncertainty tubes using the new function
uncertainty_tube(trajectories, colormap='viridis', plotter=plotter, e_proj=0.5, n_jobs=2)

plotter.add_axes()
plotter.add_text("Uncertainty Tubes", font_size=12)

plotter.show()