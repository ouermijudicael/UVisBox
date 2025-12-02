import numpy as np
import pyvista as pv

from uvisbox.Datasets import create_swirl_ensemble
from uvisbox import uncertainty_tubes  # New import


# Define a straight line from (0,0,0) to (10,10,10)
start_point = [0.0, 0.0, 0.0]
end_point = [10.0, 10.0, 10.0]

uncertainty_samples = 50
mesh_resolution = 256

# Create the ensemble
curves, center_line = create_swirl_ensemble(
    start_point, end_point,
    num_curves=uncertainty_samples,
    num_points=100,
    major_axis_length=1.0,
    minor_axis_length=0.7,
    swirl_frequency=0.4,
    distribution='interior',
    noise_level=0,
    expansion_factor=1.0  # Controls how much ellipse expands
)

np_curves = np.array(curves)
np_curves = np.expand_dims(np_curves, axis=1)


trajectories = np.transpose(np_curves, (2, 1, 0, 3))

font_size = 32
# Create PyVista plotter with two subplots
plotter = pv.Plotter(shape=(2, 2))

# Left subplot: e_proj = 0.5 (tau = 2.0)
plotter.subplot(0, 0)
plotter = uncertainty_tubes(trajectories, colormap='coolwarm',
                 plotter=plotter, resolution=mesh_resolution, e_proj=0.5, n_jobs=2)
plotter.remove_scalar_bar()
plotter.add_scalar_bar("Uncertainty Scale", vertical=False, title_font_size=font_size, label_font_size=font_size)
plotter.add_text("Tau=2.0", font_size=font_size)


plotter.subplot(0, 1)
plotter = uncertainty_tubes(trajectories, colormap='coolwarm',
                 plotter=plotter, resolution=mesh_resolution, e_proj=1.0, n_jobs=2)

plotter.add_text("Tau=1.0", font_size=font_size)


# Right subplot: e_proj = 1.0 (tau = 1.0)


plotter.subplot(1, 0)

plotter = uncertainty_tubes(trajectories, colormap='coolwarm',
                 plotter=plotter, resolution=mesh_resolution, e_proj=0.5, clim=[0,0.5], n_jobs=2)
plotter.add_text("Color Range (0,0.5)", font_size=font_size)


plotter.subplot(1, 1)
plotter = uncertainty_tubes(trajectories, colormap='coolwarm',
                 plotter=plotter, resolution=mesh_resolution, e_proj=0.5, clim=[0.5,1.0], n_jobs=2)
plotter.add_text("Color Range (0.5,1.0)", font_size=font_size)

plotter.link_views()
plotter.show()
