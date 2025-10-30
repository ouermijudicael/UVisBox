import xarray as xr
import matplotlib.pyplot as plt 
from uvisbox.Modules.SquidGlyphs import squid_glyph_3D 
import numpy as np
import uvisbox.Datasets.temperature_and_wind_data as twd
import pyvista as pv


ds = twd.load_dataset()
print(ds)
ensemble_members = ds.coords["number"].values
latitudes = ds.coords["latitude"].values
longitudes = ds.coords["longitude"].values
pressure_levels = ds.coords["pressure_level"].values
time = ds.coords["valid_time"].values


time_idx = 3

temp = ds['r'].isel(valid_time=time_idx).values
u = ds['u'].isel(valid_time=time_idx).values
v = ds['v'].isel(valid_time=time_idx).values
w = ds['w'].isel(valid_time=time_idx).values


grid_points = np.zeros((len(longitudes)*len(latitudes)*len(pressure_levels), 3))
ensemble_vectors = np.zeros((len(longitudes)*len(latitudes)*len(pressure_levels), len(ensemble_members), 3))
ensemble_temps = np.zeros((len(longitudes)*len(latitudes)*len(pressure_levels), len(ensemble_members)))
ensemble_divergences = np.zeros((len(longitudes)*len(latitudes)*len(pressure_levels), len(ensemble_members)))
for i_ens in range(len(ensemble_members)):
    for i_plev in range(len(pressure_levels)):
        for i_lat in range(len(latitudes)):
            for i_lon in range(len(longitudes)):
                g_idx = (i_lon * len(latitudes) * len(pressure_levels)) + (i_lat * len(pressure_levels)) + i_plev
                if i_ens == 0:
                    grid_points[g_idx, 0] = longitudes[i_lon]
                    grid_points[g_idx, 1] = latitudes[i_lat]
                    grid_points[g_idx, 2] = pressure_levels[i_plev]

                ensemble_vectors[g_idx, i_ens, 0] = u[i_ens, i_plev, i_lat, i_lon]
                ensemble_vectors[g_idx, i_ens, 1] = v[i_ens, i_plev, i_lat, i_lon]
                ensemble_vectors[g_idx, i_ens, 2] = w[i_ens, i_plev, i_lat, i_lon]
                ensemble_temps[g_idx, i_ens] = temp[i_ens, i_plev, i_lat, i_lon]

# calculate mean and stddev of temperature ensembles
mean_temps = np.mean(ensemble_temps, axis=1)
stddev_temps = np.std(ensemble_temps, axis=1)
stddev_temps[stddev_temps < 0.3] = 0.3

# calculate ensemble divergence along latitudinal and longitudinal directions
for i_ens in range(len(ensemble_members)):
    u_comp = ensemble_vectors[:, i_ens, 0].reshape((len(longitudes), len(latitudes), len(pressure_levels)))
    v_comp = ensemble_vectors[:, i_ens, 1].reshape((len(longitudes), len(latitudes), len(pressure_levels)))
    div = np.gradient(u_comp, axis=0) + np.gradient(v_comp, axis=1)
    ensemble_divergences[:, i_ens] = div.flatten()

# calculate mean and stddev of divergence ensembles
mean_divergences = np.mean(ensemble_divergences, axis=1)
stddev_divergences = np.std(ensemble_divergences, axis=1)
plotter = pv.Plotter()
plotter, points, triangles = squid_glyph_3D(grid_points, ensemble_vectors, point_values=stddev_temps, 
                                            percentile=95, scale=0.02, ax=plotter)
plotter.add_text("3D Squid Glyphs for Wind Uncertainty")
plotter.show()
plotter.screenshot("squid_glyph_temp_wind_example.png")
