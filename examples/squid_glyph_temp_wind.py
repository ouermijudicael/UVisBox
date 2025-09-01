import xarray as xr
import matplotlib.pyplot as plt 
from uvisbox.Glyphs.squid_glyph import  uncertainty_squid_glyphs_3D
import numpy as np
import uvisbox.Datasets.temperature_and_wind_data as twd


ds = twd.load_dataset()
# print(ds)
ensemble_members = ds.coords["number"].values
latitudes = ds.coords["latitude"].values
longitudes = ds.coords["longitude"].values
pressure_levels = ds.coords["pressure_level"].values
time = ds.coords["valid_time"].values


time_idx = 1

temp = ds['t'].isel(valid_time=time_idx).values
u = ds['u'].isel(valid_time=time_idx).values
v = ds['v'].isel(valid_time=time_idx).values
w = ds['w'].isel(valid_time=time_idx).values
grid_points = np.zeros((5*5*len(pressure_levels), 3))
ensemble_vectors = np.zeros((5*5*len(pressure_levels), len(ensemble_members), 3))

# Plot ensemble vectors using arrows
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for i_ens in range(len(ensemble_members)):
    for i_lon in range(5):#len(longitudes)):
        for i_lat in range(5):#len(latitudes)):
            for i_plev in range(len(pressure_levels)):
                g_idx = i_lon*5*len(pressure_levels) + i_lat*len(pressure_levels) + i_plev
                if i_ens == 0:
                    grid_points[g_idx, 0] = longitudes[i_lon]
                    grid_points[g_idx, 1] = latitudes[i_lat]
                    grid_points[g_idx, 2] = pressure_levels[i_plev]

                ensemble_vectors[g_idx, i_ens, 0] = u[i_ens, i_plev, i_lat, i_lon]
                ensemble_vectors[g_idx, i_ens, 1] = v[i_ens, i_plev, i_lat, i_lon]
                ensemble_vectors[g_idx, i_ens, 2] = w[i_ens, i_plev, i_lat, i_lon]
                
                ax.quiver(longitudes[i_lon], latitudes[i_lat], pressure_levels[i_plev],
                           u[i_ens, i_plev, i_lat, i_lon], v[i_ens, i_plev, i_lat, i_lon], w[i_ens, i_plev, i_lat, i_lon],
                           color='r', length=0.1)
ax.set_title("3D Wind Vectors")

ax2 = uncertainty_squid_glyphs_3D(grid_points, ensemble_vectors, 0.5, 0.25, ax=ax2)
plt.savefig("3d_wind_vectors.png")
