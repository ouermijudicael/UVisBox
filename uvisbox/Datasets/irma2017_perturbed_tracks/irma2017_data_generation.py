# This is adapted from climada examples from 
# https://climada-python.readthedocs.io/en/stable/tutorial/climada_hazard_TropCyclone.html
#

from climada.hazard import TCTracks
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

tr_irma = TCTracks.from_ibtracs_netcdf(
    provider="usa", storm_id="2017242N16333"
)  # IRMA 2017
# ax = tr_irma.plot()
# ax.set_title("IRMA")  # set title

# # other ibtracs selection options
# from climada.hazard import TCTracks

# # years 1993 and 1994 in basin EP.
# # correct_pres ignores tracks with not enough data. For statistics (frequency of events), these should be considered as well
# sel_ibtracs = TCTracks.from_ibtracs_netcdf(
#     provider="usa", year_range=(1993, 1994), basin="EP", correct_pres=False
# )
# print("Number of tracks:", sel_ibtracs.size)
# ax = sel_ibtracs.plot()
# ax.get_legend()._loc = 2  # correct legend location
# ax.set_title("1993-1994, EP")  # set title

# track1 = TCTracks.from_ibtracs_netcdf(
#     provider="usa", storm_id="2007314N10093"
# )  # SIDR 2007
# track2 = TCTracks.from_ibtracs_netcdf(
#     provider="usa", storm_id="2016138N10081"
# )  # ROANU 2016
# track1.append(track2.data)  # put both tracks together
# ax = track1.plot()
# ax.get_legend()._loc = 2  # correct legend location
# ax.set_title("SIDR and ROANU");  # set title


# tr_irma.get_track("2017242N16333")


# here we use tr_irma retrieved from IBTrACS with the function above
# select number of synthetic tracks (nb_synth_tracks) to generate per present tracks.
num_perturbed_tracks = 40
tr_irma.equal_timestep()
tr_irma.calc_perturbed_trajectories(nb_synth_tracks=num_perturbed_tracks)

num_time_steps = len(tr_irma.data[0]["time"])
print(f"Number of time steps: {num_time_steps}")

# get trajectories as numpy array
# shape is (n_steps, n_tracks, n_samples, n_dims)
lon_lat_coords = np.zeros((num_perturbed_tracks, num_time_steps, 2))
for i in range(num_perturbed_tracks):
    lon_lat_coords[i, :, 0] = tr_irma.data[i]["lon"].to_numpy()
    lon_lat_coords[i, :, 1] = tr_irma.data[i]["lat"].to_numpy()

print(f"Saving lon_lat_coords  of shape: {lon_lat_coords.shape}")
np.save("irma2017_perturbed_tracks.npy", lon_lat_coords)

# plot the perturbed tracks with Basemap
fig, ax = plt.subplots(figsize=(10, 10))

# Set up the Basemap
m = Basemap(projection='merc', 
            llcrnrlat=np.min(lon_lat_coords[:, :, 1]) - 5, 
            urcrnrlat=np.max(lon_lat_coords[:, :, 1]) + 5,
            llcrnrlon=np.min(lon_lat_coords[:, :, 0]) - 5, 
            urcrnrlon=np.max(lon_lat_coords[:, :, 0]) + 5,
            resolution='i', ax=ax)

m.drawcoastlines()
m.drawcountries()
m.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])

# Plot the perturbed tracks
for i in range(num_perturbed_tracks):
    x, y = m(lon_lat_coords[i, :, 0], lon_lat_coords[i, :, 1])
    ax.plot(x, y, label=f"Perturbed Track {i+1}")

plt.title("Perturbed Tracks of IRMA 2017")
plt.show()