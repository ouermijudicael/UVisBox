"""
This example demonstrates how to create a curve band depth plot using hurricane track data.
It visualizes the tracks on a map and highlights the most central tracks based on band depth.

Import necessary libraries and modules.

.. code-block:: python

    from usivbox.Datasets import irma2017_perturbed_tracks
    from uvisbox.BandDepths import curve_banddepth_plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np

Load the hurricane track dataset.

.. code-block:: python
    lon_lat_coords = irma2017_perturbed_tracks.load_dataset()
    # lon_lat_coords = lon_lat_coords[:20, :, :2]  # Keep only longitude and latitude
    # Create the curve band depth plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

Set up the Basemap

.. code-block:: python
    m1 = Basemap(projection='merc',
                llcrnrlat=np.min(lon_lat_coords[:, :, 1]) - 5, 
                urcrnrlat=np.max(lon_lat_coords[:, :, 1]) + 5,
                llcrnrlon=np.min(lon_lat_coords[:, :, 0]) - 5, 
                urcrnrlon=np.max(lon_lat_coords[:, :, 0]) + 5,
                resolution='i', ax=ax1)
    m1.drawcoastlines()
    m1.drawcountries()
    m1.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
    m1.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])

Plot all curves in light gray

.. code-block:: python

    for curve in lon_lat_coords:
        x, y = m1(curve[:, 0], curve[:, 1])
        ax1.plot(x, y, color='lightgray', alpha=0.5)    
    ax1.set_title('Hurricane Tracks')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

Plot the curve band depth

.. code-block:: python
    ax2 = curve_banddepth_plot(lon_lat_coords, ax=ax2, percentile=75)

    ax2.set_title('Curve Band Depth Plot')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.legend()
    # plt.savefig("curve_banddepth.png")
    plt.show()

"""


from uvisbox.Datasets import irma2017_perturbed_tracks
from uvisbox.BandDepths import curve_banddepth_plot
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Load the dataset
lon_lat_coords = irma2017_perturbed_tracks.load_dataset()
lon_lat_coords = lon_lat_coords[:10, :10, :2]  # Keep only longitude and latitude
# Create the curve band depth plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Set up the Basemap
m1 = Basemap(projection='merc', 
            llcrnrlat=np.min(lon_lat_coords[:, :, 1]) - 5, 
            urcrnrlat=np.max(lon_lat_coords[:, :, 1]) + 5,
            llcrnrlon=np.min(lon_lat_coords[:, :, 0]) - 5, 
            urcrnrlon=np.max(lon_lat_coords[:, :, 0]) + 5,
            resolution='i', ax=ax1)
m1.drawcoastlines()
m1.drawcountries()
m1.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
m1.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])


# plot all curves in light gray
for curve in lon_lat_coords:
    x, y = m1(curve[:, 0], curve[:, 1])
    ax1.plot(x, y, color='lightgray', alpha=0.5)
ax1.set_title('Hurricane Tracks')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# plot the curve band depth
ax2 = curve_banddepth_plot(lon_lat_coords, ax=ax2, percentile=75)

ax2.set_title('Curve Band Depth Plot')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
plt.legend()
plt.savefig("curve_banddepth.png")
plt.show()