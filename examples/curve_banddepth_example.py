"""
This example demonstrates how to create a curve band depth plot using hurricane track data.
It visualizes the tracks on a map and highlights the most central tracks based on band depth.

Import necessary libraries and load dataset

.. code-block:: python

    from uvisbox.Datasets import irma2017_perturbed_tracks
    from uvisbox.BandDepths import curve_banddepth_plot, curve_banddepth_meshing, curve_banddepths
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    lon_lat_coords = irma2017_perturbed_tracks.load_dataset()
    lon_lat_coords = lon_lat_coords[:10, :10, :2]  # Keep only longitude and latitude


Create figure with 2 subplots and set up Basemaps for geographic map visualization. In addition, 
plot all hurricane tracks for each ensemble member in light gray on the left subplot.

.. code-block:: python

    # create a figure with 2 subplots  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # set up Basemaps for geographic map visualization on left subplot
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

    # set up Basemaps for geographic map visualization on right subplot
    m2 = Basemap(projection='merc', 
                llcrnrlat=np.min(lon_lat_coords[:, :, 1]) - 5, 
                urcrnrlat=np.max(lon_lat_coords[:, :, 1]) + 5,
                llcrnrlon=np.min(lon_lat_coords[:, :, 0]) - 5, 
                urcrnrlon=np.max(lon_lat_coords[:, :, 0]) + 5,
                resolution='i', ax=ax2)
    m2.drawcoastlines()
    m2.drawcountries()
    m2.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
    m2.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])

    # plot all curves in light gray
    for curve in lon_lat_coords:
        x, y = m1(curve[:, 0], curve[:, 1])
        ax1.plot(x, y, color='lightgray', alpha=0.75)
    ax1.set_title('Hurricane Tracks')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # plot the curve band depth
    # ax2 = curve_banddepth_plot(lon_lat_coords, ax=ax2, percentile=75)

Calculate curve band depths, sort the curves in descending order,
generate the mesh for the 75th percentile band depth, and plot 
the mesh and median curve.

.. code-block:: python

    # calculate curve band depths for all curves
    cur_depths = curve_banddepths(lon_lat_coords)

    # sort curves by depth in descending order
    sorted_indices = np.argsort(-cur_depths)  
    sorted_curves = lon_lat_coords[sorted_indices]

    # get mesh for the 75th percentile band depth
    points, triangles = curve_banddepth_meshing(sorted_curves, percentile=75)

    # plot the mesh
    x, y = m2(points[:, 0], points[:, 1])
    f_colors = np.ones(len(triangles)) 
    ax2.tripcolor(x, y, triangles, facecolors=f_colors)

    # plot median curve
    median_curve = sorted_curves[0]
    x, y = m2(median_curve[:, 0], median_curve[:, 1])
    ax2.plot(x, y, color='red', label='Median Curve', linewidth=2)

    ax2.set_title('Curve Band Depth Plot')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.savefig("curve_banddepth_example.png")
    plt.show()

.. image:: ../curve_banddepth_example.png
   :alt: Curve Band Depth Example
   :align: center

"""

# Import necessary libraries and load dataset
from uvisbox.Datasets import irma2017_perturbed_tracks
from uvisbox.BandDepths import curve_banddepth_plot, curve_banddepth_meshing, curve_banddepths
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

lon_lat_coords = irma2017_perturbed_tracks.load_dataset()
# lon_lat_coords = lon_lat_coords[:10, :10, :2]  # Keep only longitude and latitude


# Create figure with 2 subplots and set up Basemaps for geographic map visualization. In addition, 
# plot all hurricane tracks for each ensemble member in light gray on the left subplot.
#

# create a figure with 2 subplots  
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# set up Basemaps for geographic map visualization on left subplot
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

# set up Basemaps for geographic map visualization on right subplot
m2 = Basemap(projection='merc', 
            llcrnrlat=np.min(lon_lat_coords[:, :, 1]) - 5, 
            urcrnrlat=np.max(lon_lat_coords[:, :, 1]) + 5,
            llcrnrlon=np.min(lon_lat_coords[:, :, 0]) - 5, 
            urcrnrlon=np.max(lon_lat_coords[:, :, 0]) + 5,
            resolution='i', ax=ax2)
m2.drawcoastlines()
m2.drawcountries()
m2.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
m2.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])

# plot all curves in light gray
for curve in lon_lat_coords:
    x, y = m1(curve[:, 0], curve[:, 1])
    ax1.plot(x, y, color='lightgray', alpha=0.75)
ax1.set_title('Hurricane Tracks')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# plot the curve band depth
# ax2 = curve_banddepth_plot(lon_lat_coords, ax=ax2, percentile=75)

# Calculate curve band depths, sort the curves in descending order,
# generate the mesh for the 75th percentile band depth, and plot 
# the mesh and median curve.

# calculate curve band depths for all curves
cur_depths = curve_banddepths(lon_lat_coords)

# sort curves by depth in descending order
sorted_indices = np.argsort(-cur_depths)  
sorted_curves = lon_lat_coords[sorted_indices]

# get mesh for the 75th percentile band depth
points, triangles = curve_banddepth_meshing(sorted_curves, percentile=75)

# plot the mesh
x, y = m2(points[:, 0], points[:, 1])
f_colors = np.ones(len(triangles)) 
ax2.tripcolor(x, y, triangles, facecolors=f_colors)

# plot median curve
median_curve = sorted_curves[0]
x, y = m2(median_curve[:, 0], median_curve[:, 1])
ax2.plot(x, y, color='red', label='Median Curve', linewidth=2)

ax2.set_title('Curve Band Depth Plot')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
plt.savefig("curve_banddepth_example.png")
plt.show()