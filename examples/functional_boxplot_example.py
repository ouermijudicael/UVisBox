"""
This example demonstrates how to create functional boxplots using the ``uvisbox`` library.
It uses the sea surface temperature dataset from  Sun, Y., & Genton, M. G. (2011). 
Functional Boxplots. Journal of Computational and Graphical Statistics, 20(2), 316–334. 
https://doi.org/10.1198/jcgs.2011.09224 and computes both the functional band depth and 
modified functional band depth to visualize the centrality and variability of the time series data.
The data consist of monthly sea surface temperatures (SST) measured in degrees Celsius over
the east-central tropical Paciﬁc Ocean in degrees Celsius from January 1951 to December 2007.

Import necessary libraries and load dataset

.. code-block:: python

    from uvisbox.Datasets import sea_surface_temp_data
    from uvisbox.Core.CommonInterface import BoxplotStyleConfig
    import matplotlib.pyplot as plt
    from uvisbox.Modules.FunctionalBoxplot import functional_boxplot

    data = sea_surface_temp_data.load_dataset()
    X = data.T

Create figure with three subplots and plot the original sea surface temperature time series

.. code-block:: python

    # create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    # plot the original time series in the first subplot
    for i in range(X.shape[0]):
        ax1.plot(X[i,:], alpha=1.)
    ax1.set_title("Sea Surface Temperature Time Series")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Temperature")

plot the functional boxplot for the time series data with different methods at [25,50,90,100] percentiles (default)

.. code-block:: python

    # Create style configuration to show outliers
    style = BoxplotStyleConfig(
        percentiles=[25, 50, 75, 90],
        percentile_colormap='viridis',
        show_median=True,
        median_color='red',
        show_outliers=True,  # Show outliers beyond the largest percentile
        outliers_color='gray',
        outliers_alpha=0.5
    )

    ax2 = functional_boxplot(X, boxplot_style=style, ax=ax2)
    ax2.set_title("Functional Boxplot (FBD) with Outliers")

    ax3 = functional_boxplot(X, boxplot_style=style, method='mfbd', ax=ax3)
    ax3.set_title("Functional Boxplot (MFBD) with Outliers")

Add colorbar to show percentile mapping

.. code-block:: python

    import matplotlib.cm as cm
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = Normalize(vmin=min(style.percentiles), vmax=max(style.percentiles))
    cbar = ColorbarBase(cbar_ax, cmap=cm.get_cmap(style.percentile_colormap), norm=norm, orientation='vertical')
    cbar.set_label('Percentile (%)', rotation=270, labelpad=20)

    plt.show()

.. image:: _static/sea_surface_temp_functional_banddepth_example.png
    :alt: Sea Surface Temperature Functional Band Depth Example
    :align: center
    
"""
# import necessary libraries and load dataset

from uvisbox.Datasets import sea_surface_temp_data
from uvisbox.Core.CommonInterface import BoxplotStyleConfig
import matplotlib.pyplot as plt
from uvisbox.Modules.FunctionalBoxplot import functional_boxplot

data = sea_surface_temp_data.load_dataset()
X = data.T

# Create figure with three subplots and plot the original sea surface temperature time series

# create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
# plot the original time series in the first subplot
for i in range(X.shape[0]):
    ax1.plot(X[i,:], alpha=1.)
ax1.set_title("Sea Surface Temperature Time Series")
ax1.set_xlabel("Month")
ax1.set_ylabel("Temperature")

# plot the functional boxplot for the time series data with different methods at [25,50,90,100] percentiles (default)

# Create style configuration to show outliers
style = BoxplotStyleConfig(
    percentiles=[25, 50, 75, 90],
    percentile_colormap='viridis',
    show_median=True,
    median_color='red',
    show_outliers=True,  # Show outliers beyond the largest percentile
    outliers_color='gray',
    outliers_alpha=0.5
)

ax2 = functional_boxplot(X, boxplot_style=style, ax=ax2)
ax2.set_title("Functional Boxplot (FBD) with Outliers")

ax3 = functional_boxplot(X, boxplot_style=style, method='mfbd', ax=ax3)
ax3.set_title("Functional Boxplot (MFBD) with Outliers")

# Add colorbar to show percentile mapping
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Create a new axis for the colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
norm = Normalize(vmin=min(style.percentiles), vmax=max(style.percentiles))
cbar = ColorbarBase(cbar_ax, cmap=cm.get_cmap(style.percentile_colormap), norm=norm, orientation='vertical')
cbar.set_label('Percentile (%)', rotation=270, labelpad=20)

# plt.savefig("sea_surface_temp_functional_banddepth_example.png")
plt.show()