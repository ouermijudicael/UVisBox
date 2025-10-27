"""
This example demonstrates how to create functional boxplots using the ``uvisbox`` library.
It uses the sea surface temperature dataset from  Sun, Y., & Genton, M. G. (2011). 
Functional Boxplots. Journal of Computational and Graphical Statistics, 20(2), 316–334. 
https://doi.org/10.1198/jcgs.2011.09224 and computes both the functional band depth and 
modified functional band depth to visualize the centrality and variability of the time series data.
The data consist of monthly sea surface temperatures (SST) measured in degrees Celsius over
the east-central tropical Paciﬁc Ocean in degrees Celsius from January 1951 to December 2007.

import necessary libraries and load dataset

.. code-block:: python

    from uvisbox.Datasets import sea_surface_temp_data
    import matplotlib.pyplot as plt
    from uvisbox.Modules.FunctionalBoxplot import functional_boxplot

    data = sea_surface_temp_data.load_dataset()
    X = data.T

Create figure with three subplots and plot the original sea surface temperature time series

.. code-block:: python

    # create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    # plot the original time series in the first subplot
    for i in range(X.shape[0]):
        ax1.plot(X[i,:], alpha=1.)
    ax1.set_title("Sea Surface Temperature Time Series")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Temperature")

plot the functional boxplot for the time series data with different methods at [25,50,90,100] percentiles (default)

.. code-block:: python

    ax2 = functional_boxplot(X, ax=ax2)
    ax3 = functional_boxplot(X, method='mfdb', ax=ax3)
    plt.show()

.. image:: _static/sea_surface_temp_functional_banddepth_example.png
    :alt: Sea Surface Temperature Functional Band Depth Example
    :align: center
    
"""
# import necessary libraries and load dataset

from uvisbox.Datasets import sea_surface_temp_data
import matplotlib.pyplot as plt
from uvisbox.Modules.FunctionalBoxplot import functional_boxplot

data = sea_surface_temp_data.load_dataset()
X = data.T

# Create figure with three subplots and plot the original sea surface temperature time series

# create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
# plot the original time series in the first subplot
for i in range(X.shape[0]):
    ax1.plot(X[i,:], alpha=1.)
ax1.set_title("Sea Surface Temperature Time Series")
ax1.set_xlabel("Month")
ax1.set_ylabel("Temperature")

# plot the functional boxplot for the time series data with different methods at [25,50,90,100] percentiles (default)

ax2 = functional_boxplot(X, ax=ax2)
ax3 = functional_boxplot(X, method='mfbd', ax=ax3)

# plt.savefig("sea_surface_temp_functional_banddepth_example.png")
plt.show()