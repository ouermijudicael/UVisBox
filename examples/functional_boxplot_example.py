"""
This example demonstrates how to create functional boxplots using the ``uvisbox`` library.
It uses the sea surface temperature dataset from  Sun, Y., & Genton, M. G. (2011). 
Functional Boxplots. Journal of Computational and Graphical Statistics, 20(2), 316–334. 
https://doi.org/10.1198/jcgs.2011.09224 and computes both the functional band depth and 
modified functional band depth to visualize the centrality and variability of the time series data.
The data consist of monthly sea surface temperatures (SST) measured in degrees Celsius over
the east-central tropical Paciﬁc Ocean in degrees Celsius from January 1951 to December 2007.

Import necessary libraries

.. code-block:: python

    from uvisbox.Datasets import sea_surface_temp_data
    from uvisbox.BandDepths import functional_banddepth_plot
    from uvisbox.BandDepths import modified_functional_banddepth, modified_functional
    import matplotlib.pyplot as plt

Load the sea surface temperature dataset and prepare the data

.. code-block:: python

    data = sea_surface_temp_data.load_dataset()
    X = data.T
    print(X.shape)
    # Plot the original time series a
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    for i in range(X.shape[0]):
        ax1.plot(X[i,:], alpha=1.)
    ax1.set_title("Sea Surface Temperature Time Series")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Temperature")
    
    # Compute functional band depth and plot
    ax2 = functional_banddepth_plot(X, percentil=50, scale=1.0, ax=ax2)

    # Compute modified functional band depth 
    mfbd_depths = modified_functional_banddepth(X)

    # Plot modified functional band depth
    ax3 = modified_functional_banddepth_plot(X, curves_depths=mfbd_depths, percentil=10, scale=1.0, ax=ax3)

    plt.savefig("sea_surface_temp_functional_banddepth.png")
    plt.show()

"""

from uvisbox.Datasets import sea_surface_temp_data
from uvisbox.BandDepths import functional_banddepth_plot
from uvisbox.BandDepths import modified_functional_banddepth, modified_functional_banddepth_plot
import matplotlib.pyplot as plt

# This example uses the sea surface temperature dataset taken from Sun, Y., & Genton, M. G. (2011). 
# Functional Boxplots. Journal of Computational and Graphical Statistics, 20(2), 316–334. 
# https://doi.org/10.1198/jcgs.2011.09224
# The data consist of monthly sea surface temperatures (SST) measured in degrees Celsius over 
# the east-central tropical Paciﬁc Ocean in degrees Celsius from January 1951 to December 2007.
#

# load data
data = sea_surface_temp_data.load_dataset()
X = data.T
print(X.shape)

# Plot the original time series a
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
for i in range(X.shape[0]):
    ax1.plot(X[i,:], alpha=1.)
ax1.set_title("Sea Surface Temperature Time Series")
ax1.set_xlabel("Month")
ax1.set_ylabel("Temperature")

# Compute functional band depth and plot
ax2 = functional_banddepth_plot(X, percentil=50, scale=1.0, ax=ax2)

# Compute modified functional band depth 
mfbd_depths = modified_functional_banddepth(X)

# Plot modified functional band depth
ax3 = modified_functional_banddepth_plot(X, curves_depths=mfbd_depths, percentil=10, scale=1.0, ax=ax3)

plt.savefig("sea_surface_temp_functional_banddepth_example.png")
plt.show()