"""
This example demonstrates the use of contour boxplots to visualize the variability in an ensemble of 2D scalar fields.
It generates a synthetic ensemble of scalar fields with Gaussian blobs and visualizes the ensemble contours
using spaghetti plots and contour boxplots.

import necessary libraries

.. code-block:: python

    from flask.config import T
    import numpy as np
    import matplotlib.pyplot as plt
    from uvisbox.Modules.ContourBoxplot import contour_boxplot
    from uvisbox.Core.CommonInterface import BoxplotStyleConfig

.. code-block:: python

    def create_ensemble_scalarfield(image_res=256, n_ensembles=30, sigma_min=5, sigma_max=50):
        # 
        # Create an ensemble of 2D scalar fields with Gaussian blobs in the center.
        # Args:
        #     image_res (int): Resolution of the image (image_res x image_res).
        #     n_ensembles (int): Number of ensemble members.
        #     sigma_min (float): Minimum sigma for Gaussian.
        #     sigma_max (float): Maximum sigma for Gaussian.
        # Returns:
        #     np.ndarray: Array of shape (n_ensembles, image_res, image_res).
        #
        x = np.linspace(0, image_res-1, image_res)
        y = np.linspace(0, image_res-1, image_res)
        xx, yy = np.meshgrid(x, y)
        grid = np.stack([xx, yy], axis=-1)
        ensemble = []
        for i in range(n_ensembles):
            sigma = np.random.uniform(sigma_min, sigma_max)
            cov = np.array([[sigma**2, 0], [0, sigma**2]])
            mu = np.array([image_res/2, image_res/2])
            inv_cov = np.linalg.inv(cov)
            diff = grid - mu
            pdf = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=-1))
            # Normalize to [-1, 1]
            pdf = 2 * (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf)) - 1
            ensemble.append(pdf)

        return np.array(ensemble)

Generate a synthetic ensemble of scalar fields and visualize using spaghetti 

.. code-block:: python

    # Generate synthetic ensemble of scalar fields
    ensemble = create_ensemble_scalarfield(image_res=256, n_ensembles=100, sigma_min=20, sigma_max=100) 

    # Visualize the ensemble contours using spaghetti 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # Spaghetti plot - show contours at isovalue = 0.7
    for i in range(ensemble.shape[0]):
        ax[0].contour(ensemble[i], levels=[0.7], colors='black', linewidths=1, alpha=0.3) 
    ax[0].set_title("Ensemble Contours (Spaghetti Plot)")
    ax[0].set_aspect('equal', adjustable='box')

Calculate and plot the contour boxplot using the new interface

.. code-block:: python

    style = BoxplotStyleConfig(
        percentiles=[25, 50, 75, 95],
        percentile_colormap='Oranges',  # Use magma colormap for visualization
        show_median=True,
        show_outliers=True
    )
    ax[1] = contour_boxplot(
        ensemble, 
        isovalue=0.7,
        boxplot_style=style,
        ax=ax[1],
        workers=6
    )
    ax[1].set_title("Contour Boxplot")
    ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


.. image:: _static/contour_boxplot_example.png
   :alt: Contour Boxplot Example
   :align: center

"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from uvisbox.Modules.ContourBoxplot import contour_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

def create_ensemble_scalarfield(image_res=256, n_ensembles=30, sigma_min=5, sigma_max=50):
    """
    Create an ensemble of 2D scalar fields with Gaussian blobs in the center.
    Args:
        image_res (int): Resolution of the image (image_res x image_res).
        n_ensembles (int): Number of ensemble members.
        sigma_min (float): Minimum sigma for Gaussian.
        sigma_max (float): Maximum sigma for Gaussian.
    Returns:
        np.ndarray: Array of shape (n_ensembles, image_res, image_res).
    """
    x = np.linspace(0, image_res-1, image_res)
    y = np.linspace(0, image_res-1, image_res)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)
    ensemble = []
    for i in range(n_ensembles):
        sigma = np.random.uniform(sigma_min, sigma_max)
        cov = np.array([[sigma**2, 0], [0, sigma**2]])
        mu = np.array([image_res/2, image_res/2])
        inv_cov = np.linalg.inv(cov)
        diff = grid - mu
        pdf = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=-1))
        # Normalize to [-1, 1]
        pdf = 2 * (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf)) - 1
        ensemble.append(pdf)

    return np.array(ensemble)

# Generate a synthetic ensemble of scalar fields and visualize using spaghetti 

# Generate synthetic ensemble of scalar fields
ensemble = create_ensemble_scalarfield(image_res=256, n_ensembles=100, sigma_min=20, sigma_max=100) 

# Visualize the ensemble contours using spaghetti 
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

# Spaghetti plot - show contours at isovalue = 0.7
for i in range(ensemble.shape[0]):
    ax[0].contour(ensemble[i], levels=[0.7], colors='black', linewidths=1, alpha=0.3) 
ax[0].set_title("Ensemble Contours (Spaghetti Plot)")
ax[0].set_aspect('equal', adjustable='box')

# Calculate and plot the contour boxplot using the new interface
style = BoxplotStyleConfig(
    percentiles=[25, 50, 75, 95],
    percentile_colormap='Oranges',  # Use magma colormap for visualization
    show_median=True,
    show_outliers=True
)
ax[1] = contour_boxplot(
    ensemble, 
    isovalue=0.7,
    boxplot_style=style,
    ax=ax[1],
    workers=6
)
ax[1].set_title("Contour Boxplot")
ax[1].set_aspect('equal', adjustable='box')

plt.tight_layout()
# plt.savefig("contour_boxplot_example.png", dpi=300)
plt.show()
