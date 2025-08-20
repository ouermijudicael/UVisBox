import numpy as np
import matplotlib.pyplot as plt
from uvisbox.BandDepths.contour_boxplot import contour_boxplot


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

if __name__ == "__main__":
    # Example usage
    ensemble = create_ensemble_scalarfield(sigma_min=30)
    binary_images = (ensemble > 0.5).astype(np.bool_)
    print(f"Ensemble shape: {binary_images.shape}")
    fig, ax = plt.subplots()
    ax = contour_boxplot(binary_images, ax=ax, show_median=True, show_iqr=True, show_non_outliers=True, show_outliers=True, show_firstquartile=True, outlier_percentile=95)
    plt.show()