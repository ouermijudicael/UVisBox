import numpy as np
from uvisbox import functional_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig
import matplotlib.pyplot as plt


def gen_sine_curves(n, steps, noise_scale=0.2, seed=None):
    """
    Generate n noisy sine curves, each with 'steps' points over domain [-pi, pi].
    Noise increases as sine(x) approaches zero.

    Parameters:
    -----------
    n : int
        Number of curves to generate.
    steps : int
        Number of points per curve.
    noise_scale : float
        Base scale for noise (default: 0.2).
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    curves : np.ndarray
        Array of shape (n, steps) with noisy sine curves.
    x : np.ndarray
        Array of shape (steps,) with domain values.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(-np.pi, np.pi, steps)
    base = np.sin(x)
    # Noise level increases as |sin(x)| approaches zero
    noise_levels = noise_scale * (1 - np.abs(base)) ** 0.5 + 0.01
    curves = np.zeros((n, steps))
    noise_points = 10  # Number of anchor points for smooth noise
    anchor_x = np.linspace(-np.pi, np.pi, noise_points)
    for i in range(n):
        anchor_noise = np.random.randn(noise_points)
        smooth_noise = np.interp(x, anchor_x, anchor_noise)
        noise = smooth_noise * noise_levels
        curves[i] = base + noise
    return curves, x

if __name__ == "__main__":
    data = gen_sine_curves(n=500, steps=256, noise_scale=0.4, seed=42)[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax0,ax1 = ax

    ax0.set_title("Noisy Sine Curves")
    for curve in data:
        ax0.plot(curve, color='black', alpha=0.1)

    boxplot_style = BoxplotStyleConfig(
        percentiles=[10, 25, 50, 75],
        percentile_colormap='viridis',
        show_median=True,
        median_color='red',
        median_width=4,
        median_alpha=0.9,
        show_outliers=True,
        outliers_color='gray',
        outliers_width=1,
        outliers_alpha=0.2
    )
    functional_boxplot(data, ax=ax1, boxplot_style=boxplot_style)
    ax1.set_title("Functional Boxplot of Noisy Sine Curves")
    plt.show()