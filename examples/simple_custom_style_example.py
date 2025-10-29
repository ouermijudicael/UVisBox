"""
This example demonstrates basic customization of functional boxplots using the
``BoxplotStyleConfig`` class. It shows how to:

- Use different colormaps (viridis vs hot)
- Customize median appearance (color, width, transparency)
- Control which elements are displayed (median without outliers)
- Add colorbars to indicate percentile values

The example uses sea surface temperature data and creates two side-by-side plots
to compare different colormap and styling choices.

Import necessary libraries and load dataset

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from uvisbox.Datasets import sea_surface_temp_data
    from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
    from uvisbox.Core.CommonInterface import BoxplotStyleConfig

    # Load data
    data = sea_surface_temp_data.load_dataset()
    X = data.T

Create figure with 2 subplots side by side

.. code-block:: python

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

Left plot: Default viridis colormap

.. code-block:: python

    style_viridis = BoxplotStyleConfig(
        percentiles=[25, 50, 75, 90],
        percentile_colormap='viridis',
        show_median=True,
        median_color='red',
        median_width=2.5,
        show_outliers=False
    )
    functional_boxplot(X, boxplot_style=style_viridis, ax=ax1)
    ax1.set_title('Viridis Colormap')

Right plot: Hot colormap with custom median styling

.. code-block:: python

    style_hot = BoxplotStyleConfig(
        percentiles=[10, 30, 50, 70, 90],
        percentile_colormap='hot',
        show_median=True,
        median_color='darkblue',
        median_width=3.0,
        median_alpha=0.9,
        show_outliers=False
    )
    functional_boxplot(X, boxplot_style=style_hot, ax=ax2)
    ax2.set_title('Hot Colormap')
    
    plt.tight_layout()
    plt.show()

.. image:: _static/simple_custom_style_example.png
    :alt: Simple Custom Boxplot Style Example
    :align: center

"""

import numpy as np
import matplotlib.pyplot as plt
from uvisbox.Datasets import sea_surface_temp_data
from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

# Load data
print("Loading sea surface temperature data...")
data = sea_surface_temp_data.load_dataset()
X = data.T
print(f"Dataset: {X.shape[0]} curves, {X.shape[1]} time points\n")

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ============================================================================
# Left plot: Default viridis colormap
# ============================================================================
print("Creating plot 1: Viridis colormap (default)")
style_viridis = BoxplotStyleConfig(
    percentiles=[25, 50, 75, 90],      # Percentile bands to show
    percentile_colormap='viridis',     # Colormap for bands
    show_median=True,                  # Show the median curve
    median_color='red',                # Median color
    median_width=2.5,                  # Median line width
    show_outliers=False                # Hide outliers
)

functional_boxplot(X, boxplot_style=style_viridis, ax=ax1)
ax1.set_title('Viridis Colormap', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Temperature (°C)')

# Add colorbar to show percentile mapping
sm1 = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=25, vmax=90))
sm1.set_array([])
cbar1 = plt.colorbar(sm1, ax=ax1)
cbar1.set_label('Percentile', rotation=270, labelpad=15)

# ============================================================================
# Right plot: Hot colormap with custom styling
# ============================================================================
print("Creating plot 2: Hot colormap with custom median styling")
style_hot = BoxplotStyleConfig(
    percentiles=[10, 30, 50, 70, 90],  # Different percentiles
    percentile_colormap='hot',         # Different colormap
    show_median=True,
    median_color='darkblue',           # Different median color
    median_width=3.0,                  # Thicker median line
    median_alpha=0.9,                  # Slightly transparent
    show_outliers=False
)

functional_boxplot(X, boxplot_style=style_hot, ax=ax2)
ax2.set_title('Hot Colormap', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (months)')
ax2.set_ylabel('Temperature (°C)')

# Add colorbar
sm2 = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=10, vmax=90))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2)
cbar2.set_label('Percentile', rotation=270, labelpad=15)

# Overall title
fig.suptitle('Functional Boxplot - Custom Styling with Colorbars', 
             fontsize=13, fontweight='bold')

plt.tight_layout()

print("\n" + "="*60)
print("Key customization options demonstrated:")
print("  • percentiles: Which bands to display")
print("  • percentile_colormap: Visual appearance of bands")
print("  • show_median: Whether to show median curve")
print("  • median_color/width/alpha: Median styling")
print("  • show_outliers: Whether to show outlier curves")
print("="*60)
print("\nDisplaying plot...")

# plt.savefig("simple_custom_style_example.png", dpi=150, bbox_inches='tight')
plt.show()
