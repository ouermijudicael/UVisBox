"""
This example demonstrates advanced customization of functional boxplots using the
``BoxplotStyleConfig`` class. It showcases:

- Different colormap options for percentile bands (viridis, plasma, coolwarm, custom)
- Custom median styling (color, width, transparency)
- Colorbars to indicate percentile values
- Comparison of multiple styling approaches in a 2x2 grid

The example uses sea surface temperature data and creates four subplots with different
visual styles to demonstrate the flexibility of the configuration system.

Import necessary libraries and load dataset

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from uvisbox.Datasets import sea_surface_temp_data
    from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
    from uvisbox.Core.CommonInterface import BoxplotStyleConfig

    # Load sea surface temperature data
    data = sea_surface_temp_data.load_dataset()
    X = data.T

Create figure with 4 subplots (2x2 grid) showing different colormap styles

.. code-block:: python

    fig = plt.figure(figsize=(16, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

Plot 1: Viridis colormap with custom percentiles

.. code-block:: python

    ax1 = fig.add_subplot(gs[0, 0])
    style1 = BoxplotStyleConfig(
        percentiles=[10, 25, 50, 75, 90],
        percentile_colormap='viridis',
        show_median=True,
        median_color='red',
        median_width=2.5,
        show_outliers=False
    )
    functional_boxplot(X, boxplot_style=style1, ax=ax1, method='fdb')
    ax1.set_title('Viridis Colormap - Multiple Percentiles')

Plot 2: Plasma colormap with fewer percentiles

.. code-block:: python

    ax2 = fig.add_subplot(gs[0, 1])
    style2 = BoxplotStyleConfig(
        percentiles=[25, 50, 75],
        percentile_colormap='plasma',
        show_median=True,
        median_color='cyan',
        median_width=3.0,
        median_alpha=0.9,
        show_outliers=False
    )
    functional_boxplot(X, boxplot_style=style2, ax=ax2, method='fdb')
    ax2.set_title('Plasma Colormap - Fewer Bands')

Plot 3: Coolwarm colormap for diverging representation

.. code-block:: python

    ax3 = fig.add_subplot(gs[1, 0])
    style3 = BoxplotStyleConfig(
        percentiles=[10, 30, 50, 70, 90],
        percentile_colormap='coolwarm',
        show_median=True,
        median_color='black',
        median_width=2.0,
        show_outliers=False
    )
    functional_boxplot(X, boxplot_style=style3, ax=ax3, method='fdb')
    ax3.set_title('Coolwarm Colormap - Diverging Colors')

Plot 4: Custom LinearSegmentedColormap for specialized visualization

.. code-block:: python

    ax4 = fig.add_subplot(gs[1, 1])
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_ocean', 
        ['lightblue', 'steelblue', 'darkblue', 'midnightblue']
    )
    style4 = BoxplotStyleConfig(
        percentiles=[20, 40, 60, 80],
        percentile_colormap=custom_cmap,
        show_median=True,
        median_color='orange',
        median_width=2.5,
        median_alpha=1.0,
        show_outliers=False
    )
    functional_boxplot(X, boxplot_style=style4, ax=ax4, method='fdb')
    ax4.set_title('Custom Ocean Colormap')
    
    plt.tight_layout()
    plt.show()

.. image:: _static/custom_boxplot_style_example.png
    :alt: Custom Boxplot Styling Example
    :align: center

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from uvisbox.Datasets import sea_surface_temp_data
from uvisbox.Modules.FunctionalBoxplot import functional_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

# Load sea surface temperature data
print("=" * 70)
print("Custom Boxplot Styling Example")
print("=" * 70)
print("\nLoading sea surface temperature data...")
data = sea_surface_temp_data.load_dataset()
X = data.T
print(f"Dataset shape: {X.shape} (curves × time points)")

# Create figure with 4 subplots (2x2)
fig = plt.figure(figsize=(16, 12))

# Create a grid for better layout control
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

print("\nCreating functional boxplots with different styles...")

# ============================================================================
# Plot 1: Viridis colormap (default) with custom percentiles
# ============================================================================
print("\n1. Viridis colormap with custom percentiles [10, 25, 50, 75, 90]")
ax1 = fig.add_subplot(gs[0, 0])

style1 = BoxplotStyleConfig(
    percentiles=[10, 25, 50, 75, 90],
    percentile_colormap='viridis',
    show_median=True,
    median_color='red',
    median_width=2.5,
    median_alpha=1.0,
    show_outliers=False
)

functional_boxplot(X, boxplot_style=style1, ax=ax1, method='fdb')
ax1.set_title('Viridis Colormap (Default)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Temperature (°C)')

# Add colorbar
sm1 = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=10, vmax=90))
sm1.set_array([])
cbar1 = plt.colorbar(sm1, ax=ax1, orientation='vertical', pad=0.02)
cbar1.set_label('Percentile', rotation=270, labelpad=20)

# ============================================================================
# Plot 2: Plasma colormap with tight percentiles
# ============================================================================
print("2. Plasma colormap with tight percentiles [25, 50, 75]")
ax2 = fig.add_subplot(gs[0, 1])

style2 = BoxplotStyleConfig(
    percentiles=[25, 50, 75],
    percentile_colormap='plasma',
    show_median=True,
    median_color='darkblue',
    median_width=3.0,
    median_alpha=0.9,
    show_outliers=False
)

functional_boxplot(X, boxplot_style=style2, ax=ax2, method='fdb')
ax2.set_title('Plasma Colormap - Tight Bands', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (months)')
ax2.set_ylabel('Temperature (°C)')

# Add colorbar
sm2 = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=25, vmax=75))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', pad=0.02)
cbar2.set_label('Percentile', rotation=270, labelpad=20)

# ============================================================================
# Plot 3: Coolwarm colormap with wide percentiles
# ============================================================================
print("3. Coolwarm colormap with wide percentiles [5, 25, 50, 75, 95]")
ax3 = fig.add_subplot(gs[1, 0])

style3 = BoxplotStyleConfig(
    percentiles=[5, 25, 50, 75, 95],
    percentile_colormap='coolwarm',
    show_median=True,
    median_color='black',
    median_width=2.0,
    median_alpha=1.0,
    show_outliers=False
)

functional_boxplot(X, boxplot_style=style3, ax=ax3, method='fdb')
ax3.set_title('Coolwarm Colormap - Wide Bands', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time (months)')
ax3.set_ylabel('Temperature (°C)')

# Add colorbar
sm3 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=5, vmax=95))
sm3.set_array([])
cbar3 = plt.colorbar(sm3, ax=ax3, orientation='vertical', pad=0.02)
cbar3.set_label('Percentile', rotation=270, labelpad=20)

# ============================================================================
# Plot 4: Custom colormap (blue gradient)
# ============================================================================
print("4. Custom blue gradient colormap")
ax4 = fig.add_subplot(gs[1, 1])

# Create custom colormap from light blue to dark blue
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_blue',
    ['#e0f3ff', '#a0d8ef', '#5fa8d3', '#2e5f8a', '#1a3a52']
)

style4 = BoxplotStyleConfig(
    percentiles=[10, 30, 50, 70, 90],
    percentile_colormap=custom_cmap,
    show_median=True,
    median_color='crimson',
    median_width=2.5,
    median_alpha=1.0,
    show_outliers=False
)

functional_boxplot(X, boxplot_style=style4, ax=ax4, method='fdb')
ax4.set_title('Custom Blue Gradient', fontsize=12, fontweight='bold')
ax4.set_xlabel('Time (months)')
ax4.set_ylabel('Temperature (°C)')

# Add colorbar with custom colormap
sm4 = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=10, vmax=90))
sm4.set_array([])
cbar4 = plt.colorbar(sm4, ax=ax4, orientation='vertical', pad=0.02)
cbar4.set_label('Percentile', rotation=270, labelpad=20)

# Add overall title
fig.suptitle('Custom Boxplot Styling Examples\nSea Surface Temperature Data', 
             fontsize=14, fontweight='bold', y=0.98)

print("\n" + "=" * 70)
print("Styling Summary:")
print("  • All plots show median (red/darkblue/black/crimson)")
print("  • No outliers shown (show_outliers=False)")
print("  • Each plot uses different colormap and percentile ranges")
print("  • Colorbars show percentile mapping for each plot")
print("=" * 70)

plt.tight_layout()
# plt.savefig("custom_boxplot_style_example.png", dpi=300, bbox_inches='tight')
print("\nDisplaying plot...")
plt.show()
