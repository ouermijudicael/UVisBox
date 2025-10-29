"""
Test that all three boxplot modules work with BoxplotStyleConfig
"""
import numpy as np
from uvisbox.Core.CommonInterface import BoxplotStyleConfig
from uvisbox.Modules.CurveBoxplot import curve_boxplot
from uvisbox.Modules.ContourBoxplot import contour_boxplot
from uvisbox.Modules.FunctionalBoxplot import functional_boxplot


def test_curve_boxplot_with_config():
    """Test curve_boxplot with BoxplotStyleConfig"""
    np.random.seed(42)
    n_curves = 20
    n_steps = 50
    t = np.linspace(0, 2 * np.pi, n_steps)
    base_curve = np.zeros((n_steps, 2))
    base_curve[:, 0] = t
    base_curve[:, 1] = np.sin(t)
    
    curves = np.zeros((n_curves, n_steps, 2))
    for i in range(n_curves):
        curve = base_curve.copy()
        curve[:, 1] += np.random.normal(0, 0.2, n_steps)
        curves[i] = curve
    
    style = BoxplotStyleConfig(
        percentiles=[50, 90],
        percentile_colormap='plasma',
        show_median=True,
        show_outliers=False
    )
    
    ax = curve_boxplot(curves, boxplot_style=style)
    assert ax is not None
    print("✓ curve_boxplot works with BoxplotStyleConfig and colormap")


def test_contour_boxplot_with_config():
    """Test contour_boxplot with BoxplotStyleConfig"""
    np.random.seed(42)
    ensemble = np.random.randn(20, 50, 50)
    
    style = BoxplotStyleConfig(
        percentiles=[25, 50, 75],
        percentile_colormap='hot',
        show_median=True,
        show_outliers=False
    )
    
    ax = contour_boxplot(ensemble, isovalue=0.5, boxplot_style=style, workers=2)
    assert ax is not None
    print("✓ contour_boxplot works with BoxplotStyleConfig and colormap")


def test_functional_boxplot_with_config():
    """Test functional_boxplot with BoxplotStyleConfig"""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    data = np.array([np.sin(2*np.pi*t) + 0.1*np.random.randn(100) for _ in range(30)])
    
    style = BoxplotStyleConfig(
        percentiles=[25, 50, 75],
        percentile_colormap='coolwarm',
        show_median=True
    )
    
    ax = functional_boxplot(data, boxplot_style=style)
    assert ax is not None
    print("✓ functional_boxplot works with BoxplotStyleConfig and colormap")


def test_all_with_defaults():
    """Test all boxplots work with default config (None)"""
    np.random.seed(42)
    
    # Curve
    t = np.linspace(0, 2 * np.pi, 30)
    curves = np.stack([np.column_stack([t, np.sin(t) + np.random.randn(30)*0.1]) for _ in range(15)])
    ax1 = curve_boxplot(curves)
    assert ax1 is not None
    
    # Contour
    ensemble = np.random.randn(15, 30, 30)
    ax2 = contour_boxplot(ensemble, isovalue=0.0, workers=2)
    assert ax2 is not None
    
    # Functional
    data = np.random.randn(15, 50)
    ax3 = functional_boxplot(data)
    assert ax3 is not None
    
    print("✓ All boxplots work with default config (None)")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    
    print("Testing BoxplotStyleConfig integration...")
    print()
    
    test_curve_boxplot_with_config()
    plt.close('all')
    
    test_contour_boxplot_with_config()
    plt.close('all')
    
    test_functional_boxplot_with_config()
    plt.close('all')
    
    test_all_with_defaults()
    plt.close('all')
    
    print()
    print("=" * 60)
    print("All tests passed! BoxplotStyleConfig successfully integrated.")
    print("=" * 60)
