"""
Integration tests for CurveBoxplot module.

Tests the full pipeline: stats -> mesh -> vis
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from uvisbox.Modules.CurveBoxplot import curve_boxplot
from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curve_boxplot_mesh
from uvisbox.Modules.CurveBoxplot.curve_boxplot_vis import visualize_curve_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class TestCurveBoxplotIntegration:
    """Integration tests for the complete curve boxplot pipeline."""
    
    def test_full_pipeline_2d(self):
        """Test complete pipeline from curves to visualization (2D)."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # Run through full pipeline
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        ax = visualize_curve_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_full_pipeline_3d(self):
        """Test complete pipeline from curves to visualization (3D)."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        # Run through full pipeline
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        ax = visualize_curve_boxplot(mesh_data)
        
        # Could be either Axes3D (matplotlib) or PyVista Plotter
        if PYVISTA_AVAILABLE:
            assert isinstance(ax, (Axes3D, pv.Plotter))
        else:
            assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_main_curve_boxplot_function_2d(self):
        """Test main curve_boxplot function with 2D curves."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        ax = curve_boxplot(curves)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_main_curve_boxplot_function_3d(self):
        """Test main curve_boxplot function with 3D curves."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        ax = curve_boxplot(curves)
        
        # Could be either Axes3D (matplotlib) or PyVista Plotter
        if PYVISTA_AVAILABLE:
            assert isinstance(ax, (Axes3D, pv.Plotter))
        else:
            assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_curve_boxplot_with_custom_style(self):
        """Test curve_boxplot with custom BoxplotStyleConfig."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='gray'
        )
        
        ax = curve_boxplot(curves, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_curve_boxplot_on_existing_axes_2d(self):
        """Test curve_boxplot on existing 2D axes."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        fig, ax = plt.subplots()
        result_ax = curve_boxplot(curves, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_curve_boxplot_on_existing_axes_3d(self):
        """Test curve_boxplot on existing 3D axes."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        result_ax = curve_boxplot(curves, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_curve_boxplot_with_workers_parameter(self):
        """Test curve_boxplot with different worker counts."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # Test with single worker
        ax1 = curve_boxplot(curves, workers=1)
        assert isinstance(ax1, Axes)
        
        # Test with multiple workers
        ax2 = curve_boxplot(curves, workers=4)
        assert isinstance(ax2, Axes)
        
        plt.close('all')
    
    def test_pipeline_preserves_data_consistency(self):
        """Test that data flows consistently through the pipeline."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(percentiles=[25, 50, 75])
        
        # Run through pipeline
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Check data consistency
        assert mesh_data['n_dims'] == stats['n_dims']
        np.testing.assert_array_equal(mesh_data['median_curve'], stats['median_curve'])
        np.testing.assert_array_equal(mesh_data['outliers'], stats['outliers'])
        
        # Check that all percentile meshes were created
        for percentile in style.percentiles:
            mesh_key = f'{int(percentile)}_percentile_mesh'
            assert mesh_key in mesh_data['percentile_meshes']
        
        plt.close('all')
    
    def test_input_validation_invalid_dimensions(self):
        """Test that invalid input dimensions raise appropriate errors."""
        # 2D array instead of 3D
        invalid_curves = np.random.randn(30, 50)
        
        with pytest.raises(ValueError, match="3D array"):
            curve_boxplot(invalid_curves)
        
        plt.close('all')
    
    def test_multiple_percentiles_all_rendered(self):
        """Test that multiple percentiles are all rendered correctly."""
        n_curves, n_steps = 40, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        percentiles = [10, 30, 50, 70, 90]
        style = BoxplotStyleConfig(percentiles=percentiles, show_median=False, show_outliers=False)
        
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Verify all percentile meshes exist
        assert len(mesh_data['percentile_meshes']) == len(percentiles)
        for p in percentiles:
            assert f'{int(p)}_percentile_mesh' in mesh_data['percentile_meshes']
        
        # Visualize
        ax = visualize_curve_boxplot(mesh_data, boxplot_style=style)
        assert isinstance(ax, Axes)
        
        plt.close('all')
    
    def test_median_and_outliers_rendered(self):
        """Test that median and outliers are rendered when enabled."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(
            percentiles=[50],
            show_median=True,
            show_outliers=True
        )
        
        fig, ax = plt.subplots()
        curve_boxplot(curves, boxplot_style=style, ax=ax)
        
        # Should have lines (median + outliers)
        assert len(ax.get_lines()) > 0
        
        plt.close('all')
    
    def test_no_median_no_outliers_rendered(self):
        """Test visualization with median and outliers disabled."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(
            percentiles=[50],
            show_median=False,
            show_outliers=False
        )
        
        fig, ax = plt.subplots()
        curve_boxplot(curves, boxplot_style=style, ax=ax)
        
        # Should have no lines
        assert len(ax.get_lines()) == 0
        
        plt.close('all')
    
    def test_realistic_curve_data_2d(self):
        """Test with realistic 2D curve data (sine waves with noise)."""
        n_curves, n_steps = 50, 100
        t = np.linspace(0, 4 * np.pi, n_steps)
        
        curves = np.zeros((n_curves, n_steps, 2))
        for i in range(n_curves):
            curves[i, :, 0] = t
            curves[i, :, 1] = np.sin(t) + np.random.normal(0, 0.2, n_steps)
        
        style = BoxplotStyleConfig(percentiles=[25, 50, 75], show_median=True)
        ax = curve_boxplot(curves, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_realistic_curve_data_3d(self):
        """Test with realistic 3D curve data (spiral with noise)."""
        n_curves, n_steps = 30, 100
        t = np.linspace(0, 4 * np.pi, n_steps)
        
        curves = np.zeros((n_curves, n_steps, 3))
        for i in range(n_curves):
            radius = 1 + np.random.normal(0, 0.1)
            curves[i, :, 0] = radius * np.cos(t) + np.random.normal(0, 0.05, n_steps)
            curves[i, :, 1] = radius * np.sin(t) + np.random.normal(0, 0.05, n_steps)
            curves[i, :, 2] = t + np.random.normal(0, 0.1, n_steps)
        
        style = BoxplotStyleConfig(percentiles=[50], show_median=True)
        ax = curve_boxplot(curves, boxplot_style=style)
        
        # Could be either Axes3D (matplotlib) or PyVista Plotter
        if PYVISTA_AVAILABLE:
            assert isinstance(ax, (Axes3D, pv.Plotter))
        else:
            assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_small_curve_set(self):
        """Test with very small number of curves (edge case)."""
        n_curves, n_steps = 5, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        ax = curve_boxplot(curves)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_large_curve_set(self):
        """Test with large number of curves."""
        n_curves, n_steps = 100, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(percentiles=[50])
        ax = curve_boxplot(curves, boxplot_style=style, workers=4)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_default_configuration(self):
        """Test curve_boxplot with all default parameters."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # Call with only curves parameter
        ax = curve_boxplot(curves)
        
        assert isinstance(ax, Axes)
        plt.close('all')
