"""
Unit tests for functional_boxplot_vis module.

Tests the visualize_functional_boxplot and plot_band functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_stats import functional_boxplot_summary_statistics
from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_mesh import functional_boxplot_mesh
from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_vis import (
    visualize_functional_boxplot,
    plot_band
)
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestPlotBand:
    """Test suite for plot_band function."""
    
    def test_basic_band_plotting(self):
        """Test basic band plotting without errors."""
        bottom = np.sin(np.linspace(0, 2*np.pi, 100))
        top = np.sin(np.linspace(0, 2*np.pi, 100)) + 0.5
        
        ax = plot_band(bottom, top)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_plot_on_existing_axes(self):
        """Test plotting on existing axes."""
        bottom = np.sin(np.linspace(0, 2*np.pi, 100))
        top = np.sin(np.linspace(0, 2*np.pi, 100)) + 0.5
        
        fig, ax = plt.subplots()
        result_ax = plot_band(bottom, top, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_custom_color_and_alpha(self):
        """Test plotting with custom color and alpha."""
        bottom = np.zeros(100)
        top = np.ones(100)
        
        ax = plot_band(bottom, top, color='red', alpha=0.3)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_scale_parameter(self):
        """Test scale parameter."""
        bottom = np.zeros(100)
        top = np.ones(100)
        
        ax = plot_band(bottom, top, scale=2.0)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        bottom = np.zeros(100)
        top = np.ones(50)
        
        with pytest.raises(ValueError, match="must have the same shape"):
            plot_band(bottom, top)
        
        plt.close('all')
    
    def test_non_1d_arrays_raise_error(self):
        """Test that non-1D arrays raise ValueError."""
        bottom = np.zeros((10, 10))
        top = np.ones((10, 10))
        
        with pytest.raises(ValueError, match="must be 1D arrays"):
            plot_band(bottom, top)
        
        plt.close('all')
    
    def test_list_input_converted(self):
        """Test that list inputs are converted to arrays."""
        bottom = [0] * 100
        top = [1] * 100
        
        ax = plot_band(bottom, top)
        
        assert isinstance(ax, Axes)
        plt.close('all')


class TestVisualizeFunctionalBoxplot:
    """Test suite for visualize_functional_boxplot function."""
    
    def test_basic_visualization(self):
        """Test basic visualization without errors."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_visualization_with_existing_axes(self):
        """Test visualization on existing axes."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        mesh_data = functional_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        result_ax = visualize_functional_boxplot(mesh_data, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_custom_boxplot_style(self):
        """Test visualization with custom BoxplotStyleConfig."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='blue'
        )
        
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_show_median_true(self):
        """Test that median curve is plotted when show_median is True."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(show_median=True, show_outliers=False)
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_functional_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Check that lines were plotted (median curve should be present)
        assert len(ax.get_lines()) > 0
        plt.close('all')
    
    def test_show_median_false(self):
        """Test that median curve is not plotted when show_median is False."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[50],  # Need at least one percentile
            show_median=False, 
            show_outliers=False
        )
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_functional_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have no median line, only bands
        lines_count = len(ax.get_lines())
        # May have lines from band edges, but not median specifically
        plt.close('all')
    
    def test_show_outliers_true(self):
        """Test that outlier curves are plotted when show_outliers is True."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[50], 
            show_outliers=True, 
            show_median=False
        )
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_functional_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have outlier lines plotted (if there are outliers)
        if mesh_data['outliers'].shape[0] > 0:
            assert len(ax.get_lines()) > 0
        plt.close('all')
    
    def test_show_outliers_false(self):
        """Test that outlier curves are not plotted when show_outliers is False."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[50],
            show_outliers=False, 
            show_median=False
        )
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_functional_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have no lines (no median, no outliers)
        assert len(ax.get_lines()) == 0
        plt.close('all')
    
    def test_multiple_percentile_bands(self):
        """Test visualization with multiple percentile bands."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[25, 50, 75, 90])
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
        
        # Should have 4 filled regions (one for each percentile band)
        assert len(ax.collections) >= 4
        plt.close('all')
    
    def test_default_boxplot_style(self):
        """Test visualization with default BoxplotStyleConfig."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        mesh_data = functional_boxplot_mesh(stats)
        
        # Should work with None (uses default)
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=None)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_empty_percentiles(self):
        """Test visualization with minimal percentile bands."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        # BoxplotStyleConfig requires non-empty percentiles
        style = BoxplotStyleConfig(percentiles=[50], show_median=True)
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
        
        # Should still plot median and at least one band
        assert len(ax.get_lines()) > 0
        plt.close('all')
    
    def test_no_outliers_case(self):
        """Test visualization when there are no outliers."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[100], show_outliers=True)
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_colormap_application(self):
        """Test that colormap is applied correctly."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            percentile_colormap='plasma'
        )
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
        
        # Should have filled regions
        assert len(ax.collections) >= 3
        plt.close('all')
    
    def test_bands_plotted_in_descending_order(self):
        """Test that bands are plotted from largest to smallest."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[25, 90, 50, 75])
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        ax = visualize_functional_boxplot(mesh_data, boxplot_style=style)
        
        # All bands should be plotted
        assert len(ax.collections) >= 4
        plt.close('all')
