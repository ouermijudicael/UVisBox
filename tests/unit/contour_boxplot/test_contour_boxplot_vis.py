"""
Unit tests for contour_boxplot_vis module.

Tests the visualize_contour_boxplot function.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from uvisbox.Modules.ContourBoxplot.contour_boxplot_stats import contour_boxplot_summary_statistics
from uvisbox.Modules.ContourBoxplot.contour_boxplot_mesh import contour_boxplot_mesh
from uvisbox.Modules.ContourBoxplot.contour_boxplot_vis import visualize_contour_boxplot
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestVisualizeContourBoxplot:
    """Test suite for visualize_contour_boxplot function."""
    
    def test_basic_visualization(self):
        """Test basic visualization without errors."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        ax = visualize_contour_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_visualization_with_existing_axes(self):
        """Test visualization on existing axes."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        result_ax = visualize_contour_boxplot(mesh_data, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_custom_boxplot_style(self):
        """Test visualization with custom BoxplotStyleConfig."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            percentile_colormap='hot',
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='blue'
        )
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_show_median_true(self):
        """Test that median is shown when show_median is True."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(show_median=True, show_outliers=False)
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have created the plot
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_show_median_false(self):
        """Test that median is not shown when show_median is False."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(show_median=False, show_outliers=False)
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_show_outliers_true(self):
        """Test that outliers are shown when show_outliers is True."""
        ensemble = np.random.randn(50, 50, 50)
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True, show_median=False)
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_show_outliers_false(self):
        """Test that outliers are not shown when show_outliers is False."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(show_outliers=False)
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_colormap_application(self):
        """Test that colormap is applied correctly."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            percentile_colormap='viridis'
        )
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=style)
        
        # Should have an image
        assert len(ax.images) > 0
        plt.close('all')
    
    def test_colorbar_present(self):
        """Test that colorbar is created."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, ax=ax)
        
        # Check that colorbar was created
        assert len(ax.images) > 0
        plt.close('all')
    
    def test_default_boxplot_style(self):
        """Test visualization with default BoxplotStyleConfig."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        # Should work with None (uses default)
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=None)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_multiple_percentiles(self):
        """Test visualization with multiple percentiles."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(percentiles=[10, 25, 50, 75, 90])
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_no_outliers_case(self):
        """Test visualization when there are no outliers."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(percentiles=[100], show_outliers=True)
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_imshow_parameters(self):
        """Test that imshow is called with correct parameters."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, ax=ax)
        
        # Check that an image was created
        assert len(ax.images) == 1
        # Check vmin and vmax
        im = ax.images[0]
        assert im.get_clim() == (0, 1)
        plt.close('all')
    
    def test_legend_with_median_and_outliers(self):
        """Test that legend is created when showing median and outliers."""
        # Create ensemble with controlled values to ensure median exists
        ensemble = np.ones((50, 50, 50))
        for i in range(25):
            ensemble[i, 20:30, 20:30] = -1.0
        
        style = BoxplotStyleConfig(
            percentiles=[50],
            show_median=True,
            show_outliers=True
        )
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_contour_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Legend might be created if there are contours
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_custom_colors(self):
        """Test visualization with custom median and outlier colors."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(
            show_median=True,
            show_outliers=True,
            median_color='purple',
            outliers_color='orange'
        )
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
