"""
Unit tests for curve_boxplot_vis module.

Tests the visualize_curve_boxplot function and rendering functionality.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curve_boxplot_mesh
from uvisbox.Modules.CurveBoxplot.curve_boxplot_vis import (
    visualize_curve_boxplot, 
    _plot_band_mesh,
    matplotlib_plot_band
)
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestVisualizeCurveBoxplot:
    """Test suite for visualize_curve_boxplot function."""
    
    def test_basic_2d_visualization(self):
        """Test basic 2D visualization without errors."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        ax = visualize_curve_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_basic_3d_visualization(self):
        """Test basic 3D visualization without errors."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        ax = visualize_curve_boxplot(mesh_data)
        
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_visualization_with_existing_2d_axes(self):
        """Test visualization on existing 2D axes."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        result_ax = visualize_curve_boxplot(mesh_data, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_visualization_with_existing_3d_axes(self):
        """Test visualization on existing 3D axes."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        result_ax = visualize_curve_boxplot(mesh_data, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_custom_boxplot_style(self):
        """Test visualization with custom BoxplotStyleConfig."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='blue'
        )
        
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        ax = visualize_curve_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_show_median_true(self):
        """Test that median curve is plotted when show_median is True."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(show_median=True)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_curve_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Check that lines were plotted (median curve should be present)
        assert len(ax.get_lines()) > 0
        plt.close('all')
    
    def test_show_median_false(self):
        """Test that median curve is not plotted when show_median is False."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(show_median=False, show_outliers=False)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_curve_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have no lines plotted
        assert len(ax.get_lines()) == 0
        plt.close('all')
    
    def test_show_outliers_true(self):
        """Test that outlier curves are plotted when show_outliers is True."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True, show_median=False)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_curve_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have outlier lines plotted
        # Number of outliers should be > 0 for 50th percentile with 30 curves
        assert len(ax.get_lines()) > 0
        plt.close('all')
    
    def test_show_outliers_false(self):
        """Test that outlier curves are not plotted when show_outliers is False."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(show_outliers=False, show_median=False)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        fig, ax = plt.subplots()
        visualize_curve_boxplot(mesh_data, boxplot_style=style, ax=ax)
        
        # Should have no lines (no median, no outliers)
        assert len(ax.get_lines()) == 0
        plt.close('all')
    
    def test_default_boxplot_style(self):
        """Test visualization with default BoxplotStyleConfig."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Should work with None (uses default)
        ax = visualize_curve_boxplot(mesh_data, boxplot_style=None)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_invalid_n_dims_raises_error(self):
        """Test that invalid n_dims raises ValueError."""
        # Create fake mesh_data with invalid n_dims
        mesh_data = {
            'percentile_meshes': {},
            'median_curve': np.array([]),
            'outliers': np.array([]),
            'n_dims': 4  # Invalid
        }
        
        with pytest.raises(ValueError, match="Unsupported curve dimension"):
            visualize_curve_boxplot(mesh_data)
        
        plt.close('all')


class TestPlotBandMesh:
    """Test suite for _plot_band_mesh helper function."""
    
    def test_plot_2d_mesh(self):
        """Test plotting 2D mesh."""
        # Create simple triangular mesh
        points = np.array([[0, 0], [1, 0], [0.5, 1], [1, 1]])
        triangles = np.array([[0, 1, 2], [1, 2, 3]])
        
        fig, ax = plt.subplots()
        result_ax = _plot_band_mesh(points, triangles, ax, color='red', alpha=0.5, n_dims=2)
        
        assert result_ax is ax
        # Check that polygons were added
        assert len(ax.patches) == 2  # Two triangles
        plt.close('all')
    
    def test_plot_3d_mesh(self):
        """Test plotting 3D mesh."""
        # Create simple 3D triangular mesh
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.5], [1, 1, 1]])
        triangles = np.array([[0, 1, 2], [1, 2, 3]])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        result_ax = _plot_band_mesh(points, triangles, ax, color='blue', alpha=0.7, n_dims=3)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_invalid_n_dims_raises_error(self):
        """Test that invalid n_dims raises ValueError."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        triangles = np.array([[0, 1, 2]])
        
        fig, ax = plt.subplots()
        
        with pytest.raises(ValueError, match="n_dims must be 2 or 3"):
            _plot_band_mesh(points, triangles, ax, color='red', alpha=1.0, n_dims=4)
        
        plt.close('all')
    
    def test_different_colors(self):
        """Test plotting with different colors."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        triangles = np.array([[0, 1, 2]])
        
        fig, ax = plt.subplots()
        _plot_band_mesh(points, triangles, ax, color='green', alpha=1.0, n_dims=2)
        
        # Check that patch was added
        assert len(ax.patches) == 1
        plt.close('all')
    
    def test_different_alpha(self):
        """Test plotting with different alpha values."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        triangles = np.array([[0, 1, 2]])
        
        fig, ax = plt.subplots()
        _plot_band_mesh(points, triangles, ax, color='red', alpha=0.3, n_dims=2)
        
        # Check that patch was added with correct alpha
        assert len(ax.patches) == 1
        assert ax.patches[0].get_alpha() == 0.3
        plt.close('all')


class TestMatplotlibPlotBand:
    """Test suite for matplotlib_plot_band function (deprecated but kept for compatibility)."""
    
    def test_plot_band_2d(self):
        """Test plotting 2D band."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        triangles = np.array([[0, 1, 2]])
        
        ax = matplotlib_plot_band(points, triangles, color='blue', alpha=0.5)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_plot_band_3d(self):
        """Test plotting 3D band."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.5]])
        triangles = np.array([[0, 1, 2]])
        
        ax = matplotlib_plot_band(points, triangles, color='green', alpha=0.7)
        
        assert isinstance(ax, Axes3D)
        plt.close('all')
    
    def test_plot_band_with_existing_axes_2d(self):
        """Test plotting on existing 2D axes."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        triangles = np.array([[0, 1, 2]])
        
        fig, ax = plt.subplots()
        result_ax = matplotlib_plot_band(points, triangles, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_plot_band_with_existing_axes_3d(self):
        """Test plotting on existing 3D axes."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.5]])
        triangles = np.array([[0, 1, 2]])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        result_ax = matplotlib_plot_band(points, triangles, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_invalid_points_dimension_raises_error(self):
        """Test that invalid points dimension raises ValueError."""
        # 1D points array
        points = np.array([0, 1, 2])
        triangles = np.array([[0, 1, 2]])
        
        with pytest.raises(ValueError, match="points must be a 2D array"):
            matplotlib_plot_band(points, triangles)
        
        plt.close('all')
    
    def test_invalid_point_columns_raises_error(self):
        """Test that invalid number of point columns raises ValueError."""
        # 4D points (invalid)
        points = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0.5, 1, 0.5, 0.5]])
        triangles = np.array([[0, 1, 2]])
        
        with pytest.raises(ValueError, match="2 or 3 columns"):
            matplotlib_plot_band(points, triangles)
        
        plt.close('all')
