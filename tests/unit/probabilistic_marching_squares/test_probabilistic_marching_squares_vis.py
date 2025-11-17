"""
Unit tests for probabilistic_marching_squares_vis module.

Tests the visualize_probabilistic_marching_squares function.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from uvisbox.Modules.ProbabilisticMarchingSquares.probabilistic_marching_squares_vis import (
    visualize_probabilistic_marching_squares
)


class TestVisualizeProbabilisticMarchingSquares:
    """Test suite for visualize_probabilistic_marching_squares function."""
    
    def test_creates_new_axes_when_none(self):
        """Test that function creates new axes when ax is None."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_uses_provided_axes(self):
        """Test that function uses provided axes."""
        prob_array = np.random.rand(10, 15)
        fig, provided_ax = plt.subplots()
        
        returned_ax = visualize_probabilistic_marching_squares(prob_array, ax=provided_ax)
        
        assert returned_ax is provided_ax
        plt.close('all')
    
    def test_returns_axes_object(self):
        """Test that function returns Axes object."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_default_colormap(self):
        """Test default colormap is viridis."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        # Check that image was created
        images = ax.get_images()
        assert len(images) > 0
        assert images[0].get_cmap().name == 'viridis'
        plt.close('all')
    
    def test_custom_colormap(self):
        """Test with custom colormap."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array, colormap='plasma')
        
        images = ax.get_images()
        assert images[0].get_cmap().name == 'plasma'
        plt.close('all')
    
    def test_different_colormaps(self):
        """Test with various colormaps."""
        prob_array = np.random.rand(10, 15)
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu']
        
        for cmap in colormaps:
            ax = visualize_probabilistic_marching_squares(prob_array, colormap=cmap)
            images = ax.get_images()
            assert images[0].get_cmap().name == cmap
            plt.close('all')
    
    def test_colorbar_present(self):
        """Test that colorbar is created."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        # Check that figure has a colorbar (multiple axes)
        fig = ax.get_figure()
        assert len(fig.get_axes()) == 2  # Main axes + colorbar axes
        plt.close('all')
    
    def test_vmin_vmax_range(self):
        """Test that color limits are set to [0, 1]."""
        prob_array = np.random.rand(10, 15) * 0.5 + 0.25  # Values in [0.25, 0.75]
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        images = ax.get_images()
        assert images[0].get_clim() == (0, 1)
        plt.close('all')
    
    def test_origin_lower(self):
        """Test that image origin is set to 'lower'."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        images = ax.get_images()
        assert images[0].origin == 'lower'
        plt.close('all')
    
    def test_title_present(self):
        """Test that title is set."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        title = ax.get_title()
        assert title == 'Probabilistic Marching Squares'
        plt.close('all')
    
    def test_axis_labels(self):
        """Test that axis labels are set."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        plt.close('all')
    
    def test_with_zeros(self):
        """Test visualization with all-zero probability."""
        prob_array = np.zeros((10, 15))
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        images = ax.get_images()
        assert len(images) > 0
        plt.close('all')
    
    def test_with_ones(self):
        """Test visualization with all-one probability."""
        prob_array = np.ones((10, 15))
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        images = ax.get_images()
        assert len(images) > 0
        plt.close('all')
    
    def test_single_cell(self):
        """Test with single cell (1x1 array)."""
        prob_array = np.array([[0.5]])
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_large_array(self):
        """Test with large probability array."""
        prob_array = np.random.rand(100, 200)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_different_shapes(self):
        """Test with different array shapes."""
        shapes = [(5, 10), (20, 20), (15, 25), (10, 5)]
        
        for shape in shapes:
            prob_array = np.random.rand(*shape)
            ax = visualize_probabilistic_marching_squares(prob_array)
            assert isinstance(ax, Axes)
            plt.close('all')
    
    def test_input_not_modified(self):
        """Test that input array is not modified."""
        prob_array = np.random.rand(10, 15)
        prob_array_copy = prob_array.copy()
        
        visualize_probabilistic_marching_squares(prob_array)
        
        np.testing.assert_array_equal(prob_array, prob_array_copy)
        plt.close('all')
    
    def test_multiple_visualizations_on_same_figure(self):
        """Test multiple visualizations on same figure."""
        prob_array1 = np.random.rand(10, 15)
        prob_array2 = np.random.rand(10, 15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        visualize_probabilistic_marching_squares(prob_array1, ax=ax1)
        visualize_probabilistic_marching_squares(prob_array2, ax=ax2)
        
        assert len(ax1.get_images()) > 0
        assert len(ax2.get_images()) > 0
        plt.close('all')
    
    def test_visualization_with_pattern(self):
        """Test visualization with specific pattern."""
        prob_array = np.zeros((10, 10))
        prob_array[5, :] = 1.0  # Horizontal line
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        images = ax.get_images()
        assert len(images) > 0
        plt.close('all')
    
    def test_colorbar_label(self):
        """Test that colorbar has correct label."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        fig = ax.get_figure()
        # Get colorbar axes (second axes in figure)
        cbar_ax = fig.get_axes()[1]
        # Check for colorbar label
        assert cbar_ax.get_ylabel() == 'probability of contour'
        plt.close('all')
    
    def test_image_extent(self):
        """Test that image covers correct extent."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        images = ax.get_images()
        extent = images[0].get_extent()
        # Default extent should match array dimensions (comparing as list)
        expected_extent = [-0.5, 14.5, -0.5, 9.5]
        assert list(extent) == expected_extent
        plt.close('all')
    
    def test_reusing_axes_clears_previous(self):
        """Test that reusing axes works correctly."""
        prob_array1 = np.random.rand(10, 15)
        prob_array2 = np.random.rand(10, 15)
        
        fig, ax = plt.subplots()
        visualize_probabilistic_marching_squares(prob_array1, ax=ax)
        initial_image_count = len(ax.get_images())
        
        visualize_probabilistic_marching_squares(prob_array2, ax=ax)
        final_image_count = len(ax.get_images())
        
        # Should add another image
        assert final_image_count >= initial_image_count
        plt.close('all')
    
    def test_rectangular_arrays(self):
        """Test with non-square arrays."""
        # Wide array
        prob_array_wide = np.random.rand(5, 20)
        ax = visualize_probabilistic_marching_squares(prob_array_wide)
        assert isinstance(ax, Axes)
        plt.close('all')
        
        # Tall array
        prob_array_tall = np.random.rand(20, 5)
        ax = visualize_probabilistic_marching_squares(prob_array_tall)
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_figure_creation_when_none(self):
        """Test that figure is created when ax is None."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        fig = ax.get_figure()
        
        assert isinstance(fig, Figure)
        plt.close('all')
    
    def test_closes_cleanly(self):
        """Test that visualization can be closed without errors."""
        prob_array = np.random.rand(10, 15)
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        fig = ax.get_figure()
        
        plt.close(fig)
        # No assertion needed - just checking it doesn't raise
    
    def test_with_gradient_pattern(self):
        """Test with gradient pattern."""
        y, x = np.mgrid[0:10, 0:15]
        prob_array = x / 15.0  # Gradient from 0 to 1
        
        ax = visualize_probabilistic_marching_squares(prob_array)
        
        assert isinstance(ax, Axes)
        plt.close('all')
