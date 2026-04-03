"""
Integration tests for the ProbabilisticMarchingSquares module.

Tests the full pipeline: stats → mesh → visualization → main function.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from uvisbox.Modules.ProbabilisticMarchingSquares import (
    probabilistic_marching_squares,
    probabilistic_marching_squares_summary_statistics,
    probabilistic_marching_squares_mesh,
    visualize_probabilistic_marching_squares
)


class TestProbabilisticMarchingSquaresPipeline:
    """Test the complete ProbabilisticMarchingSquares pipeline."""
    
    def test_full_pipeline_stats_mesh_vis(self):
        """Test complete pipeline: stats → mesh → vis."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        stats = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        mesh_data = probabilistic_marching_squares_mesh(stats)
        ax = visualize_probabilistic_marching_squares(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_pipeline_preserves_shape(self):
        """Test that shape is preserved through pipeline."""
        y_dim, x_dim, n_ensemble = 20, 30, 50
        ensemble = np.random.randn(y_dim, x_dim, n_ensemble)
        isovalue = 0.0

        stats = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        mesh_data = probabilistic_marching_squares_mesh(stats)

        assert mesh_data['level_crossing_probability'].shape == (y_dim - 1, x_dim - 1)
        plt.close('all')

    def test_pipeline_preserves_data_integrity(self):
        """Test that data is preserved through the pipeline."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0

        stats = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob_from_stats = stats['level_crossing_probability']

        mesh_data = probabilistic_marching_squares_mesh(stats)

        # Verify data integrity
        np.testing.assert_array_equal(mesh_data['level_crossing_probability'], prob_from_stats)
        plt.close('all')
    
    def test_pipeline_with_different_isovalues(self):
        """Test pipeline with different isovalues."""
        ensemble = np.random.randn(20, 30, 50)
        
        for isovalue in [-1.0, 0.0, 1.0]:
            stats = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
            mesh_data = probabilistic_marching_squares_mesh(stats)
            ax = visualize_probabilistic_marching_squares(mesh_data)
            
            assert isinstance(ax, Axes)
            plt.close('all')
    
    def test_pipeline_with_different_grid_sizes(self):
        """Test pipeline with different grid sizes."""
        for y_dim, x_dim in [(10, 15), (25, 25), (30, 20)]:
            ensemble = np.random.randn(y_dim, x_dim, 30)
            isovalue = 0.0
            
            stats = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
            mesh_data = probabilistic_marching_squares_mesh(stats)
            ax = visualize_probabilistic_marching_squares(mesh_data)
            
            assert isinstance(ax, Axes)
            assert mesh_data['level_crossing_probability'].shape == (y_dim - 1, x_dim - 1)
            plt.close('all')


class TestProbabilisticMarchingSquaresMainFunction:
    """Test the main probabilistic_marching_squares function."""
    
    def test_basic_probabilistic_marching_squares(self):
        """Test basic probabilistic_marching_squares call."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_on_existing_axes(self):
        """Test probabilistic_marching_squares on existing axes."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        fig, provided_ax = plt.subplots()
        returned_ax = probabilistic_marching_squares(ensemble, isovalue, ax=provided_ax)
        
        assert returned_ax is provided_ax
        plt.close('all')
    
    def test_with_custom_colormap(self):
        """Test with custom colormap."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue, colormap='plasma')
        
        assert isinstance(ax, Axes)
        images = ax.get_images()
        assert images[0].get_cmap().name == 'plasma'
        plt.close('all')
    
    def test_with_different_colormaps(self):
        """Test with various colormaps."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        colormaps = ['viridis', 'plasma', 'inferno', 'coolwarm']
        
        for cmap in colormaps:
            ax = probabilistic_marching_squares(ensemble, isovalue, colormap=cmap)
            assert isinstance(ax, Axes)
            plt.close('all')
    
    def test_deterministic_result(self):
        """Test that results are similar with same input (Monte Carlo has inherent randomness)."""
        ensemble = np.random.RandomState(42).randn(20, 30, 50)
        isovalue = 0.0
        
        # Compute stats directly - note Monte Carlo sampling has randomness
        stats1 = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        stats2 = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        
        # Results should be similar but not identical due to Monte Carlo randomness
        np.testing.assert_allclose(
            stats1['level_crossing_probability'],
            stats2['level_crossing_probability'],
            rtol=0.2  # 20% relative tolerance for Monte Carlo variability
        )
        plt.close('all')
    
    def test_small_ensemble(self):
        """Test with small ensemble."""
        ensemble = np.random.randn(20, 30, 5)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_large_ensemble(self):
        """Test with large ensemble."""
        ensemble = np.random.randn(20, 30, 200)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_all_above_isovalue(self):
        """Test when all ensemble members are above isovalue."""
        ensemble = np.ones((20, 30, 50)) * 2.0
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_all_below_isovalue(self):
        """Test when all ensemble members are below isovalue."""
        ensemble = np.ones((20, 30, 50)) * -2.0
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_known_pattern(self):
        """Test with known crossing pattern."""
        y_dim, x_dim, n_ensemble = 20, 30, 50
        ensemble = np.zeros((y_dim, x_dim, n_ensemble))
        ensemble[:, :x_dim//2, :] = -1.0
        ensemble[:, x_dim//2:, :] = 1.0
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        # Verify visualization was created
        images = ax.get_images()
        assert len(images) > 0
        plt.close('all')
    
    def test_input_not_modified(self):
        """Test that input ensemble is not modified."""
        ensemble = np.random.randn(20, 30, 50)
        ensemble_copy = ensemble.copy()
        isovalue = 0.0
        
        probabilistic_marching_squares(ensemble, isovalue)
        
        np.testing.assert_array_equal(ensemble, ensemble_copy)
        plt.close('all')
    
    def test_multiple_calls(self):
        """Test multiple calls to probabilistic_marching_squares."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        ax1 = probabilistic_marching_squares(ensemble, isovalue)
        ax2 = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)
        assert ax1 is not ax2  # Different axes objects
        plt.close('all')
    
    def test_with_negative_isovalue(self):
        """Test with negative isovalue."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = -1.5
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_with_positive_isovalue(self):
        """Test with positive isovalue."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 1.5
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_square_grid(self):
        """Test with square grid."""
        ensemble = np.random.randn(25, 25, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_rectangular_grid_wide(self):
        """Test with wide rectangular grid."""
        ensemble = np.random.randn(10, 40, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_rectangular_grid_tall(self):
        """Test with tall rectangular grid."""
        ensemble = np.random.randn(40, 10, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_minimal_grid(self):
        """Test with minimal grid (2x2)."""
        ensemble = np.random.randn(2, 2, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_extreme_values(self):
        """Test with extreme values in ensemble."""
        ensemble = np.random.randn(20, 30, 50) * 1e6
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_with_sine_wave_pattern(self):
        """Test with sine wave pattern (similar to example)."""
        n, m, n_ens = 20, 30, 50
        x = np.linspace(0, 4 * np.pi, n)
        y = np.linspace(0, 4 * np.pi, m)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create ensemble with correct shape (n, m, n_ens)
        ensemble = np.zeros((n, m, n_ens))
        for i in range(n_ens):
            ensemble[:, :, i] = np.sin(X) * np.cos(Y) + 0.2 * np.random.randn(n, m)
        
        isovalue = 0.5
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_visualization_components_present(self):
        """Test that all visualization components are present."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        # Check image is present
        assert len(ax.get_images()) > 0
        
        # Check title
        assert ax.get_title() == 'Probabilistic Marching Squares'
        
        # Check labels
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        
        # Check colorbar (additional axes in figure)
        fig = ax.get_figure()
        assert len(fig.get_axes()) == 2
        
        plt.close('all')
    
    def test_probability_values_in_range(self):
        """Test that visualized probability values are in [0, 1]."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        ax = probabilistic_marching_squares(ensemble, isovalue)
        
        # Get image data
        image_data = ax.get_images()[0].get_array()
        
        assert np.all(image_data >= 0.0)
        assert np.all(image_data <= 1.0)
        
        plt.close('all')
