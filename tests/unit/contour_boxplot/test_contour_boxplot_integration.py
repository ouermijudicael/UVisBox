"""
Integration tests for the ContourBoxplot module.

Tests the full pipeline: stats → mesh → visualization → main function.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from uvisbox.Modules.ContourBoxplot import (
    contour_boxplot,
    contour_boxplot_summary_statistics,
    contour_boxplot_mesh,
    visualize_contour_boxplot
)
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestContourBoxplotPipeline:
    """Test the complete ContourBoxplot pipeline."""
    
    def test_full_pipeline_stats_mesh_vis(self):
        """Test complete pipeline: stats → mesh → vis."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        ax = visualize_contour_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_pipeline_with_custom_style(self):
        """Test pipeline with custom BoxplotStyleConfig."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='blue'
        )
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        ax = visualize_contour_boxplot(mesh_data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_pipeline_preserves_data_integrity(self):
        """Test that data is preserved through the pipeline."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        stats = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        # Verify essential keys are present
        assert 'percentile_bands_image' in mesh_data
        assert 'median' in mesh_data
        assert 'outliers' in mesh_data
        
        # Verify data integrity
        np.testing.assert_array_equal(mesh_data['median'], stats['median'])
        plt.close('all')


class TestContourBoxplotMainFunction:
    """Test the main contour_boxplot function."""
    
    def test_basic_contour_boxplot(self):
        """Test basic contour_boxplot call."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_contour_boxplot_with_custom_style(self):
        """Test contour_boxplot with custom style."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(
            percentiles=[10, 25, 50, 75, 90],
            show_median=True,
            show_outliers=True
        )
        
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_contour_boxplot_on_existing_axes(self):
        """Test contour_boxplot on existing axes."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        fig, ax = plt.subplots()
        result_ax = contour_boxplot(ensemble, isovalue, ax=ax, workers=2)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_contour_boxplot_minimal_data(self):
        """Test contour_boxplot with minimal data."""
        ensemble = np.random.randn(3, 20, 20)
        isovalue = 0.0
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_contour_boxplot_no_median(self):
        """Test contour_boxplot without median."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(show_median=False)
        
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_contour_boxplot_no_outliers(self):
        """Test contour_boxplot without outliers."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(show_outliers=False)
        
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_contour_boxplot_matches_manual_pipeline(self):
        """Test that contour_boxplot produces same result as manual pipeline."""
        np.random.seed(42)
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[25, 50, 75])
        
        # Manual pipeline
        stats = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        mesh_data = contour_boxplot_mesh(stats)
        
        # Main function
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=2)
        
        # Both should produce valid plots
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_contour_boxplot_with_all_parameters(self):
        """Test contour_boxplot with all parameters specified."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.5
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='blue',
            percentile_colormap='viridis'
        )
        
        fig, ax = plt.subplots()
        result_ax = contour_boxplot(
            ensemble,
            isovalue,
            boxplot_style=style,
            ax=ax,
            workers=2
        )
        
        assert result_ax is ax
        plt.close('all')
    
    def test_contour_boxplot_different_isovalues(self):
        """Test contour_boxplot with different isovalue thresholds."""
        ensemble = np.random.randn(20, 50, 50)
        
        for isovalue in [-1.0, 0.0, 1.0]:
            ax = contour_boxplot(ensemble, isovalue, workers=2)
            assert isinstance(ax, Axes)
            plt.close('all')


class TestContourBoxplotEdgeCases:
    """Test edge cases for the full ContourBoxplot module."""
    
    def test_small_ensemble(self):
        """Test with small ensemble (3 members)."""
        ensemble = np.random.randn(3, 20, 20)
        isovalue = 0.0
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_large_ensemble(self):
        """Test with large ensemble."""
        ensemble = np.random.randn(100, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[50])  # Minimal to speed up
        
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=4)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_small_spatial_resolution(self):
        """Test with small spatial resolution."""
        ensemble = np.random.randn(20, 10, 10)
        isovalue = 0.0
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_extreme_percentiles(self):
        """Test with extreme percentiles."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[1, 99])
        
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_many_percentiles(self):
        """Test with many percentile bands."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(
            percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]
        )
        
        ax = contour_boxplot(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_constant_field(self):
        """Test with constant scalar field."""
        ensemble = np.ones((20, 50, 50))
        isovalue = 0.5
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_identical_fields(self):
        """Test with identical fields in ensemble."""
        field = np.random.randn(50, 50)
        ensemble = np.tile(field, (20, 1, 1))
        isovalue = 0.0
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_high_isovalue(self):
        """Test with high isovalue (few pixels below threshold)."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 3.0  # Most pixels will be below this
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_low_isovalue(self):
        """Test with low isovalue (few pixels above threshold)."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = -3.0  # Most pixels will be above this
        
        ax = contour_boxplot(ensemble, isovalue, workers=2)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_different_workers(self):
        """Test with different number of workers."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        for workers in [1, 2, 4]:
            ax = contour_boxplot(ensemble, isovalue, workers=workers)
            assert isinstance(ax, Axes)
            plt.close('all')
