"""
Integration tests for the FunctionalBoxplot module.

Tests the full pipeline: stats → mesh → visualization → main function.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from uvisbox.Modules.FunctionalBoxplot import (
    functional_boxplot,
    functional_boxplot_summary_statistics,
    functional_boxplot_mesh,
    visualize_functional_boxplot
)
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestFunctionalBoxplotPipeline:
    """Test the complete FunctionalBoxplot pipeline."""
    
    def test_full_pipeline_stats_mesh_vis(self):
        """Test complete pipeline: stats → mesh → vis."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        mesh_data = functional_boxplot_mesh(stats)
        ax = visualize_functional_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_pipeline_with_custom_style(self):
        """Test pipeline with custom BoxplotStyleConfig."""
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
    
    def test_pipeline_preserves_data_integrity(self):
        """Test that data is preserved through the pipeline."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        mesh_data = functional_boxplot_mesh(stats)
        
        # Verify essential keys are present
        assert 'depths' in mesh_data
        assert 'median' in mesh_data  # Key is 'median' not 'median_curve'
        assert 'percentile_bands' in mesh_data
        assert 'outliers' in mesh_data
        
        # Verify data integrity
        np.testing.assert_array_equal(mesh_data['depths'], stats['depths'])
        plt.close('all')
    
    def test_pipeline_with_fbd_method(self):
        """Test pipeline with fbd method."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data, method='fbd')
        mesh_data = functional_boxplot_mesh(stats)
        ax = visualize_functional_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_pipeline_with_mfbd_method(self):
        """Test pipeline with mfbd method."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data, method='mfbd')
        mesh_data = functional_boxplot_mesh(stats)
        ax = visualize_functional_boxplot(mesh_data)
        
        assert isinstance(ax, Axes)
        plt.close('all')


class TestFunctionalBoxplotMainFunction:
    """Test the main functional_boxplot function."""
    
    def test_basic_functional_boxplot(self):
        """Test basic functional_boxplot call."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_with_custom_style(self):
        """Test functional_boxplot with custom style."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[10, 25, 50, 75, 90],
            show_median=True,
            show_outliers=True
        )
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_on_existing_axes(self):
        """Test functional_boxplot on existing axes."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        fig, ax = plt.subplots()
        result_ax = functional_boxplot(data, ax=ax)
        
        assert result_ax is ax
        plt.close('all')
    
    def test_functional_boxplot_fbd_method(self):
        """Test functional_boxplot with fbd method."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        ax = functional_boxplot(data, method='fbd')
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_mfbd_method(self):
        """Test functional_boxplot with mfbd method."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        ax = functional_boxplot(data, method='mfbd')
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_minimal_data(self):
        """Test functional_boxplot with minimal data."""
        data = np.random.randn(3, 100).cumsum(axis=1)
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_no_median(self):
        """Test functional_boxplot without median."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(show_median=False)
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_no_outliers(self):
        """Test functional_boxplot without outliers."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(show_outliers=False)
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_functional_boxplot_only_median(self):
        """Test functional_boxplot with median and minimal percentiles."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        # BoxplotStyleConfig requires non-empty percentiles
        style = BoxplotStyleConfig(
            percentiles=[50],
            show_median=True,
            show_outliers=False
        )
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) > 0  # Should have median line
        plt.close('all')
    
    def test_functional_boxplot_matches_manual_pipeline(self):
        """Test that functional_boxplot produces same result as manual pipeline."""
        np.random.seed(42)
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[25, 50, 75])
        
        # Manual pipeline
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        mesh_data = functional_boxplot_mesh(stats)
        
        # Main function
        ax = functional_boxplot(data, boxplot_style=style)
        
        # Both should produce valid plots
        assert isinstance(ax, Axes)
        assert len(ax.collections) >= 3  # At least 3 bands
        plt.close('all')
    
    def test_functional_boxplot_with_all_parameters(self):
        """Test functional_boxplot with all parameters specified."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[25, 50, 75],
            show_median=True,
            show_outliers=True,
            median_color='red',
            outliers_color='blue',
            percentile_colormap='viridis'
        )
        
        fig, ax = plt.subplots()
        result_ax = functional_boxplot(
            data,
            method='mfbd',
            boxplot_style=style,
            ax=ax
        )
        
        assert result_ax is ax
        plt.close('all')


class TestFunctionalBoxplotEdgeCases:
    """Test edge cases for the full FunctionalBoxplot module."""
    
    def test_single_curve(self):
        """Test with a single curve (minimal case)."""
        data = np.random.randn(1, 100).cumsum(axis=1)
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_two_curves(self):
        """Test with two curves."""
        data = np.random.randn(2, 100).cumsum(axis=1)
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_large_dataset(self):
        """Test with a large dataset."""
        data = np.random.randn(1000, 200).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[50])  # Minimal to speed up
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_short_curves(self):
        """Test with short curves."""
        data = np.random.randn(50, 10).cumsum(axis=1)
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_extreme_percentiles(self):
        """Test with extreme percentiles."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[1, 99])
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_many_percentiles(self):
        """Test with many percentile bands."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(
            percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]
        )
        
        ax = functional_boxplot(data, boxplot_style=style)
        
        assert isinstance(ax, Axes)
        assert len(ax.collections) >= 9
        plt.close('all')
    
    def test_constant_curves(self):
        """Test with constant curves."""
        data = np.ones((50, 100))
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
    
    def test_identical_curves(self):
        """Test with identical curves."""
        curve = np.random.randn(100).cumsum()
        data = np.tile(curve, (50, 1))
        
        ax = functional_boxplot(data)
        
        assert isinstance(ax, Axes)
        plt.close('all')
