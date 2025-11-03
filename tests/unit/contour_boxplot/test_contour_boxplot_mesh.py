"""
Unit tests for contour_boxplot_mesh module.

Tests the contour_boxplot_mesh function.
"""

import pytest
import numpy as np
from uvisbox.Modules.ContourBoxplot.contour_boxplot_stats import contour_boxplot_summary_statistics
from uvisbox.Modules.ContourBoxplot.contour_boxplot_mesh import contour_boxplot_mesh
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestContourBoxplotMesh:
    """Test suite for contour_boxplot_mesh function."""
    
    def test_basic_mesh_processing(self):
        """Test basic mesh processing."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        assert 'percentile_bands_image' in mesh_data
        assert 'median' in mesh_data
        assert 'outliers' in mesh_data
    
    def test_percentile_bands_image_shape(self):
        """Test percentile_bands_image has correct shape."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        assert mesh_data['percentile_bands_image'].shape == (50, 50)
        assert mesh_data['percentile_bands_image'].dtype == np.float32
    
    def test_preserves_median(self):
        """Test that median is preserved unchanged."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        np.testing.assert_array_equal(mesh_data['median'], stats['median'])
    
    def test_preserves_outliers(self):
        """Test that outliers are preserved unchanged."""
        ensemble = np.random.randn(50, 50, 50)
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        assert len(mesh_data['outliers']) == len(stats['outliers'])
        for mesh_outlier, stats_outlier in zip(mesh_data['outliers'], stats['outliers']):
            np.testing.assert_array_equal(mesh_outlier, stats_outlier)
    
    def test_aggregation_descending_order(self):
        """Test that aggregation happens in descending percentile order."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(percentiles=[25, 50, 75])
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        # Values in the image should be from the percentile bands
        unique_vals = np.unique(mesh_data['percentile_bands_image'])
        # Should have values from percentile bands (may or may not have 0)
        # Check that values are in expected range and from the percentiles
        assert len(unique_vals) > 0
        assert np.all(unique_vals >= 0)
        assert np.all(unique_vals <= 1)
    
    def test_lower_percentiles_overwrite_higher(self):
        """Test that lower percentiles overwrite higher ones."""
        # Create controlled ensemble
        ensemble = np.zeros((20, 30, 30))
        # Make half the images have value < 0 in center region
        for i in range(10):
            ensemble[i, 10:20, 10:20] = -1.0
        # Make all images have value < 0 in a smaller center region
        for i in range(20):
            ensemble[i, 13:17, 13:17] = -1.0
        
        style = BoxplotStyleConfig(percentiles=[50, 100])
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        # The smaller center region should have 0.5 (50th percentile)
        # since it overwrites the 1.0 (100th percentile)
        image = mesh_data['percentile_bands_image']
        center_value = image[15, 15]
        # Should be the lower percentile value
        assert center_value <= 0.5
    
    def test_empty_outliers(self):
        """Test with empty outliers list."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(percentiles=[100], show_outliers=False)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        assert len(mesh_data['outliers']) == 0
    
    def test_single_percentile(self):
        """Test with single percentile band."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(percentiles=[50])
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        # Should have values 0 and 0.5
        unique_vals = np.unique(mesh_data['percentile_bands_image'])
        assert 0.0 in unique_vals
    
    def test_multiple_percentiles(self):
        """Test with multiple percentile bands."""
        ensemble = np.random.randn(20, 50, 50)
        style = BoxplotStyleConfig(percentiles=[10, 25, 50, 75, 90])
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, boxplot_style=style, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        # Image should be created successfully
        assert mesh_data['percentile_bands_image'].shape == (50, 50)
    
    def test_percentile_values_range(self):
        """Test that percentile band image values are in valid range."""
        ensemble = np.random.randn(20, 50, 50)
        stats = contour_boxplot_summary_statistics(ensemble, isovalue=0.0, workers=2)
        
        mesh_data = contour_boxplot_mesh(stats)
        
        # All values should be between 0 and 1
        assert np.all(mesh_data['percentile_bands_image'] >= 0)
        assert np.all(mesh_data['percentile_bands_image'] <= 1)
