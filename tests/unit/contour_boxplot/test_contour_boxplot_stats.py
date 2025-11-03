"""
Unit tests for contour_boxplot_stats module.

Tests the contour_boxplot_summary_statistics function.
"""

import pytest
import numpy as np
from uvisbox.Modules.ContourBoxplot.contour_boxplot_stats import contour_boxplot_summary_statistics
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestContourBoxplotSummaryStatistics:
    """Test suite for contour_boxplot_summary_statistics function."""
    
    def test_basic_computation(self):
        """Test basic summary statistics computation."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        assert 'median' in result
        assert 'percentile_bands' in result
        assert 'outliers' in result
        assert 'sorted_contours' in result
        assert 'sorted_indices' in result
        assert 'depths' in result
    
    def test_median_shape(self):
        """Test that median has correct shape."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        assert result['median'].shape == (50, 50)
        assert result['median'].dtype == np.uint8
    
    def test_median_is_binary(self):
        """Test that median is binary (0 or 1)."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        assert np.all((result['median'] == 0) | (result['median'] == 1))
    
    def test_percentile_bands_structure(self):
        """Test percentile_bands has correct structure."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[25, 50, 75])
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert len(result['percentile_bands']) == 3
        for percentile, band_image in result['percentile_bands']:
            assert percentile in [25, 50, 75]
            assert band_image.shape == (50, 50)
            assert band_image.dtype == np.float32
    
    def test_percentile_band_values(self):
        """Test percentile band images have correct values."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[50, 75])
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        
        for percentile, band_image in result['percentile_bands']:
            # Values should be either 0 or percentile/100
            unique_vals = np.unique(band_image)
            # Check that the percentile value is present
            assert np.any(np.isclose(unique_vals, percentile / 100.0)) or np.any(np.isclose(unique_vals, 0.0))
            # Check that values are in valid range
            assert np.all(band_image >= 0)
            assert np.all(band_image <= 1)
    
    def test_outliers_with_show_outliers_true(self):
        """Test outliers are identified when show_outliers is True."""
        ensemble = np.random.randn(50, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True)
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        
        # With 50th percentile, should have some outliers
        assert 'outliers' in result
        assert isinstance(result['outliers'], list)
    
    def test_outliers_with_show_outliers_false(self):
        """Test no outliers when show_outliers is False."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=False)
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert len(result['outliers']) == 0
    
    def test_outliers_are_binary(self):
        """Test that outliers are binary images."""
        ensemble = np.random.randn(50, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True)
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        
        for outlier in result['outliers']:
            assert outlier.dtype == np.uint8
            assert np.all((outlier == 0) | (outlier == 1))
    
    def test_sorted_contours_shape(self):
        """Test sorted_contours has correct shape."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        assert result['sorted_contours'].shape == (20, 50, 50)
    
    def test_sorted_indices_validity(self):
        """Test sorted_indices are valid permutation."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        assert len(result['sorted_indices']) == 20
        assert set(result['sorted_indices']) == set(range(20))
    
    def test_depths_sorted_descending(self):
        """Test depths are in descending order."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        depths = result['depths']
        assert np.all(depths[:-1] >= depths[1:])
    
    def test_input_not_modified(self):
        """Test that input ensemble is not modified."""
        ensemble = np.random.randn(20, 50, 50)
        ensemble_copy = ensemble.copy()
        isovalue = 0.0
        
        contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        np.testing.assert_array_equal(ensemble, ensemble_copy)
    
    def test_default_boxplot_style(self):
        """Test with default BoxplotStyleConfig."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        # Should work with None (uses default)
        assert 'percentile_bands' in result
        assert len(result['percentile_bands']) > 0
    
    def test_invalid_input_dimensions(self):
        """Test that invalid input dimensions raise ValueError."""
        # 2D array should fail
        ensemble = np.random.randn(50, 50)
        isovalue = 0.0
        
        with pytest.raises(ValueError, match="must be 3D array"):
            contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
    
    def test_custom_percentiles(self):
        """Test with custom percentiles."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        style = BoxplotStyleConfig(percentiles=[10, 25, 50, 75, 90])
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, boxplot_style=style, workers=2)
        
        assert len(result['percentile_bands']) == 5
    
    def test_list_input_converted_to_array(self):
        """Test that list inputs are converted to arrays."""
        ensemble_list = [np.random.randn(50, 50) for _ in range(20)]
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble_list, isovalue, workers=2)
        
        assert 'median' in result
    
    def test_isovalue_threshold(self):
        """Test that isovalue correctly creates binary images."""
        # Create ensemble with known values
        ensemble = np.zeros((10, 20, 20))
        for i in range(10):
            ensemble[i, :10, :] = -1.0  # Below isovalue
            ensemble[i, 10:, :] = 1.0   # Above isovalue
        
        isovalue = 0.0
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        # Median should have 1s where values < isovalue
        median = result['median']
        assert np.all(median[:10, :] == 1)  # Top half should be 1
        assert np.all(median[10:, :] == 0)  # Bottom half should be 0
    
    def test_small_ensemble(self):
        """Test with small ensemble."""
        ensemble = np.random.randn(3, 20, 20)
        isovalue = 0.0
        
        result = contour_boxplot_summary_statistics(ensemble, isovalue, workers=2)
        
        assert result['median'].shape == (20, 20)
    
    def test_different_workers(self):
        """Test with different number of workers."""
        ensemble = np.random.randn(20, 50, 50)
        isovalue = 0.0
        
        result1 = contour_boxplot_summary_statistics(ensemble, isovalue, workers=1)
        result2 = contour_boxplot_summary_statistics(ensemble, isovalue, workers=4)
        
        # Results should be identical regardless of worker count
        np.testing.assert_array_equal(result1['median'], result2['median'])
        assert len(result1['percentile_bands']) == len(result2['percentile_bands'])
