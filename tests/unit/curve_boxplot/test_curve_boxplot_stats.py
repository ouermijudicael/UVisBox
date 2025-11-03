"""
Unit tests for curve_boxplot_stats module.

Tests the curve_boxplot_summary_statistics function and related functionality.
"""

import pytest
import numpy as np
from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestCurveBoxplotSummaryStatistics:
    """Test suite for curve_boxplot_summary_statistics function."""
    
    def test_basic_2d_computation(self):
        """Test basic statistics computation with 2D curves."""
        # Generate simple 2D curves
        n_curves, n_steps = 20, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        
        # Check all expected keys are present
        assert 'depths' in stats
        assert 'sorted_indices' in stats
        assert 'sorted_curves' in stats
        assert 'median_curve' in stats
        assert 'percentiles' in stats
        assert 'outliers' in stats
        assert 'n_dims' in stats
        
        # Check shapes
        assert stats['depths'].shape == (n_curves,)
        assert stats['sorted_indices'].shape == (n_curves,)
        assert stats['sorted_curves'].shape == (n_curves, n_steps, 2)
        assert stats['median_curve'].shape == (n_steps, 2)
        assert stats['n_dims'] == 2
    
    def test_basic_3d_computation(self):
        """Test basic statistics computation with 3D curves."""
        # Generate simple 3D curves
        n_curves, n_steps = 20, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        
        # Check shapes for 3D
        assert stats['sorted_curves'].shape == (n_curves, n_steps, 3)
        assert stats['median_curve'].shape == (n_steps, 3)
        assert stats['n_dims'] == 3
    
    def test_depths_sorted_descending(self):
        """Test that curves are sorted by depth in descending order."""
        n_curves, n_steps = 30, 40
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        
        # Get depths in sorted order
        original_depths = stats['depths']
        sorted_depths = original_depths[stats['sorted_indices']]
        
        # Check that sorted depths are in descending order
        assert np.all(sorted_depths[:-1] >= sorted_depths[1:])
        
        # Check that the median curve has the highest depth
        assert sorted_depths[0] == np.max(original_depths)
    
    def test_custom_percentiles(self):
        """Test with custom percentile configuration."""
        n_curves, n_steps = 25, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        custom_percentiles = [10, 50, 90]
        style = BoxplotStyleConfig(percentiles=custom_percentiles)
        
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        
        assert stats['percentiles'] == custom_percentiles
    
    def test_outliers_with_show_outliers_true(self):
        """Test outlier detection when show_outliers is True."""
        n_curves, n_steps = 50, 40
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(percentiles=[25, 50, 75], show_outliers=True)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        
        # Calculate expected number of outliers (beyond 75th percentile)
        largest_percentile = 75
        expected_outlier_count = n_curves - int(np.ceil(n_curves * largest_percentile / 100))
        
        assert stats['outliers'].shape[0] == expected_outlier_count
        assert stats['outliers'].shape[1] == n_steps
        assert stats['outliers'].shape[2] == 2
    
    def test_outliers_with_show_outliers_false(self):
        """Test that no outliers are computed when show_outliers is False."""
        n_curves, n_steps = 30, 40
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(show_outliers=False)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        
        assert stats['outliers'].shape[0] == 0
        assert stats['outliers'].shape == (0, n_steps, 2)
    
    def test_no_percentiles_no_outliers(self):
        """Test that empty percentiles raises ValueError."""
        n_curves, n_steps = 20, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # BoxplotStyleConfig requires non-empty percentiles list
        with pytest.raises(ValueError, match="non-empty list"):
            BoxplotStyleConfig(percentiles=[], show_outliers=True)
    
    def test_input_not_modified(self):
        """Test that the input curves array is not modified."""
        n_curves, n_steps = 20, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        curves_copy = curves.copy()
        
        _ = curve_boxplot_summary_statistics(curves)
        
        np.testing.assert_array_equal(curves, curves_copy)
    
    def test_default_boxplot_style(self):
        """Test with default BoxplotStyleConfig."""
        n_curves, n_steps = 20, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        
        # Should use default percentiles
        default_config = BoxplotStyleConfig()
        assert stats['percentiles'] == default_config.percentiles
    
    def test_workers_parameter(self):
        """Test that workers parameter is accepted."""
        n_curves, n_steps = 20, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # Should work with different worker counts
        stats1 = curve_boxplot_summary_statistics(curves, workers=1)
        stats12 = curve_boxplot_summary_statistics(curves, workers=12)
        
        # Results should be the same regardless of worker count
        np.testing.assert_array_equal(stats1['depths'], stats12['depths'])
    
    def test_invalid_input_dimensions(self):
        """Test error handling for invalid input dimensions."""
        # 2D array instead of 3D
        invalid_curves = np.random.randn(20, 30)
        
        with pytest.raises(ValueError, match="3D array"):
            curve_boxplot_summary_statistics(invalid_curves)
        
        # 4D array
        invalid_curves_4d = np.random.randn(20, 30, 2, 2)
        
        with pytest.raises(ValueError, match="3D array"):
            curve_boxplot_summary_statistics(invalid_curves_4d)
    
    def test_invalid_curve_dimensionality(self):
        """Test error handling for invalid curve dimensionality (not 2D or 3D)."""
        # 1D curves
        curves_1d = np.random.randn(20, 30, 1)
        
        with pytest.raises(ValueError, match="2D or 3D"):
            curve_boxplot_summary_statistics(curves_1d)
        
        # 4D curves
        curves_4d = np.random.randn(20, 30, 4)
        
        with pytest.raises(ValueError, match="2D or 3D"):
            curve_boxplot_summary_statistics(curves_4d)
    
    def test_median_is_deepest_curve(self):
        """Test that median curve is the curve with maximum depth."""
        n_curves, n_steps = 30, 40
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        
        # Median should be the first curve in sorted_curves
        np.testing.assert_array_equal(stats['median_curve'], stats['sorted_curves'][0])
        
        # And should have the highest depth
        max_depth_idx = np.argmax(stats['depths'])
        np.testing.assert_array_equal(stats['median_curve'], curves[max_depth_idx])
    
    def test_small_number_of_curves(self):
        """Test with very small number of curves."""
        n_curves, n_steps = 3, 20
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        
        assert stats['depths'].shape == (n_curves,)
        assert stats['sorted_curves'].shape == (n_curves, n_steps, 2)
    
    def test_large_percentile_high_outliers(self):
        """Test outlier computation with large percentile (many outliers)."""
        n_curves, n_steps = 100, 30
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # Use small percentile, so many curves should be outliers
        style = BoxplotStyleConfig(percentiles=[10], show_outliers=True)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        
        # Should have 90 outliers (100 - ceil(100 * 0.1))
        expected_outliers = n_curves - int(np.ceil(n_curves * 10 / 100))
        assert stats['outliers'].shape[0] == expected_outliers
    
    def test_list_input_converted_to_array(self):
        """Test that list input is converted to numpy array."""
        # Create curves as nested lists
        curves_list = [[[i, i+1] for i in range(20)] for _ in range(10)]
        
        stats = curve_boxplot_summary_statistics(curves_list)
        
        # Should work and return proper shapes
        assert stats['depths'].shape == (10,)
        assert stats['sorted_curves'].shape == (10, 20, 2)
