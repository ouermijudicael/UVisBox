"""
Unit tests for functional_boxplot_stats module.

Tests the functional_boxplot_summary_statistics function and helper functions.
"""

import pytest
import numpy as np
from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_stats import (
    functional_boxplot_band_depths,
    functional_boxplot_get_band,
    functional_boxplot_summary_statistics
)
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestFunctionalBoxplotBandDepths:
    """Test suite for functional_boxplot_band_depths function."""
    
    def test_fbd_method(self):
        """Test functional band depth calculation."""
        n_curves, n_points = 50, 100
        data = np.random.randn(n_curves, n_points).cumsum(axis=1)
        
        depths = functional_boxplot_band_depths(data, method='fbd')
        
        assert depths.shape == (n_curves,)
        assert np.all(depths >= 0)
        # FBD returns integer counts, not normalized values
        assert np.issubdtype(depths.dtype, np.integer)
    
    def test_mfbd_method(self):
        """Test modified functional band depth calculation."""
        n_curves, n_points = 50, 100
        data = np.random.randn(n_curves, n_points).cumsum(axis=1)
        
        depths = functional_boxplot_band_depths(data, method='mfbd')
        
        assert depths.shape == (n_curves,)
        assert np.all(depths >= 0)
        # MFBD returns float values, not normalized to 0-1
        assert np.issubdtype(depths.dtype, np.floating)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        data = np.random.randn(50, 100)
        
        with pytest.raises(ValueError, match="Unknown method"):
            functional_boxplot_band_depths(data, method='invalid')


class TestFunctionalBoxplotGetBand:
    """Test suite for functional_boxplot_get_band function."""
    
    def test_basic_band_computation(self):
        """Test basic band computation."""
        n_curves, n_points = 50, 100
        data = np.random.randn(n_curves, n_points).cumsum(axis=1)
        
        bottom, top = functional_boxplot_get_band(data, 50, method='fbd')
        
        assert bottom.shape == (n_points,)
        assert top.shape == (n_points,)
        assert np.all(bottom <= top)
    
    def test_different_percentiles(self):
        """Test bands with different percentiles."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        bottom_25, top_25 = functional_boxplot_get_band(data, 25)
        bottom_75, top_75 = functional_boxplot_get_band(data, 75)
        
        # Larger percentile should have wider band
        assert np.mean(top_75 - bottom_75) >= np.mean(top_25 - bottom_25)
    
    def test_100_percentile_includes_all(self):
        """Test that 100th percentile includes all curves."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        bottom, top = functional_boxplot_get_band(data, 100)
        
        # Should match global min/max
        assert np.allclose(bottom, np.min(data, axis=0), atol=1e-10)
        assert np.allclose(top, np.max(data, axis=0), atol=1e-10)
    
    def test_invalid_percentile_raises_error(self):
        """Test that invalid percentile raises ValueError."""
        data = np.random.randn(50, 100)
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            functional_boxplot_get_band(data, -10)
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            functional_boxplot_get_band(data, 150)
    
    def test_invalid_data_dimension_raises_error(self):
        """Test that invalid data dimension raises ValueError."""
        data_1d = np.random.randn(100)
        data_3d = np.random.randn(10, 50, 3)
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            functional_boxplot_get_band(data_1d, 50)
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            functional_boxplot_get_band(data_3d, 50)
    
    def test_mfbd_method(self):
        """Test band computation with modified functional band depth."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        bottom, top = functional_boxplot_get_band(data, 50, method='mfbd')
        
        assert bottom.shape == (100,)
        assert top.shape == (100,)
        assert np.all(bottom <= top)


class TestFunctionalBoxplotSummaryStatistics:
    """Test suite for functional_boxplot_summary_statistics function."""
    
    def test_basic_computation(self):
        """Test basic summary statistics computation."""
        n_curves, n_points = 50, 100
        data = np.random.randn(n_curves, n_points).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        
        assert 'depths' in stats
        assert 'median' in stats
        assert 'percentile_bands' in stats
        assert 'outliers' in stats
        assert 'sorted_curves' in stats
        assert 'sorted_indices' in stats
    
    def test_depths_sorted_descending(self):
        """Test that depths are sorted in descending order."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        
        # Check sorted curves correspond to descending depths
        sorted_depths = stats['depths'][stats['sorted_indices']]
        assert np.all(sorted_depths[:-1] >= sorted_depths[1:])
    
    def test_median_is_deepest_curve(self):
        """Test that median curve is the curve with highest depth."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        
        # Median should be the first sorted curve (highest depth)
        assert np.array_equal(stats['median'], stats['sorted_curves'][0])
        
        # Median's depth should be the maximum
        max_depth_idx = np.argmax(stats['depths'])
        assert np.array_equal(stats['median'], data[max_depth_idx])
    
    def test_custom_percentiles(self):
        """Test with custom percentiles."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[10, 50, 90])
        
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        assert len(stats['percentile_bands']) == 3
        assert '10_percentile_band' in stats['percentile_bands']
        assert '50_percentile_band' in stats['percentile_bands']
        assert '90_percentile_band' in stats['percentile_bands']
    
    def test_percentile_bands_structure(self):
        """Test that percentile bands have correct structure."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        
        for band_key, (bottom, top) in stats['percentile_bands'].items():
            assert bottom.shape == (100,)
            assert top.shape == (100,)
            assert np.all(bottom <= top)
    
    def test_outliers_with_show_outliers_true(self):
        """Test outliers detection when show_outliers is True."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True)
        
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        # Should have outliers beyond 50th percentile
        assert stats['outliers'].shape[0] > 0
        assert stats['outliers'].shape[1] == 100
    
    def test_outliers_with_show_outliers_false(self):
        """Test that no outliers when show_outliers is False."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(show_outliers=False)
        
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        assert stats['outliers'].shape[0] == 0
    
    def test_no_percentiles_no_outliers(self):
        """Test with minimal percentiles list."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        # BoxplotStyleConfig requires non-empty percentiles, use minimal config
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True)
        
        result = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        assert 'percentile_bands' in result
        assert len(result['percentile_bands']) == 1
    
    def test_input_not_modified(self):
        """Test that input data is not modified."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        data_copy = data.copy()
        
        _ = functional_boxplot_summary_statistics(data)
        
        assert np.array_equal(data, data_copy)
    
    def test_default_boxplot_style(self):
        """Test with default BoxplotStyleConfig."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data, boxplot_style=None)
        
        # Should use default percentiles
        assert len(stats['percentile_bands']) > 0
    
    def test_invalid_input_dimensions(self):
        """Test that invalid input dimensions raise ValueError."""
        data_1d = np.random.randn(100)
        data_3d = np.random.randn(10, 50, 3)
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            functional_boxplot_summary_statistics(data_1d)
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            functional_boxplot_summary_statistics(data_3d)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        data = np.random.randn(50, 100)
        
        with pytest.raises(ValueError, match="Unknown method"):
            functional_boxplot_summary_statistics(data, method='invalid')
    
    def test_mfbd_method(self):
        """Test with modified functional band depth method."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data, method='mfbd')
        
        assert 'depths' in stats
        assert stats['depths'].shape == (50,)
    
    def test_list_input_converted_to_array(self):
        """Test that list input is converted to numpy array."""
        data_list = np.random.randn(50, 100).cumsum(axis=1).tolist()
        
        stats = functional_boxplot_summary_statistics(data_list)
        
        assert 'depths' in stats
        assert isinstance(stats['depths'], np.ndarray)
    
    def test_larger_percentile_more_curves(self):
        """Test that larger percentiles include more curves."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats_25 = functional_boxplot_summary_statistics(
            data, boxplot_style=BoxplotStyleConfig(percentiles=[25], show_outliers=True)
        )
        stats_75 = functional_boxplot_summary_statistics(
            data, boxplot_style=BoxplotStyleConfig(percentiles=[75], show_outliers=True)
        )
        
        # 75th percentile should have fewer outliers than 25th
        assert stats_75['outliers'].shape[0] <= stats_25['outliers'].shape[0]
    
    def test_sorted_indices_validity(self):
        """Test that sorted_indices are valid indices."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        
        # Check all indices are valid
        assert np.all(stats['sorted_indices'] >= 0)
        assert np.all(stats['sorted_indices'] < 50)
        
        # Check indices are unique
        assert len(np.unique(stats['sorted_indices'])) == 50
    
    def test_small_dataset(self):
        """Test with a small dataset."""
        data = np.random.randn(5, 20).cumsum(axis=1)
        
        stats = functional_boxplot_summary_statistics(data)
        
        assert stats['depths'].shape == (5,)
        assert stats['median'].shape == (20,)
