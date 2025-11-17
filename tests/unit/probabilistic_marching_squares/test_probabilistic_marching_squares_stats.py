"""
Unit tests for probabilistic_marching_squares_stats module.

Tests the probabilistic_marching_squares_summary_statistics function.
"""

import pytest
import numpy as np
from uvisbox.Modules.ProbabilisticMarchingSquares.probabilistic_marching_squares_stats import (
    probabilistic_marching_squares_summary_statistics
)


class TestProbabilisticMarchingSquaresSummaryStatistics:
    """Test suite for probabilistic_marching_squares_summary_statistics function."""
    
    def test_basic_computation(self):
        """Test basic summary statistics computation."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        
        assert isinstance(result, dict)
        assert 'level_crossing_probability' in result
    
    def test_output_shape(self):
        """Test that output has correct shape (y_dim-1, x_dim-1)."""
        y_dim, x_dim, n_ensemble = 20, 30, 50
        ensemble = np.random.randn(y_dim, x_dim, n_ensemble)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert prob.shape == (y_dim - 1, x_dim - 1)
    
    def test_probability_range(self):
        """Test that probabilities are in [0, 1] range."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)
    
    def test_probability_dtype(self):
        """Test that probability array has correct dtype."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert np.issubdtype(prob.dtype, np.floating)
    
    def test_all_above_isovalue(self):
        """Test when all ensemble members are above isovalue."""
        ensemble = np.ones((20, 30, 50)) * 2.0
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        # When all values are above isovalue, no crossing should occur
        assert np.all(prob == 0.0)
    
    def test_all_below_isovalue(self):
        """Test when all ensemble members are below isovalue."""
        ensemble = np.ones((20, 30, 50)) * -2.0
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        # When all values are below isovalue, no crossing should occur
        assert np.all(prob == 0.0)
    
    def test_deterministic_crossing(self):
        """Test with deterministic crossing pattern."""
        # Create ensemble where left half is below and right half is above
        y_dim, x_dim, n_ensemble = 20, 30, 50
        ensemble = np.zeros((y_dim, x_dim, n_ensemble))
        ensemble[:, :x_dim//2, :] = -1.0
        ensemble[:, x_dim//2:, :] = 1.0
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        # Cells at the boundary should have high probability
        boundary_col = x_dim // 2 - 1
        assert np.all(prob[:, boundary_col] > 0.5)
    
    def test_varying_isovalue(self):
        """Test with different isovalues."""
        ensemble = np.random.randn(20, 30, 50)
        
        result1 = probabilistic_marching_squares_summary_statistics(ensemble, -1.0)
        result2 = probabilistic_marching_squares_summary_statistics(ensemble, 0.0)
        result3 = probabilistic_marching_squares_summary_statistics(ensemble, 1.0)
        
        prob1 = result1['level_crossing_probability']
        prob2 = result2['level_crossing_probability']
        prob3 = result3['level_crossing_probability']
        
        # Different isovalues should give different results
        assert not np.allclose(prob1, prob2)
        assert not np.allclose(prob2, prob3)
    
    def test_small_ensemble(self):
        """Test with small ensemble size."""
        ensemble = np.random.randn(20, 30, 5)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert prob.shape == (19, 29)
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)
    
    def test_large_ensemble(self):
        """Test with large ensemble size."""
        ensemble = np.random.randn(20, 30, 200)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert prob.shape == (19, 29)
    
    def test_single_cell(self):
        """Test with minimal grid (single cell)."""
        ensemble = np.random.randn(2, 2, 50)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert prob.shape == (1, 1)
    
    def test_input_not_modified(self):
        """Test that input ensemble is not modified."""
        ensemble = np.random.randn(20, 30, 50)
        ensemble_copy = ensemble.copy()
        isovalue = 0.0
        
        probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        
        np.testing.assert_array_equal(ensemble, ensemble_copy)
    
    def test_negative_isovalue(self):
        """Test with negative isovalue."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = -1.5
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert prob.shape == (19, 29)
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)
    
    def test_positive_isovalue(self):
        """Test with positive isovalue."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 1.5
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert prob.shape == (19, 29)
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)
    
    def test_different_grid_sizes(self):
        """Test with different grid sizes."""
        for y_dim, x_dim in [(10, 15), (25, 25), (30, 20), (50, 100)]:
            ensemble = np.random.randn(y_dim, x_dim, 30)
            isovalue = 0.0
            
            result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
            prob = result['level_crossing_probability']
            
            assert prob.shape == (y_dim - 1, x_dim - 1)
    
    def test_known_crossing_probability(self):
        """Test with known crossing probability."""
        # Create ensemble where exactly half have crossings
        y_dim, x_dim, n_ensemble = 10, 10, 100
        ensemble = np.zeros((y_dim, x_dim, n_ensemble))
        
        # First 50 members: left half below, right half above
        ensemble[:, :x_dim//2, :50] = -1.0
        ensemble[:, x_dim//2:, :50] = 1.0
        
        # Last 50 members: all above
        ensemble[:, :, 50:] = 1.0
        
        isovalue = 0.0
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        # Boundary cells should have ~0.5 probability (relaxed tolerance due to Monte Carlo)
        boundary_col = x_dim // 2 - 1
        assert np.allclose(prob[:, boundary_col], 0.5, atol=0.1)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same input."""
        ensemble = np.random.RandomState(42).randn(20, 30, 50)
        isovalue = 0.0
        
        result1 = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        result2 = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        
        # Results should be similar (Monte Carlo may have small variations)
        np.testing.assert_allclose(
            result1['level_crossing_probability'],
            result2['level_crossing_probability'],
            rtol=0.15  # 15% relative tolerance for Monte Carlo
        )
    
    def test_float_input_types(self):
        """Test with different float types."""
        for dtype in [np.float32, np.float64]:
            ensemble = np.random.randn(20, 30, 50).astype(dtype)
            isovalue = 0.0
            
            result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
            prob = result['level_crossing_probability']
            
            assert prob.shape == (19, 29)
            assert np.all(prob >= 0.0)
            assert np.all(prob <= 1.0)
    
    def test_zero_isovalue(self):
        """Test with zero isovalue."""
        ensemble = np.random.randn(20, 30, 50)
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        
        assert 'level_crossing_probability' in result
    
    def test_extreme_values(self):
        """Test with extreme values in ensemble."""
        ensemble = np.random.randn(20, 30, 50) * 1e6
        isovalue = 0.0
        
        result = probabilistic_marching_squares_summary_statistics(ensemble, isovalue)
        prob = result['level_crossing_probability']
        
        assert np.all(np.isfinite(prob))
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)
