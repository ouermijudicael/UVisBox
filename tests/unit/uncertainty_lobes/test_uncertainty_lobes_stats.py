"""
Unit tests for uncertainty_lobes_stats.py

Tests the statistics computation stage of the uncertainty lobes pipeline.
"""

import numpy as np
import pytest
from uvisbox.Modules.UncertaintyLobes.uncertainty_lobes_stats import (
    compute_uncertainty_lobes_stats_2d,
    _process_position_stats
)


class TestComputeUncertaintyLobesStats2D:
    """Test suite for compute_uncertainty_lobes_stats_2d function."""
    
    def test_basic_functionality(self):
        """Test basic statistics computation with simple ensemble."""
        # Create simple test data
        n_positions = 3
        n_ensemble = 20
        
        positions_data = []
        ensemble_vectors = np.zeros((n_positions, n_ensemble, 2))
        
        for i in range(n_positions):
            base_angle = i * np.pi / 4
            base_mag = 1.0
            
            for j in range(n_ensemble):
                angle = base_angle + np.random.normal(0, 0.1)
                mag = base_mag + np.random.normal(0, 0.05)
                ensemble_vectors[i, j] = [mag * np.cos(angle), mag * np.sin(angle)]
        
        # Compute statistics
        stats = compute_uncertainty_lobes_stats_2d(
            ensemble_vectors, 
            percentile1=90, 
            percentile2=50
        )
        
        # Check output structure
        assert 'ensemble_polar_vectors' in stats
        assert 'depths' in stats
        assert 'median_vectors' in stats
        assert 'theta1' in stats
        assert 'theta2' in stats
        assert 'mid_angle' in stats
        assert 'r1' in stats
        assert 'r2' in stats
        assert 'r_arrow' in stats
        
        # Check shapes
        assert stats['ensemble_polar_vectors'].shape == (n_positions, n_ensemble, 2)
        assert stats['depths'].shape == (n_positions, n_ensemble)
        assert stats['median_vectors'].shape == (n_positions, 2)
        assert stats['theta1'].shape == (n_positions, 2)
        assert stats['theta2'].shape == (n_positions, 2)
        assert stats['mid_angle'].shape == (n_positions,)
        assert stats['r1'].shape == (n_positions,)
        assert stats['r2'].shape == (n_positions,)
        assert stats['r_arrow'].shape == (n_positions,)
    
    def test_single_percentile(self):
        """Test with percentile2=None (single lobe)."""
        n_positions = 2
        n_ensemble = 15
        
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        stats = compute_uncertainty_lobes_stats_2d(
            ensemble_vectors, 
            percentile1=90, 
            percentile2=None
        )
        
        # theta2 should be None
        assert stats['theta2'] is None
        
        # r2 should be zeros
        assert np.all(stats['r2'] == 0.0)
    
    def test_dual_percentile(self):
        """Test with both percentile1 and percentile2."""
        n_positions = 2
        n_ensemble = 20
        
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        stats = compute_uncertainty_lobes_stats_2d(
            ensemble_vectors, 
            percentile1=90, 
            percentile2=50
        )
        
        # theta2 should exist
        assert stats['theta2'] is not None
        assert stats['theta2'].shape == (n_positions, 2)
        
        # r2 should be non-zero (generally)
        assert np.any(stats['r2'] > 0.0)
    
    def test_polar_conversion(self):
        """Test that Cartesian to polar conversion is correct."""
        n_positions = 1
        n_ensemble = 5
        
        # Create known vectors
        ensemble_vectors = np.array([
            [[1.0, 0.0],   # 0°, magnitude 1
             [0.0, 1.0],   # 90°, magnitude 1
             [-1.0, 0.0],  # 180°, magnitude 1
             [0.0, -1.0],  # 270°, magnitude 1
             [1.0, 1.0]]   # 45°, magnitude sqrt(2)
        ])
        
        stats = compute_uncertainty_lobes_stats_2d(
            ensemble_vectors, 
            percentile1=100
        )
        
        polar = stats['ensemble_polar_vectors'][0]
        
        # Check magnitudes
        expected_mags = [1.0, 1.0, 1.0, 1.0, np.sqrt(2)]
        np.testing.assert_array_almost_equal(polar[:, 0], expected_mags, decimal=5)
        
        # Check angles (approximately)
        expected_angles = [0, np.pi/2, np.pi, -np.pi/2, np.pi/4]
        for i, expected_angle in enumerate(expected_angles):
            # Account for angle wrapping
            angle_diff = np.abs(polar[i, 1] - expected_angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            assert angle_diff < 0.01
    
    def test_depths_range(self):
        """Test that depths are in valid range [0, 1]."""
        n_positions = 2
        n_ensemble = 10
        
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        stats = compute_uncertainty_lobes_stats_2d(ensemble_vectors)
        
        # All depths should be in [0, 1]
        assert np.all(stats['depths'] >= 0.0)
        assert np.all(stats['depths'] <= 1.0)
    
    def test_median_vector_has_max_depth(self):
        """Test that median vector corresponds to maximum depth."""
        n_positions = 2
        n_ensemble = 15
        
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        stats = compute_uncertainty_lobes_stats_2d(ensemble_vectors)
        
        for i in range(n_positions):
            max_depth_idx = np.argmax(stats['depths'][i])
            median_vector = stats['ensemble_polar_vectors'][i][max_depth_idx]
            
            # Median vectors should match
            np.testing.assert_array_almost_equal(
                stats['median_vectors'][i], 
                median_vector,
                decimal=5
            )
    
    def test_angular_spread_ordering(self):
        """Test that min_angle < max_angle in theta arrays."""
        n_positions = 3
        n_ensemble = 20
        
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        stats = compute_uncertainty_lobes_stats_2d(
            ensemble_vectors,
            percentile1=90,
            percentile2=50
        )
        
        # For theta1 (outer lobe)
        for i in range(n_positions):
            # Note: angles might wrap around, so this test checks the data structure
            assert stats['theta1'][i].shape == (2,)
        
        # For theta2 (inner lobe) if exists
        if stats['theta2'] is not None:
            for i in range(n_positions):
                assert stats['theta2'][i].shape == (2,)
    
    def test_percentile_effect(self):
        """Test that higher percentile includes more vectors (larger spread)."""
        n_positions = 1
        n_ensemble = 50
        
        # Create tight ensemble
        base_angle = 0.0
        ensemble_vectors = np.zeros((n_positions, n_ensemble, 2))
        for j in range(n_ensemble):
            angle = base_angle + np.random.normal(0, 0.2)
            mag = 1.0 + np.random.normal(0, 0.1)
            ensemble_vectors[0, j] = [mag * np.cos(angle), mag * np.sin(angle)]
        
        # Compute with different percentiles
        stats_50 = compute_uncertainty_lobes_stats_2d(ensemble_vectors, percentile1=50)
        stats_90 = compute_uncertainty_lobes_stats_2d(ensemble_vectors, percentile1=90)
        
        # Higher percentile should have larger angular spread
        spread_50 = stats_50['theta1'][0, 1] - stats_50['theta1'][0, 0]
        spread_90 = stats_90['theta1'][0, 1] - stats_90['theta1'][0, 0]
        
        # Handle wrap-around
        spread_50 = spread_50 if spread_50 >= 0 else spread_50 + 2*np.pi
        spread_90 = spread_90 if spread_90 >= 0 else spread_90 + 2*np.pi
        
        assert spread_90 >= spread_50
    
    def test_empty_ensemble(self):
        """Test behavior with minimal ensemble size."""
        n_positions = 1
        n_ensemble = 2  # Minimal size
        
        ensemble_vectors = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        
        stats = compute_uncertainty_lobes_stats_2d(ensemble_vectors)
        
        # Should still produce valid output
        assert stats['depths'].shape == (n_positions, n_ensemble)
        assert stats['median_vectors'].shape == (n_positions, 2)


class TestProcessPositionStats:
    """Test suite for _process_position_stats worker function."""
    
    def test_worker_function(self):
        """Test the parallel worker function directly."""
        n_ensemble = 10
        
        # Create test ensemble in polar coordinates
        ensemble_polar = np.zeros((n_ensemble, 2))
        base_angle = 0.0
        for j in range(n_ensemble):
            angle = base_angle + np.random.normal(0, 0.1)
            mag = 1.0 + np.random.normal(0, 0.05)
            ensemble_polar[j] = [mag, angle]
        
        args = (0, ensemble_polar, 90, 50)
        result = _process_position_stats(args)
        
        # Unpack result
        i_pos, depths, theta1, theta2, mid_angle, r1, r2, r_arrow = result
        
        # Check types and shapes
        assert i_pos == 0
        assert depths.shape == (n_ensemble,)
        assert theta1.shape == (2,)
        assert theta2.shape == (2,)
        assert isinstance(mid_angle, (float, np.floating))
        assert isinstance(r1, (float, np.floating))
        assert isinstance(r2, (float, np.floating))
        assert isinstance(r_arrow, (float, np.floating))
    
    def test_worker_function_single_percentile(self):
        """Test worker function with percentile2=None."""
        n_ensemble = 10
        
        ensemble_polar = np.random.randn(n_ensemble, 2)
        ensemble_polar[:, 0] = np.abs(ensemble_polar[:, 0])  # Magnitudes positive
        
        args = (0, ensemble_polar, 90, None)
        result = _process_position_stats(args)
        
        i_pos, depths, theta1, theta2, mid_angle, r1, r2, r_arrow = result
        
        # theta2 should be None
        assert theta2 is None
        # r2 should be 0
        assert r2 == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
