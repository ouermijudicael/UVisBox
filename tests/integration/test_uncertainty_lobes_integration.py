"""
Integration tests for the complete uncertainty lobes pipeline.

Tests the end-to-end functionality from ensemble vectors to visualization.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from uvisbox.Modules.UncertaintyLobes import (
    uncertainty_lobes,
    compute_uncertainty_lobes_stats_2d,
    build_uncertainty_lobes_mesh_2d,
    render_uncertainty_lobes_2d
)


class TestUncertaintyLobesPipeline:
    """Integration tests for the complete uncertainty lobes pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from ensemble vectors to visualization."""
        # Generate test data
        n_positions = 5
        n_ensemble = 20
        
        positions = np.array([[i * 2, 0] for i in range(n_positions)])
        ensemble_vectors = np.zeros((n_positions, n_ensemble, 2))
        
        for i in range(n_positions):
            base_angle = i * np.pi / 4
            for j in range(n_ensemble):
                angle = base_angle + np.random.normal(0, 0.1)
                mag = 1.0 + np.random.normal(0, 0.05)
                ensemble_vectors[i, j] = [mag * np.cos(angle), mag * np.sin(angle)]
        
        # Test complete pipeline
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions, 
            ensemble_vectors, 
            percentile1=90, 
            percentile2=50,
            scale=0.2,
            ax=ax
        )
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_pipeline_stages_separately(self):
        """Test that all three pipeline stages work together."""
        # Generate test data
        n_positions = 3
        n_ensemble = 15
        
        positions = np.array([[0, 0], [2, 0], [4, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        # Stage 1: Compute statistics
        stats = compute_uncertainty_lobes_stats_2d(
            ensemble_vectors,
            percentile1=90,
            percentile2=50
        )
        
        assert stats is not None
        assert 'theta1' in stats
        
        # Stage 2: Build mesh
        mesh = build_uncertainty_lobes_mesh_2d(positions, stats, scale=0.2)
        
        assert mesh is not None
        assert 'wedges' in mesh
        
        # Stage 3: Render
        fig, ax = plt.subplots()
        result_ax = render_uncertainty_lobes_2d(mesh, show_median=True, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_single_lobe_pipeline(self):
        """Test pipeline with single lobe (percentile2=None)."""
        n_positions = 3
        n_ensemble = 15
        
        positions = np.array([[i, 0] for i in range(n_positions)])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            percentile1=90,
            percentile2=None,
            scale=0.2,
            ax=ax
        )
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_arc_direction_correctness(self):
        """
        Integration test for arc direction correctness.
        
        This is the critical test that verifies the fix for the arc direction bug.
        Tests that wedges are always created in the direction that includes the median.
        """
        test_cases = [
            # (median_angle_deg, angle_spread_deg)
            (0, 10),      # Wrap around 0°
            (45, 20),     # First quadrant
            (90, 15),     # Vertical
            (180, 20),    # Horizontal left
            (270, 15),    # Vertical down
            (350, 10),    # Near 0° (wrap-around)
        ]
        
        n_ensemble = 30
        
        for median_deg, spread_deg in test_cases:
            median_angle = np.deg2rad(median_deg)
            angle_spread = np.deg2rad(spread_deg)
            
            # Create ensemble around median
            ensemble_vectors = np.zeros((1, n_ensemble, 2))
            for j in range(n_ensemble):
                angle = median_angle + np.random.uniform(-angle_spread, angle_spread)
                mag = 1.0 + np.random.normal(0, 0.05)
                ensemble_vectors[0, j] = [mag * np.cos(angle), mag * np.sin(angle)]
            
            positions = np.array([[0, 0]])
            
            # Run through pipeline
            stats = compute_uncertainty_lobes_stats_2d(ensemble_vectors, percentile1=90)
            mesh = build_uncertainty_lobes_mesh_2d(positions, stats, scale=1.0)
            
            # Verify arc includes median
            wedge = mesh['wedges'][0]
            vertices = wedge['vertices']
            center = vertices[0]
            arc_points = vertices[1:]
            
            # Calculate angles of arc points
            arc_angles = np.arctan2(arc_points[:, 1] - center[1], arc_points[:, 0] - center[0])
            
            # Median direction
            median_vec = np.array([np.cos(median_angle), np.sin(median_angle)])
            
            # Check that some arc points point in median direction
            # (dot product > 0.7 means angle < ~45°)
            dot_products = np.dot(arc_points - center, median_vec) / np.linalg.norm(arc_points - center, axis=1)
            
            assert np.any(dot_products > 0.7), \
                f"Arc direction test failed for median={median_deg}°, spread={spread_deg}°"
    
    def test_different_scales(self):
        """Test pipeline with different scale factors."""
        n_positions = 2
        n_ensemble = 10
        
        positions = np.array([[0, 0], [3, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        scales = [0.1, 0.5, 1.0, 2.0]
        
        for scale in scales:
            fig, ax = plt.subplots()
            result_ax = uncertainty_lobes(
                positions,
                ensemble_vectors,
                percentile1=90,
                percentile2=50,
                scale=scale,
                ax=ax
            )
            
            assert result_ax is not None
            plt.close(fig)
    
    def test_different_percentiles(self):
        """Test pipeline with different percentile combinations."""
        n_positions = 2
        n_ensemble = 30
        
        positions = np.array([[0, 0], [3, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        percentile_combinations = [
            (50, None),
            (90, None),
            (90, 50),
            (95, 75),
            (100, 90),
        ]
        
        for p1, p2 in percentile_combinations:
            fig, ax = plt.subplots()
            result_ax = uncertainty_lobes(
                positions,
                ensemble_vectors,
                percentile1=p1,
                percentile2=p2,
                scale=0.2,
                ax=ax
            )
            
            assert result_ax is not None
            plt.close(fig)
    
    def test_show_median_flag(self):
        """Test that show_median flag works correctly."""
        n_positions = 2
        n_ensemble = 10
        
        positions = np.array([[0, 0], [2, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        # Test with show_median=True
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            show_median=True,
            ax=ax
        )
        assert result_ax is not None
        plt.close(fig)
        
        # Test with show_median=False
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            show_median=False,
            ax=ax
        )
        assert result_ax is not None
        plt.close(fig)
    
    def test_minimal_ensemble(self):
        """Test pipeline with minimal ensemble size."""
        n_positions = 2
        n_ensemble = 3  # Minimal ensemble
        
        positions = np.array([[0, 0], [2, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            percentile1=100,  # Include all vectors
            percentile2=None,
            ax=ax
        )
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_large_ensemble(self):
        """Test pipeline with large ensemble."""
        n_positions = 3
        n_ensemble = 100  # Large ensemble
        
        positions = np.array([[0, 0], [2, 0], [4, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            percentile1=90,
            percentile2=50,
            ax=ax
        )
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_deterministic_output(self):
        """Test that same input produces consistent output."""
        n_positions = 2
        n_ensemble = 10
        
        # Create fixed ensemble
        np.random.seed(42)
        positions = np.array([[0, 0], [2, 0]])
        ensemble_vectors = np.random.randn(n_positions, n_ensemble, 2)
        
        # Run pipeline twice
        stats1 = compute_uncertainty_lobes_stats_2d(ensemble_vectors, percentile1=90, percentile2=50)
        stats2 = compute_uncertainty_lobes_stats_2d(ensemble_vectors, percentile1=90, percentile2=50)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(stats1['depths'], stats2['depths'])
        np.testing.assert_array_almost_equal(stats1['theta1'], stats2['theta1'])
        np.testing.assert_array_almost_equal(stats1['mid_angle'], stats2['mid_angle'])


class TestUncertaintyLobesEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_magnitude_vectors(self):
        """Test handling of zero-magnitude vectors in ensemble."""
        n_positions = 1
        n_ensemble = 5
        
        positions = np.array([[0, 0]])
        ensemble_vectors = np.zeros((n_positions, n_ensemble, 2))
        ensemble_vectors[0, 0] = [1.0, 0.0]  # At least one non-zero vector
        
        # Should not crash
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            percentile1=100,
            ax=ax
        )
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_all_identical_vectors(self):
        """Test ensemble with all identical vectors."""
        n_positions = 1
        n_ensemble = 10
        
        positions = np.array([[0, 0]])
        ensemble_vectors = np.ones((n_positions, n_ensemble, 2))
        
        # Should produce very narrow lobes
        fig, ax = plt.subplots()
        result_ax = uncertainty_lobes(
            positions,
            ensemble_vectors,
            percentile1=90,
            ax=ax
        )
        
        assert result_ax is not None
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
