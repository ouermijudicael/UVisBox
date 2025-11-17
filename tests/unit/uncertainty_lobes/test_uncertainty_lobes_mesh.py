"""
Unit tests for uncertainty_lobes_mesh.py

Tests the mesh generation stage of the uncertainty lobes pipeline,
including the critical arc direction logic.
"""

import numpy as np
import pytest
from uvisbox.Modules.UncertaintyLobes.uncertainty_lobes_mesh import (
    build_uncertainty_lobes_mesh_2d,
    _create_wedge_vertices,
    _triangulate_wedge
)


class TestCreateWedgeVertices:
    """Test suite for _create_wedge_vertices function."""
    
    def test_basic_wedge_creation(self):
        """Test basic wedge vertex generation."""
        center = np.array([0.0, 0.0])
        r = 1.0
        theta_start = 0.0
        theta_end = np.pi / 2
        median_angle = np.pi / 4  # 45°
        num_points = 10
        
        vertices = _create_wedge_vertices(center, r, theta_start, theta_end, median_angle, num_points)
        
        # Should have num_points + 1 vertices (center + arc points)
        assert vertices.shape == (num_points + 1, 2)
        
        # First vertex should be center
        np.testing.assert_array_almost_equal(vertices[0], center)
        
        # All arc points should be at radius r from center
        arc_points = vertices[1:]
        distances = np.linalg.norm(arc_points - center, axis=1)
        np.testing.assert_array_almost_equal(distances, r, decimal=5)
    
    def test_arc_includes_median_normal_case(self):
        """Test that arc includes median angle (normal case, no wrap-around)."""
        center = np.array([0.0, 0.0])
        r = 1.0
        theta_start = np.deg2rad(30)
        theta_end = np.deg2rad(60)
        median_angle = np.deg2rad(45)  # Should be in the arc
        num_points = 20
        
        vertices = _create_wedge_vertices(center, r, theta_start, theta_end, median_angle, num_points)
        
        # Calculate angles of arc points
        arc_points = vertices[1:]
        arc_angles = np.arctan2(arc_points[:, 1] - center[1], arc_points[:, 0] - center[0])
        
        # Normalize to [0, 2π)
        arc_angles = arc_angles % (2 * np.pi)
        median_angle_norm = median_angle % (2 * np.pi)
        
        # The median should be within the range of arc angles
        min_arc = np.min(arc_angles)
        max_arc = np.max(arc_angles)
        
        # For non-wrapping case
        if max_arc - min_arc < np.pi:
            assert min_arc <= median_angle_norm <= max_arc
    
    def test_arc_includes_median_wrap_around(self):
        """Test that arc includes median angle (wrap-around case)."""
        center = np.array([0.0, 0.0])
        r = 1.0
        theta_start = np.deg2rad(350)  # 350°
        theta_end = np.deg2rad(10)     # 10°
        median_angle = np.deg2rad(0)   # 0° (should be in arc)
        num_points = 20
        
        vertices = _create_wedge_vertices(center, r, theta_start, theta_end, median_angle, num_points)
        
        # Calculate angles of arc points
        arc_points = vertices[1:]
        arc_angles = np.arctan2(arc_points[:, 1] - center[1], arc_points[:, 0] - center[0])
        arc_angles = arc_angles % (2 * np.pi)
        
        # For wrap-around, check that some angles are near 0 and some near 2π
        near_zero = np.any(arc_angles < np.deg2rad(20))
        near_360 = np.any(arc_angles > np.deg2rad(340))
        
        assert near_zero or near_360
    
    def test_arc_direction_consistency(self):
        """Test that arc always goes in the direction containing median."""
        test_cases = [
            # (theta_start, theta_end, median) in degrees
            (0, 90, 45),      # Normal case
            (350, 10, 0),     # Wrap around 0
            (170, 190, 180),  # Southern hemisphere
            (30, 150, 90),    # Large spread
            (270, 330, 300),  # Western hemisphere
        ]
        
        center = np.array([0.0, 0.0])
        r = 1.0
        num_points = 30
        
        for theta_s_deg, theta_e_deg, med_deg in test_cases:
            theta_start = np.deg2rad(theta_s_deg)
            theta_end = np.deg2rad(theta_e_deg)
            median_angle = np.deg2rad(med_deg)
            
            vertices = _create_wedge_vertices(center, r, theta_start, theta_end, median_angle, num_points)
            
            # Calculate arc angles
            arc_points = vertices[1:]
            arc_angles = np.arctan2(arc_points[:, 1], arc_points[:, 0])
            
            # Check that the median direction is "reachable" from the arc
            # by checking the dot product with median vector
            median_vec = np.array([np.cos(median_angle), np.sin(median_angle)])
            
            # At least some arc points should have positive dot product with median
            dot_products = np.dot(arc_points, median_vec)
            assert np.any(dot_products > 0.5), \
                f"Failed for case: start={theta_s_deg}°, end={theta_e_deg}°, median={med_deg}°"
    
    def test_different_radii(self):
        """Test wedge creation with different radii."""
        center = np.array([1.0, 2.0])
        radii = [0.5, 1.0, 2.0, 5.0]
        theta_start = 0.0
        theta_end = np.pi / 2
        median_angle = np.pi / 4
        
        for r in radii:
            vertices = _create_wedge_vertices(center, r, theta_start, theta_end, median_angle)
            
            arc_points = vertices[1:]
            distances = np.linalg.norm(arc_points - center, axis=1)
            np.testing.assert_array_almost_equal(distances, r, decimal=5)
    
    def test_different_centers(self):
        """Test wedge creation at different center positions."""
        centers = [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([-2.0, 3.0]),
            np.array([10.0, -5.0])
        ]
        r = 1.0
        theta_start = 0.0
        theta_end = np.pi / 2
        median_angle = np.pi / 4
        
        for center in centers:
            vertices = _create_wedge_vertices(center, r, theta_start, theta_end, median_angle)
            
            # First vertex should be center
            np.testing.assert_array_almost_equal(vertices[0], center)


class TestTriangulateWedge:
    """Test suite for _triangulate_wedge function."""
    
    def test_triangle_count(self):
        """Test that correct number of triangles are generated."""
        num_points = 20
        triangles = _triangulate_wedge(num_points)
        
        # Should have num_points - 1 triangles
        assert triangles.shape == (num_points - 1, 3)
    
    def test_triangle_indices(self):
        """Test that triangle indices are valid."""
        num_points = 10
        triangles = _triangulate_wedge(num_points)
        
        # All indices should be in valid range [0, num_points]
        assert np.all(triangles >= 0)
        assert np.all(triangles <= num_points)
        
        # First column should all be 0 (center point)
        assert np.all(triangles[:, 0] == 0)
    
    def test_triangle_connectivity(self):
        """Test that triangles form a continuous fan."""
        num_points = 5
        triangles = _triangulate_wedge(num_points)
        
        # Each triangle should connect center (0) to two consecutive arc points
        for i in range(num_points - 1):
            assert triangles[i, 0] == 0
            assert triangles[i, 1] == i + 1
            assert triangles[i, 2] == i + 2


class TestBuildUncertaintyLobesMesh2D:
    """Test suite for build_uncertainty_lobes_mesh_2d function."""
    
    def test_basic_mesh_generation(self):
        """Test basic mesh generation from statistics."""
        n_positions = 3
        
        # Create mock statistics
        stats = {
            'theta1': np.array([[0.0, np.pi/2], [np.pi/2, np.pi], [np.pi, 3*np.pi/2]]),
            'theta2': np.array([[0.0, np.pi/4], [np.pi/2, 3*np.pi/4], [np.pi, 5*np.pi/4]]),
            'mid_angle': np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4]),
            'r1': np.array([1.0, 1.0, 1.0]),
            'r2': np.array([0.5, 0.5, 0.5]),
            'r_arrow': np.array([1.0, 1.0, 1.0])
        }
        
        positions = np.array([[0, 0], [2, 0], [4, 0]])
        
        mesh = build_uncertainty_lobes_mesh_2d(positions, stats, scale=0.2)
        
        # Check structure
        assert 'wedges' in mesh
        assert 'inner_wedges' in mesh
        assert 'arrows' in mesh
        
        # Check wedges
        assert len(mesh['wedges']) == n_positions
        
        # Check inner wedges
        assert mesh['inner_wedges'] is not None
        assert len(mesh['inner_wedges']) == n_positions
        
        # Check arrows
        assert mesh['arrows']['positions'].shape == (n_positions, 2)
        assert mesh['arrows']['directions'].shape == (n_positions, 2)
        assert mesh['arrows']['lengths'].shape == (n_positions,)
    
    def test_single_lobe_mesh(self):
        """Test mesh generation with only outer lobe (no percentile2)."""
        n_positions = 2
        
        stats = {
            'theta1': np.array([[0.0, np.pi/2], [np.pi/2, np.pi]]),
            'theta2': None,  # No inner lobe
            'mid_angle': np.array([np.pi/4, 3*np.pi/4]),
            'r1': np.array([1.0, 1.0]),
            'r2': np.array([0.0, 0.0]),  # Zero radius for inner lobe
            'r_arrow': np.array([1.0, 1.0])
        }
        
        positions = np.array([[0, 0], [2, 0]])
        
        mesh = build_uncertainty_lobes_mesh_2d(positions, stats)
        
        # Should have outer wedges
        assert len(mesh['wedges']) == n_positions
        
        # Should not have inner wedges
        assert mesh['inner_wedges'] is None
    
    def test_scale_application(self):
        """Test that scale factor is correctly applied."""
        n_positions = 1
        
        stats = {
            'theta1': np.array([[0.0, np.pi/2]]),
            'theta2': np.array([[0.0, np.pi/4]]),
            'mid_angle': np.array([np.pi/4]),
            'r1': np.array([1.0]),
            'r2': np.array([0.5]),
            'r_arrow': np.array([1.0])
        }
        
        positions = np.array([[0, 0]])
        
        # Test different scales
        scales = [0.1, 0.5, 1.0, 2.0]
        
        for scale in scales:
            mesh = build_uncertainty_lobes_mesh_2d(positions, stats, scale=scale)
            
            # Check arrow length
            assert mesh['arrows']['lengths'][0] == pytest.approx(1.0 * scale)
            
            # Check wedge vertices are scaled
            wedge = mesh['wedges'][0]
            center = wedge['vertices'][0]
            arc_points = wedge['vertices'][1:]
            distances = np.linalg.norm(arc_points - center, axis=1)
            
            # Should all be approximately r1 * scale
            np.testing.assert_array_almost_equal(distances, 1.0 * scale, decimal=5)
    
    def test_arrow_directions(self):
        """Test that arrow directions are computed correctly."""
        n_positions = 4
        
        # Test angles at cardinal directions
        mid_angles = np.array([0, np.pi/2, np.pi, -np.pi/2])
        
        stats = {
            'theta1': np.zeros((n_positions, 2)),
            'theta2': np.zeros((n_positions, 2)),
            'mid_angle': mid_angles,
            'r1': np.ones(n_positions),
            'r2': np.ones(n_positions) * 0.5,
            'r_arrow': np.ones(n_positions)
        }
        
        positions = np.array([[i, 0] for i in range(n_positions)])
        
        mesh = build_uncertainty_lobes_mesh_2d(positions, stats)
        
        # Check arrow directions
        directions = mesh['arrows']['directions']
        
        expected_directions = np.array([
            [1, 0],      # 0 rad
            [0, 1],      # π/2 rad
            [-1, 0],     # π rad
            [0, -1]      # -π/2 rad
        ])
        
        np.testing.assert_array_almost_equal(directions, expected_directions, decimal=5)
    
    def test_arc_resolution(self):
        """Test different arc resolutions."""
        n_positions = 1
        
        stats = {
            'theta1': np.array([[0.0, np.pi/2]]),
            'theta2': None,
            'mid_angle': np.array([np.pi/4]),
            'r1': np.array([1.0]),
            'r2': np.array([0.0]),
            'r_arrow': np.array([1.0])
        }
        
        positions = np.array([[0, 0]])
        
        resolutions = [5, 10, 20, 50]
        
        for res in resolutions:
            mesh = build_uncertainty_lobes_mesh_2d(positions, stats, arc_resolution=res)
            
            # Check number of vertices
            wedge = mesh['wedges'][0]
            assert wedge['vertices'].shape[0] == res + 1  # res arc points + center
            
            # Check number of triangles
            assert wedge['triangles'].shape[0] == res - 1
    
    def test_zero_radius_handling(self):
        """Test handling of zero or very small radii."""
        n_positions = 2
        
        stats = {
            'theta1': np.array([[0.0, np.pi/2], [0.0, np.pi/2]]),
            'theta2': np.array([[0.0, np.pi/4], [0.0, np.pi/4]]),
            'mid_angle': np.array([np.pi/4, np.pi/4]),
            'r1': np.array([1.0, 1.0]),
            'r2': np.array([0.5, 0.0]),  # Second position has zero inner radius
            'r_arrow': np.array([1.0, 1.0])
        }
        
        positions = np.array([[0, 0], [2, 0]])
        
        mesh = build_uncertainty_lobes_mesh_2d(positions, stats)
        
        # Should have 2 outer wedges
        assert len(mesh['wedges']) == 2
        
        # Should have only 1 inner wedge (second position filtered out)
        assert len(mesh['inner_wedges']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
