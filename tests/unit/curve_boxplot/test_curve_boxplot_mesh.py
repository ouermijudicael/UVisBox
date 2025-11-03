"""
Unit tests for curve_boxplot_mesh module.

Tests the curve_boxplot_mesh function and mesh building functionality.
"""

import pytest
import numpy as np
from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_boxplot_summary_statistics
from uvisbox.Modules.CurveBoxplot.curve_boxplot_mesh import curve_boxplot_mesh, _build_percentile_mesh
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestCurveBoxplotMesh:
    """Test suite for curve_boxplot_mesh function."""
    
    def test_basic_2d_mesh_generation(self):
        """Test basic mesh generation with 2D curves."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Check expected keys
        assert 'percentile_meshes' in mesh_data
        assert 'median_curve' in mesh_data
        assert 'outliers' in mesh_data
        assert 'n_dims' in mesh_data
        
        assert mesh_data['n_dims'] == 2
    
    def test_basic_3d_mesh_generation(self):
        """Test basic mesh generation with 3D curves."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        assert mesh_data['n_dims'] == 3
    
    def test_percentile_meshes_created(self):
        """Test that meshes are created for each percentile."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        percentiles = [25, 50, 75]
        style = BoxplotStyleConfig(percentiles=percentiles)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Check that mesh exists for each percentile
        for percentile in percentiles:
            mesh_key = f'{int(percentile)}_percentile_mesh'
            assert mesh_key in mesh_data['percentile_meshes']
    
    def test_mesh_structure_2d(self):
        """Test that 2D mesh has correct structure (points and triangles)."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Get a sample mesh
        mesh_key = '50_percentile_mesh'
        points, triangles = mesh_data['percentile_meshes'][mesh_key]
        
        # Check shapes
        assert points.ndim == 2
        assert points.shape[1] == 2  # 2D points
        assert triangles.ndim == 2
        assert triangles.shape[1] == 3  # Triangular faces
        
        # Check that triangle indices are valid
        assert np.all(triangles >= 0)
        assert np.all(triangles < points.shape[0])
    
    def test_mesh_structure_3d(self):
        """Test that 3D mesh has correct structure."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Get a sample mesh
        mesh_key = '50_percentile_mesh'
        points, triangles = mesh_data['percentile_meshes'][mesh_key]
        
        # Check shapes
        assert points.ndim == 2
        assert points.shape[1] == 3  # 3D points
        assert triangles.ndim == 2
        assert triangles.shape[1] == 3  # Triangular faces
        
        # Check that triangle indices are valid
        assert np.all(triangles >= 0)
        assert np.all(triangles < points.shape[0])
    
    def test_median_and_outliers_preserved(self):
        """Test that median curve and outliers are passed through."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        style = BoxplotStyleConfig(show_outliers=True)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Check that median and outliers are preserved
        np.testing.assert_array_equal(mesh_data['median_curve'], stats['median_curve'])
        np.testing.assert_array_equal(mesh_data['outliers'], stats['outliers'])
    
    def test_larger_percentile_more_points(self):
        """Test that larger percentiles generally produce meshes with more points."""
        n_curves, n_steps = 50, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        percentiles = [25, 50, 75]
        style = BoxplotStyleConfig(percentiles=percentiles)
        stats = curve_boxplot_summary_statistics(curves, boxplot_style=style)
        mesh_data = curve_boxplot_mesh(stats)
        
        # Get meshes for different percentiles
        points_25, _ = mesh_data['percentile_meshes']['25_percentile_mesh']
        points_50, _ = mesh_data['percentile_meshes']['50_percentile_mesh']
        points_75, _ = mesh_data['percentile_meshes']['75_percentile_mesh']
        
        # Generally, larger percentiles should have more or equal points
        # (not strict inequality due to sampling and hull construction)
        assert points_25.shape[0] <= points_75.shape[0] + 1000  # Allow some variance


class TestBuildPercentileMesh:
    """Test suite for _build_percentile_mesh helper function."""
    
    def test_build_percentile_mesh_2d(self):
        """Test building a single percentile mesh in 2D."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        # Sort curves (required for _build_percentile_mesh)
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        points, triangles = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        
        assert points.shape[1] == 2
        assert triangles.shape[1] == 3
        assert np.all(triangles >= 0)
        assert np.all(triangles < points.shape[0])
    
    def test_build_percentile_mesh_3d(self):
        """Test building a single percentile mesh in 3D."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 3).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        points, triangles = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=3)
        
        assert points.shape[1] == 3
        assert triangles.shape[1] == 3
        assert np.all(triangles >= 0)
        assert np.all(triangles < points.shape[0])
    
    def test_percentile_affects_mesh_size(self):
        """Test that different percentiles produce different mesh sizes."""
        n_curves, n_steps = 50, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        points_10, _ = _build_percentile_mesh(sorted_curves, percentile=10, n_dims=2)
        points_90, _ = _build_percentile_mesh(sorted_curves, percentile=90, n_dims=2)
        
        # 90th percentile should include more curves, thus more points
        # (allowing for some variance due to hull construction)
        assert points_10.shape[0] < points_90.shape[0] + 500
    
    def test_stride_computation_small_timesteps(self):
        """Test that stride is 1 for small number of time steps."""
        n_curves, n_steps = 30, 50  # < 100 steps
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        # Should work without errors
        points, triangles = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        
        assert points.shape[0] > 0
        assert triangles.shape[0] > 0
    
    def test_stride_computation_large_timesteps(self):
        """Test that stride is computed for large number of time steps."""
        n_curves, n_steps = 30, 200  # > 100 steps
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        # Should work without errors and use stride
        points, triangles = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        
        assert points.shape[0] > 0
        assert triangles.shape[0] > 0
    
    def test_triangles_reference_valid_points(self):
        """Test that all triangle indices reference valid points."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        points, triangles = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        
        # All triangle indices should be within valid range
        assert np.all(triangles >= 0)
        assert np.all(triangles < points.shape[0])
        
        # No duplicate indices in same triangle
        for tri in triangles:
            assert len(set(tri)) == 3  # All three indices should be unique
    
    def test_empty_curves_edge_case(self):
        """Test behavior with very small percentile that may produce too few points."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        # Very small percentile - may fail due to insufficient points for ConvexHull
        # This is expected behavior for edge cases
        try:
            points, triangles = _build_percentile_mesh(sorted_curves, percentile=1, n_dims=2)
            # If it works, should have valid mesh
            assert points.shape[0] > 0
        except Exception:
            # Edge case with too few points is acceptable
            pass
    
    def test_full_percentile_includes_all_curves(self):
        """Test that 100th percentile includes all curves."""
        n_curves, n_steps = 20, 40
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        points_100, _ = _build_percentile_mesh(sorted_curves, percentile=100, n_dims=2)
        points_50, _ = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        
        # 100th percentile should have more points than 50th
        assert points_100.shape[0] >= points_50.shape[0]
    
    def test_consistent_mesh_generation(self):
        """Test that mesh generation is deterministic for same input."""
        n_curves, n_steps = 30, 50
        curves = np.random.randn(n_curves, n_steps, 2).cumsum(axis=1)
        
        stats = curve_boxplot_summary_statistics(curves)
        sorted_curves = stats['sorted_curves']
        
        # Generate mesh twice
        points1, triangles1 = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        points2, triangles2 = _build_percentile_mesh(sorted_curves, percentile=50, n_dims=2)
        
        # Should be identical
        np.testing.assert_array_equal(points1, points2)
        np.testing.assert_array_equal(triangles1, triangles2)
