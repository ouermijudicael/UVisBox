"""
Unit tests for functional_boxplot_mesh module.

Tests the functional_boxplot_mesh function.
"""

import pytest
import numpy as np
from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_stats import functional_boxplot_summary_statistics
from uvisbox.Modules.FunctionalBoxplot.functional_boxplot_mesh import functional_boxplot_mesh
from uvisbox.Core.CommonInterface import BoxplotStyleConfig


class TestFunctionalBoxplotMesh:
    """Test suite for functional_boxplot_mesh function."""
    
    def test_identity_pass_through(self):
        """Test that mesh function is an identity pass-through."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        # Should return the same object
        assert mesh_data is stats
    
    def test_preserves_all_keys(self):
        """Test that all keys are preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        
        original_keys = set(stats.keys())
        mesh_data = functional_boxplot_mesh(stats)
        
        assert set(mesh_data.keys()) == original_keys
    
    def test_preserves_depths(self):
        """Test that depths are preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert np.array_equal(mesh_data['depths'], stats['depths'])
    
    def test_preserves_median(self):
        """Test that median curve is preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert np.array_equal(mesh_data['median'], stats['median'])
    
    def test_preserves_percentile_bands(self):
        """Test that percentile bands are preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[25, 50, 75])
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert len(mesh_data['percentile_bands']) == 3
        for key in stats['percentile_bands']:
            assert key in mesh_data['percentile_bands']
            bottom_orig, top_orig = stats['percentile_bands'][key]
            bottom_mesh, top_mesh = mesh_data['percentile_bands'][key]
            assert np.array_equal(bottom_orig, bottom_mesh)
            assert np.array_equal(top_orig, top_mesh)
    
    def test_preserves_outliers(self):
        """Test that outliers are preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(percentiles=[50], show_outliers=True)
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert np.array_equal(mesh_data['outliers'], stats['outliers'])
    
    def test_preserves_sorted_curves(self):
        """Test that sorted curves are preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert np.array_equal(mesh_data['sorted_curves'], stats['sorted_curves'])
    
    def test_preserves_sorted_indices(self):
        """Test that sorted indices are preserved."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert np.array_equal(mesh_data['sorted_indices'], stats['sorted_indices'])
    
    def test_works_with_empty_outliers(self):
        """Test with empty outliers array."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        style = BoxplotStyleConfig(show_outliers=False)
        stats = functional_boxplot_summary_statistics(data, boxplot_style=style)
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert mesh_data['outliers'].shape[0] == 0
    
    def test_works_with_mfbd_method(self):
        """Test that mesh works with MFBD method statistics."""
        data = np.random.randn(50, 100).cumsum(axis=1)
        stats = functional_boxplot_summary_statistics(data, method='mfbd')
        
        mesh_data = functional_boxplot_mesh(stats)
        
        assert mesh_data is stats
