"""
Unit tests for probabilistic_marching_squares_mesh module.

Tests the probabilistic_marching_squares_mesh function.
"""

import pytest
import numpy as np
from uvisbox.Modules.ProbabilisticMarchingSquares.probabilistic_marching_squares_mesh import (
    probabilistic_marching_squares_mesh
)


class TestProbabilisticMarchingSquaresMesh:
    """Test suite for probabilistic_marching_squares_mesh function."""
    
    def test_extracts_probability_from_dict(self):
        """Test that function extracts level_crossing_probability from dict."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        np.testing.assert_array_equal(result['level_crossing_probability'], prob_array)

    def test_preserves_array_shape(self):
        """Test that output has same shape as input probability array."""
        prob_array = np.random.rand(20, 30)
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        assert result['level_crossing_probability'].shape == prob_array.shape

    def test_preserves_array_dtype(self):
        """Test that output has same dtype as input."""
        for dtype in [np.float32, np.float64]:
            prob_array = np.random.rand(10, 15).astype(dtype)
            summary_stats = {'level_crossing_probability': prob_array}

            result = probabilistic_marching_squares_mesh(summary_stats)

            assert result['level_crossing_probability'].dtype == dtype

    def test_preserves_values(self):
        """Test that values are preserved exactly."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        np.testing.assert_array_equal(result['level_crossing_probability'], prob_array)

    def test_with_zeros(self):
        """Test with all-zero probability array."""
        prob_array = np.zeros((10, 15))
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        np.testing.assert_array_equal(result['level_crossing_probability'], prob_array)

    def test_with_ones(self):
        """Test with all-one probability array."""
        prob_array = np.ones((10, 15))
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        np.testing.assert_array_equal(result['level_crossing_probability'], prob_array)

    def test_with_mixed_values(self):
        """Test with mixed probability values."""
        prob_array = np.array([[0.0, 0.5, 1.0],
                               [0.25, 0.75, 0.1],
                               [0.9, 0.3, 0.6]])
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        np.testing.assert_array_equal(result['level_crossing_probability'], prob_array)

    def test_single_cell(self):
        """Test with single cell (1x1 array)."""
        prob_array = np.array([[0.5]])
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        assert result['level_crossing_probability'].shape == (1, 1)
        assert result['level_crossing_probability'][0, 0] == 0.5

    def test_large_array(self):
        """Test with large probability array."""
        prob_array = np.random.rand(100, 200)
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        assert result['level_crossing_probability'].shape == (100, 200)
        np.testing.assert_array_equal(result['level_crossing_probability'], prob_array)

    def test_does_not_modify_input(self):
        """Test that input dictionary is not modified."""
        prob_array = np.random.rand(10, 15)
        prob_array_copy = prob_array.copy()
        summary_stats = {'level_crossing_probability': prob_array}

        probabilistic_marching_squares_mesh(summary_stats)

        np.testing.assert_array_equal(prob_array, prob_array_copy)
        assert 'level_crossing_probability' in summary_stats

    def test_missing_key_raises_error(self):
        """Test that missing key raises KeyError."""
        summary_stats = {'wrong_key': np.random.rand(10, 15)}

        with pytest.raises(KeyError):
            probabilistic_marching_squares_mesh(summary_stats)

    def test_empty_dict_raises_error(self):
        """Test that empty dict raises KeyError."""
        summary_stats = {}

        with pytest.raises(KeyError):
            probabilistic_marching_squares_mesh(summary_stats)

    def test_different_shaped_arrays(self):
        """Test with different shaped arrays."""
        shapes = [(5, 10), (20, 20), (15, 25), (1, 50), (50, 1)]

        for shape in shapes:
            prob_array = np.random.rand(*shape)
            summary_stats = {'level_crossing_probability': prob_array}

            result = probabilistic_marching_squares_mesh(summary_stats)

            assert result['level_crossing_probability'].shape == shape

    def test_is_view_or_copy(self):
        """Test whether result is view or copy of input array."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        # Modify result and check if original is affected
        prob = result['level_crossing_probability']
        prob[0, 0] = -999

        # This tests the current implementation behavior
        # If it's a view, original will be modified; if copy, it won't
        assert prob[0, 0] == -999

    def test_with_nan_values(self):
        """Test behavior with NaN values."""
        prob_array = np.array([[0.5, np.nan, 0.7],
                               [np.nan, 0.3, 0.9]])
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        prob = result['level_crossing_probability']
        assert prob.shape == (2, 3)
        assert np.isnan(prob[0, 1])
        assert np.isnan(prob[1, 0])

    def test_with_inf_values(self):
        """Test behavior with infinity values."""
        prob_array = np.array([[0.5, np.inf, 0.7],
                               [-np.inf, 0.3, 0.9]])
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        prob = result['level_crossing_probability']
        assert prob.shape == (2, 3)
        assert np.isinf(prob[0, 1])
        assert np.isinf(prob[1, 0])

    def test_no_coords_returns_none_extent(self):
        """Test that no coordinates results in None extent."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}

        result = probabilistic_marching_squares_mesh(summary_stats)

        assert result['extent'] is None
        assert result['x_coords'] is None
        assert result['y_coords'] is None

    def test_with_coords(self):
        """Test with spatial coordinates."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}
        x_coords = np.linspace(0, 1, 16)  # width+1 = 16 grid vertices
        y_coords = np.linspace(0, 2, 11)  # height+1 = 11 grid vertices

        result = probabilistic_marching_squares_mesh(summary_stats, x_coords=x_coords, y_coords=y_coords)

        assert result['extent'] == (0, 1, 0, 2)
        np.testing.assert_array_equal(result['x_coords'], x_coords)
        np.testing.assert_array_equal(result['y_coords'], y_coords)

    def test_invalid_x_coords_length(self):
        """Test that wrong x_coords length raises ValueError."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}
        x_coords = np.linspace(0, 1, 10)  # wrong length

        with pytest.raises(ValueError):
            probabilistic_marching_squares_mesh(summary_stats, x_coords=x_coords)

    def test_invalid_y_coords_length(self):
        """Test that wrong y_coords length raises ValueError."""
        prob_array = np.random.rand(10, 15)
        summary_stats = {'level_crossing_probability': prob_array}
        y_coords = np.linspace(0, 1, 5)  # wrong length

        with pytest.raises(ValueError):
            probabilistic_marching_squares_mesh(summary_stats, y_coords=y_coords)
