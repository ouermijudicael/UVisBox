"""
Unit tests for uncertainty_lobes_vis.py

Tests the visualization rendering stage of the uncertainty lobes pipeline.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from uvisbox.Modules.UncertaintyLobes.uncertainty_lobes_vis import (
    visualize_uncertainty_lobes,
)


class TestVisualizeUncertaintyLobes:
    """Test suite for visualize_uncertainty_lobes function."""
    
    def test_basic_rendering(self):
        """Test basic rendering with minimal mesh data."""
        # Create minimal mesh data
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[0.707, 0.707]]),
                'lengths': np.array([1.0])
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=ax)
        
        # Should return an axes object
        assert result_ax is not None
        assert isinstance(result_ax, matplotlib.axes.Axes)
        
        plt.close(fig)
    
    def test_rendering_without_median(self):
        """Test rendering without median arrows."""
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[0.707, 0.707]]),
                'lengths': np.array([1.0])
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=False, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_rendering_with_inner_wedges(self):
        """Test rendering with both outer and inner wedges."""
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'inner_wedges': [
                {
                    'vertices': np.array([[0, 0], [0.5, 0], [0.354, 0.354], [0, 0.5]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[0.707, 0.707]]),
                'lengths': np.array([1.0])
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_rendering_multiple_wedges(self):
        """Test rendering with multiple wedges at different positions."""
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                },
                {
                    'vertices': np.array([[2, 0], [3, 0], [2.707, 0.707], [2, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 1
                }
            ],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([[0, 0], [2, 0]]),
                'directions': np.array([[0.707, 0.707], [0.707, 0.707]]),
                'lengths': np.array([1.0, 1.0])
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=ax)
        
        assert result_ax is not None
        plt.close(fig)
    
    def test_create_new_axes(self):
        """Test that new figure and axes are created when ax=None."""
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[0.707, 0.707]]),
                'lengths': np.array([1.0])
            }
        }
        
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=None)
        
        assert result_ax is not None
        assert isinstance(result_ax, matplotlib.axes.Axes)
        plt.close('all')
    
    def test_arrow_rendering(self):
        """Test that arrows are rendered with correct properties."""
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[1, 0]]),  # Right direction
                'lengths': np.array([2.0])
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=ax)
        
        # Check that annotations (arrows) were added
        # Arrows in matplotlib are added as annotations
        assert len(result_ax.texts) > 0 or len(result_ax.patches) > 0
        
        plt.close(fig)
    
    def test_empty_wedges(self):
        """Test rendering with no wedges."""
        mesh = {
            'wedges': [],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([]).reshape(0, 2),
                'directions': np.array([]).reshape(0, 2),
                'lengths': np.array([])
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=ax)
        
        # Should not crash, just return empty plot
        assert result_ax is not None
        plt.close(fig)
    
    def test_zero_length_arrows(self):
        """Test rendering with zero-length arrows."""
        mesh = {
            'wedges': [
                {
                    'vertices': np.array([[0, 0], [1, 0], [0.707, 0.707], [0, 1]]),
                    'triangles': np.array([[0, 1, 2], [0, 2, 3]]),
                    'position_idx': 0
                }
            ],
            'inner_wedges': None,
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[1, 0]]),
                'lengths': np.array([0.0])  # Zero length
            }
        }
        
        fig, ax = plt.subplots()
        result_ax = visualize_uncertainty_lobes(mesh, show_median=True, ax=ax)
        
        # Should not crash
        assert result_ax is not None
        plt.close(fig)

    def test_custom_styles_apply_to_wedges_and_median_arrow(self):
        """Custom colors and opacities are applied to every visual element."""
        mesh = {
            'wedges': [{'vertices': np.array([[0, 0], [1, 0], [0, 1]])}],
            'inner_wedges': [{'vertices': np.array([[0, 0], [0.5, 0], [0, 0.5]])}],
            'arrows': {
                'positions': np.array([[0, 0]]),
                'directions': np.array([[1, 0]]),
                'lengths': np.array([1.0]),
            },
        }
        fig, ax = plt.subplots()

        visualize_uncertainty_lobes(
            mesh,
            ax=ax,
            outer_lobe_edge_color='black',
            outer_lobe_face_color='orange',
            inner_lobe_edge_color='navy',
            inner_lobe_face_color='yellow',
            median_arrow_color='purple',
            outer_lobe_alpha=0.2,
            inner_lobe_alpha=0.7,
            median_arrow_alpha=0.4,
        )

        outer_wedge, inner_wedge = ax.patches[:2]
        np.testing.assert_allclose(outer_wedge.get_edgecolor(), mcolors.to_rgba('black', 0.2))
        np.testing.assert_allclose(outer_wedge.get_facecolor(), mcolors.to_rgba('orange', 0.2))
        np.testing.assert_allclose(inner_wedge.get_edgecolor(), mcolors.to_rgba('navy', 0.7))
        np.testing.assert_allclose(inner_wedge.get_facecolor(), mcolors.to_rgba('yellow', 0.7))

        arrow = ax.texts[0].arrow_patch
        np.testing.assert_allclose(arrow.get_facecolor(), mcolors.to_rgba('purple', 0.4))
        assert arrow.get_alpha() == 0.4
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
