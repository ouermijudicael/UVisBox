import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from .color_interpolator import interpolate_lab

default_coloru = np.array([185, 203, 201]) / 255


class ColorTree:
    """
    A colormap class for visualizing data with uncertainty using a tree-based approach,
    inspired by VSUP (Value-Suppressing Uncertainty Palettes). It generates colors based on value
    and uncertainty levels using a colormap and uncertainty color, supporting both continuous and discrete modes.

    Attributes:
        depth (int): The depth of the color tree.
        color_u (np.ndarray): RGB color for uncertainty.
        cmap (str): Matplotlib colormap name for leaf nodes (required).
        v_min, v_max, u_min, u_max (float): Ranges for value and uncertainty.
        invert_u (bool): If False (default), high uncertainty maps to root (uncertainty color); if True, low uncertainty maps to root.
        colormap (mpl.colors.Colormap): Resampled colormap.
        nodes (np.ndarray): Array of RGB colors for tree nodes.
    """
    def __init__(self, depth=4, cmap="viridis", color_u=default_coloru, v_min=0, v_max=1, u_min=0, u_max=1, invert_u=False):
        self._validate_inputs(depth, cmap, color_u, v_min, v_max, u_min, u_max)

        self.depth = depth
        self.color_u = color_u
        self.v_min = v_min
        self.v_max = v_max
        self.u_min = u_min
        self.u_max = u_max
        self.invert_u = invert_u
        self.cmap = cmap

        self.colormap = mpl.colormaps[self.cmap].resampled(1000) # a sufficiently large magical number
        n_nodes = (2**self.depth) - 1
        self.nodes = np.zeros([n_nodes, 3])
        self.create_tree()

    def _validate_inputs(self, depth, cmap, color_u, v_min, v_max, u_min, u_max):
        """
        Validates input parameters for the ColorTree class.
        """
        if depth < 1:
            raise ValueError("Depth must be at least 1.")
        if v_min >= v_max:
            raise ValueError("v_min must be less than v_max.")
        if u_min >= u_max:
            raise ValueError("u_min must be less than u_max.")
        if not isinstance(color_u, np.ndarray) or color_u.shape != (3,):
            raise ValueError("color_u must be a 3-element numpy array.")
        if cmap not in mpl.colormaps:
            raise ValueError(f"Invalid colormap name: {cmap}")

    def create_tree(self):
        """
        Builds the color tree by interpolating colors at each node based on depth and value using the colormap and uncertainty color.
        """
        for current_depth in np.arange(self.depth):
            for node_index in np.arange(2**current_depth):
                if current_depth > 0:
                    # Calculate node index in the tree array
                    tree_index = 2**current_depth + node_index - 1
                    # Uncertainty decreases with depth (higher depth = lower uncertainty)
                    uncertainty_level = 1 - current_depth / (self.depth - 1)
                    # Value position within the current depth level
                    value_position = node_index / (2**current_depth - 1)

                    # Always use the colormap, resampled for current depth
                    resampled_colormap = mpl.colormaps[self.cmap].resampled(2**current_depth)
                    base_color = list(resampled_colormap(value_position))[:-1]
                    uncertainty_color = self.color_u
                    self.nodes[tree_index] = interpolate_lab(
                        base_color, uncertainty_color, uncertainty_level, 0, 1)
                else:
                    # Root node (depth 0) is set to uncertainty color
                    self.nodes[0] = self.color_u

    def get_colors(self, image, show_uncertainty=True, discrete=False):
        """
        Generates colors for an image array based on uncertainty (first channel) and value (second channel).

        Args:
            image (np.ndarray): Input array with shape (..., 2), where last dim is [uncertainty, value].
            show_uncertainty (bool): If False, ignores uncertainty and uses colormap directly.
            discrete (bool): If True, treats as discrete levels using the tree.

        Returns:
            np.ndarray: RGB color array with shape (..., 3).
        """
        v = image[..., 1]
        u = image[..., 0]

        v_ratio, u_ratio = self._clip_and_normalize(v, u)
        color = np.zeros(list(image.shape[:-1]) + [3], dtype=np.float32)

        if not show_uncertainty:
            if not discrete:
                return self.colormap(v_ratio)[..., :3]
            else:
                u = np.full_like(u, self.u_max)
                _, u_ratio = self._clip_and_normalize(v, u)  # Recalculate with max u

        if discrete:
            depths, idx = self._get_depth_and_index_vectorized(v_ratio, u_ratio)
            # Handle depth 0 case
            color[depths == 0] = self.nodes[0]
            # Compute indices for non-zero depths
            mask = depths > 0
            if np.any(mask):
                color[mask] = self.nodes[idx[mask]]
        else:
            # For continuous (discrete=False), always interpolate colormap color with uncertainty color
            base_colors = self.colormap(v_ratio)[..., :3]
            color = np.array([
                interpolate_lab(base, self.color_u, uncert, 1, 0)
                for base, uncert in zip(base_colors.reshape(-1, 3), u_ratio.flatten())
            ]).reshape(color.shape)

        return color

    def __call__(self, value, show_uncertainty=True, discrete=False):
        """
        Alias for get_colors to support callable interface.
        """
        return self.get_colors(value, show_uncertainty, discrete)

    def _clip_and_normalize(self, value, uncertainty):
        """
        Clips and normalizes value and uncertainty to [0, 1] ratios.
        Works for both scalars and arrays.
        """
        value_clipped = np.clip(value, self.v_min, self.v_max)
        uncertainty_clipped = np.clip(uncertainty, self.u_min, self.u_max)
        value_ratio = (value_clipped - self.v_min) / (self.v_max - self.v_min)
        uncertainty_ratio = (uncertainty_clipped - self.u_min) / (self.u_max - self.u_min)
        if not self.invert_u:
            uncertainty_ratio = 1 - uncertainty_ratio
        return value_ratio, uncertainty_ratio

    def _get_depth_and_index(self, value_ratio, uncertainty_ratio):
        """
        Computes depth and node index based on ratios.
        """
        depth = int(np.floor(uncertainty_ratio * self.depth))
        if depth == self.depth:
            depth -= 1
        if depth == 0:
            return 0, 0
        value_index = int(np.floor(value_ratio * (2 ** depth)))
        if value_index == 2 ** depth:
            value_index -= 1
        node_index = (2 ** depth) + value_index - 1
        return depth, node_index

    def _get_depth_and_index_vectorized(self, value_ratio, uncertainty_ratio):
        """
        Vectorized version: Computes depth and node index arrays based on ratios.
        """
        depth = np.floor(uncertainty_ratio * self.depth).astype(int)
        depth = np.clip(depth, 0, self.depth - 1)
        value_index = np.floor(value_ratio * (2 ** depth)).astype(int)
        value_index = np.clip(value_index, 0, (2 ** depth) - 1)
        node_index = (2 ** depth) + value_index - 1
        return depth, node_index

    def get_color(self, value, uncertainty, debug_print=False):
        """
        Retrieve the color for a given value and uncertainty based on the color tree.

        Args:
            value (float): The primary value (e.g., data point value).
            uncertainty (float): The uncertainty level.
            debug_print (bool): If True, print debug information.

        Returns:
            np.ndarray: RGB color array.
        """
        value_ratio, uncertainty_ratio = self._clip_and_normalize(value, uncertainty)
        depth, node_index = self._get_depth_and_index(value_ratio, uncertainty_ratio)

        if depth == 0:
            return self.nodes[0]

        if depth == self.depth - 1 and self.cmap is not None:
            return self.colormap(value_ratio)[:3]

        if debug_print:
            print(f"value {value:.2f}, uncertainty {uncertainty:.2f}, depth {depth}, value_index {int(np.floor(value_ratio * (2 ** depth)))}, node_index {node_index}, color: {self.nodes[node_index]}")

        return self.nodes[node_index]

    def update_colors(self, color_u=None):
        """
        Updates the uncertainty color and rebuilds the tree if it changes.
        """
        if color_u is not None:
            self.color_u = color_u
        self.create_tree()

    def update_ranges(self, v_min=None, v_max=None, u_min=None, u_max=None):
        """
        Updates the value and uncertainty ranges.
        """
        if v_min is not None:
            self.v_min = v_min
        if v_max is not None:
            self.v_max = v_max
        if u_min is not None:
            self.u_min = u_min
        if u_max is not None:
            self.u_max = u_max

    def set_invert_u(self, invert):
        """
        Sets whether to invert the uncertainty mapping.
        """
        self.invert_u = invert
