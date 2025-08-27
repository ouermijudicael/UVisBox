import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from .color_interpolator import interpolate_lab


def _get_color(x, x_min, x_max, y, y_min, y_max, c1, c2, c3):
    c = interpolate_lab(c1, c2, x, x_min, x_max)
    return interpolate_lab(c, c3, y, y_min, y_max)


default_color2 = np.array([255, 165, 0]) / 255
default_color1 = np.array([75, 0, 130]) / 255
default_coloru = np.array([185, 203, 201]) / 255


class ColorTree(mpl.colors.Colormap):
    def __init__(self, depth=4, color_a=default_color1, color_b=default_color2, cmap="viridis", color_u=default_coloru, v_min=0, v_max=1, u_min=0, u_max=1):
        self.depth = depth
        self.color_a = color_a
        self.color_b = color_b
        self.color_u = color_u
        self.v_min = v_min
        self.v_max = v_max
        self.u_min = u_min
        self.u_max = u_max
        self.invert_u = False
        self.cmap = cmap

        self.colormap = mpl.colormaps[self.cmap].resampled(
            1000) if self.cmap is not None else None
        n_nodes = (2**self.depth) - 1
        self.nodes = np.zeros([n_nodes, 3])
        self.create_tree()

    def create_tree(self):
        for d in np.arange(self.depth):
            for i in np.arange(2**d):
                if d > 0:
                    idx = 2**d + i - 1
                    uncertainty = 1 - d / (self.depth - 1)
                    value = i / (2**d - 1)

                    if self.cmap is not None:
                        colormap = mpl.colormaps[self.cmap].resampled(2**d)
                        color1 = list(colormap(value))[:-1]
                        color2 = self.color_u
                        self.nodes[idx] = interpolate_lab(
                            color1, color2, uncertainty, 0, 1)
                    else:
                        self.nodes[idx] = _get_color(
                            value,
                            0,
                            1,
                            uncertainty,
                            0,
                            1,
                            self.color_a,
                            self.color_b,
                            self.color_u,
                        )

                else:
                    self.nodes[0] = self.color_u

    def get_color(self, v, u, debugprint=False):
        v = np.clip(v, self.v_min, self.v_max)
        u = np.clip(u, self.u_min, self.u_max)

        ratio = (u - self.u_min) / (self.u_max - self.u_min)
        if self.invert_u:
            ratio = 1 - ratio
        depth = np.floor(ratio * (self.depth))
        if depth == self.depth:
            depth = depth - 1

        if depth == 0:
            return self.nodes[0]

        v_ratio = (v - self.v_min) / (self.v_max - self.v_min)
        if depth == self.depth - 1 and self.cmap is not None:
            return list(self.colormap(v_ratio))[:-1]

        v_idx = np.floor(v_ratio * (2**depth))
        if v_idx == 2**depth:
            v_idx = v_idx - 1
        idx = 2**depth + v_idx - 1

        if debugprint:
            print(
                f'v {v:.2f}, u {u:.2f}, depth {depth}, v_idx {v_idx}, idx {idx} color: {self.nodes[int(idx)]}')

        return self.nodes[int(idx)]

    def get_colors(self, image, show_uncertainty=True, discrete=False):
        v = image[..., 1].copy()
        u = image[..., 0].copy()
        v = np.clip(v, self.v_min, self.v_max)
        u = np.clip(u, self.u_min, self.u_max)

        v_ratio = (v - self.v_min) / (self.v_max - self.v_min)
        color = np.zeros(list(image.shape[:-1])+[3], dtype=np.float32)
        if not show_uncertainty:
            if not discrete:
                return self.colormap(v_ratio)[..., :3]
            else:
                u = np.zeros_like(u) + self.u_max

        u_ratio = (u - self.u_min) / (self.u_max - self.u_min)
        if self.invert_u:
            u_ratio = 1 - u_ratio
        depths = np.floor(u_ratio * (self.depth))
        depths[depths == self.depth] -= 1

        color[depths == 0] = self.nodes[0]

        v_idx = np.floor(v_ratio * (2**depths))
        v_idx[v_idx == 2**depths] -= 1
        idx = (2**depths + v_idx - 1).astype(int)

        if self.colormap is not None:
            if not discrete:
                not_leaf_indices = depths != self.depth-1
                color[not_leaf_indices] = self.nodes[idx[not_leaf_indices]]

                leaf_indices = depths == self.depth-1
                color[leaf_indices] = self.colormap(
                    v_ratio[leaf_indices])[:, :3]
            else:
                color[depths != 0] = self.nodes[idx[depths != 0]]
        else:
            color[depths != 0] = self.nodes[idx[depths != 0]]
        return color

    def __call__(self, value, show_uncertainty=True, discrete=False):
        return self.get_colors(value, show_uncertainty, discrete)
