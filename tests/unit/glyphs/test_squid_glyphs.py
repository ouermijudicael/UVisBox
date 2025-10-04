
# from uvisbox.Glyphs.Vis.uncertainty_lobes import uncertainty_lobe_glyphs_2D
# from uvisbox.Glyphs.Stat.squid_glyphs import uncertainty_squid_glyphs_2D
from uvisbox.Modules.SquidGlyphs2D.plot import plot as uncertainty_squid_glyphs_2D
import numpy as np
import matplotlib.pyplot as plt
num_points = 15

# create 5x3 grid points
x = np.linspace(0, 4, 5)
y = np.linspace(0, 2, 3)
X, Y = np.meshgrid(x, y)
grid_points = np.vstack((X.flatten(), Y.flatten())).T

np.random.seed(72)
ensemble_size = 30
ensemble_vectors = np.zeros((num_points, ensemble_size, 2))
count = 0
for i in range(num_points):
    # add random random vectors with Gaussian noise in the first quadrant 0 to pi/2
    if count == 0:
        angles = np.random.uniform(0, np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    elif count == 1:
        angles = np.random.uniform(np.pi / 2, np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    elif count == 2:
        angles = np.random.uniform(-np.pi, - np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    elif count == 3:
        angles = np.random.uniform(-np.pi / 2, 0, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in the first and second quadrant
    elif count == 4:
        angles = np.random.uniform(0, np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in the second and third quadrant
    elif count == 5:
        angles = np.random.uniform(np.pi / 2, 3*np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in the third and fourth quadrant
    elif count == 6:
        angles = np.random.uniform(np.pi, 2*np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in the fourth and first quadrant
    elif count == 7:
        angles = np.random.uniform(-np.pi / 2, np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in first, second, and third quadrants
    elif count == 8:
        angles = np.random.uniform(0, -np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in second, third, and fourth quadrants
    elif count == 9:
        angles = np.random.uniform(np.pi / 2, -np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in third, fourth, and first quadrants
    elif count == 10:
        angles = np.random.uniform(-np.pi, np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place random vectors in fourth, first, and second quadrants
    elif count == 11:
        angles = np.random.uniform(-np.pi / 2, np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place vectors first and third quadrants
    elif count == 12:
        angles = np.random.uniform(-np.pi / 2, np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place vectors in second and fourth quadrants
    elif count == 13:
        angles = np.random.uniform(np.pi / 2, -np.pi / 2, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    # place vectors in all four quadrants
    elif count == 14:
        angles = np.random.uniform(-np.pi, np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    else:
        angles = np.random.uniform(0, 2 * np.pi, ensemble_size)
        magnitudes = np.random.uniform(0.5, 1.5, ensemble_size)
        ensemble_vectors[i, :, 0] = magnitudes * np.cos(angles)
        ensemble_vectors[i, :, 1] = magnitudes * np.sin(angles)
    count += 1

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 30))

# First subplot: Original Vector Field with Ensemble Members
for i in range(num_points):
    for j in range(ensemble_size):
        u, v = ensemble_vectors[i, j]
        x, y = grid_points[i]
        ax1.arrow(x, y, u * 0.5, v * 0.5, head_width=0.03, head_length=0.06, fc='blue', ec='blue', alpha=1, length_includes_head=True)
ax1.set_xlim(-1, 5)
ax1.set_ylim(-1, 3)
ax1.set_title("Original Vector Field with Ensemble Members")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid()

# # Second subplot: Uncertainty Lobe Glyphs
# ax2 = uncertainty_lobe_glyphs_2D(grid_points, ensemble_vectors, 1.0, 0.5, scale=0.4, ax=ax2)
# ax2.set_title("Uncertainty Lobe Glyphs")
# ax2.set_xlim(-1, 5)
# ax2.set_ylim(-1, 3)
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.grid()

# # Third subplot: Uncertainty Lobe Glyphs
# ax3 = uncertainty_lobe_glyphs_2D(grid_points, ensemble_vectors, 0.5, scale=0.4, ax=ax3)
# ax3.set_title("Uncertainty Lobe Glyphs")
# ax3.set_xlim(-1, 5)
# ax3.set_ylim(-1, 3)
# ax3.set_xlabel("X")
# ax3.set_ylabel("Y")
# ax3.grid()

# plt.tight_layout()
# plt.savefig("test_squid_glyphs.png", dpi=300)

fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(8, 18))
# First subplot: Original Vector Field with Ensemble Members
for i in range(num_points):
    for j in range(ensemble_size):
        u, v = ensemble_vectors[i, j]
        x, y = grid_points[i]
        ax4.arrow(x, y, u * 0.5, v * 0.5, head_width=0.03, head_length=0.06, fc='blue', ec='blue', alpha=1, length_includes_head=True)
ax4.set_xlim(-1, 5)
ax4.set_ylim(-1, 3)
ax4.set_title("Original Vector Field with Ensemble Members")
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.grid()

# Second subplot: Uncertainty Lobe Glyphs
ax5 = uncertainty_squid_glyphs_2D(grid_points, ensemble_vectors, 1.0, scale=0.4, ax=ax5)
ax5.set_title("Uncertainty Lobe Glyphs")
ax5.set_xlim(-1, 5)
ax5.set_ylim(-1, 3)
ax5.set_xlabel("X")
ax5.set_ylabel("Y")
ax5.grid()
plt.tight_layout()
plt.savefig("test_squid_glyphs_debug.png", dpi=300)
plt.show()