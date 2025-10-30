from tkinter import W
from uvisbox.Datasets import ens_uv
import matplotlib.pyplot as plt
import numpy as np
from uvisbox.Modules import squid_glyph_2D, uncertainty_lobes
import time
# Load UV ensemble data
uv_data = ens_uv.load_data()
x_size, y_size, num_ens, n_dim = uv_data.shape
print(f"Loaded UV data with {num_ens} ensemble members, each of size {x_size}x{y_size} and {n_dim} dimensions.")

# create a uniform grid for plotting
x = np.linspace(0, 1, x_size)
y = np.linspace(0, 1, y_size)
X, Y = np.meshgrid(x, y)

# plot the mean vector field
mean_u = np.mean(uv_data[:, :, :, 0], axis=2)
mean_v = np.mean(uv_data[:, :, :, 1], axis=2)

fig_w = 16
fig_h = 12
plt.figure(figsize=(fig_w, fig_h))
# plt.quiver(X, Y, mean_u, mean_v, color='blue', alpha=0.6)
scale = 1.0e-3
for i_e in range(num_ens):
    uplot = uv_data[:, :, i_e, 0]* scale
    vplot = uv_data[:, :, i_e, 1]* scale
    plt.quiver(X, Y, uplot, vplot, alpha=1.0, scale=1.0)
# plt.quiver(X, Y, mean_u, mean_v, color='blue', alpha=0.8)

plt.title('Mean Wind Vector Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show(block=False)
plt.savefig("mean_wind_vector_field.pdf", bbox_inches='tight')

# plot uncertainty glyphs
# flatten the grid
X_flat = X.flatten()
Y_flat = Y.flatten()
positions = np.vstack((X_flat, Y_flat)).T   
# flatten the uv_data to shape (n_points, n_ensemble, 2)
uv_data_flat = uv_data.reshape(-1, num_ens, n_dim)

# Plot uncertainty lobe glyphs
fig, ax2 = plt.subplots(1, 1, figsize=(fig_w, fig_h))
print(f" calculating uncertainty lobe glyphs for percentile=95 ...")
start_time = time.time()
ax2 = uncertainty_lobes(positions, uv_data_flat, 95, 50, scale=scale, ax=ax2)
end_time = time.time()
print(f"✓ Uncertainty lobes completed in {end_time - start_time:.2f} seconds")
plt.title('Uncertainty Lobe Glyphs for Wind Ensemble Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("uncertainty_lobe_glyphs_wind_ensemble.pdf", bbox_inches='tight')
plt.show(block=False)

fig, ax3 = plt.subplots(1, 1, figsize=(fig_w, fig_h))
print(f" calculating uncertainty squid glyphs for percentil=95 ...")
start_time = time.time()
ax3 = squid_glyph_2D(positions, uv_data_flat, percentile=95, scale=scale, ax=ax3)
end_time = time.time()
print(f"✓ Squid glyphs completed in {end_time - start_time:.2f} seconds")
plt.title('Uncertainty Squid Glyphs for Wind Ensemble Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("uncertainty_squid_glyphs_wind_ensemble.pdf", bbox_inches='tight')
plt.show()