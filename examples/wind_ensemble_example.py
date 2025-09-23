from uvisbox.Datasets import ens_uv
import matplotlib.pyplot as plt
import numpy as np
from uvisbox.Glyphs import uncertainty_squid_glyphs_2D, uncertainty_lobe_glyphs_2D
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
print(f" calculating uncertainty lobe glyphs for percentil=1.0 ...")
ax2 = uncertainty_lobe_glyphs_2D(positions, uv_data_flat, 1.0, 0.5, scale=scale, ax=ax2)
plt.title('Uncertainty Lobe Glyphs for Wind Ensemble Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("uncertainty_lobe_glyphs_wind_ensemble.pdf", bbox_inches='tight')
plt.show()

fig, ax3 = plt.subplots(1, 1, figsize=(fig_w, fig_h))
print(f" calculating uncertainty squid glyphs for percentil=1.0 ...")
ax3 = uncertainty_squid_glyphs_2D(positions, uv_data_flat, percentil1=1.0, scale=scale, ax=ax3)
plt.title('Uncertainty Squid Glyphs for Wind Ensemble Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("uncertainty_squid_glyphs_wind_ensemble.pdf", bbox_inches='tight')
plt.show()