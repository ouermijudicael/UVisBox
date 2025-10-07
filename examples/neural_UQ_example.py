from uvisbox.Colors import ColorTree
import numpy as np
import matplotlib.pyplot as plt
from uvisbox.Datasets import darcy_flow_NN
from uvisbox.Colors import ColorTree

data = darcy_flow_NN.load_data()
print(f"data keys: {data.keys()}")
u_pred = data['u_pred']
u_test = data['u_test']
x_test = data['x_test']
# plt.figure(figsize=(9,6))
u_mean = np.mean(u_pred, axis=0)
u_std = np.std(u_pred, axis=0)
print(f"umean shape: {u_mean.shape}")
print(f"u_test shape: {u_test.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"f_pred shape: {u_pred.shape}")

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
# Reshape u_mean to match the grid for a heatmap
N = int(np.sqrt(x_test.shape[0]))
x1 = x_test[:, 0].reshape(N, N)
x2 = x_test[:, 1].reshape(N, N)
u_mean_grid = u_mean.reshape(N, N)
u_std_grid = u_std.reshape(N, N)

plt.pcolormesh(x1, x2, u_mean_grid, shading='auto', cmap='jet')
plt.title('u_mean')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.pcolormesh(x1, x2, u_test.reshape(N, N), shading='auto', cmap='jet')
plt.title('u_test')
plt.colorbar()

# plt.savefig('darcy_flow_u_mean_vs_u_test.png', dpi=300)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
u_error = np.abs(u_mean - u_test)
u_error_grid = u_error.reshape(N, N)

plt.pcolormesh(x1, x2, u_error_grid, shading='auto', cmap='jet')
plt.title('u_error')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.pcolormesh(x1, x2, u_std_grid, shading='auto', cmap='jet')
plt.title('u_std')
plt.colorbar()
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

image = np.stack([u_std_grid, u_error_grid], axis=-1)
# Initialize ColorTree with depth=4 and default settings
colormap = ColorTree(depth=10, cmap="jet")

# Generate colors for discrete mode (uses tree nodes)
colors = colormap.get_colors(image, discrete=False)

# Plot the discrete color map
ax.imshow(colors, origin='lower', extent=(0, 1, 0, 1))
ax.set_title("Continous Color Map")
ax.set_xlabel("Mean Error")
ax.set_ylabel("Uncertainty")



plt.show()
