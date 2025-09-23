from uvisbox.Colors import ColorTree
import numpy as np
import matplotlib.pyplot as plt
from uvisbox.Datasets import darcy_flow_NN

data = darcy_flow_NN.load_data()

u_pred = data['u_pred']
u_test = data['u_test']
x_test = data['x_test']
# plt.figure(figsize=(9,6))
u_mean = np.mean(u_pred, axis=0)
print(f"umean shape: {u_mean.shape}")
print(f"u_test shape: {u_test.shape}")
print(f"x_test shape: {x_test.shape}")

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
# Reshape u_mean to match the grid for a heatmap
N = int(np.sqrt(x_test.shape[0]))
x1 = x_test[:, 0].reshape(N, N)
x2 = x_test[:, 1].reshape(N, N)
u_mean_grid = u_mean.reshape(N, N)

plt.pcolormesh(x1, x2, u_mean_grid, shading='auto', cmap='jet')
plt.title('u_mean')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.pcolormesh(x1, x2, u_test.reshape(N, N), shading='auto', cmap='jet')
plt.title('u_test')
plt.colorbar()

plt.savefig('darcy_flow_u_mean_vs_u_test.png', dpi=300)

plt.show()
