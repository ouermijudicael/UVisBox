
from uvisbox.Glyphs import uncertainty_lobe_glyphs_2D
import numpy as np
import matplotlib.pyplot as plt
A = 0.1
omega = np.pi
epsilon = 0.25
def double_gyre(x, y, t=0):
    a = epsilon * np.sin(omega * t)
    b = 1 - 2 * a
    f = a * x**2 + b * x
    df_dx = 2 * a * x + b
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df_dx
    return u, v


# Change domain to [0,2] x [0,1]
X, Y = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 1, 5))
U, V = double_gyre(X, Y)
# Create ensemble data by perturbing vector magnitude and direction with Gaussian noise
n_ensemble = 20  # number of ensemble members
rng = np.random.default_rng(seed=42)

# Flatten the grid for easier perturbation
X_flat = X.flatten()
Y_flat = Y.flatten()
U_flat = U.flatten()
V_flat = V.flatten()
n_points = X_flat.size

# Compute magnitude and angle
mag = np.sqrt(U_flat**2 + V_flat**2)
angle = np.arctan2(V_flat, U_flat)

# Standard deviations for noise (tune as needed)
mag_noise_std = 0.20 * mag.max()
angle_noise_std = np.deg2rad(10)  # 5 degree std


n_ensemble = 20
ensemble_vectors = np.zeros((n_points, n_ensemble, 2))

for i in range(n_points):
    for j in range(n_ensemble):
        # Perturb magnitude and anglels
        mag_perturbed = mag[i] + rng.normal(0, mag_noise_std)
        angle_perturbed = angle[i] + rng.normal(0, angle_noise_std)

        # Convert back to Cartesian coordinates
        ensemble_vectors[i, j] = [
            mag_perturbed * np.cos(angle_perturbed),
            mag_perturbed * np.sin(angle_perturbed)
        ]
positions = np.vstack((X_flat, Y_flat)).T
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))

# Plot original vector field with arrows
for i in range(n_points):
    for j in range(n_ensemble):
        u, v = ensemble_vectors[i, j]
        x, y = positions[i]
        ax1.arrow(x, y, u, v, head_width=0.03, head_length=0.06, fc='blue', ec='blue', alpha=0.1, length_includes_head=True)

ax1.set_title("Original Vector Field with Ensemble Members")
ax1.set_xlim(-0.25, 2.25)
ax1.set_ylim(-0.25, 1.25)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid()

# Plot uncertainty squid glyphs
ax2 = uncertainty_lobe_glyphs_2D(positions, ensemble_vectors, 1.0, 0.5, scale=0.4, ax=ax2)
ax2.set_title("Uncertainty Squid Glyphs for Double Gyre Flow")
ax2.set_xlim(-0.25, 2.25)
ax2.set_ylim(-0.25, 1.25)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid()

plt.tight_layout()
# plt.savefig("combined_vector_field_and_uncertainty_glyphs.png", dpi=300)
plt.show()