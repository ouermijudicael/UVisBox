import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..Colors.colortree import ColorTree

def plot_uncertainty_path_3d_from_mesh(vertices, faces, mean_trajectories, uv_coords):
    """
    Plot 3D uncertainty tubes from pre-generated mesh data.

    Args:
        vertices (np.ndarray): Global vertex array with shape (total_vertices, 3)
        faces (np.ndarray): Triangle face indices with shape (n_seeds, triangles_per_seed, 3)
        mean_trajectories (np.ndarray): Mean trajectory positions with shape (n_steps+1, n_seeds, 3)
    """

    color_tree = ColorTree()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    n_steps, n_seeds = mean_trajectories.shape[:2]

    # Plot mean trajectory lines
    for i in range(n_seeds):
        for j in range(1, n_steps):
            segment = mean_trajectories[j-1:j+1, i, :]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                    color='black', alpha=1.0, linewidth=2)

    # Plot uncertainty tubes using triangular faces
    all_tube_faces = []
    all_face_color = []
    # Process faces for each seed
    for i in range(n_seeds):
        seed_faces = faces[i]  # Shape: (triangles_per_seed, 3)
        # Convert face indices to vertex coordinates
        triangle_vertices = vertices[seed_faces]  # Shape: (triangles_per_seed, 3, 3)
        color_values = uv_coords[seed_faces]
        face_colors = color_tree(color_values.mean(axis=1), discrete=True)
        all_tube_faces.extend(triangle_vertices)
        all_face_color.extend(face_colors)

    all_tube_faces = np.array(all_tube_faces)
    all_face_color = np.array(all_face_color)
    tube_collection = Poly3DCollection(all_tube_faces,
                                        facecolors=all_face_color)
    ax.add_collection3d(tube_collection)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Trajectories of Points with Uncertainty')
    plt.grid()
    plt.show()
