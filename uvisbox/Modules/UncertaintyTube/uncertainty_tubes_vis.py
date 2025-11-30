import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uvisbox.Core.Colors.colortree import ColorTree # Assuming this is available and needed

def visualize_uncertainty_tube(mesh_data, colormap="viridis", plotter=None):
    """
    Visualize 3D uncertainty tubes using either Matplotlib or PyVista.

    Parameters:
    -----------
    mesh_data (dict):
        Dictionary containing "vertices", "faces", and "uv_coords" from uncertainty_tube_mesh.
    colormap (str, optional):
        Colormap to use for rendering the tube. Defaults to "viridis".
    plotter (matplotlib.axes.Axes or pyvista.Plotter, optional):
        The plotting object to use. If None, a new Matplotlib figure/axis is created.
        If a Matplotlib Axes3D object, it plots on that.
        If a PyVista Plotter object, it adds the mesh to it.

    Returns:
    --------
    matplotlib.axes.Axes or pyvista.Plotter: The plotting object with the visualization.
    """
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    uv_coords = mesh_data["uv_coords"]

    if isinstance(plotter, plt.Axes):
        ax = plotter
        # Matplotlib specific rendering
        color_tree = ColorTree(invert_u=True, depth=4, cmap=colormap)
        
        # Each "face" in the input is a list of vertex indices. We need to get the actual vertex coordinates.
        triangle_vertices_for_mpl = vertices[faces] # (num_triangles, 3, 3)
        
        # Calculate face colors from uv_coords
        # Each face has 3 vertices, so take the mean UV of the face for coloring
        face_uv_coords = uv_coords[faces] # (num_triangles, 3, 2)
        face_colors = color_tree(face_uv_coords.mean(axis=1), discrete=True)

        tube_collection = Poly3DCollection(triangle_vertices_for_mpl, facecolors=face_colors)
        ax.add_collection3d(tube_collection)

        # Set labels and title if it's the main plotting call
        if ax.get_xlabel() == '': ax.set_xlabel('X-axis')
        if ax.get_ylabel() == '': ax.set_ylabel('Y-axis')
        if ax.get_zlabel() == '': ax.set_zlabel('Z-axis')
        if ax.get_title() == '': ax.set_title('3D Trajectories with Uncertainty')
        
        # Need to ensure the view is 3D if not already. This is usually handled by `fig.add_subplot(projection='3d')`
        if not hasattr(ax, 'get_proj') or ax.get_proj().shape != (4, 4): # Check if it's a 3D axis
            # This indicates the axis was not created as a 3D axis.
            # Matplotlib requires 'projection='3d' at subplot creation.
            # We can't change it here, so we will just warn.
            print("Warning: Provided Matplotlib Axes object is not 3D. Please create it with `fig.add_subplot(projection='3d')`.")
            # Adjust aspect ratio for potentially better view in non-3D mode, though it won't be true 3D.
            ax.set_box_aspect([np.ptp(c) for c in ax.get_xyz_limits()]) 
            ax.autoscale_view()

        return ax

    elif isinstance(plotter, pv.Plotter):
        # PyVista specific rendering
        n_faces = len(faces)
        # Add the number of points for each face (3 for triangles)
        faces_with_count = np.hstack((np.full((n_faces, 1), 3), faces)).flatten()
        tube_mesh = pv.PolyData(vertices, faces_with_count)
        tube_mesh.point_data["uv"] = uv_coords # Store UVs as point data

        # Apply colormap based on UVs
        # We can map the 'v' component of UV to scalar for coloring along the tube.
        tube_mesh.point_data["scalar_color"] = uv_coords[:, 1] # Use v-component for coloring

        plotter.add_mesh(tube_mesh, scalars="scalar_color", cmap=colormap, show_edges=False)
        return plotter

    else:
        # Default to new Matplotlib plot if no valid plotter provided
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Call self recursively with the newly created ax
        return visualize_uncertainty_tube(mesh_data, colormap=colormap, plotter=ax)