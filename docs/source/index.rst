.. UVisBox documentation master file, created by
   sphinx-quickstart on Fri Aug 29 09:20:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UVisBox (Still in Development)
===============================
UVisBox is an open-source Python based toolbox for visualizing uncertainty from scientific data. UVisBox provides a collection of methods for representing and exploring uncertainty in various scientific datasets.

Currently implemented methods include:

* **Uncertainty Tube:** For visualizing uncertainty in trajectory data. `arxiv <https://www.arxiv.org/abs/2508.13505>`__
* **Contour Boxplot:** For summarizing isocontours. `doi <https://doi.org/10.1109/TVCG.2013.143>`__
* **Curve Boxplot:** For summarizing 2D curves. `doi <https://doi.org/10.1109/TVCG.2014.2346455>`__
* **Functional Band Depth:** For summarizing functional data. `doi <https://doi.org/10.1002/sta4.8>`__
* **VSUP:** A colormap designed for uncertain data. `link <https://medium.com/@uwdata/value-suppressing-uncertainty-palettes-426130122ce9>`__
* **Squid Glyph:** A new glyph for visualizing vector field uncertainty. `doi <https://doi.ieeecomputersociety.org/10.1109/UncertaintyVisualization63963.2024.00014>`__
* **Uncertainty Lobes:** A glyph for visualizing uncertainty in 2D vector fields. `doi <https://doi.org/10.1109/VAST.2015.7347634>`__
* **Probabilistic Marching Squares:** For visualizing uncertainty in 2D scalar fields. `doi <https://doi.org/10.1109/TVCG.2010.247>`__
* **Probabilistic Marching Triangles:** For visualizing uncertainty in 2D scalar fields on triangulated meshes. `doi <https://doi.org/10.1615/Int.J.UncertaintyQuantification.2012003958>`__
* **Probabilistic Marching cubes:** For visualizing uncertainty in 3D scalar fields. `doi <https://doi.org/10.1111/j.1467-8659.2011.01942.x>`__
* **Probabilistic Marching Tetrahedra:** For visualizing uncertainty in 3D scalar fields on tetrahedral meshes.
* **Uncertainty Tube:** For visualizing uncertainty in trajectory data. `arxiv <https://www.arxiv.org/abs/2508.13505>`__

Future plans include the implementation of:

* Other novel uncertainty visualization methods

The project is built using ``poetry`` for dependency management and relies on several scientific Python libraries:

* **numpy:** For numerical operations and data structures.
* **scipy:** For scientific computing.
* **matplotlib:** For plotting and visualization.
* **scikit-learn:** For machine learning algorithms.
* **scikit-image:** For image processing.
* **pyvista:** For 3D visualization and mesh handling.
* **basemap:** For geographical plotting. Just required for examples not needed for core functionality.

The codebase is organized into modules, each handling a specific uncertainty visualization technique.:

Project Folder Structure:

.. code-block:: markdown

   UVisBox/
   ├── Core/
   │   ├── Band_Depths/           # For Calculating Band Depths
   │   ├── Cells_Crossing_Prob/   # For Computing Cell Crossing Probabilities
   │   └── Interpolation/         # For Interpolation Methods
   |
   ├── Modules/
   │   ├── Contour_Boxplot/               # For Summarizing Isocontours
   │   ├── Curve_Boxplot/                 # For Summarizing Curves
   │   ├── Functional_Boxplot/            # For Summarizing Curves
   │   ├── Squid_Glyphs/                  # 2D and 3D Vector Field Uncertainty
   │   ├── Uncertainty_Lobes/             # 2D Vector Uncertainty Lobes
   │   ├── Probabilistic_Marching_Cubes/          # Uncertainty in 3D Scalar Fields
   │   ├── Probabilistic_Marching_Tetrahedra/     # Uncertainty in 3D Scalar Fields on Tetrahedral Meshes
   │   ├── Probabilistic_Marching_Squares/        # Uncertainty in 2D Scalar Fields
   │   └── Probabilistic_Marching_Triangles/      # Uncertainty in 2D Scalar Fields on Triangulated Meshes

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   modules
