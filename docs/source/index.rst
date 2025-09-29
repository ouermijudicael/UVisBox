.. UVisBox documentation master file, created by
   sphinx-quickstart on Fri Aug 29 09:20:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UVisBox (Still in Development)
===============================
UVisBox is an open-source Python based toolbox for visulazing uncertainty from scientific data. UVisBox provides a collection of methods for representing and exploring uncertainty in various scientific datasets.

Currently implemented methods include:

* **Uncertainty Tube:** For visualizing uncertainty in trajectory data. `arxiv <https://www.arxiv.org/abs/2508.13505>`__
* **Contour Boxplot:** For summarizing isocontours. `doi <https://doi.org/10.1109/TVCG.2013.143>`__
* **VSUP:** A colormap designed for uncertain data. `link <https://medium.com/@uwdata/value-suppressing-uncertainty-palettes-426130122ce9>`__

Work in progress:

* **Squid Glyph:** A new glyph for visualizing vector field uncertainty. `doi <https://doi.ieeecomputersociety.org/10.1109/UncertaintyVisualization63963.2024.00014>`__

Future plans include the implementation of:

* Curve band depth and curve boxplots
* Probabilistic marching cubes
* Other novel uncertainty visualization methods

The project is built using ``poetry`` for dependency management and relies on several scientific Python libraries:

* **numpy:** For numerical operations and data structures.
* **scipy:** For scientific computing.
* **matplotlib:** For plotting and visualization.
* **scikit-learn:** For machine learning algorithms.
* **scikit-image:** For image processing.

The codebase is organized into modules, each handling a specific aspect of the visualization process:

* ``BandDepths``: For calculating band depths.
* ``Colors``: For color mapping and interpolation.
* ``Datasets``: For loading and handling datasets.
* ``Glyphs``: For creating glyphs.
* ``Interpolations``: For interpolation methods.
* ``UncertaintyTube``: For generating and visualizing uncertainty tubes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   modules
