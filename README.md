# UVisBox

**Uncertainty Visualization Toolbox** — a Python library for visualizing scientific uncertainty data, developed at the [SCI Institute](https://www.sci.utah.edu/), University of Utah.

## Implemented Methods

*   **Uncertainty Tube** — Uncertainty in 3D trajectory data. [arxiv](https://www.arxiv.org/abs/2508.13505)
*   **Contour Boxplot** — Summarizing isocontours in ensemble scalar fields. [doi](https://doi.org/10.1109/TVCG.2013.143)
*   **Functional Boxplot** — Band envelopes for functional data.
*   **Curve Boxplot** — Band envelopes for curve data.
*   **Squid Glyphs (2D & 3D)** — Vector field uncertainty visualization. [doi](https://doi.ieeecomputersociety.org/10.1109/UncertaintyVisualization63963.2024.00014)
*   **Cone Glyphs** — 3D cone glyph visualization for vector field uncertainty.
*   **Uncertainty Lobes** — Angular wedge representation of vector uncertainty.
*   **Probabilistic Marching Squares / Cubes / Triangles / Tetrahedra** — Isosurface extraction with crossing probabilities.
*   **VSUP** — A colormap designed for uncertain data. [link](https://medium.com/@uwdata/value-suppressing-uncertainty-palettes-426130122ce9)

## Installation

Requires Python >= 3.11, < 3.14. Uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

Optional parallel computing dependencies (numba, joblib):

```bash
poetry install --with parallel
```

### Dependencies

Core: numpy, scipy, matplotlib, scikit-learn, scikit-image, pyvista, basemap, xarray, dsapi

## Quick Start

```python
from uvisbox import uncertainty_tubes, contour_boxplot, functional_boxplot
from uvisbox import squid_glyph_2D, squid_glyph_3D, cone_glyph
from uvisbox import curve_boxplot, uncertainty_lobes
from uvisbox import probabilistic_marching_squares, probabilistic_marching_cubes
```

See the `examples/` directory for usage demos, including:

| Example | Description |
|---------|-------------|
| `uncertainty_tube_example.py` | 3D trajectory uncertainty tubes |
| `contour_boxplot_example.py` | Ensemble scalar field contours |
| `functional_boxplot_example.py` | Functional data boxplots |
| `curve_boxplot_example.py` | Curve data boxplots |
| `squid_glyphs_2D_example.py` | 2D vector field glyphs |
| `squid_glyphs_3D_example.py` | 3D vector field glyphs |
| `uncertainty_lobes_2D_example.py` | Angular uncertainty representation |
| `probabilistic_marching_squares_example.py` | Probabilistic contour extraction |
| `custom_boxplot_style_example.py` | Custom styling with `BoxplotStyleConfig` |
| `wind_ensemble_example.py` | Real climate data visualization |
| `neural_UQ_example.py` | Neural network uncertainty quantification |
| `vsup_example.py` | VSUP colormap demo |

Run any example with:

```bash
poetry run python examples/uncertainty_tube_example.py
```

## Project Structure

```
uvisbox/
├── Core/
│   ├── BandDepths/        # Band depth calculations (functional, contour, curve, vector)
│   ├── CellsCrossingProb/ # Cell crossing probability algorithms
│   ├── Colors/            # LAB color interpolation and ColorTree
│   ├── CommonInterface/   # Shared BoxplotStyleConfig
│   └── Interpolations/    # Linear interpolation utilities
├── Modules/
│   ├── UncertaintyTube/
│   ├── ContourBoxplot/
│   ├── FunctionalBoxplot/
│   ├── CurveBoxplot/
│   ├── SquidGlyphs/
│   ├── ConeGlyphs/
│   ├── UncertaintyLobes/
│   ├── ProbabilisticMarchingSquares/
│   ├── ProbabilisticMarchingCubes/
│   ├── ProbabilisticMarchingTriangles/
│   └── ProbabilisticMarchingTetrahedra/
└── Datasets/              # Dataset loaders (double gyre, hurricane, ERA5, etc.)
```

Each visualization module follows a 3-stage pipeline: **Stats → Mesh → Visualization**.

## Testing

```bash
poetry run pytest
```

## License

MIT
