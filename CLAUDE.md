# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UVisBox is a Python toolbox for visualizing scientific uncertainty data. It implements specialized statistical visualization methods including uncertainty tubes, contour boxplots, functional boxplots, curve boxplots, squid glyphs, uncertainty lobes, and probabilistic marching algorithms (squares, triangles, cubes, tetrahedra).

## Development Commands

### Setup and Dependencies
```bash
# Install all dependencies (including dev dependencies)
poetry install

# Install with optional parallel computing support
poetry install --with parallel
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run unit tests only
poetry run pytest tests/unit

# Run integration tests only
poetry run pytest tests/integration

# Run tests for a specific module
poetry run pytest tests/unit/contour_boxplot

# Run a single test file
poetry run pytest tests/unit/contour_boxplot/test_contour_boxplot_stats.py

# Run with verbose output
poetry run pytest -v
```

### Code Formatting and Linting
```bash
# Format code with black
poetry run black .

# Sort imports with isort
poetry run isort .

# Format and sort in one go
poetry run black . && poetry run isort .
```

### Running Examples
```bash
# Run any example script
poetry run python examples/<example_name>.py

# Common examples:
poetry run python examples/uncertainty_tube_example_vsup.py
poetry run python examples/contour_boxplot_example.py
poetry run python examples/functional_boxplot_example.py
```

## Architecture

### Package Structure

The codebase follows a two-tier architecture:

**Core Layer** (`uvisbox/Core/`): Fundamental algorithms and utilities used across modules
- `BandDepths/`: Core statistical depth calculations (contour, curve, functional, vector depths)
- `CellsCrossingProb/`: Cell crossing probability algorithms for probabilistic marching methods
- `Colors/`: Color interpolation, color trees, and VSUP colormap implementation
- `CommonInterface/`: Shared configurations like `BoxplotStyleConfig`
- `Interpolations/`: Core interpolation algorithms

**Modules Layer** (`uvisbox/Modules/`): Self-contained visualization methods that consume Core functionality
- `ContourBoxplot/`: 2D isocontour uncertainty visualization
- `CurveBoxplot/`: 1D curve ensemble uncertainty visualization
- `FunctionalBoxplot/`: Functional data boxplots
- `UncertaintyTube/`: Trajectory uncertainty visualization with VSUP colormap
- `UncertaintyLobes/`: Vector field uncertainty visualization
- `SquidGlyphs/`: Vector field uncertainty glyphs
- `ProbabilisticMarchingSquares/`: 2D level-crossing probability fields
- `ProbabilisticMarchingTriangles/`: Triangular mesh probabilistic extraction
- `ProbabilisticMarchingCubes/`: 3D level-crossing probability volumes
- `ProbabilisticMarchingTetrahedra/`: Tetrahedral mesh probabilistic extraction

### Module Internal Pattern

Each module in `Modules/` follows a consistent structure with four components:

1. **`<module>_stats.py`**: Computes summary statistics from ensemble data
   - Takes raw ensemble data as input
   - Returns dictionary with keys like `'median'`, `'percentiles'`, `'outliers'`, `'depths'`
   - Function name: `<module>_summary_statistics()`

2. **`<module>_mesh.py`**: Generates geometric representations from statistics
   - Takes summary statistics dictionary as input
   - Returns mesh data structures (coordinates, faces, values)
   - Function name: `<module>_mesh()`

3. **`<module>_vis.py`**: Renders visualizations from mesh data
   - Takes mesh data and matplotlib axes as input
   - Creates the actual plot
   - Function name: `visualize_<module>()`

4. **`<module>.py`**: High-level convenience function
   - Chains stats → mesh → vis pipeline
   - Provides simple API for end users
   - Function name: `<module>()`

This separation allows users to work at different levels of abstraction:
- Use the high-level function for quick visualizations
- Access intermediate steps (stats, mesh) for custom processing
- Compose with other tools by extracting just the statistics or mesh

### Test Organization

Tests mirror the package structure:
- `tests/unit/`: Unit tests organized by module matching `uvisbox/Modules/` structure
  - Each module has its own subdirectory with separate test files for stats, mesh, and vis components
  - Example: `tests/unit/contour_boxplot/test_contour_boxplot_stats.py`
- `tests/integration/`: Integration tests for complete workflows
  - Test the full pipeline from ensemble data to visualization
  - Example: `tests/integration/test_curve_boxplot_integration.py`

### Key Abstractions

**BoxplotStyleConfig**: Centralized configuration object for all boxplot-style visualizations. Controls percentiles to display, colors, line styles, and median/outlier rendering. Used consistently across ContourBoxplot, CurveBoxplot, FunctionalBoxplot, and UncertaintyLobes.

**Summary Statistics Interface**: All boxplot modules return dictionaries with standardized keys:
- `'median'`: The median curve/contour/function
- `'percentiles'`: Dict mapping percentile values to their curves
- `'depths'`: Band depth values for each ensemble member
- `'outliers'`: Ensemble members classified as outliers (optional)

**Band Depth Algorithms**: Core statistical concept throughout the package. Each algorithm (contour, curve, functional, vector) computes depth values indicating how "central" each ensemble member is. Higher depth = more central/typical.

## Important Development Notes

### Python Version
Requires Python >=3.11, <3.14

### Key Dependencies
- NumPy >=2.0: Core numerical operations
- SciPy: Scientific computing and distance calculations
- Matplotlib: All 2D visualizations
- PyVista: 3D visualizations (glyphs, volumes)
- scikit-learn: PCA and statistical algorithms
- scikit-image: Image processing for contour extraction

### Parallel Computing
The `parallel` dependency group (numba, joblib) provides optional performance improvements for band depth calculations. Most functions accept a `workers` parameter to control parallelization.

### Dataset Module
`uvisbox/Datasets/` contains data loaders and generators for examples and testing. This is separate from Core/Modules as it's not part of the core visualization functionality.
