# Gemini Code Assistant Context

This document provides context for the Gemini code assistant to understand the UVisBox project.

## Project Overview

UVisBox is a Python toolbox for visualizing scientific uncertainty data. It provides a collection of methods for representing and exploring uncertainty in various scientific datasets. The project is built using `poetry` for dependency management and testing.

The main visualization modules are located in `uvisbox/Modules`. Each module follows a `Stats -> Mesh -> Vis` pipeline, where data is first processed to compute summary statistics, then a mesh is generated for visualization, and finally, the visualization is rendered.

Key implemented methods include:
- **Contour Boxplot**: For summarizing isocontours.
- **Uncertainty Tube**: For visualizing uncertainty in trajectory data.
- **Squid Glyph**: A glyph for visualizing vector field uncertainty.

## Building and Running

### Installation

This project uses `poetry` for dependency management. To install the required dependencies, run:
```bash
poetry install
```

### Running Examples

The `examples` directory contains several Python scripts that demonstrate how to use the `uvisbox` library. To run an example, use `poetry run`:
```bash
poetry run python examples/contour_boxplot_example.py
```

### Running Tests

The project uses `pytest` for testing. The tests are located in the `tests` directory, with unit tests in `tests/unit` and integration tests in `tests/integration`.

To run all tests, use:
```bash
poetry run pytest
```

To run tests for a specific module, you can specify the path:
```bash
poetry run pytest tests/unit/contour_boxplot/
```

## Development Conventions

- **Code Style**: The project uses `black` for code formatting and `isort` for import sorting.
- **Testing**: The project has a comprehensive test suite using `pytest`. New features should include corresponding tests.
- **Modularity**: The project is organized into modules, each responsible for a specific visualization technique. The `Core` directory contains common components shared across modules.
- **Pipeline**: The visualization modules follow a `Stats -> Mesh -> Vis` pipeline.
  - `*_stats.py`: Computes summary statistics from the input data.
  - `*_mesh.py`: Generates a mesh from the statistics.
  - `*_vis.py`: Renders the visualization from the mesh.
- **Styling**: A common `BoxplotStyleConfig` class is used to configure the appearance of boxplot-like visualizations. This class is located in `uvisbox/Core/CommonInterface/boxplot_style_config.py`.
