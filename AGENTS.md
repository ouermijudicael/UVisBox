# UVisBox Agent Guide

This document gives coding agents repository-specific guidance for working on UVisBox. Prefer evidence from the current code and configuration over assumptions. If documentation, dependency files, and implementation disagree, report the discrepancy rather than silently choosing one.

## Project Overview

UVisBox is a Python library for scientific uncertainty visualization. It supports Python 3.11 through 3.13 and uses Poetry as its primary dependency and packaging workflow.

The repository is organized as follows:

```text
uvisbox/
  Core/       Shared statistics, colors, interpolation, configuration, and map helpers
  Modules/    Individual uncertainty-visualization methods
  Datasets/   Dataset loaders, generators, and bundled data
examples/     Runnable examples, notebooks, and performance benchmarks
tests/
  unit/       Unit and per-stage pipeline tests
  integration/
docs/         Sphinx documentation
```

UVisBox is a library rather than a command-line application. There is no declared console-script entry point or `uvisbox/__main__.py`.

## Installation

Use Poetry unless the task explicitly requires another environment manager:

```bash
poetry install
```

Install optional parallel-computing dependencies with:

```bash
poetry install --with parallel
```

The repository also contains `requirements.txt` and `environment.yml`, but they are not exact equivalents of the Poetry configuration:

- `requirements.txt` combines runtime, development, parallel, and Jupyter dependencies.
- `environment.yml` creates the `uvisbox` Conda environment, includes documentation and Jupyter dependencies, and installs the local package in editable mode.
- Cartopy is required by `pyproject.toml` and `environment.yml`, but remains commented out in `requirements.txt`.

Do not reconcile or edit these dependency definitions unless the task specifically calls for it.

## Architecture and Public APIs

Visualization methods normally use a three-stage pipeline:

```text
summary statistics -> mesh/geometry construction -> visualization
```

A typical method package has this structure:

```text
uvisbox/Modules/MethodName/
  __init__.py
  method_name.py
  method_name_stats.py
  method_name_mesh.py
  method_name_vis.py
```

Responsibilities should remain separated:

- `*_stats.py`: validate and summarize input ensembles; perform statistical calculations.
- `*_mesh.py`: convert summary statistics into renderable geometry or plotting data.
- `*_vis.py`: render existing mesh data with Matplotlib or PyVista.
- The unsuffixed module: expose a convenient high-level function that coordinates all three stages and returns an Axes or Plotter where appropriate.
- `__init__.py`: explicitly export the supported high- and low-level APIs, preferably through `__all__`.

Public method packages are aggregated in `uvisbox/Modules/__init__.py`, then re-exported from `uvisbox/__init__.py`. When adding a public method, update the module aggregator explicitly. Do not assume package discovery alone makes it available from `uvisbox`.

`ConeGlyphs` is currently not imported by `uvisbox/Modules/__init__.py`, despite being implemented and documented. Treat this as an existing inconsistency, not as a pattern to follow.

## Shared Core Functionality

Before duplicating an algorithm, check these packages:

- `uvisbox/Core/BandDepths/`: contour, curve, functional, and vector depth algorithms.
- `uvisbox/Core/CellsCrossingProb/`: cell-crossing probability calculations.
- `uvisbox/Core/Colors/`: LAB interpolation and `ColorTree`.
- `uvisbox/Core/CommonInterface/`: shared `BoxplotStyleConfig`.
- `uvisbox/Core/Interpolations/`: interpolation helpers.
- `uvisbox/Core/MapSetup/`: Cartopy-based map setup.

The `CellsCrossingProb/__init__.py` description appears stale relative to its implementation. Inspect the actual source before concluding that functionality is absent.

## Testing

Run the complete test suite from the repository root with:

```bash
poetry run pytest
```

Useful narrower commands include:

```bash
poetry run pytest tests/unit
poetry run pytest tests/integration
poetry run pytest tests/unit/<method_name>
poetry run pytest -k <expression>
```

Tests use pytest conventions:

- Files are named `test_*.py`.
- Test functions are named `test_*` and may be grouped in `Test*` classes.
- Test individual statistics, mesh, and visualization stages separately where practical.
- Add an integration test for the complete high-level pipeline.
- Assert array shapes, output keys and values, error handling, and returned plotting objects as applicable.
- Use deterministic seeds for generated numerical data.
- Use a non-interactive Matplotlib backend for tests that render figures.

There is no repository-level pytest configuration, coverage configuration, tox/nox setup, or CI workflow that runs tests. Do not claim a test passes unless it was actually run. Be aware that some rendering tests may create figures or artifacts.

## Examples and Experiments

Run an example from the repository root with Poetry:

```bash
poetry run python examples/uncertainty_tube_example.py
```

Benchmarks are also ordinary Python scripts, for example:

```bash
poetry run python examples/contour_boxplot_benchmark.py
poetry run python examples/curve_boxplot_benchmark.py
poetry run python examples/curve_banddepth_benchmark.py
poetry run python examples/uncertainty_tube_benchmark.py
```

Some benchmark scripts write PNG results. Confirm that generated output is acceptable before running them during a read-only or diagnostic task. Many example scripts execute code at import time, so do not import them merely to inspect their APIs.

The repository includes Jupyter notebooks, but it does not document a canonical notebook launch command. State this uncertainty if notebook execution matters.

## Documentation

Build the documentation from the repository root using the Conda environment:

```bash
conda run -n uvisbox python docs/build_docs.py
```

This command regenerates canonical API stubs under `docs/source/_generated/api/`
and performs a strict Sphinx build. Files under `docs/source/_generated/` are
ignored by Git and must not be edited manually. Narrative pages such as
`docs/source/examples.rst` remain hand-written. The examples page links to
runnable scripts and checked-in result images without importing or executing
the scripts during the Sphinx build.

## Coding Conventions

Follow the nearest well-maintained method package, especially `ContourBoxplot`, `CurveBoxplot`, `FunctionalBoxplot`, or `UncertaintyLobes`.

- Use four-space indentation.
- Use `snake_case` for functions, variables, and Python filenames.
- Existing visualization package directories use `PascalCase`.
- Keep the statistics, mesh, visualization, and orchestration stages separate.
- Document expected NumPy array dimensions and return structures.
- Prefer explicit validation with clear `ValueError`, `TypeError`, or `KeyError` behavior.
- Allow callers to inject Matplotlib Axes or PyVista Plotters when consistent with neighboring modules.
- Return the plotting object from high-level visualization functions.
- Use explicit package exports and `__all__` for new modules.
- Prefer package-relative imports inside a visualization module and existing public Core imports for shared functionality.

Black and isort are development dependencies, but the repository has no checked-in configuration or documented formatting command, and existing formatting is inconsistent. Keep changes stylistically consistent with adjacent code; avoid unrelated bulk reformatting. Docstrings currently mix NumPy and Google conventions, so match the surrounding module.

## Adding an Uncertainty Visualization Method

For a new method:

1. Create `uvisbox/Modules/MethodName/` with the standard five files.
2. Implement and test statistics, mesh, and visualization stages independently.
3. Add a high-level wrapper that composes the stages.
4. Define public exports in the method package's `__init__.py`.
5. Import the new method package from `uvisbox/Modules/__init__.py`.
6. Reuse or extend Core algorithms where appropriate.
7. Add unit tests under `tests/unit/<method_name>/`.
8. Add an end-to-end test under `tests/integration/` when practical.
9. Add a runnable example under `examples/`.
10. Update `README.md` and the Sphinx pages under `docs/source/`.

Sphinx AutoAPI pages are regenerated automatically by `docs/build_docs.py`
without importing UVisBox. They are temporary build inputs rather than tracked documentation. Follow
`docs/dev-doc-generation-readme.md` when changing hand-written documentation.

## Known Repository Inconsistencies

Keep these in mind and report them when relevant:

- Package version values differ between `pyproject.toml` and `uvisbox/__init__.py`; Sphinx obtains the installed package version dynamically.
- The Sphinx installation page refers to a nonexistent `examples/uncertainty_tube.py`; the actual filename is `examples/uncertainty_tube_example.py`.
- Test coverage varies substantially among visualization methods.
- `ConeGlyphs` is not exported by the main Modules aggregator.
- `requirements.txt` still differs from `pyproject.toml` and `environment.yml`, particularly around Cartopy.
- Documentation and some package docstrings contain stale structure descriptions.

Do not fix these opportunistically unless they are part of the requested task.

## Change Discipline

- Preserve unrelated user changes and avoid broad cleanup.
- Inspect neighboring implementations and tests before introducing a new abstraction.
- Keep generated images, benchmark results, caches, and documentation builds out of changes unless requested.
- Update tests and user-facing documentation whenever a public API changes.
- Report unverified behavior and repository inconsistencies explicitly rather than guessing.
