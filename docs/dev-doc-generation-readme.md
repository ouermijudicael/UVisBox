# Documentation Generation Guide

This document describes how to generate the API documentation for UVisBox using Sphinx.

## Prerequisites

Make sure you have Sphinx installed in your environment:

```bash
pip install sphinx
# or
conda install sphinx
```

## Generating Documentation

The UVisBox documentation is generated using `sphinx-apidoc` to automatically create API documentation from the source code. Run the following commands from the `docs/` directory:

### 1. Generate Module Documentation

```bash
sphinx-apidoc -o source/ ../uvisbox/Modules/
```

This command generates documentation for all visualization modules including:
- ContourBoxplot
- CurveBoxplot
- FunctionalBoxplot
- SquidGlyphs
- UncertaintyLobes
- UncertaintyTube
- ProbabilisticMarchingSquares
- ProbabilisticMarchingTriangles
- ProbabilisticMarchingCubes
- ProbabilisticMarchingTetrahedra

### 2. Generate Core Documentation

```bash
sphinx-apidoc -o source/ ../uvisbox/Core
```

This command generates documentation for core functionality including:
- BandDepths
- CellsCrossingProb
- Colors
- Interpolations

### 3. Generate Datasets Documentation

```bash
sphinx-apidoc -o source/ ../uvisbox/Datasets
```

This command generates documentation for the datasets module including:
- Sample scientific datasets
- Data loading utilities
- Preprocessing functions

## Complete Documentation Generation Workflow

From the `docs/` directory, run the following commands in sequence:

```bash
# Generate API documentation from source code
sphinx-apidoc -o source/ ../uvisbox/Modules/
sphinx-apidoc -o source/ ../uvisbox/Core
sphinx-apidoc -o source/ ../uvisbox/Datasets

# Build HTML documentation
sphinx-build -b html source/ build/html/
```

## Complete Documentation Build Process

After running the `sphinx-apidoc` commands above, you can build the complete documentation:

### Build HTML Documentation

```bash
sphinx-build -b html source/ build/html/
```

Alternative methods (if Makefile is available):

```bash
make html
```

or on Windows:

```bash
make.bat html
```

### View Documentation

The generated HTML documentation will be available in the `build/html/` directory. Open `build/html/index.html` in your web browser to view the documentation.

## File Structure

After running the generation commands, your documentation structure should look like:

```
docs/
├── source/
│   ├── conf.py                    # Sphinx configuration
│   ├── index.rst                  # Main documentation index
│   ├── Core.rst                   # Core module documentation
│   ├── Core.*.rst                 # Individual core submodules
│   ├── Modules.rst                # Modules documentation
│   ├── Modules.*.rst              # Individual module documentation
│   ├── Datasets.rst               # Datasets documentation
│   └── ...
├── build/
│   └── html/                      # Generated HTML documentation
├── Makefile                       # Unix build commands
└── make.bat                       # Windows build commands
```

## Updating Documentation

When you add new modules, classes, or functions to UVisBox:

1. Re-run the appropriate `sphinx-apidoc` command(s) above
2. Rebuild the documentation with `sphinx-build -b html source/ build/html/`
3. Review the generated documentation to ensure it looks correct

## Notes

- The `-o source/` flag specifies the output directory for the generated `.rst` files
- The `sphinx-apidoc` command automatically discovers Python modules and creates corresponding documentation files
- Make sure to run these commands from the `docs/` directory so the relative paths work correctly
- If you encounter import errors during documentation generation, ensure that UVisBox and all its dependencies are properly installed in your environment

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure UVisBox is installed in your current environment
2. **Missing Dependencies**: Install all required packages listed in `pyproject.toml`
3. **Path Issues**: Ensure you're running commands from the `docs/` directory
4. **Outdated Files**: Delete old `.rst` files in `source/` if you're restructuring modules

### Clean Build

To perform a clean build:

```bash
# Remove previous build
rm -rf build/html/

# Rebuild documentation
sphinx-build -b html source/ build/html/
```

Alternative with Makefile (if available):

```bash
make clean
make html
```

This removes all previously generated files and rebuilds the documentation from scratch.