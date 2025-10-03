# UVisBox Package Restructuring Plan

## Overview
This document outlines the comprehensive plan to restructure the UVisBox package according to the specified hierarchy:

```
1. UVisBox
1.1 uvisbox
1.1.1 Core
1.1.2 Modules  
1.1.3 Datasets
1.2 tests
1.3 examples
1.4 docs
```

## Current Structure Analysis

### Current uvisbox Directory Structure
```
uvisbox/
├── BandDepths/
│   ├── Meshing/
│   ├── Stat/
│   └── Vis/
├── Colors/
├── Datasets/
├── Glyphs/
│   ├── Meshing/
│   ├── Stat/
│   └── Vis/
├── Interpolations/
├── ProbabilisticContours/
│   ├── Stat/
│   └── Vis/
├── ProbabilisticSurfaces/
│   ├── Stat/
│   └── Vis/
└── UncertaintyTube/
    ├── Meshing/
    ├── Stat/
    └── Vis/
```

### Other Directories (Already Correctly Positioned)
- `tests/` - Unit tests directory
- `examples/` - Example scripts and demonstrations  
- `docs/` - Documentation and build files

## Proposed New Structure

### Target Structure
```
UVisBox/
├── uvisbox/
│   ├── Core/
│   │   ├── __init__.py
│   │   ├── BandDepths/
│   │   │   ├── __init__.py
│   │   │   ├── contour_banddepth.py
│   │   │   ├── curve_banddepth.py
│   │   │   ├── functional_banddepth.py
│   │   │   └── vector_depths.py
│   │   ├── CellCrossingProb/
│   │   │   └── __init__.py
│   │   ├── Colors/
│   │   │   ├── __init__.py
│   │   │   ├── color_interpolator.py
│   │   │   └── colortree.py
│   │   └── Interpolations/
│   │       ├── __init__.py
│   │       └── linear_interpolation.py
│   ├── Modules/
│   │   ├── __init__.py
│   │   ├── ContourBoxplot/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── Curve_Boxplot/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── FunctionalBoxplot/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── ProbabilisticMarchingCubes/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── ProbabilisticMarchingSquares/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── ProbabiliticMarchingTetrahedra/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── ProbabiliticMarchingTriangles/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── SquidGlyph/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   ├── UncertaintyLobes/
│   │   │   ├── __init__.py
│   │   │   ├── Mesh/
│   │   │   ├── Stats/
│   │   │   └── Vis/
│   │   └── UncertaintyTube/
│   │       ├── __init__.py
│   │       ├── Mesh/
│   │       ├── Stats/
│   │       └── Vis/
│   ├── Datasets/
│   │   └── [existing structure remains]
│   └── __init__.py
├── tests/
├── examples/
└── docs/
```

## Detailed Restructuring Steps

### Phase 1: Core Module Creation and Population

#### 1.1 Create Core Directory Structure
- Create `uvisbox/Core/` directory
- Create `uvisbox/Core/__init__.py`
- Create `uvisbox/Core/BandDepths/` directory
- Create `uvisbox/Core/BandDepths/__init__.py`
- Create `uvisbox/Core/CellCrossingProb/` directory
- Create `uvisbox/Core/CellCrossingProb/__init__.py`
- Create `uvisbox/Core/Colors/` directory
- Create `uvisbox/Core/Colors/__init__.py`
- Create `uvisbox/Core/Interpolations/` directory
- Create `uvisbox/Core/Interpolations/__init__.py`

#### 1.2 Identify and Move Core Functionality
**Analysis needed to identify:**
- Core package initialization and configuration
- Common constants and package-level settings
- Fundamental functionality that other modules depend on
- Core BandDepths statistical methods
- Core CellCrossingProb functionality
- Core Colors functionality
- Core Interpolations functionality

**Potential Core Components:**
- Package-level initialization code
- Common constants and configuration settings
- Core functionality that spans multiple modules
- Core BandDepths statistical methods:
  - `contour_banddepth.py` - Core contour band depth algorithms
  - `curve_banddepth.py` - Core curve band depth algorithms  
  - `functional_banddepth.py` - Core functional band depth algorithms
  - `vector_depths.py` - Core vector depth algorithms
- Core CellCrossingProb functionality:
  - Cell crossing probability algorithms and data structures
- Core Colors functionality:
  - `color_interpolator.py` - Core color interpolation methods
  - `colortree.py` - Core color tree data structures
- Core Interpolations functionality:
  - `linear_interpolation.py` - Core linear interpolation algorithms

### Phase 2: Modules Directory Creation and Reorganization

#### 2.1 Create Modules Directory Structure
- Create `uvisbox/Modules/` directory
- Create `uvisbox/Modules/__init__.py`

#### 2.2 Reorganize Feature Modules
Reorganize existing functionality into new module structure:
- `BandDepths/` → Split between Core and new specialized Modules:
  - `BandDepths/Stat/` files → `uvisbox/Core/BandDepths/` (core statistical methods)  
  - `BandDepths/Meshing/curve_banddepth_meshing.py` → `uvisbox/Modules/Curve_Boxplot/Mesh/`
  - `BandDepths/Vis/curve_banddepth_plot.py` → `uvisbox/Modules/Curve_Boxplot/Vis/`
  - `BandDepths/Vis/contour_boxplot.py` → `uvisbox/Modules/ContourBoxplot/Vis/`
  - `BandDepths/Vis/functional_banddepth_plot.py` → `uvisbox/Modules/FunctionalBoxplot/Vis/`
- `Colors/` → `uvisbox/Core/Colors/` (core color functionality)
- `Interpolations/` → `uvisbox/Core/Interpolations/` (core interpolation functionality)  
- `Glyphs/` → Split into specialized modules:
  - `Glyphs/Vis/squid_glyphs_plot.py` → `uvisbox/Modules/SquidGlyph/Vis/`
  - `Glyphs/Meshing/squid_glyphs_meshing.py` → `uvisbox/Modules/SquidGlyph/Mesh/`
  - `Glyphs/Stat/squid_glyphs.py` → `uvisbox/Modules/SquidGlyph/Stats/`
  - `Glyphs/Vis/uncertainty_lobes.py` → `uvisbox/Modules/UncertaintyLobes/Vis/`
- `ProbabilisticContours/` → Split into specialized modules:
  - `ProbabilisticContours/Stat/probabilistic_marching_squares.py` → `uvisbox/Modules/ProbabilisticMarchingSquares/Stats/`
  - `ProbabilisticContours/Stat/probabilistic_marching_triangles.py` → `uvisbox/Modules/ProbabiliticMarchingTriangles/Stats/`
- `ProbabilisticSurfaces/` → Split into specialized modules:
  - `ProbabilisticSurfaces/Stat/probabilistic_marching_cubes.py` → `uvisbox/Modules/ProbabilisticMarchingCubes/Stats/`
  - `ProbabilisticSurfaces/Vis/probabilistic_marching_cubes_plot.py` → `uvisbox/Modules/ProbabilisticMarchingCubes/Vis/`
  - `ProbabilisticSurfaces/Stat/probabilistic_marching_tetrahedra.py` → `uvisbox/Modules/ProbabiliticMarchingTetrahedra/Stats/`
  - `ProbabilisticSurfaces/Vis/probabilistic_marching_tetrahedra_plot.py` → `uvisbox/Modules/ProbabiliticMarchingTetrahedra/Vis/`
- `UncertaintyTube/` → `uvisbox/Modules/UncertaintyTube/` (keep existing structure but rename subdirectories)

#### 2.3 New Module Structure
Each new module will have a consistent internal structure:
- `Mesh/` subdirectory for meshing/geometry functionality
- `Stats/` subdirectory for statistical computation functionality  
- `Vis/` subdirectory for visualization functionality
- All modules will have appropriate `__init__.py` files

### Phase 3: Datasets Organization
- Verify `uvisbox/Datasets/` is correctly positioned (already correct)
- Ensure all dataset modules are properly accessible
- Review and update dataset `__init__.py` files if needed

### Phase 4: Import Statement Updates

#### 4.1 Update Module __init__.py Files
**Main uvisbox/__init__.py:**
```python
# New structure imports
from .Core import *
from .Modules import *
from .Datasets import *
```

**Modules/__init__.py:**
```python
# Import all feature modules
from .ContourBoxplot import *
from .Curve_Boxplot import *
from .FunctionalBoxplot import *
from .ProbabilisticMarchingCubes import *
from .ProbabilisticMarchingSquares import *
from .ProbabiliticMarchingTetrahedra import *
from .ProbabiliticMarchingTriangles import *
from .SquidGlyph import *
from .UncertaintyLobes import *
from .UncertaintyTube import *
```

**Core/__init__.py:**
```python
# Core package functionality and constants
# This module contains package-level configuration and shared constants
from .BandDepths import *
from .CellCrossingProb import *
from .Colors import *
from .Interpolations import *
```

#### 4.2 Update Internal Module Imports
- Review and update all internal imports within each module
- Update imports to use new Core utilities where applicable
- Ensure relative imports work correctly with new structure

#### 4.3 Maintain Backward Compatibility
Ensure existing import statements continue to work:
- `from uvisbox.BandDepths import ...` should still work
- `from uvisbox.Datasets import ...` should still work
- All current public APIs should remain accessible

### Phase 5: Update Examples Directory

#### 5.1 Current Import Patterns in Examples
```python
# Current imports found in examples:
from uvisbox.Datasets import irma2017_perturbed_tracks
from uvisbox.BandDepths import curve_banddepth_plot, curve_banddepth_meshing, curve_banddepths
from uvisbox.ProbabilisticSurfaces import probabilistic_marching_cubes_plot as pmc
from uvisbox.Glyphs import uncertainty_squid_glyphs_3D
from uvisbox.Colors.colortree import ColorTree
from uvisbox.UncertaintyTube import generate_cross_sections, generate_tube_mesh
```

#### 5.2 Updated Import Strategy
**Option 1: Maintain Current Imports (Recommended)**
- Keep existing import patterns working through proper `__init__.py` configuration
- No changes needed to example files

**Option 2: Update to New Structure**
- Update imports to use new paths:
  ```python
  from uvisbox.Modules.BandDepths import curve_banddepth_plot
  from uvisbox.Modules.Glyphs import uncertainty_squid_glyphs_3D
  ```

### Phase 6: Update Tests Directory

#### 6.1 Review Test Structure
- Analyze existing test files and their import patterns
- Update test imports to work with new structure
- Ensure all tests continue to pass after restructuring

#### 6.2 Test Import Updates
- Update test imports to reflect new module locations
- Verify Core module integration works correctly
- Verify integration tests work with new structure

### Phase 7: Update Documentation

#### 7.1 Documentation Files to Update
- `docs/source/*.rst` files that reference module paths
- API documentation that shows import examples
- Installation and usage guides

#### 7.2 Documentation Updates Needed
- Update module reference documentation
- Update example code in documentation
- Update API reference paths
- Regenerate Sphinx documentation

### Phase 8: Package Configuration Updates

#### 8.1 Update pyproject.toml
- Review package configuration
- Ensure new structure is properly included in package
- Update any module-specific configurations

#### 8.2 Update Setup/Build Configuration
- Verify package building works with new structure
- Test package installation from source
- Verify all modules are included in distribution

## Implementation Checklist

### Pre-Implementation
- [ ] Create backup of current codebase
- [ ] Document current import dependencies
- [ ] Identify shared/core functionality for Core module

### Core Module Implementation
- [ ] Create Core directory structure
- [ ] Create Core/BandDepths directory structure
- [ ] Create Core/CellCrossingProb directory structure
- [ ] Create Core/Colors directory structure
- [ ] Create Core/Interpolations directory structure
- [ ] Move BandDepths/Stat files to Core/BandDepths:
  - [ ] Move contour_banddepth.py to Core/BandDepths/
  - [ ] Move curve_banddepth.py to Core/BandDepths/
  - [ ] Move functional_banddepth.py to Core/BandDepths/
  - [ ] Move vector_depths.py to Core/BandDepths/
- [ ] Create or move CellCrossingProb functionality to Core/CellCrossingProb:
  - [ ] Implement cell crossing probability algorithms
- [ ] Move Colors files to Core/Colors:
  - [ ] Move color_interpolator.py to Core/Colors/
  - [ ] Move colortree.py to Core/Colors/
- [ ] Move Interpolations files to Core/Interpolations:
  - [ ] Move linear_interpolation.py to Core/Interpolations/
- [ ] Create Core/__init__.py with imports for BandDepths, CellCrossingProb, Colors, and Interpolations
- [ ] Create Core/BandDepths/__init__.py with statistical method exports
- [ ] Create Core/CellCrossingProb/__init__.py with cell crossing probability exports
- [ ] Create Core/Colors/__init__.py with color functionality exports
- [ ] Create Core/Interpolations/__init__.py with interpolation method exports

### Modules Reorganization
- [ ] Create Modules directory with new structure
- [ ] Create new module directories with Mesh/Stats/Vis subdirectories:
  - [ ] Create ContourBoxplot/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create Curve_Boxplot/ with Mesh/, Stats/, Vis/ subdirectories  
  - [ ] Create FunctionalBoxplot/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create ProbabilisticMarchingCubes/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create ProbabilisticMarchingSquares/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create ProbabiliticMarchingTetrahedra/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create ProbabiliticMarchingTriangles/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create SquidGlyph/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Create UncertaintyLobes/ with Mesh/, Stats/, Vis/ subdirectories
  - [ ] Reorganize UncertaintyTube/ to use Mesh/, Stats/, Vis/ subdirectories
- [ ] Move existing functionality to appropriate new module locations
- [ ] Update internal imports within each module
- [ ] Create Modules/__init__.py with exports for all new modules
- [ ] Create __init__.py files for each new module and subdirectory

### Import System Updates
- [ ] Update main uvisbox/__init__.py
- [ ] Update all module __init__.py files
- [ ] Test backward compatibility of imports
- [ ] Update internal cross-module imports

### Examples Updates
- [ ] Test all examples with new structure
- [ ] Update example imports if necessary
- [ ] Verify all examples still work correctly

### Tests Updates
- [ ] Update test imports
- [ ] Run full test suite
- [ ] Fix any broken tests
- [ ] Verify Core module integration works correctly

### Documentation Updates
- [ ] Update all .rst files with new import paths
- [ ] Update example code in documentation
- [ ] Regenerate documentation
- [ ] Verify documentation builds correctly

### Final Validation
- [ ] Test package installation
- [ ] Test import statements from clean environment
- [ ] Run all examples
- [ ] Run full test suite
- [ ] Verify documentation is accessible

## Risk Mitigation

### Backward Compatibility
- Maintain all existing import paths through proper `__init__.py` configuration
- Provide deprecation warnings if changing any public APIs
- Document migration path for any breaking changes

### Testing Strategy
- Run tests after each phase of implementation
- Maintain backup of working version
- Test in isolated environment before committing changes

### Rollback Plan
- Keep detailed git history of changes
- Maintain tagged version of pre-restructure code
- Document rollback procedure if issues arise

## Benefits of New Structure

### 1. Improved Organization
- Clear separation between core functionality and feature modules
- Better discoverability of functionality
- Consistent module organization

### 2. Better Maintainability
- Shared code consolidated in Core module
- Reduced code duplication
- Easier to add new feature modules

### 3. Enhanced Extensibility
- Clear place for new modules (Modules/)
- Established patterns for core functionality
- Easier integration of new features

### 4. Professional Package Structure
- Follows Python packaging best practices
- Cleaner import hierarchy
- Better suited for large-scale development

## Timeline Estimate

- **Phase 1 (Core Creation)**: 2-3 days
- **Phase 2 (Modules Reorganization)**: 1-2 days  
- **Phase 3 (Datasets Verification)**: 0.5 days
- **Phase 4 (Import Updates)**: 1-2 days
- **Phase 5 (Examples Updates)**: 1 day
- **Phase 6 (Tests Updates)**: 1-2 days
- **Phase 7 (Documentation Updates)**: 1-2 days
- **Phase 8 (Package Configuration)**: 0.5 days
- **Final Testing and Validation**: 1-2 days

**Total Estimated Time**: 8-14 days

## Conclusion

This restructuring plan will modernize the UVisBox package structure while maintaining backward compatibility and improving maintainability. The phased approach allows for incremental implementation and testing, reducing the risk of introducing breaking changes.