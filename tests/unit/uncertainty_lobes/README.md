# Uncertainty Lobes Test Suite

Comprehensive test suite for the UncertaintyLobes module, covering all three stages of the visualization pipeline.

## Test Organization

### Unit Tests (`tests/unit/uncertainty_lobes/`)

#### `test_uncertainty_lobes_stats.py` (11 tests)
Tests for the statistics computation stage:
- ✅ Basic functionality with ensemble vectors
- ✅ Single vs dual percentile modes
- ✅ Cartesian to polar coordinate conversion
- ✅ Vector depth computation and ranges
- ✅ Median vector selection (maximum depth)
- ✅ Angular spread calculations
- ✅ Percentile effects on spread size
- ✅ Edge cases (minimal ensemble, empty data)
- ✅ Parallel worker function (`_process_position_stats`)

**Key Test:** `test_percentile_effect` - Verifies higher percentiles include more vectors (larger spreads)

#### `test_uncertainty_lobes_mesh.py` (15 tests)
Tests for the mesh generation stage:
- ✅ Wedge vertex generation
- ✅ **Arc direction correctness** (critical fix validation)
- ✅ Arc includes median angle (normal and wrap-around cases)
- ✅ Arc direction consistency across all quadrants
- ✅ Triangle fan generation
- ✅ Different radii and center positions
- ✅ Scale factor application
- ✅ Arrow direction computation
- ✅ Arc resolution variations
- ✅ Zero radius handling

**Key Test:** `test_arc_direction_consistency` - Validates the arc direction fix for various angle configurations including wrap-around cases (0°/360°)

#### `test_uncertainty_lobes_vis.py` (8 tests)
Tests for the visualization rendering stage:
- ✅ Basic rendering with matplotlib
- ✅ Rendering with/without median arrows
- ✅ Dual-lobe rendering (inner + outer)
- ✅ Multiple wedges visualization
- ✅ Automatic figure/axes creation
- ✅ Arrow rendering properties
- ✅ Empty data handling
- ✅ Zero-length arrows

### Integration Tests (`tests/integration/`)

#### `test_uncertainty_lobes_integration.py` (12 tests)

**TestUncertaintyLobesPipeline** - End-to-end pipeline tests:
- ✅ Complete pipeline from ensemble vectors to visualization
- ✅ All three stages working together separately
- ✅ Single lobe mode (percentile2=None)
- ✅ **Arc direction correctness in full pipeline** (wrap-around validation)
- ✅ Different scale factors
- ✅ Different percentile combinations
- ✅ Show/hide median flag
- ✅ Minimal ensemble (n=3)
- ✅ Large ensemble (n=100)
- ✅ Deterministic output verification

**TestUncertaintyLobesEdgeCases** - Edge case handling:
- ✅ Zero-magnitude vectors
- ✅ All identical vectors (narrow lobes)

**Critical Integration Test:** `test_arc_direction_correctness` - Comprehensive validation that wedges are always created in the direction containing the median angle, testing:
- Normal cases (e.g., 45° median)
- Wrap-around 0° (e.g., 350° to 10° with 0° median)
- All quadrants (0°, 90°, 180°, 270°)
- Near-boundary cases (350° median)

## Running the Tests

### Run all uncertainty lobes tests:
```bash
pytest tests/unit/uncertainty_lobes/ tests/integration/test_uncertainty_lobes_integration.py -v
```

### Run specific test modules:
```bash
# Statistics tests
pytest tests/unit/uncertainty_lobes/test_uncertainty_lobes_stats.py -v

# Mesh generation tests (includes arc direction fix validation)
pytest tests/unit/uncertainty_lobes/test_uncertainty_lobes_mesh.py -v

# Visualization tests
pytest tests/unit/uncertainty_lobes/test_uncertainty_lobes_vis.py -v

# Integration tests
pytest tests/integration/test_uncertainty_lobes_integration.py -v
```

### Run specific test:
```bash
# Test the critical arc direction fix
pytest tests/unit/uncertainty_lobes/test_uncertainty_lobes_mesh.py::TestCreateWedgeVertices::test_arc_direction_consistency -v
```

## Test Coverage

### Statistics Module (`uncertainty_lobes_stats.py`)
- ✅ `compute_uncertainty_lobes_stats_2d()` - Main API
- ✅ `_process_position_stats()` - Parallel worker
- ✅ Cartesian to polar conversion
- ✅ Vector depth computation
- ✅ Spread calculation for dual percentiles

### Mesh Module (`uncertainty_lobes_mesh.py`)
- ✅ `build_uncertainty_lobes_mesh_2d()` - Main API
- ✅ `_create_wedge_vertices()` - **Arc direction logic** ⭐
- ✅ `_triangulate_wedge()` - Triangle generation
- ✅ Wedge construction for outer/inner lobes
- ✅ Arrow data preparation

### Visualization Module (`uncertainty_lobes_vis.py`)
- ✅ `render_uncertainty_lobes_2d()` - Main API
- ✅ Wedge polygon rendering
- ✅ Arrow rendering
- ✅ Matplotlib integration

### High-Level API (`uncertainty_lobes.py`)
- ✅ `uncertainty_lobes()` - Complete pipeline orchestration
- ✅ Integration with all three stages
- ✅ Parameter passing and defaults

## Key Validations

### Arc Direction Fix (Critical)
The test suite extensively validates the fix for the arc direction bug:

1. **Unit level** (`test_uncertainty_lobes_mesh.py`):
   - `test_arc_includes_median_normal_case` - Normal angle ranges
   - `test_arc_includes_median_wrap_around` - Wrap-around 0°/360°
   - `test_arc_direction_consistency` - 5 different configurations

2. **Integration level** (`test_uncertainty_lobes_integration.py`):
   - `test_arc_direction_correctness` - 6 angle configurations with full pipeline
   - Validates median arrow points INTO wedge
   - Tests all quadrants and boundary cases

### Pipeline Correctness
- Data flow through all three stages
- Correct shape transformations
- Consistent output structure
- Deterministic behavior

### Edge Cases
- Minimal ensemble sizes
- Zero-magnitude vectors
- Identical vectors
- Extreme percentiles (0, 100)
- Wrap-around angles
- Various scales and resolutions

## Test Results

**Total: 46 tests**
- Unit tests: 34
- Integration tests: 12
- **All tests passing** ✅

Minor warnings (2) related to division by zero in edge cases (all identical vectors) - handled gracefully by the code.

## Continuous Integration

These tests should be run:
- Before committing changes to uncertainty lobes module
- As part of CI/CD pipeline
- After refactoring or bug fixes
- When adding new features

## Notes

- Tests use `matplotlib.use('Agg')` for non-interactive backend
- Integration tests validate the critical arc direction fix comprehensively
- Tests cover normal operation and edge cases
- Deterministic random seeds used where needed for reproducibility
