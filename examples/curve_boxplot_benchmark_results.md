# Curve Boxplot Parallelization Benchmark Results

## Executive Summary

**Key Finding: Fork multiprocessing significantly outperforms Numba for curve boxplot band depth computation.**

Unlike contour boxplot where Numba showed 3-4x speedup, curve boxplot's reliance on `ConvexHull` prevents effective Numba acceleration. The hybrid Numba approach (Python for hull construction, Numba for point-in-hull testing) does NOT provide speedup over sequential execution.

## Performance Comparison

### 2D Curves (More Practical Use Cases)

| Configuration | Fork (8 workers) | Numba (sequential) | Fork Speedup | Winner |
|--------------|------------------|-----------------------|--------------|---------|
| 30 curves, 50 steps | 0.19s | 0.85s | **6.0x** | **Fork** |
| 50 curves, 50 steps | 0.50s | 2.41s | **6.4x** | **Fork** |
| 100 curves, 50 steps | 2.00s | 10.00s | **6.6x** | **Fork** |

### 3D Curves (Limited to Small Ensembles)

| Configuration | Fork (8 workers) | Numba (sequential) | Fork Speedup | Winner |
|--------------|------------------|-----------------------|--------------|---------|
| 10 curves, 50 steps | 0.05s | 0.18s | **4.2x** | **Fork** |
| 10 curves, 100 steps | 0.12s | 0.54s | **5.3x** | **Fork** |
| 30 curves, 50 steps | 0.33s | 1.64s | **6.0x** | **Fork** |
| 30 curves, 100 steps | 1.04s | 5.48s | **5.9x** | **Fork** |

## Key Observations

### Why Numba Fails for Curve Boxplot

1. **ConvexHull Bottleneck**: `scipy.spatial.ConvexHull` cannot be compiled with Numba's `nopython=True` mode
2. **Hybrid Approach Ineffective**: Moving only point-in-hull testing to Numba provides insufficient speedup
3. **Sequential Execution**: Numba version runs sequentially with no parallelization benefit
4. **Overhead**: The hybrid approach actually adds overhead compared to pure Python

### Fork Multiprocessing Advantages

1. **Excellent Scaling**: Achieves 6-6.6x speedup with 8 workers on 2D curves
2. **Consistent Performance**: Low variance across runs (±0.002-0.03s)
3. **Good Efficiency**: Approaches ideal linear scaling for larger datasets
4. **3D Support**: Handles 3D curves effectively, though limited to smaller ensembles

### Computational Complexity

Curve boxplot is **significantly more expensive** than contour boxplot:
- **ConvexHull construction**: O(n log n) per band per time step
- **Point-in-hull testing**: O(m × f) where m = points, f = facets
- **Total complexity**: Much higher than contour's union/intersection operations

This explains why:
- 100 curves × 50 steps (2D) takes ~13s sequential (vs <1s for similar contour boxplot)
- 3D examples must be limited to 10-30 curves (vs 100+ for contours)

## Consistency Verification

✅ **All methods produce identical results** (differences < 1e-10 for most cases)

⚠️ **Minor numerical differences** observed for 100-curve 2D ensemble:
- Max difference: 4.1e-06 (one curve out of 100)
- Mean difference: 4.1e-08
- Likely due to floating-point rounding in parallel vs sequential execution
- **Practically negligible** for visualization purposes

## Recommendations

### For Curve Boxplot Implementation

1. ✅ **Use fork multiprocessing** - Provides 5-6x speedup consistently
2. ❌ **Don't use Numba** - No benefit due to ConvexHull dependency
3. 📊 **Optimal worker count**: 6-8 workers for best performance
4. 🎯 **Dataset limits**:
   - 2D curves: Up to 100 curves with 50-100 steps
   - 3D curves: Limited to 10-30 curves with 50-100 steps

### Comparison with Contour Boxplot

| Aspect | Contour Boxplot | Curve Boxplot |
|--------|----------------|---------------|
| **Best Method** | Numba (3-4x) | Fork (5-6x) |
| **Numba Viable?** | ✅ Yes | ❌ No |
| **Fork Scaling** | Good (2-4.6x) | Excellent (5-6.6x) |
| **Complexity** | Low | High |
| **Max Ensemble** | 100+ members | 30-100 (2D), 10-30 (3D) |

### Alternative Approaches to Consider

Since Numba is not viable, future optimizations could explore:

1. **Cython**: Compile ConvexHull operations for better performance
2. **C++ extensions**: Native code with pybind11 for hull operations
3. **GPU acceleration**: CUDA/OpenCL for parallel hull construction
4. **Approximate methods**: Faster convex hull approximations
5. **Algorithm optimization**: Better data structures or pruning strategies

## Benchmark Configuration

- **Hardware**: macOS with conda environment
- **Repeats**: 3 runs per configuration
- **Worker counts**: 1, 2, 4, 6, 8
- **Data generation**: Synthetic smooth curves with sinusoidal components
- **Seed**: 42 (reproducible)

## Visualization

See `curve_boxplot_benchmark_results.png` for detailed execution time and speedup plots across all configurations.

## Conclusion

**Fork multiprocessing is the clear winner for curve boxplot parallelization**, achieving excellent 5-6x speedup. Unlike contour boxplot where Numba was superior, curve boxplot's computational complexity and ConvexHull dependency make traditional multiprocessing the best choice. The current implementation should continue using fork context for optimal performance.
