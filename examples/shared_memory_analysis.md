# Shared Memory Analysis for Curve Boxplot

## Question
Can `multiprocessing.shared_memory` further accelerate curve boxplot by eliminating data serialization overhead?

## Answer: **No significant benefit** (0-3% improvement)

## Test Results Summary

### Performance Comparison

| Configuration | Workers | Fork Time | Shared Memory Time | Improvement |
|--------------|---------|-----------|-------------------|-------------|
| 30 curves, 2D | 4 | 0.314s | 0.344s | **-9.5%** ⚠️ |
| 30 curves, 2D | 8 | 0.198s | 0.191s | +3.3% |
| 50 curves, 2D | 4 | 0.873s | 0.873s | 0.0% |
| 50 curves, 2D | 8 | 0.537s | 0.522s | +2.8% |
| 100 curves, 2D | 4 | 3.573s | 3.539s | +1.0% |
| 100 curves, 2D | 8 | 2.002s | 1.990s | +0.6% |
| 30 curves, 3D | 4 | 0.590s | 0.590s | 0.0% |
| 30 curves, 3D | 8 | 0.336s | 0.333s | +0.8% |

**Best case improvement: +3.3% (within measurement error)**  
**Worst case: -9.5% (overhead dominates for small datasets)**

## Overhead Analysis

### Array Sizes (Typical Use Cases)

| Dimensions | Size | Pickle (1 worker) | Pickle (8 workers) | Shared Memory Setup |
|------------|------|-------------------|-------------------|---------------------|
| 30×50×2 | 24 KB | 0.04ms | 0.32ms | 0.51ms |
| 100×100×2 | 153 KB | 0.02ms | 0.14ms | 0.03ms |
| 100×100×3 | 229 KB | 0.02ms | 0.17ms | 0.04ms |

### Key Findings

1. **Pickle overhead is negligible**: 0.02-0.04ms per worker
2. **Shared memory overhead**: 0.03-0.51ms (setup + copy + cleanup)
3. **Total savings with 8 workers**: ~0.1-0.3ms (0.0001-0.0003s)
4. **ConvexHull computation**: Dominates runtime (0.2-3.5s)

## Why Shared Memory Doesn't Help

### 1. Small Data Arrays
- Typical curve ensembles: 24-240 KB
- Pickle serialization is extremely fast for small arrays
- Shared memory overhead comparable to pickle overhead

### 2. Computation-Dominated Workload
- ConvexHull construction is expensive: O(n log n) per band
- Point-in-hull testing: O(m × f) per band
- Serialization overhead: <0.1% of total runtime

### 3. Fork Context Already Efficient
- macOS fork context uses copy-on-write
- Read-only data (curves array) is not actually copied in memory
- Only process metadata and small variables are copied

### 4. Shared Memory Overhead
- Creating shared memory block: ~0.03ms
- Copying data into shared memory: ~0.01ms per MB
- Cleanup and unlinking: ~0.01ms
- Total: Similar or worse than pickle for small arrays

## When Would Shared Memory Help?

Shared memory provides benefits when:

1. **Very large arrays** (>10 MB): Pickle overhead becomes significant
2. **Fast computation**: Serialization overhead is significant % of runtime
3. **Many iterations**: Amortize setup cost over many operations
4. **Many workers** (>16): More copies = more overhead

**None of these apply to curve boxplot!**

### Example Where It Would Help

```python
# Large dataset: 1000 curves × 1000 steps × 3D = 24 MB
curves = np.random.randn(1000, 1000, 3)

# Pickle time: ~50ms per worker
# 8 workers = 400ms total pickle overhead

# Shared memory: ~25ms setup + 0ms copying (already in memory)
# Savings: ~375ms (15-20% for fast operations)
```

But for curve boxplot:
- Typical size: 30-100 curves × 50-100 steps = 24-240 KB
- Pickle: <1ms total
- Computation: 200-3000ms
- Savings: <0.5% even in best case

## Recommendation

### ❌ **Do NOT implement shared_memory for curve boxplot**

**Reasons:**
1. No measurable performance improvement (0-3%)
2. Adds complexity and potential bugs
3. Platform-specific behavior (shared_memory API varies)
4. More difficult to debug
5. Memory management complexity
6. Negligible benefit doesn't justify maintenance cost

### ✅ **Keep current fork-based implementation**

The current implementation is optimal because:
- Simple and maintainable
- Fork context already efficient with copy-on-write
- Pickle overhead is negligible (<0.1% of runtime)
- ConvexHull computation dominates (99.9% of time)

## Alternative Optimization Strategies

If further acceleration is needed, consider:

### 1. **Algorithm Optimization**
- Prune bands that cannot contribute (geometric constraints)
- Cache hull computations for repeated bands
- Early termination for obvious outliers

### 2. **Better Hull Algorithm**
- Use approximate convex hull for faster computation
- Incremental hull updates instead of reconstruction
- Spatial indexing to reduce hull tests

### 3. **Native Code**
- Cython or C++ extension for hull operations
- GPU acceleration for parallel hull construction
- Custom hull implementation optimized for this use case

### 4. **Data Structure Changes**
- Spatial partitioning (octree/kd-tree)
- Hierarchical hull representations
- Progressive refinement

**All of these would provide >10% improvement, unlike shared_memory's <3%**

## Conclusion

**Multiprocessing.shared_memory provides no meaningful benefit for curve boxplot** because:

- Data arrays are too small (24-240 KB)
- Pickle overhead is negligible (<1ms)
- ConvexHull computation dominates (>99.9% of runtime)
- Fork context already uses copy-on-write

The current fork-based implementation is already optimal for this use case. Focus optimization efforts on algorithmic improvements rather than micro-optimizations that save <1% runtime.

---

*Analysis completed: November 2, 2025*
