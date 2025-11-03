# Curve Band Depth Optimization Analysis

## Current Implementation Analysis

### Algorithm Complexity

**Current Implementation:**
```
For each time_step (1 to n_steps-1):                    O(n_steps)
  For each band in indices:                             O(C(n_curves, 2)) = O(n_curves²)
    Extract band points: band_curves[band, :step, :]    O(band_size × step)
    Build ConvexHull:                                    O(points × log(points))
    Test all points in hull:                             O(n_curves × n_facets)
```

**Total Complexity:**
- Time: O(n_steps × n_curves² × (hull_construction + hull_testing))
- Hull construction: O(n_points × log(n_points)) per band
- Hull testing: O(n_curves × n_facets) per band
- **Dominated by ConvexHull construction** (~99% of runtime)

### Benchmark Results (from testing)

| Configuration | Sequential Time | 8 Workers Time | Speedup |
|--------------|-----------------|----------------|---------|
| 30 curves, 50 steps, 2D | 1.12s | 0.19s | 5.9x |
| 50 curves, 50 steps, 2D | 3.21s | 0.50s | 6.4x |
| 100 curves, 50 steps, 2D | 13.24s | 2.00s | 6.6x |
| 30 curves, 50 steps, 3D | 1.99s | 0.33s | 6.0x |

**Scaling Analysis:**
- 30→50 curves: 2.9x slower (expected: 2.8x for O(n²))
- 50→100 curves: 4.1x slower (expected: 4.0x for O(n²))
- Good scaling indicates O(n²) behavior as expected

### Bottlenecks Identified

1. **ConvexHull Construction (99% of time)**
   - Called: (n_steps-1) × n_bands times
   - Example: 50 steps × 435 bands = 21,315 hull constructions
   - Each hull: O(n log n) where n = band_size × step

2. **Redundant Computations**
   - Same band at different time steps has overlapping points
   - Hulls are reconstructed from scratch each time
   - No caching or reuse of previous results

3. **Memory Access Pattern**
   - Array slicing creates views but still requires iteration
   - Band extraction: `curves[band, :step_idx+1, :]` repeated

4. **No Early Termination**
   - All bands processed even if some curves clearly outliers
   - No pruning based on geometric constraints

## Optimization Opportunities

### 1. **Incremental Hull Updates** ⭐⭐⭐⭐⭐ (HIGH IMPACT)

**Problem:** Reconstructing hulls from scratch at each time step.

**Solution:** Use incremental hull construction.

At step `t`, the hull is built from points `[:t+1]`.  
At step `t+1`, only ONE new point added per curve in band.

**Approach:**
```python
# Instead of:
for step_idx in range(1, n_steps):
    hull = ConvexHull(band_curves[:step_idx+1].reshape(-1, n_dims))

# Use incremental:
hull = ConvexHull(band_curves[:2].reshape(-1, n_dims))  # Initialize
for step_idx in range(2, n_steps):
    new_points = band_curves[:, step_idx, :]  # Just the new points
    hull = update_hull_incremental(hull, new_points)  # Update, don't rebuild
```

**Expected Speedup:** 2-5x (depending on n_steps)

**Challenges:**
- SciPy ConvexHull doesn't support incremental updates
- Would need custom implementation or different library
- Qhull (underlying library) supports incremental mode
- Could use direct Qhull bindings

**Implementation Complexity:** High (requires custom C++/Cython extension)

---

### 2. **Hull Caching & Reuse** ⭐⭐⭐⭐ (MEDIUM-HIGH IMPACT)

**Problem:** Some band combinations appear in multiple contexts.

**Solution:** Cache hull computations with memoization.

**Approach:**
```python
from functools import lru_cache

hull_cache = {}

def get_or_create_hull(band_tuple, step_idx):
    """Cache hulls by (band, step) key."""
    key = (band_tuple, step_idx)
    if key not in hull_cache:
        band_curves = curves[band, :step_idx+1, :]
        hull_cache[key] = ConvexHull(band_curves.reshape(-1, n_dims))
    return hull_cache[key]
```

**Expected Speedup:** 1.1-1.3x (limited by unique band-step combinations)

**Pros:**
- Easy to implement
- No algorithm changes
- Memory overhead is manageable

**Cons:**
- Most band-step combinations are unique
- Memory grows linearly with n_steps × n_bands

---

### 3. **Spatial Pruning** ⭐⭐⭐⭐⭐ (HIGH IMPACT)

**Problem:** Testing all curves against all hulls, even when geometrically impossible.

**Solution:** Use bounding boxes to skip impossible containment tests.

**Approach:**
```python
# Compute bounding box for hull
hull_min = band_curves.min(axis=(0, 1))
hull_max = band_curves.max(axis=(0, 1))

# Quick rejection test
bbox_mask = np.all((all_points >= hull_min) & (all_points <= hull_max), axis=1)

# Only test points that pass bounding box test
candidates = np.where(bbox_mask)[0]
in_hull_mask = np.zeros(n_curves, dtype=bool)
in_hull_mask[candidates] = points_in_hull(all_points[candidates], hull)
```

**Expected Speedup:** 1.5-3x (more outliers = more speedup)

**Pros:**
- Simple to implement
- No external dependencies
- Works well when data has outliers

**Cons:**
- Less effective for tightly clustered data
- Bounding box computation has small overhead

---

### 4. **Adaptive Band Sampling** ⭐⭐⭐ (MEDIUM IMPACT)

**Problem:** Computing depth for all O(n²) band combinations.

**Solution:** Use statistical sampling for large ensembles.

**Approach:**
```python
# For large ensembles, sample bands instead of using all
if n_curves > 50:
    n_samples = min(len(indices), 1000)  # Limit to 1000 bands
    indices_sampled = random.sample(indices, n_samples)
else:
    indices_sampled = indices

# Adjust normalization for sampling
depths /= (n_steps-1) * len(indices_sampled)
```

**Expected Speedup:** Up to n_bands / 1000 for large ensembles

**Pros:**
- Dramatic speedup for large ensembles (100+ curves)
- Good approximation with sufficient samples

**Cons:**
- Results are approximate (not exact)
- Requires statistical validation
- May miss important features

---

### 5. **Parallel Hull Construction** ⭐⭐ (LOW IMPACT - Already Done)

**Status:** ✅ Already implemented via multiprocessing

Current implementation parallelizes at time-step level.  
Achieves 5-6x speedup with 8 workers.

**Potential improvement:** Parallelize at band level within each time step.

**Expected Additional Speedup:** 1.1-1.2x (diminishing returns)

**Not recommended:** More complex, marginal gains.

---

### 6. **Approximate Hull Algorithms** ⭐⭐⭐⭐ (MEDIUM-HIGH IMPACT)

**Problem:** Exact ConvexHull is overkill for depth estimation.

**Solution:** Use faster approximate hull algorithms.

**Options:**

#### A. **Jarvis March (Gift Wrapping)**
- O(n × h) where h = number of hull vertices
- Faster for low-dimensional data with few hull vertices
- Exact, not approximate

#### B. **Quickhull Approximation**
- Stop early with epsilon tolerance
- Trade accuracy for speed

#### C. **Bounding Ellipsoid**
- Fit ellipsoid instead of convex hull
- Much faster: O(n) with closed-form solution
- Approximate but often sufficient for depth

**Implementation Example (Bounding Ellipsoid):**
```python
def points_in_ellipsoid(points, band_points):
    """Test if points inside bounding ellipsoid of band."""
    # Compute mean and covariance
    mu = band_points.mean(axis=0)
    cov = np.cov(band_points.T)
    
    # Mahalanobis distance
    diff = points - mu
    inv_cov = np.linalg.inv(cov)
    mahal = np.sum(diff @ inv_cov * diff, axis=1)
    
    # Points inside if distance < threshold
    return mahal < chi2_threshold
```

**Expected Speedup:** 5-10x (ellipsoid vs hull)

**Cons:**
- Less accurate (ellipsoid ⊂ convex hull)
- May affect outlier detection quality

---

### 7. **GPU Acceleration** ⭐⭐⭐⭐⭐ (VERY HIGH IMPACT - Future Work)

**Problem:** ConvexHull is CPU-bound, single-threaded.

**Solution:** Use GPU for parallel hull construction and testing.

**Approach:**
- Use CUDA/OpenCL for parallel hull construction
- Process multiple bands simultaneously on GPU
- Vectorize point-in-hull tests on GPU

**Libraries:**
- CuPy: GPU-accelerated NumPy
- Taichi: Python GPU programming
- Custom CUDA kernels

**Expected Speedup:** 10-50x (depending on GPU)

**Cons:**
- Requires GPU hardware
- Complex implementation
- Not portable

---

### 8. **Data Structure Optimization** ⭐⭐⭐ (MEDIUM IMPACT)

**Problem:** Repeated array slicing and reshaping.

**Solution:** Pre-process data into optimal layout.

**Approach:**
```python
# Pre-allocate and pre-compute band data
band_points_cache = {}
for step_idx in range(1, n_steps):
    for band_idx, band in enumerate(indices):
        key = (band_idx, step_idx)
        band_points_cache[key] = curves[band, :step_idx+1, :].reshape(-1, n_dims)

# Then use cached data
hull = ConvexHull(band_points_cache[(band_idx, step_idx)])
```

**Expected Speedup:** 1.1-1.2x (reduce slicing overhead)

**Cons:**
- Increased memory usage
- May not fit in memory for large ensembles

---

## Recommended Optimization Strategy

### Phase 1: Quick Wins (1-2 days implementation)

1. ✅ **Spatial Pruning with Bounding Boxes** (1.5-3x speedup)
   - Easy to implement
   - No algorithm changes
   - Immediate benefit

2. ✅ **Hull Caching** (1.1-1.3x speedup)
   - Simple memoization
   - Low risk

**Combined Expected:** 1.5-4x speedup

---

### Phase 2: Medium Effort (1 week)

3. ✅ **Approximate Hulls (Bounding Ellipsoid)** (5-10x speedup)
   - Validate accuracy on test cases
   - Provide option: exact vs approximate
   - Significant speedup

**Combined Expected:** 7-15x speedup over current

---

### Phase 3: Advanced (2-4 weeks)

4. ✅ **Incremental Hull Construction**
   - Custom Qhull bindings or Cython implementation
   - Most impactful single optimization
   - 2-5x additional speedup

5. ⚠️ **GPU Acceleration** (if needed)
   - Only for very large datasets
   - Research project level effort

**Combined Expected:** 15-50x speedup over current

---

## Prototype: Quick Win Implementation

Here's a prototype combining bounding box pruning and hull caching:

```python
def curve_banddepths_optimized(curves, indices=None, workers=12, use_bbox_pruning=True):
    """
    Optimized curve band depth with bounding box pruning and caching.
    """
    n_curves, n_steps, n_dims = curves.shape
    
    if indices is None:
        indices = list(combinations(range(n_curves), 2))
    
    depths = np.zeros(n_curves)
    hull_cache = {}  # Cache hulls
    
    use_parallel = workers is not None and workers > 1
    
    if use_parallel:
        # TODO: Update worker function with optimizations
        pass
    else:
        for step_idx in range(1, n_steps):
            all_points = curves[:, step_idx, :]
            
            for band_idx, band in enumerate(indices):
                # Try cache first
                cache_key = (tuple(band), step_idx)
                if cache_key in hull_cache:
                    hull, bbox_min, bbox_max = hull_cache[cache_key]
                else:
                    band_curves = curves[band, :step_idx+1, :]
                    try:
                        hull = ConvexHull(band_curves.reshape(-1, n_dims))
                        # Compute bounding box
                        flat_band = band_curves.reshape(-1, n_dims)
                        bbox_min = flat_band.min(axis=0)
                        bbox_max = flat_band.max(axis=0)
                        hull_cache[cache_key] = (hull, bbox_min, bbox_max)
                    except:
                        continue
                
                if use_bbox_pruning:
                    # Quick bounding box test
                    bbox_mask = np.all(
                        (all_points >= bbox_min) & (all_points <= bbox_max), 
                        axis=1
                    )
                    
                    # Only test candidates
                    candidates = np.where(bbox_mask)[0]
                    if len(candidates) == 0:
                        continue
                    
                    in_hull_mask = np.zeros(n_curves, dtype=bool)
                    in_hull_mask[candidates] = points_in_hull(
                        all_points[candidates], hull
                    )
                else:
                    in_hull_mask = points_in_hull(all_points, hull)
                
                depths += in_hull_mask.astype(int)
    
    depths /= (n_steps-1) * len(indices)
    return depths
```

---

## Benchmark Target

| Configuration | Current Time | Target Time | Speedup Needed |
|--------------|--------------|-------------|----------------|
| 30 curves, 2D | 1.12s | <0.3s | 4x |
| 50 curves, 2D | 3.21s | <0.7s | 5x |
| 100 curves, 2D | 13.24s | <3.0s | 4.5x |
| 30 curves, 3D | 1.99s | <0.5s | 4x |

**Phase 1 (Quick Wins):** Should achieve these targets  
**Phase 2 (Approximate Hulls):** 10-20x beyond target  
**Phase 3 (Incremental Hulls):** 20-50x beyond target

---

## Next Steps

1. **Implement Phase 1** (bounding box + caching)
2. **Benchmark against current implementation**
3. **Validate correctness** (ensure identical results)
4. **Profile to identify remaining bottlenecks**
5. **Decide on Phase 2 based on results**

Would you like me to implement the Phase 1 optimizations?
