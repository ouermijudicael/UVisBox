"""
Test script to evaluate if multiprocessing.shared_memory can accelerate curve boxplot.

Analysis of potential benefits:
1. Current implementation: Each worker receives a COPY of the entire curves array via pickling
2. Shared memory: Workers access the same memory without copying
3. Potential savings: Eliminate serialization overhead for large arrays

For a curves array of shape (100, 100, 3):
- Size: 100 * 100 * 3 * 8 bytes (float64) = 240 KB
- With 8 workers: 8 copies = 1.92 MB total
- Serialization time: ~1-5ms per worker spawn

This tests whether shared_memory provides measurable speedup.
"""

import numpy as np
import time
from multiprocessing import get_context, shared_memory
from itertools import combinations
from scipy.spatial import ConvexHull
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uvisbox.Core.BandDepths.curve_banddepth import curve_banddepths, points_in_hull


def _process_time_step_shared(args):
    """
    Worker function using shared_memory to access curves array.
    
    This eliminates pickle serialization of the large curves array.
    """
    step_idx, shm_name, shape, dtype, n_curves, indices, n_dims = args
    
    # Attach to existing shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create numpy array view on shared memory
    curves = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    # Initialize depth increments for this time step
    step_depths = np.zeros(n_curves)
    
    # Extract all curve points at this time step
    all_points = curves[:, step_idx, :]
    
    # Process each band
    for band in indices:
        band_curves = curves[band, :step_idx+1, :]
        
        try:
            hull = ConvexHull(band_curves.reshape(-1, n_dims))
        except:
            continue
        
        in_hull_mask = points_in_hull(all_points, hull)
        step_depths += in_hull_mask.astype(int)
    
    # Cleanup
    existing_shm.close()
    
    return step_depths


def curve_banddepths_shared_memory(curves, indices=None, workers=12):
    """
    Curve band depth using shared_memory for zero-copy multiprocessing.
    
    Benefits:
    - No pickle serialization of curves array
    - Workers access shared read-only memory
    - Potential speedup for large arrays
    
    Overhead:
    - Shared memory creation/cleanup
    - Memory management complexity
    """
    n_curves, n_steps, n_dims = curves.shape
    
    if indices is None:
        indices = list(combinations(range(n_curves), 2))
    
    depths = np.zeros(n_curves)
    
    # Create shared memory block
    shm = shared_memory.SharedMemory(create=True, size=curves.nbytes)
    
    # Create numpy array backed by shared memory
    shared_curves = np.ndarray(curves.shape, dtype=curves.dtype, buffer=shm.buf)
    
    # Copy data into shared memory (one-time cost)
    shared_curves[:] = curves[:]
    
    try:
        # Prepare arguments for workers
        step_args = [
            (step_idx, shm.name, curves.shape, curves.dtype, n_curves, indices, n_dims)
            for step_idx in range(1, n_steps)
        ]
        
        # Create worker pool
        ctx = get_context('fork')
        pool = ctx.Pool(processes=workers)
        
        try:
            # Process all time steps in parallel
            step_depths_list = pool.map(_process_time_step_shared, step_args)
            
            # Sum up depth increments
            for step_depths in step_depths_list:
                depths += step_depths
        finally:
            pool.close()
            pool.join()
    finally:
        # Cleanup shared memory
        shm.close()
        shm.unlink()
    
    # Normalize
    depths /= (n_steps-1) * len(indices)
    return depths


def generate_test_curves(n_curves=30, n_steps=50, n_dims=3, seed=42):
    """Generate synthetic curves for testing."""
    np.random.seed(seed)
    curves = []
    for i in range(n_curves):
        steps = np.random.randn(n_steps, n_dims) * 0.1
        t = np.linspace(0, 2*np.pi, n_steps)
        for d in range(n_dims):
            phase = np.random.rand() * 2 * np.pi
            amplitude = np.random.rand() * 2 + 0.5
            steps[:, d] += amplitude * np.sin(t + phase)
        curve = np.cumsum(steps, axis=0)
        curves.append(curve)
    return np.array(curves)


def benchmark_comparison():
    """
    Compare standard fork vs shared_memory implementation.
    """
    print("="*70)
    print("CURVE BOXPLOT: Fork vs Shared Memory Benchmark")
    print("="*70)
    
    configs = [
        {'n_curves': 30, 'n_steps': 50, 'n_dims': 2},
        {'n_curves': 50, 'n_steps': 50, 'n_dims': 2},
        {'n_curves': 100, 'n_steps': 50, 'n_dims': 2},
        {'n_curves': 30, 'n_steps': 50, 'n_dims': 3},
    ]
    
    workers_list = [4, 8]
    
    for config in configs:
        n_curves = config['n_curves']
        n_steps = config['n_steps']
        n_dims = config['n_dims']
        
        print(f"\n{'='*70}")
        print(f"Configuration: {n_curves} curves, {n_steps} steps, {n_dims}D")
        print(f"{'='*70}")
        
        # Generate data
        curves = generate_test_curves(n_curves, n_steps, n_dims)
        indices = list(combinations(range(n_curves), 2))
        
        # Calculate memory size
        memory_mb = curves.nbytes / (1024 * 1024)
        print(f"Curves array size: {memory_mb:.2f} MB")
        print(f"Number of bands: {len(indices)}")
        
        for workers in workers_list:
            print(f"\n{'-'*70}")
            print(f"Testing with {workers} workers")
            print(f"{'-'*70}")
            
            # Test standard fork implementation
            print("\n1. Standard Fork (with pickle serialization):")
            fork_times = []
            for i in range(3):
                start = time.perf_counter()
                depths_fork = curve_banddepths(curves, indices=indices, workers=workers)
                elapsed = time.perf_counter() - start
                fork_times.append(elapsed)
                print(f"   Run {i+1}: {elapsed:.4f}s")
            
            fork_mean = np.mean(fork_times)
            fork_std = np.std(fork_times)
            
            # Test shared memory implementation
            print("\n2. Shared Memory (zero-copy):")
            shm_times = []
            for i in range(3):
                start = time.perf_counter()
                depths_shm = curve_banddepths_shared_memory(curves, indices=indices, workers=workers)
                elapsed = time.perf_counter() - start
                shm_times.append(elapsed)
                print(f"   Run {i+1}: {elapsed:.4f}s")
            
            shm_mean = np.mean(shm_times)
            shm_std = np.std(shm_times)
            
            # Verify consistency
            max_diff = np.max(np.abs(depths_fork - depths_shm))
            consistent = max_diff < 1e-10
            
            # Calculate improvement
            improvement = ((fork_mean - shm_mean) / fork_mean) * 100
            
            # Summary
            print(f"\n{'─'*70}")
            print("RESULTS:")
            print(f"{'─'*70}")
            print(f"Fork mean time:        {fork_mean:.4f}s (±{fork_std:.4f}s)")
            print(f"Shared memory time:    {shm_mean:.4f}s (±{shm_std:.4f}s)")
            print(f"Improvement:           {improvement:+.1f}%")
            print(f"Consistent results:    {'✓ Yes' if consistent else '✗ No'} (max diff: {max_diff:.2e})")
            
            if improvement > 5:
                print(f"→ Shared memory is {improvement:.1f}% FASTER")
            elif improvement < -5:
                print(f"→ Shared memory is {abs(improvement):.1f}% SLOWER (overhead dominates)")
            else:
                print(f"→ No significant difference (within measurement error)")


def analyze_overhead():
    """
    Analyze the overhead components of shared memory.
    """
    print("\n" + "="*70)
    print("OVERHEAD ANALYSIS")
    print("="*70)
    
    sizes = [
        (30, 50, 2),   # Small: ~24 KB
        (100, 100, 2), # Medium: ~160 KB
        (100, 100, 3), # Large: ~240 KB
    ]
    
    for n_curves, n_steps, n_dims in sizes:
        curves = generate_test_curves(n_curves, n_steps, n_dims)
        size_mb = curves.nbytes / (1024 * 1024)
        
        print(f"\nArray size: {n_curves}×{n_steps}×{n_dims} = {size_mb:.3f} MB")
        
        # Test shared memory creation overhead
        times = []
        for _ in range(10):
            start = time.perf_counter()
            shm = shared_memory.SharedMemory(create=True, size=curves.nbytes)
            shared_curves = np.ndarray(curves.shape, dtype=curves.dtype, buffer=shm.buf)
            shared_curves[:] = curves[:]
            shm.close()
            shm.unlink()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        overhead_ms = np.mean(times) * 1000
        print(f"  Shared memory setup+copy+cleanup: {overhead_ms:.2f}ms (±{np.std(times)*1000:.2f}ms)")
        
        # Test pickle overhead (approximate)
        import pickle
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = pickle.dumps(curves)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        pickle_ms = np.mean(times) * 1000
        print(f"  Pickle serialization (1 worker):  {pickle_ms:.2f}ms (±{np.std(times)*1000:.2f}ms)")
        print(f"  Pickle for 8 workers (estimated):  {pickle_ms * 8:.2f}ms")
        
        if overhead_ms < pickle_ms * 4:
            print(f"  → Shared memory saves ~{pickle_ms * 8 - overhead_ms:.1f}ms with 8 workers")
        else:
            print(f"  → Shared memory overhead dominates (not worth it)")


if __name__ == "__main__":
    print("\nAnalyzing shared_memory potential for curve boxplot acceleration...\n")
    
    # First analyze overheads
    analyze_overhead()
    
    # Then run full benchmark
    print("\n" + "="*70)
    print("FULL BENCHMARK")
    print("="*70)
    benchmark_comparison()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The effectiveness of shared_memory depends on:

1. Array Size vs Computation Time:
   - Small arrays (<100 KB): Pickle overhead negligible, shared_memory overhead dominates
   - Large arrays (>1 MB): Pickle overhead significant, shared_memory can help
   
2. Computation Intensity:
   - Fast computation: Serialization overhead matters
   - Slow computation (ConvexHull): Computation dominates, serialization irrelevant
   
3. Number of Workers:
   - More workers = more pickle copies = more benefit from shared_memory

For curve boxplot:
- Typical arrays are small-medium (30-100 curves × 50-100 steps = 24-240 KB)
- ConvexHull is computationally expensive (dominates runtime)
- Pickle overhead is likely <5% of total time

EXPECTED: Shared memory provides minimal improvement (<5%) for typical use cases.
""")
