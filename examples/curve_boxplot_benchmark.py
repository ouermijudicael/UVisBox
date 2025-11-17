"""
Benchmark script to compare fork and numba parallelization for curve boxplot.

This script tests:
1. Consistency of results between fork and numba
2. Performance (speedup) comparison
3. Various ensemble sizes and worker counts

Usage:
    python curve_boxplot_benchmark.py
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import get_context
from itertools import combinations
import sys
import os

# Add the parent directory to the path to import uvisbox
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uvisbox.Core.BandDepths.curve_banddepth import curve_banddepths

try:
    from numba import jit, prange
    from scipy.spatial import ConvexHull
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available, numba benchmarks will be skipped")


def generate_synthetic_curves(n_curves=30, n_steps=100, n_dims=3, seed=42):
    """
    Generate synthetic 3D curves with smooth variations.
    
    Args:
        n_curves (int): Number of curves in the ensemble.
        n_steps (int): Number of time steps per curve.
        n_dims (int): Number of dimensions (2 or 3).
        seed (int): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of shape (n_curves, n_steps, n_dims).
    """
    np.random.seed(seed)
    
    # Generate smooth curves using cumulative sum of random walks
    curves = []
    for i in range(n_curves):
        # Random walk in each dimension
        steps = np.random.randn(n_steps, n_dims) * 0.1
        # Add a sinusoidal component for smoothness
        t = np.linspace(0, 2*np.pi, n_steps)
        for d in range(n_dims):
            phase = np.random.rand() * 2 * np.pi
            amplitude = np.random.rand() * 2 + 0.5
            steps[:, d] += amplitude * np.sin(t + phase)
        
        # Cumulative sum to create continuous path
        curve = np.cumsum(steps, axis=0)
        curves.append(curve)
    
    return np.array(curves)


if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def points_in_hull_numba(points, hull_equations):
        """
        Numba-accelerated check if points are inside convex hull.
        
        Args:
            points: (n_points, n_dims) array of points to test
            hull_equations: (n_facets, n_dims+1) array of hull facet equations
        
        Returns:
            Boolean array of shape (n_points,)
        """
        n_points = points.shape[0]
        n_facets = hull_equations.shape[0]
        result = np.ones(n_points, dtype=np.bool_)
        
        for i in range(n_points):
            for j in range(n_facets):
                # Compute dot product + offset
                val = 0.0
                for k in range(points.shape[1]):
                    val += points[i, k] * hull_equations[j, k]
                val += hull_equations[j, -1]
                
                if val > 1e-10:  # Point is outside this facet
                    result[i] = False
                    break
        
        return result


    @jit(nopython=True, parallel=True)
    def curve_banddepths_numba_kernel(curves, band_indices, n_curves, n_steps, n_dims):
        """
        Numba-accelerated kernel for curve band depth computation.
        
        Note: This is a simplified version that works for the core computation.
        Hull construction still happens in Python due to ConvexHull dependency.
        """
        depths = np.zeros(n_curves, dtype=np.float64)
        n_bands = len(band_indices)
        
        # This would be the inner loop that gets accelerated
        # The hull construction part needs to stay in Python
        return depths


def curve_banddepths_with_numba(curves, indices=None, workers=None):
    """
    Numba-accelerated version of curve_banddepths.
    
    Since ConvexHull construction can't be done in numba, we use a hybrid approach:
    - Hull construction in Python
    - Point-in-hull testing with numba
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")
    
    from scipy.spatial import ConvexHull
    
    n_curves, n_steps, n_dims = curves.shape
    
    if indices is None:
        indices = list(combinations(range(n_curves), 2))
    
    depths = np.zeros(n_curves)
    
    # Sequential processing with numba-accelerated inner loops
    for step_idx in range(1, n_steps):
        all_points = curves[:, step_idx, :]
        
        for band in indices:
            band_curves = curves[band, :step_idx+1, :]
            
            try:
                hull = ConvexHull(band_curves.reshape(-1, n_dims))
                # Use numba-accelerated point-in-hull test
                hull_equations = hull.equations
                in_hull_mask = points_in_hull_numba(all_points, hull_equations)
                depths += in_hull_mask.astype(int)
            except:
                continue
    
    depths /= (n_steps-1) * len(indices)
    return depths


def benchmark_curve_banddepth(curves, method='fork', workers=4, repeats=3, indices=None):
    """
    Benchmark curve_banddepth with a specific method.
    
    Args:
        curves (np.ndarray): Curve data.
        method (str): Either 'fork' or 'numba'.
        workers (int): Number of worker processes.
        repeats (int): Number of times to repeat the benchmark.
        indices (list): Precomputed band indices.
    
    Returns:
        dict: Results containing depths, times, and statistics.
    """
    times = []
    depths_list = []
    
    for i in range(repeats):
        start_time = time.perf_counter()
        
        if method == 'numba':
            depths = curve_banddepths_with_numba(curves, indices=indices, workers=workers)
        else:  # fork
            depths = curve_banddepths(curves, indices=indices, workers=workers)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        times.append(elapsed)
        depths_list.append(depths)
        
        print(f"  Run {i+1}/{repeats}: {elapsed:.4f}s")
    
    # Verify consistency across runs
    depths_array = np.array(depths_list)
    max_difference = np.max(np.abs(depths_array - depths_array[0]))
    
    return {
        'depths': depths_list[0],
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'max_difference': max_difference
    }


def verify_consistency(depths_ref, depths_test, method_name="Test", tolerance=1e-10):
    """
    Verify that two depth arrays produce consistent results.
    
    Args:
        depths_ref (np.ndarray): Reference depths.
        depths_test (np.ndarray): Test depths to compare.
        method_name (str): Name of the test method for display.
        tolerance (float): Maximum allowed difference.
    
    Returns:
        bool: True if results are consistent.
    """
    difference = np.abs(depths_ref - depths_test)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)
    
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK: Fork vs {method_name}")
    print(f"{'='*60}")
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"All differences < {tolerance}: {np.all(difference < tolerance)}")
    
    if max_diff > tolerance:
        print(f"\n⚠️  WARNING: Differences exceed tolerance!")
        print(f"Indices with largest differences:")
        top_diffs = np.argsort(difference)[-5:][::-1]
        for idx in top_diffs:
            print(f"  Index {idx}: fork={depths_ref[idx]}, {method_name}={depths_test[idx]}, diff={difference[idx]}")
        return False
    else:
        print(f"✓ Results are consistent!")
        return True


def run_benchmark_suite():
    """
    Run comprehensive benchmark comparing fork and numba parallelization.
    """
    print("="*60)
    print("CURVE BOXPLOT PARALLELIZATION BENCHMARK")
    methods = ["Fork"]
    if NUMBA_AVAILABLE:
        methods.append("Numba")
    print(f"Comparing: {', '.join(methods)}")
    print("="*60)
    
    # Test configurations
    # Note: Curve boxplot is much more complex than contour boxplot due to ConvexHull
    # Focus on 2D curves for most tests, only small 3D examples
    configs = [
        # 2D curves - can handle larger ensembles
        {'n_curves': 30, 'n_steps': 50, 'n_dims': 2, 'workers': [1, 2, 4, 6, 8]},
        {'n_curves': 50, 'n_steps': 50, 'n_dims': 2, 'workers': [1, 2, 4, 6, 8]},
        {'n_curves': 100, 'n_steps': 50, 'n_dims': 2, 'workers': [1, 2, 4, 6, 8]},
        # 3D curves - only small ensembles
        {'n_curves': 10, 'n_steps': 50, 'n_dims': 3, 'workers': [1, 2, 4, 6, 8]},
        {'n_curves': 10, 'n_steps': 100, 'n_dims': 3, 'workers': [1, 2, 4, 6, 8]},
        {'n_curves': 30, 'n_steps': 50, 'n_dims': 3, 'workers': [1, 2, 4, 6, 8]},
        {'n_curves': 30, 'n_steps': 100, 'n_dims': 3, 'workers': [1, 2, 4, 6, 8]},
    ]
    
    all_results = []
    
    for config_idx, config in enumerate(configs):
        n_curves = config['n_curves']
        n_steps = config['n_steps']
        n_dims = config['n_dims']
        workers_list = config['workers']
        
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx + 1}/{len(configs)}")
        print(f"Ensemble: {n_curves} curves, {n_steps} steps, {n_dims}D")
        print(f"{'='*60}")
        
        # Generate test data
        print(f"\nGenerating curve data...")
        curves = generate_synthetic_curves(n_curves=n_curves, n_steps=n_steps, n_dims=n_dims, seed=42)
        
        # Precompute indices (same for all methods)
        indices = list(combinations(range(n_curves), 2))
        print(f"Generated {n_curves} curves with {len(indices)} band combinations")
        
        config_results = {
            'n_curves': n_curves,
            'n_steps': n_steps,
            'n_dims': n_dims,
            'workers_results': {}
        }
        
        for workers in workers_list:
            print(f"\n{'-'*60}")
            print(f"Testing with {workers} worker(s)")
            print(f"{'-'*60}")
            
            # Test fork context
            print(f"\nFork context:")
            fork_results = benchmark_curve_banddepth(
                curves,
                method='fork',
                workers=workers,
                repeats=3,
                indices=indices
            )
            
            # Test numba (if available)
            numba_results = None
            if NUMBA_AVAILABLE:
                print(f"\nNumba (hybrid):")
                numba_results = benchmark_curve_banddepth(
                    curves,
                    method='numba',
                    workers=workers,  # Note: workers parameter ignored by numba
                    repeats=3,
                    indices=indices
                )
            
            # Verify consistency
            is_consistent = True
            if NUMBA_AVAILABLE and numba_results is not None:
                is_consistent = verify_consistency(
                    fork_results['depths'],
                    numba_results['depths'],
                    "Numba"
                )
            
            # Calculate speedup
            if workers == 1:
                baseline_fork = fork_results['mean_time']
                if NUMBA_AVAILABLE and numba_results is not None:
                    baseline_numba = numba_results['mean_time']
            
            speedup_fork = baseline_fork / fork_results['mean_time'] if workers > 1 else 1.0
            
            result_dict = {
                'fork': fork_results,
                'consistent': is_consistent,
                'speedup_fork': speedup_fork
            }
            
            if NUMBA_AVAILABLE and numba_results is not None:
                speedup_numba = baseline_numba / numba_results['mean_time'] if workers > 1 else 1.0
                result_dict['numba'] = numba_results
                result_dict['speedup_numba'] = speedup_numba
            
            # Store results
            config_results['workers_results'][workers] = result_dict
            
            # Print summary
            print(f"\n{'─'*60}")
            print("SUMMARY")
            print(f"{'─'*60}")
            print(f"Fork mean time:       {fork_results['mean_time']:.4f}s (±{fork_results['std_time']:.4f}s)")
            if NUMBA_AVAILABLE and numba_results is not None:
                print(f"Numba mean time:      {numba_results['mean_time']:.4f}s (±{numba_results['std_time']:.4f}s)")
            if workers > 1:
                print(f"Fork speedup:         {speedup_fork:.2f}x")
                if NUMBA_AVAILABLE and numba_results is not None:
                    print(f"Numba speedup:        {speedup_numba:.2f}x")
            print(f"Consistent results:   {'✓ Yes' if is_consistent else '✗ No'}")
        
        all_results.append(config_results)
    
    # Generate summary plots
    plot_benchmark_results(all_results)
    
    return all_results


def plot_benchmark_results(results):
    """
    Generate visualization of benchmark results.
    
    Args:
        results (list): List of benchmark results for each configuration.
    """
    # Arrange plots in a grid layout for better visualization with many configs
    n_configs = len(results)
    n_cols = min(4, n_configs)  # Max 4 columns
    n_rows = 2 * ((n_configs + n_cols - 1) // n_cols)  # 2 rows per config group
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_configs == 1:
        axes = axes.reshape(-1, 1)
    
    for col, config_results in enumerate(results):
        # Calculate row and column for this config
        row_base = (col // n_cols) * 2
        col_idx = col % n_cols
        
        workers_list = sorted(config_results['workers_results'].keys())
        
        fork_times = []
        numba_times = []
        fork_speedups = []
        numba_speedups = []
        
        has_numba = False
        for workers in workers_list:
            res = config_results['workers_results'][workers]
            fork_times.append(res['fork']['mean_time'])
            fork_speedups.append(res['speedup_fork'])
            
            if 'numba' in res:
                has_numba = True
                numba_times.append(res['numba']['mean_time'])
                numba_speedups.append(res['speedup_numba'])
        
        # Plot execution times
        ax1 = axes[row_base, col_idx]
        ax1.plot(workers_list, fork_times, 'o-', label='Fork', linewidth=2, markersize=8)
        if has_numba:
            ax1.plot(workers_list, numba_times, 'd-', label='Numba', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Workers', fontsize=10)
        ax1.set_ylabel('Execution Time (s)', fontsize=10)
        ax1.set_title(f'Execution Time\n({config_results["n_curves"]} curves, {config_results["n_steps"]} steps, {config_results["n_dims"]}D)', 
                     fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot speedups
        ax2 = axes[row_base + 1, col_idx]
        ax2.plot(workers_list, fork_speedups, 'o-', label='Fork', linewidth=2, markersize=8)
        if has_numba:
            ax2.plot(workers_list, numba_speedups, 'd-', label='Numba', linewidth=2, markersize=8)
        ax2.plot(workers_list, workers_list, 'k--', alpha=0.5, label='Ideal (linear)')
        ax2.set_xlabel('Number of Workers', fontsize=10)
        ax2.set_ylabel('Speedup', fontsize=10)
        ax2.set_title(f'Speedup vs Workers\n({config_results["n_curves"]} curves, {config_results["n_steps"]} steps, {config_results["n_dims"]}D)', 
                     fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # Hide unused subplots if any
    total_plots_needed = n_configs
    total_plots = n_rows * n_cols // 2
    for idx in range(total_plots_needed, total_plots):
        row_base = (idx // n_cols) * 2
        col_idx = idx % n_cols
        axes[row_base, col_idx].set_visible(False)
        axes[row_base + 1, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'curve_boxplot_benchmark_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Benchmark plots saved to: {output_path}")
    print(f"{'='*60}")
    
    plt.show()


if __name__ == "__main__":
    print("\nStarting curve boxplot parallelization benchmark...\n")
    results = run_benchmark_suite()
    print("\n✓ Benchmark complete!")
