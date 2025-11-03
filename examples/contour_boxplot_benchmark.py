"""
Benchmark script to compare fork, joblib, and numba parallelization for contour boxplot.

This script tests:
1. Consistency of results between fork, joblib, and numba
2. Performance (speedup) comparison
3. Various ensemble sizes and worker counts

Usage:
    python contour_boxplot_benchmark.py
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import get_context
from joblib import Parallel, delayed
import sys
import os

# Add the parent directory to the path to import uvisbox
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uvisbox.Core.BandDepths.contour_banddepth import contour_banddepth

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available, numba benchmarks will be skipped")


def noop():
    """No-op function for joblib warmup."""
    pass


def create_ensemble_scalarfield(image_res=128, n_ensembles=30, sigma_min=5, sigma_max=50, seed=42):
    """
    Create an ensemble of 2D scalar fields with Gaussian blobs in the center.
    
    Args:
        image_res (int): Resolution of the image (image_res x image_res).
        n_ensembles (int): Number of ensemble members.
        sigma_min (float): Minimum sigma for Gaussian.
        sigma_max (float): Maximum sigma for Gaussian.
        seed (int): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of shape (n_ensembles, image_res, image_res).
    """
    np.random.seed(seed)
    x = np.linspace(0, image_res-1, image_res)
    y = np.linspace(0, image_res-1, image_res)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)
    ensemble = []
    
    for i in range(n_ensembles):
        sigma = np.random.uniform(sigma_min, sigma_max)
        cov = np.array([[sigma**2, 0], [0, sigma**2]])
        mu = np.array([image_res/2, image_res/2])
        inv_cov = np.linalg.inv(cov)
        diff = grid - mu
        pdf = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=-1))
        # Normalize to [-1, 1]
        pdf = 2 * (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf)) - 1
        ensemble.append(pdf)
    
    return np.array(ensemble)


def get_binary_contours(ensemble, isovalue=0.7):
    """
    Extract binary contours from ensemble at specified isovalue.
    
    Args:
        ensemble (np.ndarray): Ensemble of scalar fields.
        isovalue (float): Threshold value for contour extraction.
    
    Returns:
        np.ndarray: Binary contour data.
    """
    return ensemble >= isovalue


# Numba-accelerated functions
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_all_depths_numba(binary_data_flat, n_images, n_pixels, combination):
        """
        Numba-accelerated depth computation for all images.
        
        Args:
            binary_data_flat: Flattened binary data (n_images * n_pixels)
            n_images: Number of images
            n_pixels: Number of pixels per image
            combination: Array of (xdx, ydx) pairs
        
        Returns:
            Array of depth values
        """
        depths = np.zeros(n_images, dtype=np.int32)
        n_combinations = combination.shape[0]
        
        # Parallel over images
        for tdx in prange(n_images):
            target_start = tdx * n_pixels
            target_end = target_start + n_pixels
            target = binary_data_flat[target_start:target_end]
            depth = 0
            
            for c_idx in range(n_combinations):
                xdx = combination[c_idx, 0]
                ydx = combination[c_idx, 1]
                
                # Get images
                x_start = xdx * n_pixels
                x_end = x_start + n_pixels
                y_start = ydx * n_pixels
                y_end = y_start + n_pixels
                
                img_x = binary_data_flat[x_start:x_end]
                img_y = binary_data_flat[y_start:y_end]
                
                # Compute intersection and union
                has_intersection = False
                intersection_contained = True
                target_contained = True
                
                for p in range(n_pixels):
                    inter_val = img_x[p] and img_y[p]
                    union_val = img_x[p] or img_y[p]
                    
                    if inter_val:
                        has_intersection = True
                        if not target[p]:
                            intersection_contained = False
                    
                    if target[p] and not union_val:
                        target_contained = False
                
                if has_intersection and intersection_contained and target_contained:
                    depth += 1
            
            depths[tdx] = depth
        
        return depths


def contour_banddepth_with_numba(data, combination=None, workers=None):
    """
    Numba-accelerated version of contour_banddepth.
    
    Uses numba's parallel JIT compilation for speedup.
    The 'workers' parameter is ignored (numba uses prange automatically).
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")
    
    if isinstance(data, np.ndarray):
        if data.dtype != np.bool_:
            binary_data = data.astype(np.bool_)
        else:
            binary_data = data
    else:
        try:
            binary_data = np.array(data, dtype=np.bool_)
        except Exception as e:
            raise ValueError("Input data could not be converted to a boolean array.") from e
    
    from uvisbox.Core.BandDepths.contour_banddepth import get_combinations
    
    n_images = binary_data.shape[0]
    if combination is None:
        combination = get_combinations(n_images)
    
    # Convert combination to numpy array for numba
    combination_array = np.array(combination, dtype=np.int32)
    
    # Flatten binary data for easier indexing in numba
    binary_data_flat = binary_data.reshape(n_images, -1).ravel()
    n_pixels = binary_data.shape[1] * binary_data.shape[2]
    
    # Compute depths with numba
    depths = _compute_all_depths_numba(binary_data_flat, n_images, n_pixels, combination_array)
    
    return depths.astype(np.float64)


def benchmark_contour_banddepth_context(binary_data, context_type='fork', workers=4, repeats=3):
    """
    Benchmark contour_banddepth with a specific parallelization method.
    
    Args:
        binary_data (np.ndarray): Binary contour data.
        context_type (str): Either 'fork', 'joblib', or 'numba'.
        workers (int): Number of worker processes.
        repeats (int): Number of times to repeat the benchmark.
    
    Returns:
        dict: Results containing depths, times, and statistics.
    """
    times = []
    depths_list = []
    
    for i in range(repeats):
        # Time the execution
        start_time = time.perf_counter()
        
        # Call with specified parallelization method
        if context_type == 'joblib':
            depths = contour_banddepth_with_joblib(
                binary_data,
                workers=workers
            )
        elif context_type == 'numba':
            depths = contour_banddepth_with_numba(
                binary_data,
                workers=workers
            )
        else:
            depths = contour_banddepth_with_context(
                binary_data, 
                context_type=context_type,
                workers=workers
            )
        
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


def contour_banddepth_with_context(data, context_type='fork', combination=None, 
                                   allow_portion=False, eps=0, workers=12):
    """
    Modified version of contour_banddepth that allows specifying the multiprocessing context.
    
    This is a wrapper that modifies the context used in the parallel processing section.
    """
    if isinstance(data, np.ndarray):
        if data.dtype != np.bool_:
            binary_data = data.astype(np.bool_)
        else:
            binary_data = data
    else:
        try:
            binary_data = np.array(data, dtype=np.bool_)
        except Exception as e:
            raise ValueError("Input data could not be converted to a boolean array.") from e
    
    from uvisbox.Core.BandDepths.contour_banddepth import (
        get_combinations, 
        _compute_depth_for_image_optimized
    )
    
    n_images = binary_data.shape[0]
    if combination is None:
        combination = get_combinations(n_images)
    
    # Pre-compute cardinalities for all target images
    cardinalities = np.array([np.count_nonzero(img) for img in binary_data])
    
    if workers == 1:
        # Sequential processing - use the original function
        return contour_banddepth(data, combination, allow_portion, eps, workers)
    else:
        # Parallel processing with specified context
        ctx = get_context(context_type)
        with ctx.Pool(processes=workers) as pool:
            # Prepare arguments for each image
            args_list = [
                (tdx, binary_data[tdx], cardinalities[tdx], binary_data, combination, allow_portion, eps)
                for tdx in range(n_images)
            ]
            # Compute depths in parallel
            results = pool.map(_compute_depth_for_image_optimized, args_list)
        
        # Extract depths from (index, depth) tuples
        depths = np.zeros(n_images)
        for idx, depth in results:
            depths[idx] = depth
        return depths


def contour_banddepth_with_joblib(data, combination=None, allow_portion=False, eps=0, workers=12):
    """
    Modified version of contour_banddepth using joblib for parallelization.
    
    This uses joblib.Parallel instead of multiprocessing.Pool.
    """
    if isinstance(data, np.ndarray):
        if data.dtype != np.bool_:
            binary_data = data.astype(np.bool_)
        else:
            binary_data = data
    else:
        try:
            binary_data = np.array(data, dtype=np.bool_)
        except Exception as e:
            raise ValueError("Input data could not be converted to a boolean array.") from e
    
    from uvisbox.Core.BandDepths.contour_banddepth import (
        get_combinations, 
        _compute_depth_for_image_optimized
    )
    
    n_images = binary_data.shape[0]
    if combination is None:
        combination = get_combinations(n_images)
    
    # Pre-compute cardinalities for all target images
    cardinalities = np.array([np.count_nonzero(img) for img in binary_data])
    
    if workers == 1:
        # Sequential processing - use the original function
        return contour_banddepth(data, combination, allow_portion, eps, workers)
    else:
        # Parallel processing with joblib
        # Prepare arguments for each image
        args_list = [
            (tdx, binary_data[tdx], cardinalities[tdx], binary_data, combination, allow_portion, eps)
            for tdx in range(n_images)
        ]
        
        # Compute depths in parallel using joblib
        results = Parallel(n_jobs=workers, backend='loky')(
            delayed(_compute_depth_for_image_optimized)(args) for args in args_list
        )
        
        # Extract depths from (index, depth) tuples
        depths = np.zeros(n_images)
        for idx, depth in results:
            depths[idx] = depth
        return depths


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
    Run comprehensive benchmark comparing fork, joblib, and numba parallelization.
    """
    print("="*60)
    print("CONTOUR BOXPLOT PARALLELIZATION BENCHMARK")
    methods = ["Fork", "Joblib"]
    if NUMBA_AVAILABLE:
        methods.append("Numba")
    print(f"Comparing: {', '.join(methods)}")
    print("="*60)
    
    # Test configurations
    configs = [
        {'n_ensembles': 30, 'image_res': 128, 'workers': [1, 2, 4, 6, 8]},
        {'n_ensembles': 50, 'image_res': 128, 'workers': [1, 2, 4, 6, 8]},
        {'n_ensembles': 100, 'image_res': 128, 'workers': [1, 2, 4, 6, 8]},
    ]
    
    all_results = []
    
    for config_idx, config in enumerate(configs):
        n_ensembles = config['n_ensembles']
        image_res = config['image_res']
        workers_list = config['workers']
        
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx + 1}/{len(configs)}")
        print(f"Ensemble size: {n_ensembles}, Image resolution: {image_res}x{image_res}")
        print(f"{'='*60}")
        
        # Generate test data
        print(f"\nGenerating ensemble data...")
        ensemble = create_ensemble_scalarfield(
            image_res=image_res, 
            n_ensembles=n_ensembles,
            sigma_min=20,
            sigma_max=50,
            seed=42
        )
        binary_data = get_binary_contours(ensemble, isovalue=0.7)
        print(f"Generated {n_ensembles} binary contours of size {image_res}x{image_res}")
        
        config_results = {
            'n_ensembles': n_ensembles,
            'image_res': image_res,
            'workers_results': {}
        }
        
        for workers in workers_list:
            print(f"\n{'-'*60}")
            print(f"Testing with {workers} worker(s)")
            print(f"{'-'*60}")
            
            # Test fork context
            print(f"\nFork context:")
            fork_results = benchmark_contour_banddepth_context(
                binary_data, 
                context_type='fork',
                workers=workers,
                repeats=3
            )
            
            # Test joblib
            print(f"\nJoblib (loky backend):")
            joblib_results = benchmark_contour_banddepth_context(
                binary_data,
                context_type='joblib',
                workers=workers,
                repeats=3
            )
            
            # Test numba (if available)
            numba_results = None
            if NUMBA_AVAILABLE:
                print(f"\nNumba (JIT parallel):")
                numba_results = benchmark_contour_banddepth_context(
                    binary_data,
                    context_type='numba',
                    workers=workers,  # Note: workers parameter ignored by numba
                    repeats=3
                )
            
            # Verify consistency between all methods
            fork_vs_joblib = verify_consistency(
                fork_results['depths'],
                joblib_results['depths'],
                "Joblib"
            )
            
            is_consistent = fork_vs_joblib
            
            if NUMBA_AVAILABLE and numba_results is not None:
                fork_vs_numba = verify_consistency(
                    fork_results['depths'],
                    numba_results['depths'],
                    "Numba"
                )
                is_consistent = is_consistent and fork_vs_numba
            
            # Calculate speedup
            if workers == 1:
                baseline_fork = fork_results['mean_time']
                baseline_joblib = joblib_results['mean_time']
                if NUMBA_AVAILABLE and numba_results is not None:
                    baseline_numba = numba_results['mean_time']
            
            speedup_fork = baseline_fork / fork_results['mean_time'] if workers > 1 else 1.0
            speedup_joblib = baseline_joblib / joblib_results['mean_time'] if workers > 1 else 1.0
            
            result_dict = {
                'fork': fork_results,
                'joblib': joblib_results,
                'consistent': is_consistent,
                'speedup_fork': speedup_fork,
                'speedup_joblib': speedup_joblib
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
            print(f"Joblib mean time:     {joblib_results['mean_time']:.4f}s (±{joblib_results['std_time']:.4f}s)")
            if NUMBA_AVAILABLE and numba_results is not None:
                print(f"Numba mean time:      {numba_results['mean_time']:.4f}s (±{numba_results['std_time']:.4f}s)")
            if workers > 1:
                print(f"Fork speedup:         {speedup_fork:.2f}x")
                print(f"Joblib speedup:       {speedup_joblib:.2f}x")
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
    fig, axes = plt.subplots(2, len(results), figsize=(6*len(results), 10))
    
    if len(results) == 1:
        axes = axes.reshape(-1, 1)
    
    for col, config_results in enumerate(results):
        workers_list = sorted(config_results['workers_results'].keys())
        
        fork_times = []
        joblib_times = []
        numba_times = []
        fork_speedups = []
        joblib_speedups = []
        numba_speedups = []
        
        has_numba = False
        for workers in workers_list:
            res = config_results['workers_results'][workers]
            fork_times.append(res['fork']['mean_time'])
            joblib_times.append(res['joblib']['mean_time'])
            fork_speedups.append(res['speedup_fork'])
            joblib_speedups.append(res['speedup_joblib'])
            
            if 'numba' in res:
                has_numba = True
                numba_times.append(res['numba']['mean_time'])
                numba_speedups.append(res['speedup_numba'])
        
        # Plot execution times
        ax1 = axes[0, col]
        ax1.plot(workers_list, fork_times, 'o-', label='Fork', linewidth=2, markersize=8)
        ax1.plot(workers_list, joblib_times, '^-', label='Joblib', linewidth=2, markersize=8)
        if has_numba:
            ax1.plot(workers_list, numba_times, 'd-', label='Numba', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Workers', fontsize=12)
        ax1.set_ylabel('Execution Time (s)', fontsize=12)
        ax1.set_title(f'Execution Time\n(n={config_results["n_ensembles"]}, res={config_results["image_res"]})', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot speedups
        ax2 = axes[1, col]
        ax2.plot(workers_list, fork_speedups, 'o-', label='Fork', linewidth=2, markersize=8)
        ax2.plot(workers_list, joblib_speedups, '^-', label='Joblib', linewidth=2, markersize=8)
        if has_numba:
            ax2.plot(workers_list, numba_speedups, 'd-', label='Numba', linewidth=2, markersize=8)
        ax2.plot(workers_list, workers_list, 'k--', alpha=0.5, label='Ideal (linear)')
        ax2.set_xlabel('Number of Workers', fontsize=12)
        ax2.set_ylabel('Speedup', fontsize=12)
        ax2.set_title(f'Speedup vs Workers\n(n={config_results["n_ensembles"]}, res={config_results["image_res"]})', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'contour_boxplot_benchmark_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Benchmark plots saved to: {output_path}")
    print(f"{'='*60}")
    
    plt.show()


if __name__ == "__main__":
    # Warmup joblib - required for accurate timing
    print("Warming up joblib...")
    Parallel(n_jobs=12)(delayed(noop)() for _ in range(1))
    
    print("\nStarting contour boxplot parallelization benchmark...\n")
    results = run_benchmark_suite()
    print("\n✓ Benchmark complete!")
