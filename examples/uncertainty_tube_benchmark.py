import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes import uncertainty_tube_summary_statistics, uncertainty_tube_mesh # Updated imports

def benchmark_uncertainty_tube(n_seeds=100, n_steps=100, n_jobs_list=[1, 4, 12], repeats=3):
    """
    Benchmark the performance of uncertainty tube generation with different n_jobs values.
    
    Args:
        n_seeds (int): Number of seed points
        n_steps (int): Number of time steps
        n_jobs_list (list): List of n_jobs values to test
        repeats (int): Number of times to repeat each test
    
    Returns:
        dict: Timing results for each n_jobs value
    """
    print(f"Benchmarking with {n_seeds} seeds, {n_steps} steps")
    
    # Generate trajectories
    t0, t1 = 0, 5
    scale = np.arange(n_seeds)
    scale = linear_interpolate(scale, 0, n_seeds-1, 1.0, 2.0)
    xy_scale = np.ones(n_seeds)
    xy_scale[1::2] = 0.1
    
    # Generate seed points
    seeds = np.random.uniform(-1, 1, (n_seeds, 3))
    trajectories = flowmap_3d(seeds, t0, t1, n_steps, scale=scale, xy_scale=xy_scale)
    
    # Benchmark stats and mesh generation combined
    results = {}
    
    for n_jobs in n_jobs_list:
        times = []
        
        for i in range(repeats):
            print(f"  Run {i+1}/{repeats} with n_jobs={n_jobs}...")
            
            start_time = time.perf_counter_ns()
            # Stage 1: Statistics
            summary_statistics = uncertainty_tube_summary_statistics(trajectories, n_jobs=n_jobs)
            # Stage 2: Mesh Generation
            mesh_data = uncertainty_tube_mesh(summary_statistics, n_jobs=n_jobs)
            end_time = time.perf_counter_ns()
            
            elapsed = (end_time - start_time)/1e9  # Convert to seconds
            times.append(elapsed)
            print(f"  Completed in {elapsed:.3f} seconds")
        
        # Store results
        results[n_jobs] = {
            'times': times,
            'mean': np.mean(times),
            'std': np.std(times)
        }
        print(f"Average time with n_jobs={n_jobs}: {results[n_jobs]['mean']:.3f} ± {results[n_jobs]['std']:.3f} seconds")
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    n_jobs_values = list(results.keys())
    mean_times = [results[n_jobs]['mean'] for n_jobs in n_jobs_values]
    std_times = [results[n_jobs]['std'] for n_jobs in n_jobs_values]
    
    # Create figure with one subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot execution times
    ax[0].bar(n_jobs_values, mean_times, yerr=std_times, capsize=5)
    ax[0].set_xlabel('Number of Jobs')
    ax[0].set_ylabel('Execution Time (seconds)')
    ax[0].set_title('Uncertainty Tube Generation Time vs. Number of Jobs')
    ax[0].grid(True, linestyle='--', alpha=0.7)
    
    # Add time labels on top of bars
    for i, time_val in enumerate(mean_times):
        ax[0].text(n_jobs_values[i], time_val + std_times[i] + 0.1, 
                 f"{time_val:.2f}s", ha='center')
    
    # Calculate speedup relative to sequential (n_jobs=1)
    sequential_time = results[1]['mean']
    speedups = [sequential_time / time for time in mean_times]
    
    # Plot speedup
    ax[1].bar(n_jobs_values, speedups)
    ax[1].set_xlabel('Number of Jobs')
    ax[1].set_ylabel('Speedup (relative to sequential)')
    ax[1].set_title('Speedup vs. Number of Jobs')
    ax[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add speedup labels on top of bars
    for i, speedup in enumerate(speedups):
        ax[1].text(n_jobs_values[i], speedup + 0.1, f"{speedup:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('uncertainty_tube_benchmark.png')
    plt.show()

if __name__ == "__main__":
    n_seeds = 300
    n_steps = 100
    n_jobs_list = [1, 2, 4, multiprocessing.cpu_count()]
    repeats = 3

    # Run benchmark
    results = benchmark_uncertainty_tube(n_seeds=n_seeds, n_steps=n_steps, n_jobs_list=n_jobs_list, repeats=repeats)

    # Plot results
    plot_results(results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("-------------------")
    baseline = results[1]['mean']
    for n_jobs in n_jobs_list:
        speedup = baseline / results[n_jobs]['mean']
        print(f"n_jobs={n_jobs}: {results[n_jobs]['mean']:.3f}s ± {results[n_jobs]['std']:.3f}s (Speedup: {speedup:.2f}x)")
