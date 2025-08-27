import numpy as np
import time
import matplotlib.pyplot as plt

from uvisbox.Datasets import flowmap_3d
from uvisbox.Interpolations import linear_interpolate
from uvisbox.UncertaintyTube import generate_uncertainty_tube, generate_tube_mesh

def benchmark_uncertainty_tube(n_seeds=100, n_steps=100, n_jobs_list=[1, 4, 12], repeats=3):
    """
    Benchmark the performance of generate_tube_mesh with different n_jobs values.
    
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
    
    # Generate cross sections (do this once, outside the timing loop)
    print("Generating cross sections...")
    cross_sections, _ = generate_uncertainty_tube(trajectories, None, 16, e_proj=0.5)
    
    # Benchmark each n_jobs value
    results = {}
    
    for n_jobs in n_jobs_list:
        times = []
        
        for i in range(repeats):
            print(f"  Run {i+1}/{repeats}...")
            
            # Time the mesh generation
            start_time = time.time()
            _, _, _ = generate_tube_mesh(trajectories, cross_sections, n_jobs=n_jobs)
            end_time = time.time()
            
            elapsed = end_time - start_time
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
    
    # Calculate speedup relative to sequential (n_jobs=1)
    sequential_time = results[1]['mean']
    speedups = [sequential_time / time for time in mean_times]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot execution times
    ax1.bar(n_jobs_values, mean_times, yerr=std_times, capsize=5)
    ax1.set_xlabel('Number of Jobs')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs. Number of Jobs')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add time labels on top of bars
    for i, time_val in enumerate(mean_times):
        ax1.text(n_jobs_values[i], time_val + std_times[i] + 0.1, 
                 f"{time_val:.2f}s", ha='center')
    
    # Plot speedup
    ax2.bar(n_jobs_values, speedups)
    ax2.set_xlabel('Number of Jobs')
    ax2.set_ylabel('Speedup (relative to sequential)')
    ax2.set_title('Speedup vs. Number of Jobs')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add speedup labels on top of bars
    for i, speedup in enumerate(speedups):
        ax2.text(n_jobs_values[i], speedup + 0.1, f"{speedup:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('tube_mesh_benchmark.png')
    plt.show()

if __name__ == "__main__":
    # Set parameters
    n_seeds = 50  # More seeds means more work to parallelize
    n_steps = 100
    n_jobs_list = [1, 4, 12]
    repeats = 3
    
    # Run benchmark
    results = benchmark_uncertainty_tube(n_seeds, n_steps, n_jobs_list, repeats)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("-------------------")
    baseline = results[1]['mean']
    for n_jobs in n_jobs_list:
        speedup = baseline / results[n_jobs]['mean']
        print(f"n_jobs={n_jobs}: {results[n_jobs]['mean']:.3f}s ± {results[n_jobs]['std']:.3f}s (Speedup: {speedup:.2f}x)")