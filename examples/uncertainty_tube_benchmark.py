import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing

from uvisbox.Datasets import flowmap_3d
from uvisbox.Core.Interpolations import linear_interpolate
from uvisbox.Modules.UncertaintyTube import generate_cross_sections, generate_tube_mesh

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

    superellipse_results = {}
    for n_jobs in n_jobs_list:
        times = []
        for i in range(repeats):
            start_time = time.perf_counter_ns()
            _,_ = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=n_jobs)
            end_time = time.perf_counter_ns()
            elapsed = (end_time - start_time) / 1e9  # Convert to seconds
            times.append(elapsed)
            print(f"n_jobs={n_jobs}, Run {i+1}/{repeats}: {elapsed:.3f} seconds")
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"n_jobs={n_jobs}: Mean time = {mean_time:.3f} ± {std_time:.3f} seconds")
        superellipse_results[n_jobs] = {'times': times, 'mean': mean_time, 'std': std_time}

    cross_sections, _ = generate_cross_sections(trajectories, None, 16, e_proj=0.5, n_jobs=12)

    # Benchmark each n_jobs value
    meshing_results = {}
    
    for n_jobs in n_jobs_list:
        times = []
        
        for i in range(repeats):
            print(f"  Run {i+1}/{repeats}...")
            
            # Time the mesh generation
            start_time = time.perf_counter_ns()
            _, _, _ = generate_tube_mesh(trajectories, cross_sections, n_jobs=n_jobs)
            end_time = time.perf_counter_ns()
            
            elapsed = (end_time - start_time)/1e9  # Convert to seconds
            times.append(elapsed)
            print(f"  Completed in {elapsed:.3f} seconds")
        
        # Store results
        meshing_results[n_jobs] = {
            'times': times,
            'mean': np.mean(times),
            'std': np.std(times)
        }
        print(f"Average time with n_jobs={n_jobs}: {meshing_results[n_jobs]['mean']:.3f} ± {meshing_results[n_jobs]['std']:.3f} seconds")
    
    return superellipse_results, meshing_results

def plot_results(superellipse_results,meshing_results):
    # Create figure with two subplots
    fig, ax = plt.subplots(2, 2, figsize=(12, 5))
    sax1, sax2 = ax[0]
    """Plot benchmark results."""
    superellipse_n_jobs_values = list(superellipse_results.keys())
    superellipse_mean_times = [superellipse_results[n_jobs]['mean'] for n_jobs in superellipse_n_jobs_values]
    superellipse_std_times = [superellipse_results[n_jobs]['std'] for n_jobs in superellipse_n_jobs_values]
    # Calculate speedup relative to sequential (n_jobs=1)
    superellipse_sequential_time = superellipse_results[1]['mean']
    superellipse_speedups = [superellipse_sequential_time / time for time in superellipse_mean_times]
    # Plot superellipse generation times
    sax1.bar(superellipse_n_jobs_values, superellipse_mean_times, yerr=superellipse_std_times, capsize=5)
    sax1.set_xlabel('Number of Jobs') 
    sax1.set_ylabel('Execution Time (seconds)')
    sax1.set_title('Superellipse Generation Time vs. Number of Jobs')
    sax1.grid(True, linestyle='--', alpha=0.7)
    # Add time labels on top of bars
    for i, time_val in enumerate(superellipse_mean_times):
        sax1.text(superellipse_n_jobs_values[i], time_val + superellipse_std_times[i] + 0.1, 
                 f"{time_val:.2f}s", ha='center')
    # Plot speedup
    sax2.bar(superellipse_n_jobs_values, superellipse_speedups)
    sax2.set_xlabel('Number of Jobs')
    sax2.set_ylabel('Speedup (relative to sequential)')
    sax2.set_title('Speedup vs. Number of Jobs')
    sax2.grid(True, linestyle='--', alpha=0.7)
    # Add speedup labels on top of bars
    for i, speedup in enumerate(superellipse_speedups):
        sax2.text(superellipse_n_jobs_values[i], speedup + 0.1, f"{speedup:.2f}x", ha='center') 


    meshing_n_jobs_values = list(meshing_results.keys())
    meshing_mean_times = [meshing_results[n_jobs]['mean'] for n_jobs in meshing_n_jobs_values]
    meshing_std_times = [meshing_results[n_jobs]['std'] for n_jobs in meshing_n_jobs_values]
    
    # Calculate speedup relative to sequential (n_jobs=1)
    meshing_sequential_time = meshing_results[1]['mean']
    meshing_speedups = [meshing_sequential_time / time for time in meshing_mean_times]
    
    # Plot superellipse generation times
    max1, max2 = ax[1]
    # Plot execution times
    max1.bar(meshing_n_jobs_values, meshing_mean_times, yerr=meshing_std_times, capsize=5)
    max1.set_xlabel('Number of Jobs')
    max1.set_ylabel('Execution Time (seconds)')
    max1.set_title('Meshing Time vs. Number of Jobs')
    max1.grid(True, linestyle='--', alpha=0.7)
    
    # Add time labels on top of bars
    for i, time_val in enumerate(meshing_mean_times):
        max1.text(meshing_n_jobs_values[i], time_val + meshing_std_times[i] + 0.1, 
                 f"{time_val:.2f}s", ha='center')
    
    # Plot speedup
    max2.bar(meshing_n_jobs_values, meshing_speedups)
    max2.set_xlabel('Number of Jobs')
    max2.set_ylabel('Speedup (relative to sequential)')
    max2.set_title('Speedup vs. Number of Jobs')
    max2.grid(True, linestyle='--', alpha=0.7)
    
    # Add speedup labels on top of bars
    for i, speedup in enumerate(meshing_speedups):
        max2.text(meshing_n_jobs_values[i], speedup + 0.1, f"{speedup:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('tube_mesh_benchmark.png')
    plt.show()

if __name__ == "__main__":
    n_seeds = 300
    n_steps = 100
    n_jobs_list = [1, 2, 4, multiprocessing.cpu_count()]
    repeats = 3

    # Run benchmark
    # Run benchmark with different parallel settings
    superellipse_results, meshing_results = benchmark_uncertainty_tube(n_seeds=n_seeds, n_steps=n_steps, n_jobs_list=n_jobs_list, repeats=repeats)

    # Plot results
    plot_results(superellipse_results,meshing_results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("-------------------")
    baseline = meshing_results[1]['mean']
    for n_jobs in n_jobs_list:
        speedup = baseline / meshing_results[n_jobs]['mean']
        print(f"n_jobs={n_jobs}: {meshing_results[n_jobs]['mean']:.3f}s ± {meshing_results[n_jobs]['std']:.3f}s (Speedup: {speedup:.2f}x)")