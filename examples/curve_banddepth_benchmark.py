"""
This example benchmarks the curve_banddepths computation performance
by varying the number of worker processes. It demonstrates the speedup
achieved through parallelization and vectorization.

The benchmark generates synthetic curves and measures computation time
for different worker counts (1, 2, 4, 6, 8, 10, 12).
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from uvisbox.Modules.CurveBoxplot.curve_boxplot_stats import curve_banddepths

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for curve generation
n_curves = 50       # Number of curves (samples)
n_steps = 150        # Number of time steps
n_dims = 2           # 2D curves (x, y)

print("=" * 70)
print("Curve Band Depth Computation Benchmark")
print("=" * 70)
print(f"Dataset: {n_curves} curves, {n_steps} time steps, {n_dims} dimensions")
print(f"Total points: {n_curves * n_steps} = {n_curves * n_steps:,}")
print("=" * 70)

# Generate base curve (smooth sine wave)
t = np.linspace(0, 4 * np.pi, n_steps)
base_curve = np.zeros((n_steps, n_dims))
base_curve[:, 0] = t  # X increases linearly with time
base_curve[:, 1] = np.sin(t) * 2  # Y follows a sine wave

# Generate random curves by adding variations to the base curve
curves = np.zeros((n_curves, n_steps, n_dims))

print("\nGenerating random curves...")
for curve_idx in range(n_curves):
    # Start with the base curve
    curve = base_curve.copy()
    
    # Add a global Y-direction shift sampled from Gaussian
    y_shift = np.random.normal(0, 0.5)
    curve[:, 1] += y_shift
    
    # Add slight randomness at each step
    for step in range(n_steps):
        step_noise = np.random.normal(0, 0.15, n_dims)
        curve[step, :] += step_noise
    
    curves[curve_idx] = curve

print(f"Generated {n_curves} curves with shape: {curves.shape}")

# Test different numbers of workers
worker_counts = [1, 2, 4, 6, 8, 10, 12, 14, 16]
computation_times = []

print("\n" + "=" * 70)
print("Running benchmark with different worker counts...")
print("=" * 70)

for workers in worker_counts:
    print(f"\nWorkers: {workers:2d} ... ", end='', flush=True)
    
    # Warm-up run (not timed)
    if workers == worker_counts[0]:
        print("[warm-up] ", end='', flush=True)
        _ = curve_banddepths(curves, workers=workers)
    
    # Timed run
    start_time = time.time()
    depths = curve_banddepths(curves, workers=workers)
    end_time = time.time()
    
    elapsed = end_time - start_time
    computation_times.append(elapsed)
    
    print(f"Time: {elapsed:.3f}s", end='')
    
    # Calculate speedup relative to single worker
    if workers == 1:
        baseline_time = elapsed
        print(" (baseline)")
    else:
        speedup = baseline_time / elapsed
        efficiency = (speedup / workers) * 100
        print(f"  |  Speedup: {speedup:.2f}x  |  Efficiency: {efficiency:.1f}%")

print("\n" + "=" * 70)
print("Benchmark Results Summary")
print("=" * 70)
print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
print("-" * 70)

for i, (workers, comp_time) in enumerate(zip(worker_counts, computation_times)):
    speedup = baseline_time / comp_time
    efficiency = (speedup / workers) * 100
    speedup_str = "1.00x" if workers == 1 else f"{speedup:.2f}x"
    efficiency_str = "100.0%" if workers == 1 else f"{efficiency:.1f}%"
    print(f"{workers:<10} {comp_time:<12.3f} {speedup_str:<12} {efficiency_str:<12}")

print("=" * 70)

# Verify results are consistent
print(f"\nDepth statistics: Min={depths.min():.4f}, Max={depths.max():.4f}, Mean={depths.mean():.4f}")
print(f"Median curve index: {np.argmax(depths)}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Computation time vs number of workers
ax1.plot(worker_counts, computation_times, 'o-', linewidth=2, markersize=8, color='#2e5f8a')
ax1.set_xlabel('Number of Workers', fontsize=12)
ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
ax1.set_title('Computation Time vs Number of Workers', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(worker_counts)

# Add value labels on points
for workers, comp_time in zip(worker_counts, computation_times):
    ax1.annotate(f'{comp_time:.2f}s', 
                xy=(workers, comp_time), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9)

# Plot 2: Speedup vs number of workers
speedups = [baseline_time / t for t in computation_times]
ideal_speedup = worker_counts  # Ideal linear speedup

ax2.plot(worker_counts, speedups, 'o-', linewidth=2, markersize=8, 
         color='#2e5f8a', label='Actual Speedup')
ax2.plot(worker_counts, ideal_speedup, '--', linewidth=2, 
         color='gray', alpha=0.5, label='Ideal Linear Speedup')
ax2.set_xlabel('Number of Workers', fontsize=12)
ax2.set_ylabel('Speedup (relative to 1 worker)', fontsize=12)
ax2.set_title('Speedup vs Number of Workers', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(worker_counts)
ax2.legend(loc='best', fontsize=10)

# Add value labels on actual speedup points
for workers, speedup in zip(worker_counts, speedups):
    ax2.annotate(f'{speedup:.2f}x', 
                xy=(workers, speedup), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9)

plt.tight_layout()
# plt.savefig("curve_banddepth_benchmark.png", dpi=150, bbox_inches='tight')
print("\nPlot saved as: curve_banddepth_benchmark.png")
plt.show()

print("\n" + "=" * 70)
print("Benchmark complete!")
print("=" * 70)
