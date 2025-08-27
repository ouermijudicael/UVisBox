# Uncertainty Tube Benchmarking Script

This script provides a benchmarking framework for evaluating the performance of uncertainty tube generation and mesh construction in the UVisBox package. It is designed to help users determine the optimal parallelization settings (`n_jobs`) for their specific use case and hardware.

## Overview

- **Trajectory Generation:** Uses synthetic or real data to create ensemble trajectories for uncertainty visualization.
- **Uncertainty Tube Construction:** Computes cross-sectional uncertainty profiles along each trajectory.
- **Mesh Generation:** Builds a triangle mesh representing the uncertainty tube for visualization or analysis.
- **Benchmarking:** Measures execution time for both sequential and parallel processing modes.

## Parallelization

The script supports parallel processing using either `joblib` or Python's built-in `multiprocessing` (with the `'fork'` method for Mac compatibility). Users can specify the number of parallel jobs (`n_jobs`) for both tube generation and meshing functions.

- **Sequential Mode:** Set `n_jobs=1` (default).
- **Parallel Mode:** Set `n_jobs` to the desired number of worker processes (e.g., `n_jobs=4` or `n_jobs=-1` for all available CPUs).

## Usage Instructions

1. **Edit Parameters:** Adjust the number of seeds, time steps, and other parameters to match your typical use case.
2. **Run the Benchmark:** Execute the script to measure performance for different `n_jobs` values.
3. **Analyze Results:** Review the timing and speedup results to determine the best parallelization setting for your workflow.
4. **Set `n_jobs` Appropriately:** Use the chosen `n_jobs` value when calling `generate_uncertainty_tube` and `generate_tube_mesh` in your production code.

## Recommendations

- **Benchmark with Realistic Data:** Run the benchmark using data sizes and configurations similar to your actual analysis tasks.
- **Consider System Resources:** Higher `n_jobs` values may not always yield better performance due to memory and CPU constraints.
- **Mac Compatibility:** The script uses the `'fork'` method for multiprocessing, which is recommended for Mac systems.

## Example

```python
# Run benchmark with different parallel settings
superellipse_results, meshing_results = benchmark_uncertainty_tube(n_seeds=300, n_steps=100, n_jobs_list=[1,2,4,multiprocessing.cpu_count()], repeats=3)
# Plot results
plot_results(superellipse_results,meshing_results)
```
```python
# Use the optimal n_jobs value in your workflow
vertices, faces, mean_paths = generate_tube_mesh(trajectories, cross_sections, n_jobs=4)
```
### Note
Benchmarking is essential for optimizing performance. Results may vary depending on your hardware, data size, and system load. Always test with your typical use case before setting n_jobs for production runs.