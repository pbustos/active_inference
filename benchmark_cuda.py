#!/usr/bin/env python3
"""
CUDA Performance Benchmark for Active Inference Navigation

Analyzes GPU utilization and identifies optimization opportunities.

OPTIMIZATION SUMMARY (as of 2024):
==================================

Current Performance:
- Full step: ~32 ms (31 Hz)
- Controller (GPU): ~5 ms (optimized)
- LIDAR GPU: ~0.8 ms (4.2x faster than CPU 3.5ms)
- Overhead: ~27 ms (Room estimation + Tracking)

Applied Optimizations:
1. Increased num_seeds from 6 to 16 for better GPU utilization
2. Pre-allocated GPU buffers for common operations
3. Replaced torch.tensor(0.0, ...) with tensor.new_zeros(())
4. Batched trajectory optimization (6.6x faster than sequential)
5. GPU LIDAR simulation with vectorized ray casting (4.2x faster)

GPU Utilization:
- Memory: 17 MB / 16 GB (0.1% - heavily underutilized)
- Batch size 256 shows optimal throughput

Bottlenecks:
- Room estimation: Already on GPU but has Python overhead
- DBSCAN clustering: sklearn on CPU, could use cuML for GPU
- Obstacle tracking: Python object management

Future Optimizations:
1. torch.compile() for PyTorch 2.0+ JIT compilation
2. torch.cuda.amp for mixed precision (float16)
3. Use cuML DBSCAN for GPU-accelerated clustering
4. Fuse multiple small kernels into larger ones
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Ensure we're using GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

print(f"Running benchmarks on: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()


def benchmark_function(func, name: str, warmup: int = 5, iterations: int = 50) -> Dict:
    """Benchmark a function with warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func()

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'name': name,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def print_benchmark(result: Dict):
    """Print benchmark result."""
    print(f"  {result['name']:40s} {result['mean_ms']:8.3f} ms ± {result['std_ms']:.3f}")


# =============================================================================
# BENCHMARK 1: Tensor Creation Overhead
# =============================================================================
def benchmark_tensor_creation():
    print("=" * 60)
    print("BENCHMARK 1: Tensor Creation Overhead")
    print("=" * 60)

    results = []

    # CPU tensor then transfer
    def cpu_then_transfer():
        x = torch.tensor(np.random.randn(1000, 2), dtype=DTYPE)
        x = x.to(DEVICE)
        return x

    results.append(benchmark_function(cpu_then_transfer, "CPU tensor -> GPU transfer"))

    # Direct GPU tensor
    def direct_gpu():
        x = torch.randn(1000, 2, dtype=DTYPE, device=DEVICE)
        return x

    results.append(benchmark_function(direct_gpu, "Direct GPU tensor creation"))

    # Pre-allocated buffer
    buffer = torch.zeros(1000, 2, dtype=DTYPE, device=DEVICE)
    def preallocated():
        buffer.normal_()
        return buffer

    results.append(benchmark_function(preallocated, "Pre-allocated buffer fill"))

    for r in results:
        print_benchmark(r)
    print()

    return results


# =============================================================================
# BENCHMARK 2: Batch Size Impact
# =============================================================================
def benchmark_batch_sizes():
    print("=" * 60)
    print("BENCHMARK 2: Batch Size Impact on EFE Computation")
    print("=" * 60)

    results = []

    # Simulate EFE computation for different batch sizes
    obstacles = torch.randn(5, 3, dtype=DTYPE, device=DEVICE)  # 5 obstacles
    goal = torch.randn(2, dtype=DTYPE, device=DEVICE)

    for batch_size in [1, 4, 8, 16, 32, 64, 128, 256]:
        positions = torch.randn(batch_size, 2, dtype=DTYPE, device=DEVICE)
        actions = torch.randn(batch_size, 2, dtype=DTYPE, device=DEVICE)

        def compute_efe():
            # Goal distance
            to_goal = goal.unsqueeze(0) - positions
            G_goal = torch.norm(to_goal, dim=1)

            # Obstacle distances
            diffs = positions.unsqueeze(1) - obstacles[:, :2].unsqueeze(0)
            dist_centers = torch.norm(diffs, dim=2)
            dist_surfaces = dist_centers - obstacles[:, 2].unsqueeze(0)

            # Safety cost
            safety_margin = 20.0
            G_safety = torch.sum(torch.exp(-dist_surfaces / 10), dim=1)

            # Total
            G_total = G_goal + G_safety
            return G_total

        result = benchmark_function(compute_efe, f"Batch size {batch_size:3d}")
        result['batch_size'] = batch_size
        result['throughput'] = batch_size / result['mean_ms'] * 1000  # samples/sec
        results.append(result)

    for r in results:
        print(f"  Batch {r['batch_size']:3d}: {r['mean_ms']:8.3f} ms, throughput: {r['throughput']:,.0f} samples/sec")
    print()

    return results


# =============================================================================
# BENCHMARK 3: Trajectory Optimization Methods
# =============================================================================
def benchmark_trajectory_optimization():
    print("=" * 60)
    print("BENCHMARK 3: Trajectory Optimization Methods")
    print("=" * 60)

    results = []

    T = 20  # Trajectory length
    K = 8   # Number of seeds

    obstacles = torch.randn(5, 3, dtype=DTYPE, device=DEVICE)
    goal = torch.randn(2, dtype=DTYPE, device=DEVICE)
    pos = torch.randn(2, dtype=DTYPE, device=DEVICE)

    # Method 1: Sequential optimization
    def sequential_opt():
        best_G = float('inf')
        for k in range(K):
            trajectory = torch.randn(T, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([trajectory], lr=0.1)

            for _ in range(10):  # 10 iterations
                optimizer.zero_grad()

                # Compute G
                positions = pos + torch.cumsum(trajectory, dim=0)
                to_goal = goal - positions[-1]
                G = torch.norm(to_goal)

                G.backward()
                optimizer.step()

            if G.item() < best_G:
                best_G = G.item()
        return best_G

    results.append(benchmark_function(sequential_opt, "Sequential (K=8, iter=10)", iterations=20))

    # Method 2: Batched optimization
    def batched_opt():
        trajectories = torch.randn(K, T, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([trajectories], lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()

            # Batched G computation
            positions = pos.unsqueeze(0) + torch.cumsum(trajectories, dim=1)
            to_goal = goal.unsqueeze(0) - positions[:, -1, :]
            G = torch.norm(to_goal, dim=1)
            G_total = G.sum()

            G_total.backward()
            optimizer.step()

        return G.min().item()

    results.append(benchmark_function(batched_opt, "Batched (K=8, iter=10)", iterations=20))

    # Method 3: Fully parallel with more seeds
    def parallel_more_seeds():
        K_large = 64
        trajectories = torch.randn(K_large, T, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([trajectories], lr=0.1)

        for _ in range(5):  # Fewer iterations but more seeds
            optimizer.zero_grad()

            positions = pos.unsqueeze(0) + torch.cumsum(trajectories, dim=1)
            to_goal = goal.unsqueeze(0) - positions[:, -1, :]
            G = torch.norm(to_goal, dim=1)
            G_total = G.sum()

            G_total.backward()
            optimizer.step()

        return G.min().item()

    results.append(benchmark_function(parallel_more_seeds, "Parallel (K=64, iter=5)", iterations=20))

    for r in results:
        print_benchmark(r)
    print()

    return results


# =============================================================================
# BENCHMARK 4: Memory Transfer Overhead
# =============================================================================
def benchmark_memory_transfer():
    print("=" * 60)
    print("BENCHMARK 4: CPU-GPU Memory Transfer")
    print("=" * 60)

    results = []

    for size in [100, 1000, 10000, 100000]:
        data_cpu = np.random.randn(size, 2)

        def transfer_to_gpu():
            tensor = torch.tensor(data_cpu, dtype=DTYPE, device=DEVICE)
            return tensor

        result = benchmark_function(transfer_to_gpu, f"Transfer {size:6d} points to GPU")
        result['size'] = size
        results.append(result)

    for r in results:
        print_benchmark(r)
    print()

    return results


# =============================================================================
# BENCHMARK 5: SDF Computation (Critical Path)
# =============================================================================
def benchmark_sdf_computation():
    print("=" * 60)
    print("BENCHMARK 5: SDF Computation Performance")
    print("=" * 60)

    results = []

    # Circle SDF
    N_points = 1000
    points = torch.randn(N_points, 2, dtype=DTYPE, device=DEVICE)
    center = torch.tensor([100.0, 100.0], dtype=DTYPE, device=DEVICE)
    radius = torch.tensor(30.0, dtype=DTYPE, device=DEVICE)

    def circle_sdf():
        dist = torch.norm(points - center, dim=1)
        sdf = dist - radius
        return sdf

    results.append(benchmark_function(circle_sdf, f"Circle SDF ({N_points} points)"))

    # Rectangle SDF
    width = torch.tensor(60.0, dtype=DTYPE, device=DEVICE)
    height = torch.tensor(40.0, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.3, dtype=DTYPE, device=DEVICE)

    def rect_sdf():
        # Transform to local frame
        cos_a = torch.cos(-angle)
        sin_a = torch.sin(-angle)
        local_pts = points - center
        rotated = torch.stack([
            local_pts[:, 0] * cos_a - local_pts[:, 1] * sin_a,
            local_pts[:, 0] * sin_a + local_pts[:, 1] * cos_a
        ], dim=1)

        # Box SDF
        half_size = torch.tensor([width/2, height/2], dtype=DTYPE, device=DEVICE)
        d = torch.abs(rotated) - half_size
        outside = torch.norm(torch.clamp(d, min=0), dim=1)
        inside = torch.clamp(torch.max(d, dim=1).values, max=0)
        sdf = outside + inside
        return sdf

    results.append(benchmark_function(rect_sdf, f"Rectangle SDF ({N_points} points)"))

    # Batched SDF for multiple obstacles
    N_obs = 10
    centers = torch.randn(N_obs, 2, dtype=DTYPE, device=DEVICE) * 100
    radii = torch.rand(N_obs, dtype=DTYPE, device=DEVICE) * 30 + 20

    def batched_circle_sdf():
        # [N_points, N_obs]
        diffs = points.unsqueeze(1) - centers.unsqueeze(0)
        dists = torch.norm(diffs, dim=2)
        sdfs = dists - radii.unsqueeze(0)
        return sdfs.min(dim=1).values

    results.append(benchmark_function(batched_circle_sdf, f"Batched Circle SDF ({N_points}x{N_obs})"))

    for r in results:
        print_benchmark(r)
    print()

    return results


# =============================================================================
# BENCHMARK 6: Full Navigation Step
# =============================================================================
def benchmark_full_step():
    print("=" * 60)
    print("BENCHMARK 6: Full Navigation Step Timing")
    print("=" * 60)

    from integrated_navigator import IntegratedNavigator, IntegratedNavConfig

    config = IntegratedNavConfig()
    nav = IntegratedNavigator(config)
    nav.robot_pos = np.array([100.0, 300.0])
    nav.set_goal(np.array([700.0, 300.0]))
    nav.set_obstacles([(300, 300, 40), (500, 250, 35)])

    # Warmup
    for _ in range(10):
        nav.step()

    # Benchmark components
    times = {'total': [], 'perception': [], 'control': []}

    for _ in range(50):
        start = time.perf_counter()
        result = nav.step()
        total_time = (time.perf_counter() - start) * 1000

        times['total'].append(total_time)
        times['control'].append(result.get('elapsed_ms', 0))

    print(f"  {'Component':30s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*50}")
    for name, t in times.items():
        if t:
            print(f"  {name:30s} {np.mean(t):10.2f} ms {np.std(t):10.2f} ms")
    print()

    return times


# =============================================================================
# BENCHMARK 7: GPU Utilization Analysis
# =============================================================================
def benchmark_gpu_utilization():
    print("=" * 60)
    print("BENCHMARK 7: GPU Memory and Utilization")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available")
        return

    torch.cuda.reset_peak_memory_stats()

    # Run some operations
    from integrated_navigator import IntegratedNavigator, IntegratedNavConfig

    config = IntegratedNavConfig()
    nav = IntegratedNavigator(config)
    nav.robot_pos = np.array([100.0, 300.0])
    nav.set_goal(np.array([700.0, 300.0]))
    nav.set_obstacles([(300, 300, 40), (500, 250, 35)])

    for _ in range(20):
        nav.step()

    print(f"  Current memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  Peak memory allocated:    {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print(f"  Memory reserved:          {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    print()


# =============================================================================
# OPTIMIZATION RECOMMENDATIONS
# =============================================================================
def print_recommendations(all_results: Dict):
    print("=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    recommendations = []

    # Check batch size efficiency
    batch_results = all_results.get('batch_sizes', [])
    if batch_results:
        throughputs = [(r['batch_size'], r['throughput']) for r in batch_results]
        best_batch = max(throughputs, key=lambda x: x[1])
        if best_batch[0] > 32:
            recommendations.append(
                f"✓ Increase batch size to {best_batch[0]} for optimal GPU throughput"
            )

    # Check tensor creation
    tensor_results = all_results.get('tensor_creation', [])
    if tensor_results and len(tensor_results) >= 2:
        cpu_time = tensor_results[0]['mean_ms']
        gpu_time = tensor_results[1]['mean_ms']
        if cpu_time > gpu_time * 2:
            recommendations.append(
                "✓ Use direct GPU tensor creation instead of CPU->GPU transfer"
            )

    # Check trajectory optimization
    traj_results = all_results.get('trajectory', [])
    if traj_results and len(traj_results) >= 2:
        seq_time = traj_results[0]['mean_ms']
        batch_time = traj_results[1]['mean_ms']
        speedup = seq_time / batch_time
        if speedup > 1.5:
            recommendations.append(
                f"✓ Batched trajectory optimization is {speedup:.1f}x faster than sequential"
            )

    # Memory recommendations
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
        if peak_mem < total_mem * 0.1:
            recommendations.append(
                f"✓ GPU memory underutilized ({peak_mem:.0f}/{total_mem:.0f} MB). "
                "Consider larger batch sizes or more parallel computation."
            )

    recommendations.append("✓ Consider using torch.compile() for PyTorch 2.0+ optimizations")
    recommendations.append("✓ Use torch.cuda.amp for mixed precision if accuracy permits")

    for rec in recommendations:
        print(f"  {rec}")
    print()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CUDA PERFORMANCE BENCHMARK")
    print("=" * 60 + "\n")

    all_results = {}

    all_results['tensor_creation'] = benchmark_tensor_creation()
    all_results['batch_sizes'] = benchmark_batch_sizes()
    all_results['trajectory'] = benchmark_trajectory_optimization()
    all_results['memory_transfer'] = benchmark_memory_transfer()
    all_results['sdf'] = benchmark_sdf_computation()

    # Full step benchmark (requires imports)
    try:
        all_results['full_step'] = benchmark_full_step()
        benchmark_gpu_utilization()
    except Exception as e:
        print(f"Full step benchmark failed: {e}")

    print_recommendations(all_results)

    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

