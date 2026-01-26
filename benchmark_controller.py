#!/usr/bin/env python3
"""Detailed controller benchmark"""
import numpy as np
import time
import sys
import torch

sys.stdout.reconfigure(line_buffering=True)

print("=" * 60)
print("Controller Detailed Benchmark")
print("=" * 60)

from unified_controller import UnifiedController, UnifiedControllerConfig, GPU_AVAILABLE

print(f"GPU Available: {GPU_AVAILABLE}")
if GPU_AVAILABLE:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

robot_pos = np.array([100.0, 100.0])
goal = np.array([600.0, 300.0])
obstacles = [(350.0, 300.0, 60.0)]

for mode in ['gpu_batched', 'sequential']:
    print(f"\n--- Mode: {mode} ---")

    config = UnifiedControllerConfig(execution_mode=mode, num_seeds=6)
    controller = UnifiedController(config)
    print(f"Device: {controller.device}")

    # Warmup
    for _ in range(5):
        controller.compute_action(robot_pos, 0.0, goal, obstacles)

    # Reset position
    pos = robot_pos.copy()

    # Benchmark
    times = []
    for i in range(20):
        t0 = time.time()
        action, info = controller.compute_action(pos, 0.0, goal, obstacles)
        elapsed = time.time() - t0
        times.append(elapsed)
        pos = pos + action * 0.5

        if i < 5:
            print(f"  Step {i}: {elapsed*1000:.1f} ms, horizon={info.get('horizon', -1)}")

    print(f"  Avg: {np.mean(times)*1000:.2f} ms")
    print(f"  Min: {np.min(times)*1000:.2f} ms")
    print(f"  Max: {np.max(times)*1000:.2f} ms")

print("\n" + "=" * 60)

