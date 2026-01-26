#!/usr/bin/env python3
"""Benchmark to identify performance bottlenecks"""
import numpy as np
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

print("=" * 60)
print("Performance Benchmark")
print("=" * 60)

# Test 1: Room Estimator alone
print("\n1. Room Estimator (pose tracking):")
from concept_room import RoomPoseEstimatorV2

room_est = RoomPoseEstimatorV2(room_width=800, room_height=600, use_known_room_dimensions=True)
robot_pos = np.array([100.0, 100.0])
robot_theta = 0.0

# Warmup
for _ in range(10):
    room_est.update(robot_pos, robot_theta, np.array([1.0, 0.5]), dt=1.0)
    robot_pos = robot_pos + np.array([1.0, 0.5])

# Benchmark
times = []
for _ in range(50):
    t0 = time.time()
    room_est.update(robot_pos, robot_theta, np.array([1.0, 0.5]), dt=1.0)
    times.append(time.time() - t0)
    robot_pos = robot_pos + np.array([1.0, 0.5])

print(f"   Avg: {np.mean(times)*1000:.2f} ms")
print(f"   Max: {np.max(times)*1000:.2f} ms")

# Test 2: Controller alone
print("\n2. Controller (trajectory optimization):")
from unified_controller import UnifiedController, UnifiedControllerConfig

config = UnifiedControllerConfig(execution_mode='gpu_batched', num_seeds=6)
controller = UnifiedController(config)

robot_pos = np.array([100.0, 100.0])
goal = np.array([600.0, 300.0])
obstacles = [(350.0, 300.0, 60.0)]

# Warmup
for _ in range(5):
    controller.compute_action(robot_pos, 0.0, goal, obstacles)

# Benchmark
times = []
for _ in range(50):
    t0 = time.time()
    action, info = controller.compute_action(robot_pos, 0.0, goal, obstacles)
    times.append(time.time() - t0)
    robot_pos = robot_pos + action * 0.5

print(f"   Avg: {np.mean(times)*1000:.2f} ms")
print(f"   Max: {np.max(times)*1000:.2f} ms")

# Test 3: IntegratedNavigator (full pipeline)
print("\n3. IntegratedNavigator (full pipeline):")
from integrated_navigator import IntegratedNavigator, IntegratedNavConfig

nav_config = IntegratedNavConfig(
    room_width=800,
    room_height=600,
    use_estimated_pose=True,
    use_estimated_obstacles=True,
    execution_mode='gpu_batched',
    num_seeds=6,
)
nav = IntegratedNavigator(nav_config)
nav.set_goal(np.array([600.0, 300.0]))
nav.set_obstacles([(350.0, 300.0, 60.0)])

# Warmup
for _ in range(10):
    nav.step()

# Benchmark
times = []
for _ in range(50):
    t0 = time.time()
    nav.step()
    times.append(time.time() - t0)

print(f"   Avg: {np.mean(times)*1000:.2f} ms")
print(f"   Max: {np.max(times)*1000:.2f} ms")

# Test 4: Breakdown of IntegratedNavigator step
print("\n4. Step breakdown (IntegratedNavigator):")

# Reset navigator
nav = IntegratedNavigator(nav_config)
nav.set_goal(np.array([600.0, 300.0]))
nav.set_obstacles([(350.0, 300.0, 60.0)])

# Warmup
for _ in range(10):
    nav.step()

# Detailed timing
room_times = []
control_times = []
total_times = []

for _ in range(30):
    t_total_start = time.time()

    # Room estimation
    t0 = time.time()
    last_velocity = getattr(nav, '_last_action', np.zeros(2))
    perception_result = nav.room_estimator.update(
        robot_pos=nav.robot_pos,
        robot_theta=nav.robot_theta,
        robot_velocity=last_velocity,
        dt=1.0
    )
    room_times.append(time.time() - t0)

    # Control
    t0 = time.time()
    action, control_info = nav.controller.compute_action(
        robot_pos=nav.robot_pos,
        robot_theta=nav.robot_theta,
        goal=nav.goal,
        obstacles=nav.true_obstacles,
    )
    control_times.append(time.time() - t0)

    # Update position
    nav.robot_pos = nav.robot_pos + action
    nav._last_action = action.copy()
    nav.steps += 1

    total_times.append(time.time() - t_total_start)

print(f"   Room estimation: {np.mean(room_times)*1000:.2f} ms ({100*np.mean(room_times)/np.mean(total_times):.1f}%)")
print(f"   Control:         {np.mean(control_times)*1000:.2f} ms ({100*np.mean(control_times)/np.mean(total_times):.1f}%)")
print(f"   Total:           {np.mean(total_times)*1000:.2f} ms")

print("\n" + "=" * 60)
print(f"Target for 20 Hz: < 50 ms per step")
print(f"Current:          {np.mean(total_times)*1000:.1f} ms per step")
print("=" * 60)

