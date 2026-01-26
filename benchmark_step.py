#!/usr/bin/env python3
"""Detailed step breakdown benchmark"""
import numpy as np
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

from integrated_navigator import IntegratedNavigator, IntegratedNavConfig

print("=" * 60)
print("Detailed Step Breakdown")
print("=" * 60)

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

# Warmup - let room estimator converge
print("Warming up (letting room estimator converge)...")
for _ in range(20):
    nav.step()

print("\nBenchmarking individual components:\n")

# Now benchmark each component
room_times = []
obstacle_times = []
control_times = []
total_times = []

for i in range(30):
    t_total = time.time()

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

    # Update estimated pos
    if nav.config.use_estimated_pose and 'pose' in perception_result:
        nav.estimated_pos = perception_result['pose'][:2].copy()

    # Obstacle beliefs update
    t0 = time.time()
    if nav.room_estimator.is_converged() and len(nav.true_obstacles) > 0:
        nav._update_obstacle_beliefs()
        nav._validate_committed_trajectory()
    obstacle_times.append(time.time() - t0)

    # Control
    t0 = time.time()
    action, info = nav.controller.compute_action(
        robot_pos=nav.robot_pos,
        robot_theta=nav.robot_theta,
        goal=nav.goal,
        obstacles=nav.true_obstacles,
    )
    control_times.append(time.time() - t0)

    # Update state
    nav.robot_pos = nav.robot_pos + action
    nav._last_action = action.copy()
    nav.steps += 1

    total_times.append(time.time() - t_total)

print(f"Room estimation:     {np.mean(room_times)*1000:6.2f} ms ({100*np.mean(room_times)/np.mean(total_times):5.1f}%)")
print(f"Obstacle beliefs:    {np.mean(obstacle_times)*1000:6.2f} ms ({100*np.mean(obstacle_times)/np.mean(total_times):5.1f}%)")
print(f"Control:             {np.mean(control_times)*1000:6.2f} ms ({100*np.mean(control_times)/np.mean(total_times):5.1f}%)")
print(f"Total:               {np.mean(total_times)*1000:6.2f} ms")
print()
print(f"Target: < 50 ms for 20 Hz")
print("=" * 60)

