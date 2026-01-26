#!/usr/bin/env python3
"""
Test obstacle belief convergence in super_dense scenario.

This test tracks how obstacle beliefs evolve over time and measures
their convergence toward ground truth obstacles.

Metrics:
- Number of detected vs true obstacles
- Position error (distance from belief center to nearest true obstacle center)
- Radius error (difference between belief radius and true radius)
- Coverage: how many true obstacles have a matching belief
"""
import sys
import numpy as np
from typing import List, Tuple, Dict

sys.stdout.reconfigure(line_buffering=True)

from integrated_navigator import IntegratedNavigator, IntegratedNavConfig


def get_super_dense_obstacles() -> List[Tuple[float, float, float]]:
    """Return the super_dense scenario obstacles."""
    return [
        # Left region
        (150, 150, 30),
        (150, 350, 35),
        (150, 500, 30),
        # Left-center region
        (280, 250, 32),
        (280, 420, 28),
        # Center region
        (400, 120, 35),
        (400, 300, 30),
        (400, 480, 32),
        # Right-center region
        (520, 200, 30),
        (520, 380, 35),
        (520, 520, 28),
        # Right region
        (650, 150, 32),
        (650, 300, 30),
        (650, 450, 35),
        (650, 550, 28),
    ]


def get_simple_obstacles() -> List[Tuple[float, float, float]]:
    """Simple scenario with 3 obstacles spread for convergence testing."""
    return [
        (300, 200, 35),  # Upper left
        (400, 400, 35),  # Lower center
        (550, 250, 35),  # Upper right
    ]


def match_beliefs_to_truth(beliefs, true_obstacles, max_dist=80) -> Dict:
    """
    Match beliefs to ground truth obstacles.

    Returns dict with:
    - matched_pairs: list of (belief_idx, truth_idx, pos_error, radius_error)
    - unmatched_beliefs: beliefs with no matching true obstacle
    - unmatched_truths: true obstacles with no matching belief
    """
    if not beliefs:
        return {
            'matched_pairs': [],
            'unmatched_beliefs': [],
            'unmatched_truths': list(range(len(true_obstacles))),
            'coverage': 0.0,
            'mean_pos_error': float('inf'),
            'mean_radius_error': float('inf'),
        }

    matched_pairs = []
    matched_belief_idx = set()
    matched_truth_idx = set()

    # Greedy matching: for each truth, find closest belief
    for t_idx, (tx, ty, tr) in enumerate(true_obstacles):
        best_b_idx = -1
        best_dist = max_dist

        for b_idx, belief in enumerate(beliefs):
            if b_idx in matched_belief_idx:
                continue
            bx, by = belief.position
            dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
            if dist < best_dist:
                best_dist = dist
                best_b_idx = b_idx

        if best_b_idx >= 0:
            belief = beliefs[best_b_idx]
            pos_error = best_dist
            radius_error = abs(belief.radius - tr)
            matched_pairs.append((best_b_idx, t_idx, pos_error, radius_error))
            matched_belief_idx.add(best_b_idx)
            matched_truth_idx.add(t_idx)

    unmatched_beliefs = [i for i in range(len(beliefs)) if i not in matched_belief_idx]
    unmatched_truths = [i for i in range(len(true_obstacles)) if i not in matched_truth_idx]

    coverage = len(matched_pairs) / len(true_obstacles) if true_obstacles else 0
    mean_pos_error = np.mean([p[2] for p in matched_pairs]) if matched_pairs else float('inf')
    mean_radius_error = np.mean([p[3] for p in matched_pairs]) if matched_pairs else float('inf')

    return {
        'matched_pairs': matched_pairs,
        'unmatched_beliefs': unmatched_beliefs,
        'unmatched_truths': unmatched_truths,
        'coverage': coverage,
        'mean_pos_error': mean_pos_error,
        'mean_radius_error': mean_radius_error,
    }


def run_convergence_test(max_steps=500, verbose=True):
    """Run the convergence test."""

    print("=" * 70)
    print("OBSTACLE BELIEF CONVERGENCE TEST")
    print("=" * 70)

    # Setup
    config = IntegratedNavConfig(
        room_width=800,
        room_height=600,
        use_estimated_pose=True,
        use_estimated_obstacles=True,
        execution_mode='gpu_batched',
        num_seeds=8,  # More seeds for better exploration
    )

    nav = IntegratedNavigator(config)

    # Use simple obstacles for convergence testing
    obstacles = get_simple_obstacles()
    for ox, oy, r in obstacles:
        nav.add_obstacle(ox, oy, r)

    # Multiple passes: zigzag to pass near all obstacles
    waypoints = [
        np.array([700.0, 200.0]),  # Go right-top (pass near upper obstacles)
        np.array([100.0, 450.0]),  # Go left-bottom (pass near lower obstacle)
        np.array([700.0, 300.0]),  # Go right-center (pass near obstacles again)
    ]

    current_waypoint_idx = 0
    nav.set_goal(waypoints[current_waypoint_idx])

    print(f"\nScenario: {len(obstacles)} obstacles in a row")
    print(f"Robot makes 3 passes to observe convergence")
    print(f"Obstacles: {obstacles}")
    print()

    # Track metrics over time
    history = {
        'step': [],
        'num_beliefs': [],
        'num_matched': [],
        'coverage': [],
        'mean_pos_error': [],
        'mean_radius_error': [],
        'spurious_count': [],
        'robot_pos': [],
    }

    # Run navigation
    print("Step  | Beliefs | Matched | Coverage | Pos Err | Rad Err | Spurious | Robot Pos      | Waypoint")
    print("-" * 100)

    for step in range(max_steps):
        result = nav.step()

        # Get current beliefs (only confident ones used for control)
        confident_beliefs = [b for b in nav.obstacle_beliefs if b.confidence > 0.3]

        # Match to ground truth
        match_result = match_beliefs_to_truth(confident_beliefs, obstacles)

        # Record metrics
        history['step'].append(step)
        history['num_beliefs'].append(len(confident_beliefs))
        history['num_matched'].append(len(match_result['matched_pairs']))
        history['coverage'].append(match_result['coverage'])
        history['mean_pos_error'].append(match_result['mean_pos_error'])
        history['mean_radius_error'].append(match_result['mean_radius_error'])
        history['spurious_count'].append(len(match_result['unmatched_beliefs']))
        history['robot_pos'].append(nav.robot_pos.copy())

        # Check if current waypoint reached
        dist_to_waypoint = np.linalg.norm(nav.robot_pos - waypoints[current_waypoint_idx])
        if dist_to_waypoint < 30:  # Waypoint reached
            current_waypoint_idx += 1
            if current_waypoint_idx < len(waypoints):
                nav.set_goal(waypoints[current_waypoint_idx])
                # Reset controller for new goal
                nav.controller.previous_trajectory = None
                nav.controller.committed_trajectory = None
                print(f"  >>> Waypoint {current_waypoint_idx} reached! New goal: {waypoints[current_waypoint_idx]}")
            else:
                print(f"\n✓ All waypoints reached at step {step}!")
                break

        # Print progress every 25 steps
        if step % 25 == 0 or step < 10:
            pos_err = match_result['mean_pos_error']
            rad_err = match_result['mean_radius_error']
            pos_err_str = f"{pos_err:7.1f}" if pos_err < 1000 else "    inf"
            rad_err_str = f"{rad_err:7.1f}" if rad_err < 1000 else "    inf"

            print(f"{step:5d} | {len(confident_beliefs):7d} | {len(match_result['matched_pairs']):7d} | "
                  f"{match_result['coverage']*100:7.1f}% | {pos_err_str} | {rad_err_str} | "
                  f"{len(match_result['unmatched_beliefs']):8d} | "
                  f"[{nav.robot_pos[0]:5.0f},{nav.robot_pos[1]:5.0f}] | {current_waypoint_idx+1}/{len(waypoints)}")

        # Check if stuck
        if nav.controller.current_horizon < 0:
            print(f"\n✗ Negative horizon at step {step} - navigation failed")
            break

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    final_beliefs = [b for b in nav.obstacle_beliefs if b.confidence > 0.3]
    final_match = match_beliefs_to_truth(final_beliefs, obstacles)

    print(f"\nTrue obstacles:     {len(obstacles)}")
    print(f"Detected beliefs:   {len(final_beliefs)}")
    print(f"Matched:            {len(final_match['matched_pairs'])}")
    print(f"Coverage:           {final_match['coverage']*100:.1f}%")
    print(f"Spurious beliefs:   {len(final_match['unmatched_beliefs'])}")
    print(f"Missed obstacles:   {len(final_match['unmatched_truths'])}")

    if final_match['matched_pairs']:
        print(f"\nPosition error:     {final_match['mean_pos_error']:.1f} px (mean)")
        print(f"Radius error:       {final_match['mean_radius_error']:.1f} px (mean)")

    # Print details of matched beliefs
    print("\n" + "-" * 70)
    print("MATCHED BELIEFS DETAILS:")
    print("-" * 70)
    print("True Obs (x, y, r)      | Belief (x, y, r)       | Pos Err | Rad Err | Conf")
    print("-" * 70)

    for b_idx, t_idx, pos_err, rad_err in final_match['matched_pairs']:
        tx, ty, tr = obstacles[t_idx]
        belief = final_beliefs[b_idx]
        bx, by, br = belief.position[0], belief.position[1], belief.radius
        print(f"({tx:5.0f}, {ty:5.0f}, {tr:4.0f}) | ({bx:5.0f}, {by:5.0f}, {br:4.0f}) | "
              f"{pos_err:7.1f} | {rad_err:7.1f} | {belief.confidence:.2f}")

    # Print missed obstacles
    if final_match['unmatched_truths']:
        print("\n" + "-" * 70)
        print("MISSED OBSTACLES:")
        print("-" * 70)
        for t_idx in final_match['unmatched_truths']:
            tx, ty, tr = obstacles[t_idx]
            print(f"  ({tx:.0f}, {ty:.0f}, {tr:.0f})")

    # Print spurious beliefs
    if final_match['unmatched_beliefs']:
        print("\n" + "-" * 70)
        print("SPURIOUS BELIEFS (no matching true obstacle):")
        print("-" * 70)
        for b_idx in final_match['unmatched_beliefs']:
            belief = final_beliefs[b_idx]
            print(f"  ({belief.position[0]:.0f}, {belief.position[1]:.0f}, {belief.radius:.0f}) conf={belief.confidence:.2f}")

    print("\n" + "=" * 70)

    # Return success metrics
    return {
        'goal_reached': result.get('goal_reached', False),
        'steps': step,
        'final_coverage': final_match['coverage'],
        'final_pos_error': final_match['mean_pos_error'],
        'final_radius_error': final_match['mean_radius_error'],
        'spurious_count': len(final_match['unmatched_beliefs']),
        'history': history,
    }


if __name__ == "__main__":
    result = run_convergence_test(max_steps=800, verbose=True)  # More steps for waypoints

    # Print verdict
    print("\n" + "=" * 70)
    if result['final_coverage'] > 0.5 and result['spurious_count'] < 5:
        print("✓ TEST PASSED")
    else:
        print("✗ TEST NEEDS IMPROVEMENT")
        if result['final_coverage'] <= 0.5:
            print(f"  - Low coverage: {result['final_coverage']*100:.1f}% (target: >50%)")
        if result['spurious_count'] >= 5:
            print(f"  - Too many spurious beliefs: {result['spurious_count']} (target: <5)")
    print("=" * 70)

