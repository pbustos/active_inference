#!/usr/bin/env python3
"""
Test script to analyze belief evolution vs ground truth over time.

This test uses the dense scenario to check whether obstacle beliefs
become more accurate as more data is gathered during navigation
across MULTIPLE TARGETS (at least 5).

Expected behavior (per paper):
- Beliefs should converge toward true obstacle positions and radii
- Uncertainty should decrease as more observations are integrated
- High-confidence beliefs should closely match ground truth
- Beliefs should remain stable across multiple target changes
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from integrated_navigator import IntegratedNavigator, IntegratedNavConfig


@dataclass
class BeliefSnapshot:
    """Snapshot of a belief at a point in time"""
    step: int
    target_num: int
    belief_id: int
    position: np.ndarray
    radius: float
    confidence: float
    uncertainty: float
    matched_obstacle_idx: int  # Index of closest true obstacle
    position_error: float  # Distance to true obstacle center
    radius_error: float  # Absolute error in radius


@dataclass
class ObstacleMetrics:
    """Metrics for tracking a single true obstacle over time"""
    true_pos: np.ndarray
    true_radius: float
    snapshots: List[BeliefSnapshot]


@dataclass
class TargetMetrics:
    """Metrics at time of target change"""
    target_num: int
    step: int
    num_beliefs: int
    avg_pos_error: float
    avg_rad_error: float
    avg_confidence: float
    per_obstacle_errors: Dict[int, Tuple[float, float]]  # obs_idx -> (pos_err, rad_err)


def match_belief_to_obstacle(belief_pos: np.ndarray, belief_radius: float,
                              obstacles: List[Tuple[float, float, float]]) -> Tuple[int, float, float]:
    """
    Match a belief to the closest true obstacle.
    
    Returns: (obstacle_idx, position_error, radius_error)
    """
    best_idx = -1
    best_dist = float('inf')
    best_radius_err = 0.0
    
    for i, (ox, oy, r) in enumerate(obstacles):
        dist = np.linalg.norm(belief_pos - np.array([ox, oy]))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_radius_err = abs(belief_radius - r)
    
    return best_idx, best_dist, best_radius_err


def generate_random_target(nav, obstacles, margin=80):
    """Generate a random target avoiding obstacles"""
    room_w = nav.config.room_width
    room_h = nav.config.room_height
    
    for _ in range(100):
        new_goal = np.array([
            np.random.uniform(margin, room_w - margin),
            np.random.uniform(margin, room_h - margin)
        ])
        
        # Check distance from current position (at least 150px)
        if np.linalg.norm(new_goal - nav.robot_pos) < 150:
            continue
        
        # Check not inside any obstacle
        valid = True
        for ox, oy, r in obstacles:
            if np.linalg.norm(new_goal - np.array([ox, oy])) < r + 50:
                valid = False
                break
        
        if valid:
            return new_goal
    
    # Fallback
    return np.array([room_w - margin, room_h / 2])


def run_belief_evolution_test(num_targets: int = 5, max_steps_per_target: int = 500, 
                               verbose: bool = True) -> Dict:
    """
    Run navigation with dense obstacles across MULTIPLE TARGETS and track belief evolution.
    
    Args:
        num_targets: Number of targets to reach (at least 5)
        max_steps_per_target: Max steps allowed per target before giving up
        verbose: Print detailed output
    
    Returns metrics about how beliefs evolve over time and across targets.
    """
    print("=" * 70)
    print(f"BELIEF EVOLUTION TEST - Dense Scenario - {num_targets} Targets")
    print("=" * 70)
    print()
    
    # Dense scenario configuration
    obstacles = [
        (200, 250, 30),
        (300, 350, 35),
        (400, 280, 30),
        (500, 320, 35),
        (600, 270, 30)
    ]
    
    print(f"Ground truth obstacles ({len(obstacles)}):")
    for i, (x, y, r) in enumerate(obstacles):
        print(f"  [{i}] pos=({x}, {y}) radius={r}")
    print()
    
    # Create navigator
    config = IntegratedNavConfig(
        room_width=800,
        room_height=600,
        use_estimated_pose=False,  # Use ground truth pose for cleaner test
        use_estimated_obstacles=True  # Use belief-based obstacle tracking
    )
    nav = IntegratedNavigator(config)
    
    # Setup scenario
    nav.robot_pos = np.array([100.0, 300.0])
    nav.set_goal(np.array([700.0, 300.0]))
    nav.set_obstacles(obstacles)
    
    # Track metrics over time
    all_snapshots: List[BeliefSnapshot] = []
    obstacle_metrics: Dict[int, ObstacleMetrics] = {
        i: ObstacleMetrics(
            true_pos=np.array([ox, oy]),
            true_radius=r,
            snapshots=[]
        )
        for i, (ox, oy, r) in enumerate(obstacles)
    }
    
    # Target-level metrics
    target_metrics: List[TargetMetrics] = []
    
    # Tracking variables
    step_metrics = []
    total_steps = 0
    targets_reached = 0
    
    print("Running navigation...")
    print("-" * 70)
    
    while targets_reached < num_targets:
        steps_this_target = 0
        
        print(f"\n>>> TARGET {targets_reached + 1}/{num_targets}: "
              f"from ({nav.robot_pos[0]:.0f}, {nav.robot_pos[1]:.0f}) "
              f"to ({nav.goal[0]:.0f}, {nav.goal[1]:.0f})")
        
        while steps_this_target < max_steps_per_target:
            # Execute navigation step
            result = nav.step()
            total_steps += 1
            steps_this_target += 1
            
            # Collect belief snapshots
            for belief_id, belief in enumerate(nav.obstacle_beliefs):
                obs_idx, pos_err, rad_err = match_belief_to_obstacle(
                    belief.position, belief.radius, obstacles
                )
                
                snapshot = BeliefSnapshot(
                    step=total_steps,
                    target_num=targets_reached + 1,
                    belief_id=belief_id,
                    position=belief.position.copy(),
                    radius=belief.radius,
                    confidence=belief.confidence,
                    uncertainty=belief.uncertainty,
                    matched_obstacle_idx=obs_idx,
                    position_error=pos_err,
                    radius_error=rad_err
                )
                all_snapshots.append(snapshot)
                
                if obs_idx >= 0:
                    obstacle_metrics[obs_idx].snapshots.append(snapshot)
            
            # Record step-level metrics
            num_beliefs = len(nav.obstacle_beliefs)
            avg_pos_err = 0.0
            avg_rad_err = 0.0
            avg_confidence = 0.0
            
            if num_beliefs > 0:
                pos_errors = []
                rad_errors = []
                confidences = []
                
                for belief in nav.obstacle_beliefs:
                    _, pos_err, rad_err = match_belief_to_obstacle(
                        belief.position, belief.radius, obstacles
                    )
                    pos_errors.append(pos_err)
                    rad_errors.append(rad_err)
                    confidences.append(belief.confidence)
                
                avg_pos_err = np.mean(pos_errors)
                avg_rad_err = np.mean(rad_errors)
                avg_confidence = np.mean(confidences)
            
            step_metrics.append({
                'step': total_steps,
                'target_num': targets_reached + 1,
                'num_beliefs': num_beliefs,
                'avg_pos_error': avg_pos_err,
                'avg_rad_error': avg_rad_err,
                'avg_confidence': avg_confidence
            })
            
            # Check goal reached
            if result.get('goal_reached'):
                targets_reached += 1
                
                # Record target metrics
                per_obs_errors = {}
                for belief in nav.obstacle_beliefs:
                    obs_idx, pos_err, rad_err = match_belief_to_obstacle(
                        belief.position, belief.radius, obstacles
                    )
                    if obs_idx >= 0:
                        # Keep the best match for each obstacle
                        if obs_idx not in per_obs_errors or pos_err < per_obs_errors[obs_idx][0]:
                            per_obs_errors[obs_idx] = (pos_err, rad_err)
                
                tm = TargetMetrics(
                    target_num=targets_reached,
                    step=total_steps,
                    num_beliefs=num_beliefs,
                    avg_pos_error=avg_pos_err,
                    avg_rad_error=avg_rad_err,
                    avg_confidence=avg_confidence,
                    per_obstacle_errors=per_obs_errors
                )
                target_metrics.append(tm)
                
                print(f"\n✓ Target {targets_reached} reached at step {total_steps} "
                      f"(took {steps_this_target} steps)")
                print(f"  Beliefs: {num_beliefs}, "
                      f"Avg pos err: {avg_pos_err:.1f}, "
                      f"Avg rad err: {avg_rad_err:.1f}, "
                      f"Avg conf: {avg_confidence:.2f}")
                
                # Print individual belief status
                print(f"  Individual beliefs:")
                for belief in nav.obstacle_beliefs:
                    obs_idx, pos_err, rad_err = match_belief_to_obstacle(
                        belief.position, belief.radius, obstacles
                    )
                    true_r = obstacles[obs_idx][2] if obs_idx >= 0 else 0
                    print(f"    - pos=({belief.position[0]:.1f}, {belief.position[1]:.1f}) "
                          f"r={belief.radius:.1f} (true={true_r}) "
                          f"conf={belief.confidence:.2f} "
                          f"pos_err={pos_err:.1f} rad_err={rad_err:.1f}")
                
                # Generate new target if more to go
                if targets_reached < num_targets:
                    new_goal = generate_random_target(nav, obstacles)
                    nav.set_goal(new_goal)
                    
                    # Reset controller state for new target
                    if hasattr(nav, 'controller') and nav.controller is not None:
                        nav.controller.previous_trajectory = None
                        nav.controller.committed_trajectory = None
                
                break
            
            # Check collision
            if result.get('collision'):
                print(f"\n✗ Collision at step {total_steps}")
                # Generate new target and continue
                new_goal = generate_random_target(nav, obstacles)
                nav.set_goal(new_goal)
                break
        
        else:
            # Max steps reached without reaching target
            print(f"\n⚠ Max steps ({max_steps_per_target}) reached for target {targets_reached + 1}")
            targets_reached += 1  # Count as "attempted"
            
            if targets_reached < num_targets:
                new_goal = generate_random_target(nav, obstacles)
                nav.set_goal(new_goal)
    
    # Final analysis
    print()
    print("=" * 70)
    print("FINAL ANALYSIS - BELIEF EVOLUTION ACROSS TARGETS")
    print("=" * 70)
    
    # Target-by-target progression
    print("\nTarget-by-Target Progression:")
    print("-" * 70)
    print(f"{'Target':<8} {'Step':<8} {'Beliefs':<8} {'Pos Err':<10} {'Rad Err':<10} {'Conf':<8}")
    print("-" * 70)
    
    for tm in target_metrics:
        print(f"{tm.target_num:<8} {tm.step:<8} {tm.num_beliefs:<8} "
              f"{tm.avg_pos_error:<10.1f} {tm.avg_rad_error:<10.1f} {tm.avg_confidence:<8.2f}")
    
    # Per-obstacle analysis
    print("\n\nPer-Obstacle Belief Evolution:")
    print("-" * 70)
    
    for obs_idx, metrics in obstacle_metrics.items():
        true_pos = metrics.true_pos
        true_r = metrics.true_radius
        snapshots = metrics.snapshots
        
        print(f"\nObstacle {obs_idx}: true pos=({true_pos[0]:.0f}, {true_pos[1]:.0f}), true radius={true_r}")
        
        if not snapshots:
            print("  No matching beliefs detected")
            continue
        
        # Get snapshots at each target completion
        for tm in target_metrics:
            if obs_idx in tm.per_obstacle_errors:
                pos_err, rad_err = tm.per_obstacle_errors[obs_idx]
                print(f"  After target {tm.target_num}: pos_err={pos_err:.1f}, rad_err={rad_err:.1f}")
    
    # Overall summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if len(target_metrics) >= 2:
        first_tm = target_metrics[0]
        last_tm = target_metrics[-1]
        
        print(f"\nFirst target → Last target comparison:")
        print(f"  Position error: {first_tm.avg_pos_error:.1f} → {last_tm.avg_pos_error:.1f} "
              f"({'IMPROVED' if last_tm.avg_pos_error < first_tm.avg_pos_error else 'DEGRADED'})")
        print(f"  Radius error: {first_tm.avg_rad_error:.1f} → {last_tm.avg_rad_error:.1f} "
              f"({'IMPROVED' if last_tm.avg_rad_error < first_tm.avg_rad_error else 'DEGRADED'})")
        print(f"  Confidence: {first_tm.avg_confidence:.2f} → {last_tm.avg_confidence:.2f}")
        print(f"  Num beliefs: {first_tm.num_beliefs} → {last_tm.num_beliefs}")
    
    # Final beliefs summary
    print(f"\nFinal beliefs ({len(nav.obstacle_beliefs)}) after {total_steps} steps:")
    for belief in nav.obstacle_beliefs:
        obs_idx, pos_err, rad_err = match_belief_to_obstacle(
            belief.position, belief.radius, obstacles
        )
        true_r = obstacles[obs_idx][2] if obs_idx >= 0 else 0
        print(f"  pos=({belief.position[0]:.1f}, {belief.position[1]:.1f}) "
              f"r={belief.radius:.1f} (true={true_r}) "
              f"conf={belief.confidence:.2f} "
              f"pos_err={pos_err:.1f} rad_err={rad_err:.1f}")
    
    # Verdict
    print()
    print("=" * 70)
    if len(target_metrics) >= 2:
        pos_improved = target_metrics[-1].avg_pos_error <= target_metrics[0].avg_pos_error * 1.2
        rad_improved = target_metrics[-1].avg_rad_error <= target_metrics[0].avg_rad_error * 1.2
        beliefs_stable = abs(target_metrics[-1].num_beliefs - len(obstacles)) <= 2
        
        if pos_improved and rad_improved and beliefs_stable:
            print("VERDICT: ✓ PASS - Beliefs remain accurate across multiple targets")
        else:
            issues = []
            if not pos_improved:
                issues.append("position error increased")
            if not rad_improved:
                issues.append("radius error increased")
            if not beliefs_stable:
                issues.append(f"belief count unstable ({target_metrics[-1].num_beliefs} vs {len(obstacles)} true)")
            print(f"VERDICT: ✗ FAIL - Issues: {', '.join(issues)}")
    print("=" * 70)
    
    return {
        'step_metrics': step_metrics,
        'target_metrics': target_metrics,
        'obstacle_metrics': obstacle_metrics,
        'final_beliefs': nav.obstacle_beliefs,
        'obstacles': obstacles,
        'total_steps': total_steps,
        'targets_reached': targets_reached
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test belief evolution over multiple targets')
    parser.add_argument('--targets', type=int, default=5, help='Number of targets to reach')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per target')
    
    args = parser.parse_args()
    
    results = run_belief_evolution_test(num_targets=args.targets, 
                                         max_steps_per_target=args.max_steps,
                                         verbose=True)
