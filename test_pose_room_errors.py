"""
Test: Room Estimation and Navigation with Initial Pose Errors

This test evaluates how the integrated navigation system performs when:
1. Initial pose estimate has error (distance from ground truth)
2. Room shape estimate has error

Key insight from Active Inference:
- Higher pose uncertainty → Higher G_ambiguity → More cautious control
- The system should still navigate safely even with estimation errors
"""

import numpy as np
from typing import List, Dict, Tuple
import time
from integrated_navigator import IntegratedNavigator, IntegratedNavConfig
from concept_room import RoomPoseEstimatorV2, RoomBelief


def test_initial_pose_errors():
    """
    Test navigation with different initial pose errors.

    We manually inject pose estimation errors and see how the system copes.
    """
    print("=" * 70)
    print("Test: Navigation with Initial Pose Errors")
    print("=" * 70)
    print()

    # Test configuration
    config = IntegratedNavConfig(
        room_width=800,
        room_height=600,
        robot_radius=15.0,
        use_estimated_pose=True
    )

    # Different initial pose errors to test (in pixels)
    pose_errors = [0, 10, 25, 50, 100]

    # Simple navigation scenario - use smaller obstacle for reliability
    start_pos = np.array([100.0, 100.0])
    goal_pos = np.array([500.0, 300.0])
    obstacles = [(300.0, 200.0, 40.0)]  # Smaller obstacle

    results = []

    print(f"Scenario: Start={start_pos}, Goal={goal_pos}")
    print(f"Obstacle: {obstacles[0]}")
    print()
    print(f"{'Pose Error':>12} | {'Success':>8} | {'Steps':>6} | {'Clearance':>10} | {'Final Dist':>10}")
    print("-" * 60)

    for pose_error in pose_errors:
        nav = IntegratedNavigator(config)

        # Set true robot position
        nav.robot_pos = start_pos.copy()
        nav.robot_theta = 0.0

        # Inject initial pose error (random direction)
        if pose_error > 0:
            error_angle = np.random.uniform(0, 2 * np.pi)
            error_vec = pose_error * np.array([np.cos(error_angle), np.sin(error_angle)])
            nav.estimated_pos = start_pos + error_vec

            # Also inflate pose covariance to reflect uncertainty
            nav.pose_covariance = np.eye(2) * (pose_error ** 2 / 4)
        else:
            nav.estimated_pos = start_pos.copy()
            nav.pose_covariance = np.eye(2) * 1.0

        # Set goal and obstacles
        nav.set_goal(goal_pos)
        nav.set_obstacles(obstacles)

        # Force room estimator to tracking phase (skip detection)
        nav.room_estimator.phase = "tracking"
        nav.room_estimator.belief = RoomBelief(
            x=nav.estimated_pos[0],
            y=nav.estimated_pos[1],
            theta=0.0,
            width=config.room_width,
            length=config.room_height
        )
        nav.room_estimator._last_true_pos = nav.robot_pos.copy()
        nav.room_estimator._last_true_theta = nav.robot_theta

        # Run navigation
        result = nav.run_to_goal(max_steps=400, verbose=False)

        clearance = result.get('min_clearance', float('inf'))
        final_dist = result.get('final_dist', np.linalg.norm(nav.robot_pos - goal_pos))

        results.append({
            'pose_error': pose_error,
            'success': result['success'],
            'steps': result['steps'],
            'clearance': clearance,
            'final_dist': final_dist
        })

        status = "✓" if result['success'] else "✗"
        clearance_str = f"{clearance:.1f}px" if clearance < float('inf') else "N/A"
        final_str = f"{final_dist:.1f}px" if not result['success'] else "-"

        print(f"{pose_error:>10}px | {status:>8} | {result['steps']:>6} | {clearance_str:>10} | {final_str:>10}")

    print()

    # Summary
    successes = sum(1 for r in results if r['success'])
    print(f"Summary: {successes}/{len(results)} scenarios succeeded")

    return results


def test_room_shape_errors():
    """
    Test navigation with errors in room shape estimation.

    The room estimator may have wrong room dimensions, which affects
    the pose estimation and thus the control.
    """
    print()
    print("=" * 70)
    print("Test: Navigation with Room Shape Errors")
    print("=" * 70)
    print()

    config = IntegratedNavConfig(
        room_width=800,
        room_height=600,
        robot_radius=15.0,
        use_estimated_pose=True
    )

    # Different room size errors (percentage of true size)
    room_errors = [0, 10, 25, 50]  # percentage error

    start_pos = np.array([100.0, 100.0])
    goal_pos = np.array([500.0, 300.0])
    obstacles = [(300.0, 200.0, 35.0)]  # Smaller obstacle

    results = []

    print(f"True room: {config.room_width}x{config.room_height}")
    print(f"Scenario: Start={start_pos}, Goal={goal_pos}")
    print()
    print(f"{'Room Error':>12} | {'Est. Room':>15} | {'Success':>8} | {'Steps':>6} | {'Clearance':>10}")
    print("-" * 70)

    for room_error_pct in room_errors:
        nav = IntegratedNavigator(config)

        # Set true robot position
        nav.robot_pos = start_pos.copy()
        nav.robot_theta = 0.0
        nav.estimated_pos = start_pos.copy()

        # Calculate erroneous room dimensions
        error_factor = 1.0 + room_error_pct / 100.0
        est_width = config.room_width * error_factor
        est_height = config.room_height * error_factor

        # Force room estimator with wrong room dimensions
        nav.room_estimator.phase = "tracking"
        nav.room_estimator.belief = RoomBelief(
            x=start_pos[0],
            y=start_pos[1],
            theta=0.0,
            width=est_width,
            length=est_height
        )
        nav.room_estimator._last_true_pos = nav.robot_pos.copy()
        nav.room_estimator._last_true_theta = nav.robot_theta

        # Initial covariance reflects uncertainty
        nav.pose_covariance = np.eye(2) * (room_error_pct + 1) ** 2

        # Set goal and obstacles
        nav.set_goal(goal_pos)
        nav.set_obstacles(obstacles)

        # Run navigation
        result = nav.run_to_goal(max_steps=400, verbose=False)

        clearance = result.get('min_clearance', float('inf'))

        results.append({
            'room_error_pct': room_error_pct,
            'est_room': (est_width, est_height),
            'success': result['success'],
            'steps': result['steps'],
            'clearance': clearance
        })

        status = "✓" if result['success'] else "✗"
        clearance_str = f"{clearance:.1f}px" if clearance < float('inf') else "N/A"
        room_str = f"{est_width:.0f}x{est_height:.0f}"

        print(f"{room_error_pct:>10}% | {room_str:>15} | {status:>8} | {result['steps']:>6} | {clearance_str:>10}")

    print()

    successes = sum(1 for r in results if r['success'])
    print(f"Summary: {successes}/{len(results)} scenarios succeeded")

    return results


def test_combined_errors():
    """
    Test with both pose and room shape errors simultaneously.
    This is the most realistic scenario.
    """
    print()
    print("=" * 70)
    print("Test: Combined Pose + Room Shape Errors")
    print("=" * 70)
    print()

    config = IntegratedNavConfig(
        room_width=800,
        room_height=600,
        robot_radius=15.0,
        use_estimated_pose=True
    )

    # Test matrix: (pose_error_px, room_error_pct)
    error_combinations = [
        (0, 0),      # Perfect
        (10, 5),     # Small errors
        (25, 10),    # Medium errors
        (50, 20),    # Large errors
        (100, 30),   # Very large errors
    ]

    start_pos = np.array([150.0, 150.0])
    goal_pos = np.array([650.0, 450.0])
    obstacles = [
        (350.0, 300.0, 50.0),
        (500.0, 200.0, 40.0)
    ]

    results = []

    print(f"True room: {config.room_width}x{config.room_height}")
    print(f"Scenario: Start={start_pos}, Goal={goal_pos}, Obstacles: {len(obstacles)}")
    print()
    print(f"{'Pose Err':>10} | {'Room Err':>10} | {'Success':>8} | {'Steps':>6} | {'Clearance':>10}")
    print("-" * 60)

    np.random.seed(42)  # Reproducibility

    for pose_error, room_error_pct in error_combinations:
        nav = IntegratedNavigator(config)

        # Set true robot position
        nav.robot_pos = start_pos.copy()
        nav.robot_theta = 0.0

        # Inject pose error
        if pose_error > 0:
            error_angle = np.random.uniform(0, 2 * np.pi)
            error_vec = pose_error * np.array([np.cos(error_angle), np.sin(error_angle)])
            nav.estimated_pos = start_pos + error_vec
        else:
            nav.estimated_pos = start_pos.copy()

        # Calculate erroneous room dimensions
        error_factor = 1.0 + room_error_pct / 100.0
        est_width = config.room_width * error_factor
        est_height = config.room_height * error_factor

        # Force room estimator
        nav.room_estimator.phase = "tracking"
        nav.room_estimator.belief = RoomBelief(
            x=nav.estimated_pos[0],
            y=nav.estimated_pos[1],
            theta=0.0,
            width=est_width,
            length=est_height
        )
        nav.room_estimator._last_true_pos = nav.robot_pos.copy()
        nav.room_estimator._last_true_theta = nav.robot_theta

        # Covariance reflects combined uncertainty
        combined_uncertainty = (pose_error ** 2 / 4) + (room_error_pct + 1) ** 2
        nav.pose_covariance = np.eye(2) * combined_uncertainty

        # Set goal and obstacles
        nav.set_goal(goal_pos)
        nav.set_obstacles(obstacles)

        # Run navigation
        result = nav.run_to_goal(max_steps=500, verbose=False)

        clearance = result.get('min_clearance', float('inf'))

        results.append({
            'pose_error': pose_error,
            'room_error_pct': room_error_pct,
            'success': result['success'],
            'steps': result['steps'],
            'clearance': clearance
        })

        status = "✓" if result['success'] else "✗"
        clearance_str = f"{clearance:.1f}px" if clearance < float('inf') else "N/A"

        print(f"{pose_error:>8}px | {room_error_pct:>8}% | {status:>8} | {result['steps']:>6} | {clearance_str:>10}")

    print()

    successes = sum(1 for r in results if r['success'])
    print(f"Summary: {successes}/{len(results)} scenarios succeeded")

    return results


def test_uncertainty_effect_on_control():
    """
    Test how pose uncertainty affects control behavior.

    According to Active Inference:
    - Higher pose uncertainty → Higher G_ambiguity → Slower, more cautious movement
    """
    print()
    print("=" * 70)
    print("Test: Pose Uncertainty Effect on Control")
    print("=" * 70)
    print()

    from unified_controller import UnifiedController, UnifiedControllerConfig

    controller = UnifiedController()

    # Simple scenario
    robot_pos = np.array([100.0, 100.0])
    goal = np.array([400.0, 100.0])
    obstacles = [(250.0, 100.0, 50.0)]

    uncertainty_levels = [1.0, 25.0, 100.0, 400.0, 1000.0]  # Variance in px²

    print(f"Scenario: Navigate from {robot_pos} to {goal} with obstacle at (250, 100)")
    print()
    print(f"{'Uncertainty':>12} | {'Action':>20} | {'Speed':>8} | {'G_total':>10}")
    print("-" * 60)

    for uncertainty in uncertainty_levels:
        controller.reset()

        # Set pose covariance
        cov = np.eye(2) * uncertainty
        controller.set_pose_covariance(cov)

        # Compute action
        action, info = controller.compute_action(
            robot_pos=robot_pos,
            robot_theta=0.0,
            goal=goal,
            obstacles=obstacles,
            pose_covariance=cov
        )

        speed = np.linalg.norm(action)

        print(f"{uncertainty:>10.0f}px² | [{action[0]:>7.2f}, {action[1]:>7.2f}] | {speed:>6.2f} | {info['G_total']:>10.0f}")

    print()
    print("Observation: Higher uncertainty should lead to more cautious (slower) actions")
    print("This is mediated by G_ambiguity = (1/2) ln|Σ_o| which grows with Σ_s")


def run_all_tests():
    """Run all tests"""
    print()
    print("*" * 70)
    print("*  INTEGRATED NAVIGATION TEST SUITE: Pose & Room Estimation Errors  *")
    print("*" * 70)
    print()

    start_time = time.time()

    # Test 1: Initial pose errors
    pose_results = test_initial_pose_errors()

    # Test 2: Room shape errors
    room_results = test_room_shape_errors()

    # Test 3: Combined errors
    combined_results = test_combined_errors()

    # Test 4: Uncertainty effect on control
    test_uncertainty_effect_on_control()

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()

    total = len(pose_results) + len(room_results) + len(combined_results)
    successes = (sum(1 for r in pose_results if r['success']) +
                 sum(1 for r in room_results if r['success']) +
                 sum(1 for r in combined_results if r['success']))

    print(f"Pose Error Tests:     {sum(1 for r in pose_results if r['success'])}/{len(pose_results)}")
    print(f"Room Error Tests:     {sum(1 for r in room_results if r['success'])}/{len(room_results)}")
    print(f"Combined Error Tests: {sum(1 for r in combined_results if r['success'])}/{len(combined_results)}")
    print(f"")
    print(f"Total: {successes}/{total} scenarios passed")
    print(f"Time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    run_all_tests()

