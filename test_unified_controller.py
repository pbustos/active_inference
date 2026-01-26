"""
Test the Unified Controller integrated with navigation scenarios.

Uses multiprocessing to run scenarios in parallel for faster testing.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import sys


@dataclass
class ScenarioResult:
    """Result of a navigation scenario test"""
    name: str
    success: bool
    steps: int
    path: List[np.ndarray]
    elapsed_time: float
    min_clearance: float
    collision: bool
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[Tuple[float, float, float]]
    controller_stats: Dict[str, Any]


@dataclass
class Scenario:
    """Definition of a navigation scenario"""
    name: str
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[Tuple[float, float, float]]
    description: str = ""


def create_scenarios() -> List[Scenario]:
    """Create all test scenarios including random variations"""
    scenarios = []

    # Scenario 1: Simple navigation
    scenarios.append(Scenario(
        name="Simple navigation (no obstacles)",
        start=np.array([100.0, 100.0]),
        goal=np.array([400.0, 300.0]),
        obstacles=[],
        description="Basic goal reaching without obstacles"
    ))

    # Scenario 2: Single obstacle
    scenarios.append(Scenario(
        name="Single obstacle in path",
        start=np.array([100.0, 100.0]),
        goal=np.array([400.0, 100.0]),
        obstacles=[(250.0, 100.0, 50.0)],
        description="One obstacle directly in the path"
    ))

    # Scenario 3: Corridor
    scenarios.append(Scenario(
        name="Corridor navigation",
        start=np.array([100.0, 100.0]),
        goal=np.array([400.0, 100.0]),
        obstacles=[
            (200.0, 50.0, 30.0),
            (200.0, 150.0, 30.0),
            (300.0, 50.0, 30.0),
            (300.0, 150.0, 30.0),
        ],
        description="Navigate through a corridor formed by obstacles"
    ))

    # Scenario 4: Maze-like
    scenarios.append(Scenario(
        name="Multiple obstacles (maze-like)",
        start=np.array([50.0, 300.0]),
        goal=np.array([550.0, 300.0]),
        obstacles=[
            (150.0, 300.0, 40.0),
            (250.0, 200.0, 40.0),
            (250.0, 400.0, 40.0),
            (350.0, 300.0, 40.0),
            (450.0, 200.0, 40.0),
            (450.0, 400.0, 40.0),
        ],
        description="Navigate through a maze-like obstacle field"
    ))

    # Scenario 5: Stress test (random obstacles)
    np.random.seed(42)
    robot_pos = np.array([100.0, 100.0])
    goal = np.array([700.0, 500.0])
    obstacles = generate_random_obstacles(robot_pos, goal, 10, seed=42)

    scenarios.append(Scenario(
        name="Stress test (10 random obstacles)",
        start=robot_pos,
        goal=goal,
        obstacles=obstacles,
        description="Navigate through 10 random obstacles"
    ))

    # =========================================================================
    # Additional random scenarios to leverage parallel processing
    # =========================================================================

    # Scenarios 6-10: Random start/goal with random obstacles
    for i in range(5):
        seed = 100 + i
        np.random.seed(seed)

        start = np.array([
            np.random.uniform(50, 200),
            np.random.uniform(50, 550)
        ])
        goal = np.array([
            np.random.uniform(500, 750),
            np.random.uniform(50, 550)
        ])

        num_obstacles = np.random.randint(5, 12)
        obstacles = generate_random_obstacles(start, goal, num_obstacles, seed=seed)

        scenarios.append(Scenario(
            name=f"Random scenario {i+1} ({num_obstacles} obs)",
            start=start,
            goal=goal,
            obstacles=obstacles,
            description=f"Random start/goal with {num_obstacles} obstacles (seed={seed})"
        ))

    # Scenarios 11-15: Dense obstacle fields
    for i in range(5):
        seed = 200 + i
        np.random.seed(seed)

        start = np.array([50.0, 300.0])
        goal = np.array([750.0, 300.0])

        # More obstacles, smaller size
        obstacles = generate_random_obstacles(start, goal, 15, seed=seed,
                                               min_radius=15, max_radius=30)

        scenarios.append(Scenario(
            name=f"Dense field {i+1} (15 small obs)",
            start=start,
            goal=goal,
            obstacles=obstacles,
            description=f"Dense field with 15 small obstacles (seed={seed})"
        ))

    # Scenarios 16-20: Long distance navigation
    for i in range(5):
        seed = 300 + i
        np.random.seed(seed)

        # Diagonal traversal
        start = np.array([50.0, 50.0])
        goal = np.array([750.0, 550.0])

        obstacles = generate_random_obstacles(start, goal, 8, seed=seed)

        scenarios.append(Scenario(
            name=f"Long diagonal {i+1} (8 obs)",
            start=start,
            goal=goal,
            obstacles=obstacles,
            description=f"Long diagonal navigation with 8 obstacles (seed={seed})"
        ))

    return scenarios


def generate_random_obstacles(start: np.ndarray, goal: np.ndarray,
                               num_obstacles: int, seed: int,
                               min_radius: float = 20, max_radius: float = 40,
                               margin: float = 50) -> List[Tuple[float, float, float]]:
    """Generate random obstacles that don't overlap with start/goal"""
    np.random.seed(seed)
    obstacles = []

    # Bounds for obstacle placement
    min_x = min(start[0], goal[0]) - 50
    max_x = max(start[0], goal[0]) + 50
    min_y = min(start[1], goal[1]) - 100
    max_y = max(start[1], goal[1]) + 100

    attempts = 0
    while len(obstacles) < num_obstacles and attempts < num_obstacles * 10:
        cx = np.random.uniform(min_x + 50, max_x - 50)
        cy = np.random.uniform(max(50, min_y), min(550, max_y))
        r = np.random.uniform(min_radius, max_radius)

        # Check distance from start and goal
        if np.linalg.norm([cx - start[0], cy - start[1]]) > r + margin:
            if np.linalg.norm([cx - goal[0], cy - goal[1]]) > r + margin:
                # Check overlap with existing obstacles
                overlap = False
                for ocx, ocy, or_ in obstacles:
                    if np.linalg.norm([cx - ocx, cy - ocy]) < r + or_ + 10:
                        overlap = True
                        break
                if not overlap:
                    obstacles.append((cx, cy, r))

        attempts += 1

    return obstacles


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """
    Run a single navigation scenario.
    This function is designed to be called by multiprocessing.Pool
    """
    # Import here to avoid issues with multiprocessing
    from unified_controller import UnifiedController, UnifiedControllerConfig

    config = UnifiedControllerConfig()
    controller = UnifiedController(config)

    goal_threshold = 15.0
    max_steps = 300

    start_time = time.time()

    # Run simulation
    pos = scenario.start.copy()
    theta = 0.0
    path = [pos.copy()]
    prev_vel = np.array([0.0, 0.0])
    collision = False

    for step in range(max_steps):
        dist_to_goal = np.linalg.norm(pos - scenario.goal)
        if dist_to_goal < goal_threshold:
            elapsed = time.time() - start_time
            min_clearance = compute_min_clearance(path, scenario.obstacles)
            return ScenarioResult(
                name=scenario.name,
                success=True,
                steps=step,
                path=path,
                elapsed_time=elapsed,
                min_clearance=min_clearance,
                collision=False,
                start=scenario.start,
                goal=scenario.goal,
                obstacles=scenario.obstacles,
                controller_stats=controller.stats.copy()
            )

        action, info = controller.compute_action(
            pos, theta, scenario.goal, scenario.obstacles, prev_vel
        )

        pos = pos + action
        theta = np.arctan2(action[1], action[0])
        prev_vel = action
        path.append(pos.copy())

        # Check collision
        for cx, cy, r in scenario.obstacles:
            if np.linalg.norm(pos - np.array([cx, cy])) < r:
                collision = True
                break

        if collision:
            break

    elapsed = time.time() - start_time
    min_clearance = compute_min_clearance(path, scenario.obstacles)

    return ScenarioResult(
        name=scenario.name,
        success=False,
        steps=step + 1 if not collision else step,
        path=path,
        elapsed_time=elapsed,
        min_clearance=min_clearance,
        collision=collision,
        start=scenario.start,
        goal=scenario.goal,
        obstacles=scenario.obstacles,
        controller_stats=controller.stats.copy()
    )


def compute_min_clearance(path: List[np.ndarray],
                          obstacles: List[Tuple[float, float, float]]) -> float:
    """Compute minimum clearance from path to obstacles"""
    if not obstacles:
        return float('inf')

    min_clearance = float('inf')
    for pos in path:
        for cx, cy, r in obstacles:
            dist = np.linalg.norm(pos - np.array([cx, cy])) - r
            min_clearance = min(min_clearance, dist)
    return min_clearance


def print_result(idx: int, result: ScenarioResult):
    """Print a single scenario result"""
    print(f"\nScenario {idx + 1}: {result.name}")
    print("-" * 50)
    print(f"  Start: {result.start.astype(int)}")
    print(f"  Goal: {result.goal.astype(int)}")
    if result.obstacles:
        if len(result.obstacles) == 1:
            print(f"  Obstacle: {result.obstacles[0]}")
        else:
            print(f"  Obstacles: {len(result.obstacles)}")
    print(f"  Success: {result.success}")
    print(f"  Steps: {result.steps}")
    print(f"  Time: {result.elapsed_time:.2f}s ({result.elapsed_time/max(result.steps,1)*1000:.1f}ms/step)")
    if result.collision:
        print(f"  [COLLISION DETECTED]")
    if result.success and result.obstacles:
        print(f"    Min clearance: {result.min_clearance:.1f}px")
        print(f"  Path clear: {result.min_clearance > 0}")


def test_navigation_parallel(num_processes: int = None):
    """
    Test unified controller on navigation scenarios using multiprocessing.

    Args:
        num_processes: Number of parallel processes. Defaults to CPU count.
    """
    if num_processes is None:
        num_processes = min(cpu_count(), 5)  # At most 5 parallel scenarios

    print("=" * 70)
    print("Testing Unified Controller - Navigation Scenarios (PARALLEL)")
    print(f"Using {num_processes} processes")
    print("=" * 70)

    scenarios = create_scenarios()

    total_start = time.time()

    # Run scenarios in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_scenario, scenarios)

    total_time = time.time() - total_start

    # Print results
    for idx, result in enumerate(results):
        print_result(idx, result)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.success)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Total time (parallel): {total_time:.2f}s")

    # Estimate sequential time
    sequential_time = sum(r.elapsed_time for r in results)
    print(f"Estimated sequential time: {sequential_time:.2f}s")
    print(f"Speedup: {sequential_time/total_time:.1f}x")

    # Aggregate stats
    total_seeds = sum(r.controller_stats['seeds_evaluated'] for r in results)
    total_iters = sum(r.controller_stats['iterations_total'] for r in results)
    total_warm = sum(r.controller_stats['warm_starts_used'] for r in results)

    print()
    print("Aggregate Controller Statistics:")
    print(f"  Seeds evaluated: {total_seeds}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Warm starts used: {total_warm}")
    print("=" * 70)

    return all(r.success for r in results)


def test_navigation_sequential():
    """
    Test unified controller on navigation scenarios sequentially.
    (Original single-threaded version)
    """
    from unified_controller import UnifiedController, UnifiedControllerConfig

    print("=" * 70)
    print("Testing Unified Controller - Navigation Scenarios (SEQUENTIAL)")
    print("=" * 70)

    scenarios = create_scenarios()
    config = UnifiedControllerConfig()
    controller = UnifiedController(config)

    total_start = time.time()
    results = []

    for scenario in scenarios:
        controller.reset()

        # Run simulation
        start_time = time.time()
        pos = scenario.start.copy()
        theta = 0.0
        path = [pos.copy()]
        prev_vel = np.array([0.0, 0.0])
        collision = False
        goal_threshold = 15.0
        max_steps = 300

        for step in range(max_steps):
            dist_to_goal = np.linalg.norm(pos - scenario.goal)
            if dist_to_goal < goal_threshold:
                break

            action, info = controller.compute_action(
                pos, theta, scenario.goal, scenario.obstacles, prev_vel
            )

            pos = pos + action
            theta = np.arctan2(action[1], action[0])
            prev_vel = action
            path.append(pos.copy())

            for cx, cy, r in scenario.obstacles:
                if np.linalg.norm(pos - np.array([cx, cy])) < r:
                    collision = True
                    break
            if collision:
                break

        elapsed = time.time() - start_time
        success = np.linalg.norm(pos - scenario.goal) < goal_threshold
        min_clearance = compute_min_clearance(path, scenario.obstacles)

        result = ScenarioResult(
            name=scenario.name,
            success=success,
            steps=step,
            path=path,
            elapsed_time=elapsed,
            min_clearance=min_clearance,
            collision=collision,
            start=scenario.start,
            goal=scenario.goal,
            obstacles=scenario.obstacles,
            controller_stats=controller.stats.copy()
        )
        results.append(result)
        print_result(len(results) - 1, result)

    total_time = time.time() - total_start

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.success)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Total time (sequential): {total_time:.2f}s")

    print()
    print("Controller Statistics:")
    print(f"  Seeds evaluated: {controller.stats['seeds_evaluated']}")
    print(f"  Total iterations: {controller.stats['iterations_total']}")
    print(f"  Warm starts used: {controller.stats['warm_starts_used']}")
    print(f"  Horizon extensions: {controller.stats['horizon_extensions']}")
    print(f"  Trajectory commits: {controller.stats.get('trajectory_commits', 0)}")
    print("=" * 70)

    return all(r.success for r in results)


if __name__ == "__main__":
    # Parse command line arguments
    parallel = True
    num_processes = None

    for arg in sys.argv[1:]:
        if arg == "--sequential" or arg == "-s":
            parallel = False
        elif arg.startswith("--processes=") or arg.startswith("-p="):
            num_processes = int(arg.split("=")[1])
        elif arg == "--help" or arg == "-h":
            print("Usage: python test_unified_controller.py [options]")
            print()
            print("Options:")
            print("  --sequential, -s     Run tests sequentially (default: parallel)")
            print("  --processes=N, -p=N  Number of parallel processes (default: auto)")
            print("  --help, -h           Show this help message")
            sys.exit(0)

    if parallel:
        success = test_navigation_parallel(num_processes)
    else:
        success = test_navigation_sequential()

    sys.exit(0 if success else 1)

