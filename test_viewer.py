#!/usr/bin/env python3
"""
Test script for the Active Inference Navigation Viewer

Runs a simple navigation demo with the viewer to test visualization.
"""

import numpy as np
import time
import sys

# Check if DearPyGui is available
try:
    import dearpygui.dearpygui as dpg
    DEARPYGUI_AVAILABLE = True
except ImportError:
    print("DearPyGui not installed. Install with: pip install dearpygui")
    DEARPYGUI_AVAILABLE = False
    sys.exit(1)

from integrated_navigator import IntegratedNavigator, IntegratedNavConfig
from viewer import NavigationViewer
from observer import NavigationEvent


class NavigationDemo:
    """Demo wrapper that connects navigator to viewer"""

    def __init__(self, scenario: str = "simple", execution_mode: str = "gpu_batched",
                 num_seeds: int = 32, use_estimated: bool = True):
        # Create navigator with configurable GPU/CPU acceleration
        # execution_mode options:
        #   'gpu_batched' - Fully vectorized GPU (fastest, requires CUDA)
        #   'parallel'    - Multi-threaded CPU
        #   'sequential'  - Single-threaded CPU (debugging)
        self.nav = IntegratedNavigator(IntegratedNavConfig(
            room_width=800,
            room_height=600,
            use_estimated_pose=use_estimated,
            use_estimated_obstacles=use_estimated,
            execution_mode=execution_mode,
            num_seeds=num_seeds,
        ))

        # Create viewer
        self.viewer = NavigationViewer(width=800, height=600, scale=1.0)

        # Setup scenario
        self._setup_scenario(scenario)

        # State - start running immediately
        self.running = True
        
        # Target tracking
        self.target_num = 1
        self.success_count = 0
        self.total_targets = 0  # 0 means unlimited

    def _setup_scenario(self, scenario: str):
        """Setup navigation scenario"""
        scenarios = {
            "simple": {
                "start": [100, 100],
                "goal": [600, 400],
                "obstacles": []
            },
            "single": {
                "start": [100, 100],
                "goal": [600, 400],
                "obstacles": [(350, 300, 40)]
            },
            "corridor": {
                "start": [100, 300],
                "goal": [700, 300],
                "obstacles": [(250, 200, 40), (250, 400, 40), (450, 200, 40), (450, 400, 40)]
            },
            "dense": {
                "start": [100, 300],
                "goal": [700, 300],
                "obstacles": [(200, 250, 30), (300, 350, 35), (400, 280, 30),
                             (500, 320, 35), (600, 270, 30)]
            },
            "wall_hole": {
                # Wall at x=400 with a hole in the upper section (not centered)
                # Robot must find and navigate through the off-center gap
                "start": [100, 300],
                "goal": [700, 300],
                "obstacles": [
                    # Wall segments (vertical line of obstacles at x=400)
                    (400, 50, 40),    # Top of wall
                    (400, 130, 40),   # Upper-mid wall
                    (400, 280, 40),  # Upper-mid wall
                    # Gap here around y=210-290 (hole in upper section)
                    (400, 370, 40),   # Lower-mid wall
                    (400, 450, 40),   # Lower wall
                    (400, 530, 40),   # Bottom of wall
                ]
            },
            "super_dense": {
                # 15 obstacles scattered across the room
                # Tests navigation through highly cluttered environment
                "start": [50, 300],
                "goal": [750, 300],
                "obstacles": [
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
                    # Extra obstacle
                    (340, 380, 25),
                ]
            },
            "dense_mixed": {
                # Mixed scenario with 4 circles and 4 rectangles
                # Tests Bayesian Model Selection for obstacle type inference
                # Obstacles positioned away from start (50,300) and goal (750,300)
                "start": [50, 300],
                "goal": [750, 300],
                "obstacles": [
                    # 4 Circles - scattered positions
                    (180, 180, 35),   # Top-left area
                    (250, 450, 30),   # Bottom-left area
                    (450, 150, 32),   # Top-center area
                    (380, 420, 28),   # Bottom-center area
                ],
                "rectangles": [
                    # 4 Rectangles - scattered positions with varied angles
                    (520, 200, 55, 35, 0.2),     # Top-right, slight angle
                    (580, 400, 45, 70, -0.15),   # Bottom-right, slight angle
                    (320, 160, 60, 30, 0.0),     # Top-center-left
                    (650, 280, 35, 55, 0.1),     # Right side, vertical-ish
                    (320, 300, 35, 55, 0.1),     # Center
                ]
            },
            "dense_rect": {
                # Rectangles only scenario - tests rectangle detection
                # Rectangles scattered at random-looking positions
                "start": [50, 300],
                "goal": [750, 300],
                "obstacles": [],  # No circles
                "rectangles": [
                    (180, 200, 55, 40, 0.1),    # Left-upper area
                    (250, 420, 45, 55, -0.2),   # Left-lower area
                    (420, 160, 70, 35, 0.05),   # Center-upper
                    (380, 300, 50, 65, 0.15),   # Center-lower
                    (580, 190, 60, 40, -0.1),   # Right-upper
                    (520, 320, 55, 75, 0.0),    # Right-lower
                ]
            },
            "single_rect": {
                # Single rectangle - robot passes from one side to the other
                # Tests evidence accumulation for shape estimation
                # Rectangle positioned off-center so robot passes by it
                "start": [50, 300],
                "goal": [750, 300],
                "obstacles": [],  # No circles
                "rectangles": [
                    (400, 200, 80, 50, 0.0),    # Rectangle above path
                ]
            }
        }

        config = scenarios.get(scenario, scenarios["simple"])

        self.nav.robot_pos = np.array(config["start"], dtype=float)
        self.nav.set_goal(np.array(config["goal"], dtype=float))
        self.nav.set_obstacles(config["obstacles"])

        # Handle rectangles if present
        rectangles = config.get("rectangles", [])
        if rectangles:
            self.nav.set_rectangles(rectangles)

        # Update viewer with initial state
        self.viewer.true_obstacles = config["obstacles"]
        self.viewer.true_rectangles = rectangles  # Pass rectangles to viewer
        self.viewer.target = np.array(config["goal"])
        self.viewer.robot_pos = self.nav.robot_pos.copy()

    def run(self):
        """Main loop"""
        print("=" * 60)
        print("Active Inference Navigation Viewer Demo")
        print("=" * 60)
        print()
        print("Controls:")
        print("  ▶ Start  - Begin continuous navigation")
        print("  ⏸ Pause  - Pause navigation")
        print()
        print("Close window to exit.")
        print()

        last_step_time = time.time()
        step_interval = 0.01  # 100 Hz target (actual limited by computation ~27 Hz)

        while self.viewer.should_continue():
            # Handle controls - sync running state with viewer
            self.running = self.viewer.is_running()

            # Execute navigation step
            current_time = time.time()
            if self.running and current_time - last_step_time >= step_interval:
                self._execute_step()
                last_step_time = current_time

            # Render frame
            self.viewer.render_frame()

        self.viewer.cleanup()
        print("Demo finished.")

    def _execute_step(self):
        """Execute one navigation step and update viewer"""
        result = self.nav.step()

        # Update viewer state
        self.viewer.robot_pos = self.nav.robot_pos.copy()
        self.viewer.steps = self.nav.steps

        # Calculate robot_theta from action (direction of movement)
        action = result.get('action', np.zeros(2))
        if np.linalg.norm(action) > 0.1:
            robot_theta = np.arctan2(action[1], action[0])
        else:
            robot_theta = getattr(self, '_last_theta', 0.0)
        self._last_theta = robot_theta

        # Get controller info if available
        controller_info = {
            'horizon': result.get('horizon', 15),
            'efe_smoothed': 0.0,
            'G_total': result.get('G_total', 0),
            'elapsed_ms': result.get('elapsed_ms', 0),
            'G_components': {
                'goal': result.get('G_goal', 0),
                'obstacle': result.get('G_obstacle', 0),
                'ambiguity': result.get('G_ambiguity', 0),
                'epistemic': result.get('G_epistemic', 0),
                'structural': result.get('G_structural', 0),
                'localization': result.get('G_localization', 0),
                'velocity': result.get('G_velocity', 0),
            }
        }

        if hasattr(self.nav, 'controller') and self.nav.controller is not None:
            controller_info['efe_smoothed'] = self.nav.controller._efe_smoothed

        # Get trajectory if available
        if 'trajectory' in result:
            self.viewer.trajectory = result['trajectory']

        # Get LIDAR data if available
        lidar_points = []
        lidar_upper_points = []
        if hasattr(self.nav, 'get_lidar_points'):
            lidar_points = self.nav.get_lidar_points()
        if hasattr(self.nav, 'get_upper_lidar_points'):
            lidar_upper_points = self.nav.get_upper_lidar_points()

        # Get pose covariance if available
        pose_covariance = None
        room_covariance = None
        room_estimate = None
        room_phase = "unknown"

        if hasattr(self.nav, 'room_estimator') and self.nav.room_estimator is not None:
            room_est = self.nav.room_estimator

            # Get phase
            if hasattr(room_est, 'phase'):
                room_phase = room_est.phase

            # Get belief (contains room and pose estimates)
            if hasattr(room_est, 'belief') and room_est.belief is not None:
                belief = room_est.belief

                # Room estimate (width, height/length)
                if hasattr(belief, 'width') and hasattr(belief, 'length'):
                    room_estimate = np.array([belief.width, belief.length])

                # Pose covariance (x, y)
                if hasattr(belief, 'pose_cov') and belief.pose_cov is not None:
                    pose_covariance = belief.pose_cov[:2, :2]

                # Room covariance (W, L) - indices 3,4 in full covariance
                if hasattr(belief, 'cov') and belief.cov is not None:
                    full_cov = belief.cov
                    if full_cov.shape[0] >= 5:
                        room_covariance = full_cov[3:5, 3:5]

        # Get obstacle beliefs with uncertainty
        obstacle_beliefs = []
        if hasattr(self.nav, 'obstacle_beliefs'):
            obstacle_beliefs = self.nav.obstacle_beliefs

        # Create step data for viewer
        step_data = {
            'robot_pos': self.nav.robot_pos,
            'robot_theta': robot_theta,
            'estimated_pos': getattr(self.nav, 'estimated_pos', self.nav.robot_pos),
            'velocity': action,  # Current velocity/action [vx, vy]
            'steps': self.nav.steps,
            'target_num': self.target_num,
            'total_targets': self.total_targets,
            'success_count': self.success_count,
            'controller_info': controller_info,
            'trajectory': result.get('trajectory', []),
            'all_trajectories': result.get('all_trajectories', []),
            'best_seed': result.get('best_seed', 0),
            'lidar_points': lidar_points,
            'lidar_upper_points': lidar_upper_points,
            'pose_covariance': pose_covariance,
            'room_covariance': room_covariance,
            'room_estimate': room_estimate,
            'room_phase': room_phase,
            'obstacle_beliefs': obstacle_beliefs,
            'pose_error': result.get('pose_error', 0),
            'tracked_obstacles': getattr(self.nav, 'tracked_obstacles', []),
        }

        # Update viewer via observer pattern
        self.viewer.update(NavigationEvent.STEP, step_data)

        # Check completion - generate new target when goal reached
        if result.get('goal_reached'):
            self.success_count += 1
            print(f"✓ Goal {self.target_num} reached in {self.nav.steps} steps! (Success: {self.success_count})")
            
            # Notify viewer of target reached
            self.viewer.update(NavigationEvent.TARGET_REACHED, {
                'success_count': self.success_count,
                'total_targets': self.total_targets,
                'target_num': self.target_num,
            })
            
            self._generate_new_target()
        elif result.get('collision'):
            # On collision, push robot away from obstacle
            for cx, cy, r in self.nav.true_obstacles:
                obs_pos = np.array([cx, cy])
                dist = np.linalg.norm(self.nav.robot_pos - obs_pos)
                if dist < r + self.nav.config.robot_radius + 5:
                    # Push robot away from obstacle
                    direction = self.nav.robot_pos - obs_pos
                    if np.linalg.norm(direction) > 0.1:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0.0])
                    self.nav.robot_pos = obs_pos + direction * (r + self.nav.config.robot_radius + 10)

            # Reset controller to replan
            if hasattr(self.nav, 'controller') and self.nav.controller is not None:
                self.nav.controller.previous_trajectory = None
                self.nav.controller.committed_trajectory = None

    def _generate_new_target(self):
        """Generate a new random target avoiding obstacles"""
        # Increment target counter
        self.target_num += 1
        
        MARGIN = 80
        room_w = self.nav.config.room_width
        room_h = self.nav.config.room_height
        obstacles = self.nav.true_obstacles
        rectangles = getattr(self.nav, 'true_rectangles', [])

        # Find valid target position
        for _ in range(100):
            new_goal = np.array([
                np.random.uniform(MARGIN, room_w - MARGIN),
                np.random.uniform(MARGIN, room_h - MARGIN)
            ])

            # Check distance from current position (at least 150px)
            if np.linalg.norm(new_goal - self.nav.robot_pos) < 150:
                continue

            # Check not inside any circle obstacle
            valid = True
            for ox, oy, r in obstacles:
                if np.linalg.norm(new_goal - np.array([ox, oy])) < r + 40:
                    valid = False
                    break

            # Check not inside any rectangle obstacle
            if valid:
                for rect in rectangles:
                    cx, cy, w, h, angle = rect
                    # Transform point to rectangle local frame
                    dx = new_goal[0] - cx
                    dy = new_goal[1] - cy
                    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                    local_x = dx * cos_a - dy * sin_a
                    local_y = dx * sin_a + dy * cos_a
                    # Check if inside rectangle with margin
                    margin = 40
                    if (abs(local_x) < w/2 + margin and
                        abs(local_y) < h/2 + margin):
                        valid = False
                        break

            if valid:
                break

        # Set new goal
        self.nav.set_goal(new_goal)
        self.viewer.target = new_goal.copy()

        # Reset controller state for new target
        if hasattr(self.nav, 'controller') and self.nav.controller is not None:
            self.nav.controller.previous_trajectory = None
            self.nav.controller.committed_trajectory = None

        print(f"→ New target #{self.target_num}: ({new_goal[0]:.0f}, {new_goal[1]:.0f})")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Active Inference Navigation Viewer Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_viewer.py --scenario corridor
  python test_viewer.py --scenario dense --execution-mode gpu_batched --num-seeds 8
  python test_viewer.py --execution-mode parallel  # Use CPU multi-threading
        """
    )
    parser.add_argument('--scenario', type=str, default='single',
                       choices=['simple', 'single', 'corridor', 'dense', 'wall_hole', 'super_dense', 'dense_mixed', 'dense_rect', 'single_rect'],
                       help='Scenario to run (default: single)')
    parser.add_argument('--execution-mode', type=str, default='gpu_batched',
                       choices=['gpu_batched', 'parallel', 'sequential'],
                       help='''Execution mode for trajectory optimization:
                           gpu_batched  - Fully vectorized GPU (fastest, requires CUDA)
                           parallel     - Multi-threaded CPU with PyTorch parallelism
                           sequential   - Single-threaded CPU (for debugging)
                           (default: gpu_batched)''')
    parser.add_argument('--num-seeds', type=int, default=6,
                       help='Number of trajectory seeds for optimization. More seeds = better coverage but slower (default: 6)')
    parser.add_argument('--use-ground-truth', action='store_true',
                       help='Use ground truth pose/obstacles instead of estimated (for debugging)')

    args = parser.parse_args()

    demo = NavigationDemo(
        scenario=args.scenario,
        execution_mode=args.execution_mode,
        num_seeds=args.num_seeds,
        use_estimated=not args.use_ground_truth
    )
    demo.run()


if __name__ == "__main__":
    main()

