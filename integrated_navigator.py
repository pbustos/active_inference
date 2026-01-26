"""
Integrated Navigation System with Unified Active Inference Controller

Combines:
- Room/Pose Estimation (perception module) - Upper LIDAR
- Obstacle Perception (DBSCAN + VFE) - Lower LIDAR
- Unified Controller (planning + control via EFE minimization)

This implements the complete Active Inference navigation pipeline where
perception and control share the same probabilistic framework.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from unified_controller import UnifiedController, UnifiedControllerConfig
from concept_room import RoomPoseEstimatorV2
from perception import cluster_points_dbscan
from obstacle_base import ObstacleTracker, ObstacleBeliefConfig

# GPU LIDAR simulation
try:
    from lidar_gpu import LidarGPU, LidarGPUConfig
    LIDAR_GPU_AVAILABLE = True
except ImportError:
    LIDAR_GPU_AVAILABLE = False

# Import concept modules to trigger registration
import concept_circle
import concept_rectangle



@dataclass
class IntegratedNavConfig:
    """Configuration for integrated navigation system"""
    # Room dimensions
    room_width: float = 800.0
    room_height: float = 600.0

    # Robot parameters
    robot_radius: float = 15.0
    max_velocity: float = 4.0  # pixels per step (increased from 3.0 for faster movement)

    # Lower LIDAR for obstacle detection
    num_lower_rays: int = 360  # Full resolution for early obstacle detection
    lower_sensor_range: float = 350.0  # Increased from 300 to detect obstacles earlier

    # DBSCAN clustering parameters
    # Note: epsilon controls max distance between points in a cluster
    # Smaller values prevent merging points from nearby but distinct obstacles
    dbscan_epsilon: float = 12.0  # Larger to group distant sparse points
    dbscan_min_samples: int = 2   # Reduced to detect earlier
    min_cluster_size: int = 2     # Reduced to 2 for very early detection

    # Obstacle belief parameters
    initial_obstacle_uncertainty: float = 100.0
    obstacle_process_noise: float = 3.0
    sigma_sensor_obstacle: float = 25.0
    confidence_decay: float = 0.95  # Faster decay to remove spurious beliefs
    min_confidence: float = 0.25    # Higher threshold to remove weak beliefs

    # Goal reaching
    goal_threshold: float = 15.0

    # Integration parameters
    use_estimated_pose: bool = True  # Use room estimation vs ground truth
    use_estimated_obstacles: bool = True  # Use obstacle perception vs ground truth
    pose_uncertainty_scale: float = 1.0  # Scale factor for pose uncertainty
    use_known_room_dimensions: bool = False  # If True, use room_width/height as known; False uses Bayesian fusion

    # Controller execution mode:
    # - 'gpu_batched': Fully vectorized GPU execution (best performance)
    # - 'parallel': Multi-threaded CPU execution
    # - 'sequential': Single-threaded CPU (for debugging)
    execution_mode: str = 'gpu_batched'

    # GPU LIDAR simulation (5x faster than CPU)
    use_gpu_lidar: bool = True

    # Number of trajectory seeds for optimization (more = better coverage, slower)
    num_seeds: int = 32


class IntegratedNavigator:
    """
    Integrated Active Inference Navigator.

    This class demonstrates the key insight from the paper:
    perception and control are unified under the same Free Energy framework.

    - Perception: RoomPoseEstimator minimizes VFE to estimate robot pose
    - Control: UnifiedController minimizes EFE to select actions
    - Integration: Pose uncertainty from perception affects control via G_ambiguity
    """

    def __init__(self, config: Optional[IntegratedNavConfig] = None):
        self.config = config or IntegratedNavConfig()

        # Perception: Room and pose estimation (V2 - SDF based)
        self.room_estimator = RoomPoseEstimatorV2(
            room_width=self.config.room_width,
            room_height=self.config.room_height,
            use_known_room_dimensions=self.config.use_known_room_dimensions
        )

        # Control: Unified EFE-based controller
        controller_config = UnifiedControllerConfig(
            max_velocity=self.config.max_velocity,
            execution_mode=self.config.execution_mode,
            num_seeds=self.config.num_seeds,
        )
        self.controller = UnifiedController(controller_config)

        # Obstacle tracker with multi-hypothesis model selection
        # Uses TrackedObstacle instances that maintain ALL registered types
        obstacle_config = ObstacleBeliefConfig(
            sensor_range=self.config.lower_sensor_range,
            match_threshold=60.0,
        )
        self.obstacle_tracker = ObstacleTracker(obstacle_config)

        # GPU LIDAR for faster simulation
        self.lidar_gpu = None
        if self.config.use_gpu_lidar and LIDAR_GPU_AVAILABLE:
            lidar_config = LidarGPUConfig(
                num_rays=self.config.num_lower_rays,
                max_range=self.config.lower_sensor_range
            )
            self.lidar_gpu = LidarGPU(lidar_config)
            print("[IntegratedNavigator] Using GPU LIDAR simulation")

        # State
        self.reset()

    def reset(self):
        """Reset navigation state"""
        self.robot_pos = np.array([100.0, 100.0])
        self.robot_theta = 0.0
        self.goal = None

        # Ground truth obstacles (for simulation)
        self.true_obstacles: List[Tuple[float, float, float]] = []  # Circles: (cx, cy, r)
        self.true_rectangles: List[Tuple[float, float, float, float, float]] = []  # Rects: (cx, cy, w, h, angle)

        # Tracked obstacles with multi-hypothesis model selection
        self.tracked_obstacles: List = []  # TrackedObstacle instances

        # Estimated state (from perception)
        self.estimated_pos = self.robot_pos.copy()
        self.estimated_theta = self.robot_theta
        self.pose_covariance = np.eye(2) * 25.0  # Initial uncertainty

        # Metrics
        self.steps = 0
        self.path = [self.robot_pos.copy()]
        self._stuck_counter = 0  # Counter for collision recovery

        # Reset subsystems
        self.room_estimator = RoomPoseEstimatorV2(
            room_width=self.config.room_width,
            room_height=self.config.room_height,
            use_known_room_dimensions=self.config.use_known_room_dimensions
        )

        # Reset obstacle tracker
        if hasattr(self, 'obstacle_tracker') and self.obstacle_tracker is not None:
            self.obstacle_tracker.clear()

        self.controller.reset()

    def set_goal(self, goal: np.ndarray):
        """Set navigation goal"""
        self.goal = goal.copy()

    def add_obstacle(self, cx: float, cy: float, radius: float):
        """Add a true obstacle (for simulation)"""
        self.true_obstacles.append((cx, cy, radius))

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Set all true obstacles (circles) for simulation"""
        self.true_obstacles = list(obstacles)

    def set_rectangles(self, rectangles: List[Tuple[float, float, float, float, float]]):
        """Set all true rectangles for simulation. Format: (cx, cy, width, height, angle)"""
        self.true_rectangles = list(rectangles)

    def get_lidar_points(self) -> List[Tuple[float, float]]:
        """
        Get lower LIDAR points (obstacle detection).

        Returns list of (x, y) points in room frame.

        Uses GPU-accelerated LIDAR when available (5x faster).
        """
        # Use GPU LIDAR if available
        if self.lidar_gpu is not None:
            return self.lidar_gpu.simulate_numpy(
                self.robot_pos,
                self.true_obstacles,
                self.true_rectangles,
                self.config.room_width,
                self.config.room_height
            )

        # Fallback to CPU implementation
        return self._get_lidar_points_cpu()

    def get_lidar_points_array(self) -> np.ndarray:
        """
        Get lower LIDAR points as numpy array (faster than list).

        Returns [N, 2] array of hit points in room frame.
        """
        if self.lidar_gpu is not None:
            return self.lidar_gpu.simulate_array(
                self.robot_pos,
                self.true_obstacles,
                self.true_rectangles,
                self.config.room_width,
                self.config.room_height
            )

        # Fallback: convert list to array
        pts = self._get_lidar_points_cpu()
        return np.array(pts) if pts else np.empty((0, 2))

    def _get_lidar_points_cpu(self) -> List[Tuple[float, float]]:
        """CPU implementation of LIDAR simulation (fallback)."""
        points = []
        num_rays = 360
        max_range = self.config.lower_sensor_range

        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays

            # Find closest hit
            min_dist = max_range

            # Check obstacles (circles)
            for ox, oy, r in self.true_obstacles:
                # Ray-circle intersection
                dx, dy = np.cos(angle), np.sin(angle)
                fx, fy = self.robot_pos[0] - ox, self.robot_pos[1] - oy

                a = dx*dx + dy*dy
                b = 2 * (fx*dx + fy*dy)
                c = fx*fx + fy*fy - r*r

                disc = b*b - 4*a*c
                if disc >= 0:
                    t = (-b - np.sqrt(disc)) / (2*a)
                    if 0 < t < min_dist:
                        min_dist = t

            # Check rectangles
            dx, dy = np.cos(angle), np.sin(angle)
            for rect in self.true_rectangles:
                cx, cy, w, h, rect_angle = rect
                t = self._ray_rect_intersection(
                    self.robot_pos[0], self.robot_pos[1],
                    dx, dy, cx, cy, w, h, rect_angle
                )
                if t is not None and 0 < t < min_dist:
                    min_dist = t

            # Check walls
            for wall_dist in self._ray_wall_distances(angle):
                if 0 < wall_dist < min_dist:
                    min_dist = wall_dist

            # Add point
            if min_dist < max_range:
                px = self.robot_pos[0] + min_dist * np.cos(angle)
                py = self.robot_pos[1] + min_dist * np.sin(angle)
                points.append((px, py))

        return points

    def get_upper_lidar_points(self) -> List[Tuple[float, float]]:
        """
        Get upper LIDAR points (room/wall detection).

        Upper LIDAR is above obstacles, only sees walls.
        Returns list of (x, y) points in room frame.
        """
        points = []
        num_rays = 360
        max_range = 500

        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays

            # Only walls (upper lidar is above obstacles)
            min_dist = max_range
            for wall_dist in self._ray_wall_distances(angle):
                if 0 < wall_dist < min_dist:
                    min_dist = wall_dist

            if min_dist < max_range:
                px = self.robot_pos[0] + min_dist * np.cos(angle)
                py = self.robot_pos[1] + min_dist * np.sin(angle)
                points.append((px, py))

        return points

    def _ray_wall_distances(self, angle: float) -> List[float]:
        """Get distances to walls for a ray at given angle"""
        distances = []
        dx, dy = np.cos(angle), np.sin(angle)
        x, y = self.robot_pos

        # Left wall (x=0)
        if dx < -1e-6:
            t = -x / dx
            if t > 0:
                distances.append(t)
        # Right wall (x=room_width)
        if dx > 1e-6:
            t = (self.config.room_width - x) / dx
            if t > 0:
                distances.append(t)
        # Bottom wall (y=0)
        if dy < -1e-6:
            t = -y / dy
            if t > 0:
                distances.append(t)
        # Top wall (y=room_height)
        if dy > 1e-6:
            t = (self.config.room_height - y) / dy
            if t > 0:
                distances.append(t)

        return distances

    def _ray_rect_intersection(self, ox: float, oy: float, dx: float, dy: float,
                                cx: float, cy: float, w: float, h: float,
                                angle: float) -> Optional[float]:
        """
        Ray-rectangle intersection using slab method.

        Args:
            ox, oy: Ray origin
            dx, dy: Ray direction (unit vector)
            cx, cy: Rectangle center
            w, h: Rectangle width and height
            angle: Rectangle rotation angle (radians)

        Returns:
            Distance to intersection or None if no intersection
        """
        # Transform ray to rectangle's local frame
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)

        # Translate origin relative to rectangle center
        local_ox = ox - cx
        local_oy = oy - cy

        # Rotate to local frame
        lo_x = local_ox * cos_a - local_oy * sin_a
        lo_y = local_ox * sin_a + local_oy * cos_a

        # Rotate direction
        ld_x = dx * cos_a - dy * sin_a
        ld_y = dx * sin_a + dy * cos_a

        half_w, half_h = w / 2, h / 2
        eps = 1e-10

        # Slab method for AABB intersection
        if abs(ld_x) > eps:
            tx1 = (-half_w - lo_x) / ld_x
            tx2 = (half_w - lo_x) / ld_x
            tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
        else:
            if lo_x < -half_w or lo_x > half_w:
                return None
            tx_min, tx_max = -1e10, 1e10

        if abs(ld_y) > eps:
            ty1 = (-half_h - lo_y) / ld_y
            ty2 = (half_h - lo_y) / ld_y
            ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
        else:
            if lo_y < -half_h or lo_y > half_h:
                return None
            ty_min, ty_max = -1e10, 1e10

        t_enter = max(tx_min, ty_min)
        t_exit = min(tx_max, ty_max)

        if t_enter < t_exit and t_exit > 0:
            t = t_enter if t_enter > 0 else t_exit
            if t > 0:
                return t
        return None

    def step(self, dt: float = 1.0) -> Dict:
        """
        Execute one navigation step.

        Pipeline:
        1. Perception: Update pose estimate from room estimation
        2. Control: Compute action using UnifiedController
        3. Execute: Apply action to robot
        4. Update: Track metrics and path

        Returns:
            Dict with step information
        """
        if self.goal is None:
            return {'status': 'no_goal'}

        # =====================================================================
        # STEP 1: PERCEPTION - Update pose estimate
        # This minimizes Variational Free Energy F[q,o] (Paper Section 3)
        # Uses upper LIDAR (above obstacles) to detect room walls
        # OPTIMIZATION: After convergence, run only every 5 steps
        # =====================================================================

        # Get last action as velocity for prediction
        last_velocity = getattr(self, '_last_action', np.zeros(2))

        # Always update room estimation (uses fast 4-iteration optimization)
        perception_result = self.room_estimator.update(
            robot_pos=self.robot_pos,
            robot_theta=self.robot_theta,
            robot_velocity=last_velocity,
            dt=dt
        )

        if self.config.use_estimated_pose and 'pose' in perception_result:
            self.estimated_pos = perception_result['pose'][:2].copy()
            self.estimated_theta = perception_result['pose'][2]

            # Extract pose covariance from room estimator (V2 API)
            if 'pose_cov' in perception_result:
                self.pose_covariance = perception_result['pose_cov'][:2, :2].copy()
                self.pose_covariance *= self.config.pose_uncertainty_scale
            elif self.room_estimator.belief is not None:
                self.pose_covariance = self.room_estimator.belief.pose_cov[:2, :2].copy()
                self.pose_covariance *= self.config.pose_uncertainty_scale
            else:
                self.pose_covariance = np.eye(2) * 100.0
        else:
            # Use ground truth
            self.estimated_pos = self.robot_pos.copy()
            self.estimated_theta = self.robot_theta
            self.pose_covariance = np.eye(2) * 1.0  # Low uncertainty

        # =====================================================================
        # STEP 1b: OBSTACLE PERCEPTION - Lower LIDAR + DBSCAN + VFE
        # This minimizes VFE for each obstacle belief (Paper Section 4.2)
        # Only active after room estimation converges
        # =====================================================================

        has_obstacles = len(self.true_obstacles) > 0 or len(self.true_rectangles) > 0
        if self.room_estimator.is_converged() and has_obstacles:
            self._update_obstacle_beliefs()

            # IMPORTANT: Validate committed trajectory against new obstacle beliefs
            # If the trajectory crosses a newly detected obstacle, invalidate it
            self._validate_committed_trajectory()

        # Decide which obstacles to use for control
        # Priority: TrackedObstacles (with model-averaged prediction) > ground truth
        if self.config.use_estimated_obstacles and len(self.tracked_obstacles) > 0:
            # Use TrackedObstacle directly - controller will do model-averaged prediction
            # Paper Eq. model_averaged_prediction: p(o_{t+1}|π) = Σ_m q(m)·p(o_{t+1}|π,m)
            control_obstacles = list(self.tracked_obstacles)
        else:
            # Use ground truth obstacles
            control_obstacles = list(self.true_obstacles)

            # Add ground truth rectangles as RectangleObstacle objects
            if hasattr(self, 'true_rectangles') and len(self.true_rectangles) > 0:
                from concept_circle import CircleObstacle
                from concept_rectangle import RectangleObstacle

                # Convert circles to CircleObstacle objects
                circle_obstacles = []
                for obs in control_obstacles:
                    if isinstance(obs, tuple) and len(obs) == 3:
                        cx, cy, r = obs
                        circle_obstacles.append(CircleObstacle(cx, cy, r))
                    else:
                        circle_obstacles.append(obs)

                # Create RectangleObstacle objects
                rect_obstacles = []
                for rect in self.true_rectangles:
                    cx, cy, w, h, angle = rect
                    rect_obstacles.append(RectangleObstacle(cx, cy, w, h, angle))

                # Combine all obstacles
                control_obstacles = circle_obstacles + rect_obstacles

        # =====================================================================
        # STEP 2: CONTROL - Compute action via EFE minimization
        # This minimizes Expected Free Energy G[π] (Paper Section 5)
        # =====================================================================

        # Pass pose covariance to controller (affects G_ambiguity and G_epistemic)
        self.controller.set_pose_covariance(self.pose_covariance)

        # Select pose based on configuration
        if self.config.use_estimated_pose:
            # Use estimated pose from room estimator (realistic scenario)
            # The pose uncertainty affects control via G_ambiguity term
            control_pos = self.estimated_pos
            control_theta = self.estimated_theta
        else:
            # Use ground truth for debugging/testing
            control_pos = self.robot_pos
            control_theta = self.robot_theta

        action, control_info = self.controller.compute_action(
            robot_pos=control_pos,
            robot_theta=control_theta,
            goal=self.goal,
            obstacles=control_obstacles,
            prev_velocity=None,
            pose_covariance=self.pose_covariance
        )

        # =====================================================================
        # STEP 3: EXECUTE - Apply action to robot (with collision prevention)
        # =====================================================================
        new_pos = self.robot_pos + action * dt

        # Check if new position would cause collision
        would_collide = False
        collision_obstacle = None
        for cx, cy, r in self.true_obstacles:
            obs_pos = np.array([cx, cy])
            dist = np.linalg.norm(new_pos - obs_pos) - r
            if dist < self.config.robot_radius:
                would_collide = True
                collision_obstacle = obs_pos
                collision_radius = r
                break

        # Also check rectangle collisions
        if not would_collide and hasattr(self, 'true_rectangles'):
            for rect in self.true_rectangles:
                cx, cy, w, h, angle = rect
                # Transform robot position to rectangle's local frame
                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                local_x = (new_pos[0] - cx) * cos_a - (new_pos[1] - cy) * sin_a
                local_y = (new_pos[0] - cx) * sin_a + (new_pos[1] - cy) * cos_a
                # Check distance to rectangle (using SDF)
                half_w, half_h = w / 2, h / 2
                dx = abs(local_x) - half_w
                dy = abs(local_y) - half_h
                if dx <= 0 and dy <= 0:
                    # Inside rectangle
                    dist = max(dx, dy)
                else:
                    # Outside - compute distance to nearest edge
                    dist = np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                if dist < self.config.robot_radius:
                    would_collide = True
                    collision_obstacle = np.array([cx, cy])
                    collision_radius = max(w, h) / 2  # Use bounding radius
                    break

        if not would_collide:
            # No collision - normal movement
            self.robot_pos = new_pos
            self._last_action = action.copy()
            self._stuck_counter = 0
        else:
            # Collision prevention: try to slide along the obstacle
            if collision_obstacle is not None:
                # Compute tangent direction to slide
                to_robot = self.robot_pos - collision_obstacle
                dist_to_center = np.linalg.norm(to_robot)
                if dist_to_center > 0.1:
                    # Normalize
                    radial = to_robot / dist_to_center
                    # Tangent is perpendicular to radial
                    tangent = np.array([-radial[1], radial[0]])

                    # Project action onto tangent (slide direction)
                    slide_amount = np.dot(action, tangent)
                    slide_action = tangent * slide_amount * 0.7  # Reduce speed when sliding

                    # Try sliding
                    slide_pos = self.robot_pos + slide_action * dt

                    # Check if slide position is valid
                    slide_valid = True
                    for cx2, cy2, r2 in self.true_obstacles:
                        if np.linalg.norm(slide_pos - np.array([cx2, cy2])) - r2 < self.config.robot_radius:
                            slide_valid = False
                            break

                    # Also check rectangles for slide validity
                    if slide_valid and hasattr(self, 'true_rectangles'):
                        for rect in self.true_rectangles:
                            cx2, cy2, w2, h2, angle2 = rect
                            cos_a, sin_a = np.cos(-angle2), np.sin(-angle2)
                            local_x = (slide_pos[0] - cx2) * cos_a - (slide_pos[1] - cy2) * sin_a
                            local_y = (slide_pos[0] - cx2) * sin_a + (slide_pos[1] - cy2) * cos_a
                            dx = abs(local_x) - w2 / 2
                            dy = abs(local_y) - h2 / 2
                            if dx <= 0 and dy <= 0:
                                dist = max(dx, dy)
                            else:
                                dist = np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                            if dist < self.config.robot_radius:
                                slide_valid = False
                                break

                    if slide_valid and np.linalg.norm(slide_action) > 0.1:
                        self.robot_pos = slide_pos
                        self._last_action = slide_action.copy()
                        self._stuck_counter = 0
                    else:
                        # Can't slide - truly stuck
                        self._last_action = np.zeros(2)
                        self._stuck_counter = getattr(self, '_stuck_counter', 0) + 1
                else:
                    self._last_action = np.zeros(2)
                    self._stuck_counter = getattr(self, '_stuck_counter', 0) + 1

            # If stuck for too long, force replan
            if getattr(self, '_stuck_counter', 0) > 3:
                self.controller.previous_trajectory = None
                self.controller.committed_trajectory = None
                self._stuck_counter = 0

        # Update orientation based on movement direction
        if np.linalg.norm(self._last_action) > 0.1:
            self.robot_theta = np.arctan2(self._last_action[1], self._last_action[0])

        # =====================================================================
        # STEP 4: UPDATE - Track metrics
        # =====================================================================
        self.steps += 1
        self.path.append(self.robot_pos.copy())

        # Check goal reached
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal)
        goal_reached = dist_to_goal < self.config.goal_threshold

        # Check collision (against TRUE obstacles)
        collision = False
        min_clearance = float('inf')
        for cx, cy, r in self.true_obstacles:
            dist = np.linalg.norm(self.robot_pos - np.array([cx, cy])) - r
            min_clearance = min(min_clearance, dist)
            if dist < self.config.robot_radius:
                collision = True

        # Compute pose estimation error
        pose_error = np.linalg.norm(self.estimated_pos - self.robot_pos)

        # Count detected obstacles
        num_beliefs = len([o for o in self.tracked_obstacles if o.confidence > 0.3])

        # Extract EFE components from controller info
        G_components = control_info.get('G_components', {})

        return {
            'status': 'goal_reached' if goal_reached else ('collision' if collision else 'running'),
            'step': self.steps,
            'robot_pos': self.robot_pos.copy(),
            'estimated_pos': self.estimated_pos.copy(),
            'pose_error': pose_error,
            'pose_uncertainty': np.trace(self.pose_covariance),
            'action': action.copy(),
            'dist_to_goal': dist_to_goal,
            'min_clearance': min_clearance,
            'G_total': control_info.get('G_total', 0),
            'G_goal': G_components.get('goal', 0),
            'G_obstacle': G_components.get('obstacle', 0),
            'G_ambiguity': G_components.get('ambiguity', 0),
            'G_epistemic': G_components.get('epistemic', 0),
            'G_structural': G_components.get('structural', 0),
            'G_localization': G_components.get('localization', 0),
            'G_velocity': G_components.get('velocity', 0),
            'horizon': control_info.get('horizon', 0),
            'elapsed_ms': control_info.get('elapsed_ms', 0),
            'trajectory': control_info.get('trajectory', []),
            'all_trajectories': control_info.get('all_trajectories', []),
            'best_seed': control_info.get('best_seed', 0),
            'perception_phase': perception_result.get('phase', 'unknown'),
            'goal_reached': goal_reached,
            'collision': collision,
            'num_true_obstacles': len(self.true_obstacles),
            'num_detected_obstacles': num_beliefs,
            'control': control_info  # Pass through full control info
        }

    def run_to_goal(self, max_steps: int = 500, verbose: bool = True) -> Dict:
        """
        Run navigation until goal is reached or max steps exceeded.

        Returns:
            Dict with navigation results
        """
        if self.goal is None:
            raise ValueError("No goal set")

        for step in range(max_steps):
            result = self.step()

            if verbose and step % 50 == 0:
                print(f"Step {step}: pos=[{result['robot_pos'][0]:.0f}, {result['robot_pos'][1]:.0f}] "
                      f"dist={result['dist_to_goal']:.0f} pose_err={result['pose_error']:.1f} "
                      f"G={result['G_total']:.0f}")

            if result['goal_reached']:
                if verbose:
                    print(f"\nGoal reached in {step + 1} steps!")
                return {
                    'success': True,
                    'steps': step + 1,
                    'path': self.path.copy(),
                    'final_pos': self.robot_pos.copy(),
                    'min_clearance': self._compute_path_clearance()
                }

            if result['collision']:
                if verbose:
                    print(f"\nCollision at step {step + 1}")
                return {
                    'success': False,
                    'reason': 'collision',
                    'steps': step + 1,
                    'path': self.path.copy(),
                    'final_pos': self.robot_pos.copy()
                }

        if verbose:
            print(f"\nMax steps ({max_steps}) reached")
        return {
            'success': False,
            'reason': 'max_steps',
            'steps': max_steps,
            'path': self.path.copy(),
            'final_pos': self.robot_pos.copy(),
            'final_dist': np.linalg.norm(self.robot_pos - self.goal)
        }

    def _compute_path_clearance(self) -> float:
        """Compute minimum clearance along path"""
        if not self.true_obstacles:
            return float('inf')

        min_clearance = float('inf')
        for pos in self.path:
            for cx, cy, r in self.true_obstacles:
                dist = np.linalg.norm(pos - np.array([cx, cy])) - r
                min_clearance = min(min_clearance, dist)
        return min_clearance

    def _filter_wall_points(self, points: np.ndarray) -> np.ndarray:
        """
        Filter out LIDAR points that correspond to room walls.

        This implements a hierarchical perception approach:
        - Level 1 (high): Room geometry (walls) - already estimated
        - Level 2 (low): Obstacles (circles, rectangles)

        Points are filtered if they are close to the estimated wall positions.
        This prevents walls from being detected as rectangular obstacles.

        Args:
            points: LIDAR points [N, 2] in room frame

        Returns:
            Filtered points with wall points removed
        """
        if len(points) == 0:
            return points

        # Get room dimensions (use estimate if available, else config)
        if self.room_estimator.belief is not None:
            room_w = self.room_estimator.belief.room_params[0]
            room_h = self.room_estimator.belief.room_params[1]
        else:
            room_w = self.config.room_width
            room_h = self.config.room_height

        # Threshold for considering a point as "on the wall"
        wall_threshold = 15.0  # pixels

        # Check distance to each wall (x=0, x=room_w, y=0, y=room_h)
        dist_left = np.abs(points[:, 0])
        dist_right = np.abs(points[:, 0] - room_w)
        dist_bottom = np.abs(points[:, 1])
        dist_top = np.abs(points[:, 1] - room_h)

        # A point is on a wall if it's close to any wall
        min_wall_dist = np.minimum(
            np.minimum(dist_left, dist_right),
            np.minimum(dist_bottom, dist_top)
        )

        # Keep points that are NOT on walls
        not_wall_mask = min_wall_dist > wall_threshold

        return points[not_wall_mask]

    def _update_obstacle_beliefs(self):
        """
        Update obstacle beliefs using lower LIDAR and concept managers.

        Paper Section 4.2: Perception over obstacles

        Pipeline:
        1. Get lower LIDAR points (includes circles AND rectangles)
        2. FILTER OUT wall points using room estimate (hierarchical)
        3. Cluster remaining points with DBSCAN
        4. ObstacleTracker handles lifecycle with multi-hypothesis tracking
        """
        # Get LIDAR points (includes both circles and rectangles)
        lidar_points = self.get_lidar_points()

        if len(lidar_points) < 5:
            return

        points = np.array(lidar_points)

        # =====================================================================
        # HIERARCHICAL FILTERING: Remove wall points
        # Points that can be explained by the room walls are filtered out
        # This prevents walls from being detected as obstacles
        # =====================================================================
        points = self._filter_wall_points(points)

        if len(points) < 3:
            # All points were walls
            return

        # =====================================================================
        # OBSTACLE TRACKING via ObstacleTracker
        # Handles multi-hypothesis model selection (circle/rectangle)
        # =====================================================================

        # Cluster points with DBSCAN
        clusters = cluster_points_dbscan(
            points,
            self.config.dbscan_epsilon,
            self.config.dbscan_min_samples,
            self.config.min_cluster_size
        )

        # Update tracked obstacles
        robot_pose = np.array([self.robot_pos[0], self.robot_pos[1], self.robot_theta])
        self._update_tracked_obstacles(clusters, robot_pose)

    def _update_tracked_obstacles(self, clusters: List[np.ndarray], robot_pose: np.ndarray):
        """
        Update tracked obstacles using ObstacleTracker.

        The tracker handles:
        - Matching clusters to existing TrackedObstacle instances
        - Creating new instances with all registered model hypotheses
        - Updating instances with new observations
        - Computing model posterior q(m) via VFE comparison
        - Commitment based on paper criteria

        Args:
            clusters: List of point arrays from DBSCAN clustering
            robot_pose: Robot pose [x, y, theta]
        """
        # Update obstacle tracker with clusters
        # Tracker handles matching, creation, and lifecycle
        self.tracked_obstacles = self.obstacle_tracker.update(clusters, robot_pose)

    def _validate_committed_trajectory(self):
        """
        Validate controller's committed trajectory against current obstacle beliefs.

        If the trajectory crosses any detected obstacle, invalidate it so the
        controller will replan. This is critical when new obstacles are discovered
        that weren't known when the trajectory was planned.
        """
        # Check if controller has a committed trajectory
        if not hasattr(self.controller, 'committed_trajectory') or self.controller.committed_trajectory is None:
            return

        # Get current obstacles with high confidence
        obstacles = []
        for obs in self.tracked_obstacles:
            if obs.confidence > 0.3:
                center = obs.position[:2]
                r = obs.get_best_model().get_bounding_radius() if obs.get_best_model() else 40.0
                obstacles.append((center[0], center[1], r))

        if not obstacles:
            return

        trajectory = self.controller.committed_trajectory
        current_step = getattr(self.controller, 'committed_step', 0)

        # Check remaining trajectory points
        safety_margin = self.config.robot_radius + 5  # Add small margin

        current_pos = self.robot_pos.copy()
        for i in range(current_step, len(trajectory)):
            # Compute position at this step
            if hasattr(trajectory[i], 'numpy'):
                action = trajectory[i].numpy()
            else:
                action = np.array(trajectory[i])

            current_pos = current_pos + action

            # Check collision with each obstacle belief
            for cx, cy, r in obstacles:
                dist = np.linalg.norm(current_pos - np.array([cx, cy])) - r
                if dist < safety_margin:
                    # Trajectory crosses obstacle - invalidate it
                    self.controller.committed_trajectory = None
                    self.controller.committed_step = 0
                    if hasattr(self.controller, 'commit_reason'):
                        self.controller.commit_reason = ""
                    return


def test_integrated_navigation():
    """Test integrated navigation system"""
    print("=" * 70)
    print("Testing Integrated Navigation System")
    print("=" * 70)
    print()

    # Create navigator
    config = IntegratedNavConfig(
        room_width=800,
        room_height=600,
        use_estimated_pose=True
    )
    nav = IntegratedNavigator(config)

    # Test 1: Simple navigation
    print("Test 1: Simple navigation (no obstacles)")
    print("-" * 50)

    nav.reset()
    nav.robot_pos = np.array([100.0, 100.0])
    nav.set_goal(np.array([400.0, 300.0]))

    result = nav.run_to_goal(max_steps=200, verbose=True)
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Steps: {result['steps']}")
    print()

    # Test 2: With obstacle
    print("Test 2: Navigation with obstacle")
    print("-" * 50)

    nav.reset()
    nav.robot_pos = np.array([100.0, 100.0])
    nav.set_goal(np.array([400.0, 100.0]))
    nav.add_obstacle(250.0, 100.0, 50.0)

    result = nav.run_to_goal(max_steps=200, verbose=True)
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Steps: {result['steps']}")
    if 'min_clearance' in result:
        print(f"Min clearance: {result['min_clearance']:.1f}px")
    print()

    # Test 3: Dense obstacles
    print("Test 3: Dense obstacle field")
    print("-" * 50)

    nav.reset()
    nav.robot_pos = np.array([50.0, 300.0])
    nav.set_goal(np.array([750.0, 300.0]))

    # Add multiple obstacles
    np.random.seed(42)
    for _ in range(10):
        cx = np.random.uniform(150, 650)
        cy = np.random.uniform(150, 450)
        r = np.random.uniform(20, 35)
        # Avoid placing on start/goal
        if np.linalg.norm([cx - 50, cy - 300]) > r + 50:
            if np.linalg.norm([cx - 750, cy - 300]) > r + 50:
                nav.add_obstacle(cx, cy, r)

    print(f"Obstacles: {len(nav.obstacles)}")

    result = nav.run_to_goal(max_steps=400, verbose=True)
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Steps: {result['steps']}")
    if 'min_clearance' in result:
        print(f"Min clearance: {result['min_clearance']:.1f}px")
    print()

    # Test 4: Pose estimation integration
    print("Test 4: Pose estimation effect")
    print("-" * 50)

    # With estimated pose
    nav.reset()
    nav.config.use_estimated_pose = True
    nav.robot_pos = np.array([100.0, 100.0])
    nav.set_goal(np.array([600.0, 400.0]))
    nav.add_obstacle(350.0, 250.0, 60.0)

    result_estimated = nav.run_to_goal(max_steps=300, verbose=False)

    # With ground truth pose
    nav.reset()
    nav.config.use_estimated_pose = False
    nav.robot_pos = np.array([100.0, 100.0])
    nav.set_goal(np.array([600.0, 400.0]))
    nav.add_obstacle(350.0, 250.0, 60.0)

    result_truth = nav.run_to_goal(max_steps=300, verbose=False)

    print(f"With estimated pose: {'SUCCESS' if result_estimated['success'] else 'FAILED'} in {result_estimated['steps']} steps")
    print(f"With ground truth:   {'SUCCESS' if result_truth['success'] else 'FAILED'} in {result_truth['steps']} steps")
    print()

    print("=" * 70)
    print("Integration tests complete")
    print("=" * 70)


if __name__ == "__main__":
    test_integrated_navigation()

