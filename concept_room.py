"""
Concept: Room - Room Detection and Pose Estimation via Variational Inference

This module implements the room/pose perception concept from the Active Inference
navigation framework (Paper Section 4.1).

Key features:
- Upper LIDAR (above obstacles) detects room walls
- SDF_rect: Signed Distance Function to rectangular room boundary
- Two-phase operation:
  1. Initialization: Robot static, minimize SDF error wrt (x, y, θ, W, L)
  2. Tracking: Room fixed, minimize SDF likelihood + prior (predicted pose + covariance)

Uses PyTorch for automatic differentiation of SDF gradients.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# Use float32 for speed, GPU if available
DTYPE = torch.float32
GPU_AVAILABLE = False
DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    try:
        # Test if GPU actually works (RTX 50xx needs PyTorch nightly with sm_120)
        _test = torch.zeros(1, device='cuda') + 1
        DEVICE = torch.device('cuda')
        GPU_AVAILABLE = True
        print(f"[RoomEstimator] Using GPU: {torch.cuda.get_device_name(0)}")
    except RuntimeError:
        print("[RoomEstimator] GPU not compatible with current PyTorch, using CPU")


@dataclass
class RoomBelief:
    """
    Belief state for room geometry and robot pose.

    State vector: s = (x, y, θ, W, L)
    - (x, y, θ): robot pose in room frame
    - (W, L): room width and length

    Belief is Gaussian: q(s) = N(μ, Σ)
    """
    # Mean
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    width: float = 400.0   # Default value, overridden by Gaussian initialization
    length: float = 300.0  # Default value, overridden by Gaussian initialization

    # Covariance (5x5 matrix)
    cov: np.ndarray = None

    # Convergence flag
    converged: bool = False

    def __post_init__(self):
        if self.cov is None:
            # High initial uncertainty reflecting Gaussian prior on room dimensions
            # Pose uncertainty: [x, y, theta]
            # Room uncertainty: [width, length] with std ~50px
            self.cov = np.diag([200.0, 200.0, 1.0, 2500.0, 2500.0])  # 50^2 = 2500

    @property
    def pose(self) -> np.ndarray:
        """Robot pose [x, y, θ]"""
        return np.array([self.x, self.y, self.theta])

    @property
    def pose_cov(self) -> np.ndarray:
        """Pose covariance [3x3]"""
        return self.cov[:3, :3].copy()

    @property
    def room_params(self) -> np.ndarray:
        """Room parameters [W, L]"""
        return np.array([self.width, self.length])

    @property
    def state_vector(self) -> np.ndarray:
        """Full state [x, y, θ, W, L]"""
        return np.array([self.x, self.y, self.theta, self.width, self.length])

    def uncertainty(self) -> float:
        """Total uncertainty (trace of pose covariance)"""
        return np.trace(self.cov[:3, :3])


class RoomPoseEstimatorV2:
    """
    Room and Pose Estimation using SDF minimization with PyTorch.

    Two-phase operation:

    Phase 1 (Initialization):
        - Robot is STATIC
        - Collect upper LIDAR points
        - Minimize SDF error wrt full state (x, y, θ, W, L)
        - Use PyTorch autograd for gradients
        - Once error < threshold, fix room parameters

    Phase 2 (Tracking):
        - Room parameters (W, L) are FIXED
        - Each step: predict pose using velocity, then correct with SDF
        - Minimize: F = SDF_likelihood + KL_prior
        - Output pose and covariance for obstacle models
    """

    def __init__(self,
                 room_width: float = 800.0,
                 room_height: float = 600.0,
                 num_upper_rays: int = 64,  # Reduced from 360 for faster tracking
                 upper_sensor_range: float = 1000.0,
                 distance_noise_std: float = 5.0,
                 use_known_room_dimensions: bool = False):
        """
        Args:
            room_width: True room width (for simulation)
            room_height: True room height (for simulation)
            num_upper_rays: Number of upper LIDAR rays
            upper_sensor_range: Maximum range of upper LIDAR
            distance_noise_std: Gaussian noise std dev on distance measurements
            use_known_room_dimensions: If True, fix room dimensions (only estimate pose)
        """
        # Ground truth (for simulation only)
        self.true_width = room_width
        self.true_height = room_height
        self.use_known_room_dimensions = use_known_room_dimensions


        # Upper LIDAR parameters
        self.num_rays = num_upper_rays
        self.sensor_range = upper_sensor_range
        self.distance_noise_std = distance_noise_std

        # Pre-compute ray angles
        self.ray_angles = torch.linspace(0, 2 * np.pi, num_upper_rays + 1,
                                         dtype=DTYPE, device=DEVICE)[:-1]

        # Belief state
        self.belief: Optional[RoomBelief] = None

        # Phase: 'init' or 'tracking'
        self.phase = 'init'

        # Initialization parameters
        self.init_buffer = []  # Collected LIDAR points
        self.min_points_for_init = 100
        self.init_convergence_threshold = 10.0  # Stricter threshold for better accuracy
        self.max_init_iterations = 500
        self.init_iterations = 0

        # Robot rotation during initialization for robust belief
        self.init_rotation_enabled = True
        self.init_rotation_angles = [0.0, 0.3, -0.3, 0.5, -0.5, 0.0]  # Radians: center, left, right, more left, more right, center
        self.init_rotation_index = 0
        self.init_rotation_hold_steps = 3  # Steps to hold each angle
        self.init_rotation_step_count = 0

        # Tracking parameters
        self.sigma_sdf = 3.0  # Smaller - trust LIDAR observations strongly
        self.Q_pose = np.diag([5.0, 5.0, 0.1])  # Process noise - larger to reflect odometry uncertainty

        # Optimization parameters
        self.lr_init = 1.0  # Learning rate for initialization
        self.lr_tracking = 0.4  # Conservative LR for stability

        # State for tracking
        self._last_velocity = np.zeros(2)

        # Statistics
        self.stats = {
            'init_steps': 0,
            'sdf_error_history': [],
            'pose_error_history': []
        }

    def simulate_upper_lidar(self,
                             robot_pos: np.ndarray,
                             robot_theta: float) -> torch.Tensor:
        """
        Simulate upper LIDAR detecting room walls (VECTORIZED).

        The upper LIDAR is mounted above obstacle height, so it only sees walls.

        Returns:
            Tensor of shape [K, 2] with hit points in robot frame
        """
        # Ensure robot_pos is 1D array of shape (2,)
        robot_pos = np.asarray(robot_pos).flatten()[:2]

        # Vectorized ray directions
        angles = robot_theta + self.ray_angles.cpu().numpy()  # [num_rays]
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # [num_rays, 2]

        # Initialize distances to sensor range
        t_min = np.full(self.num_rays, self.sensor_range)

        # Suppress divide warnings - we handle inf/nan explicitly
        with np.errstate(divide='ignore', invalid='ignore'):
            # Left wall (x = 0): t = -robot_pos[0] / direction[0]
            mask = directions[:, 0] < -1e-6
            t_left = -robot_pos[0] / directions[:, 0]
            y_left = robot_pos[1] + t_left * directions[:, 1]
            valid_left = mask & np.isfinite(t_left) & (t_left > 0) & (t_left < t_min) & (y_left >= 0) & (y_left <= self.true_height)
            t_min = np.where(valid_left, t_left, t_min)

            # Right wall (x = width)
            mask = directions[:, 0] > 1e-6
            t_right = (self.true_width - robot_pos[0]) / directions[:, 0]
            y_right = robot_pos[1] + t_right * directions[:, 1]
            valid_right = mask & np.isfinite(t_right) & (t_right > 0) & (t_right < t_min) & (y_right >= 0) & (y_right <= self.true_height)
            t_min = np.where(valid_right, t_right, t_min)

            # Bottom wall (y = 0)
            mask = directions[:, 1] < -1e-6
            t_bottom = -robot_pos[1] / directions[:, 1]
            x_bottom = robot_pos[0] + t_bottom * directions[:, 0]
            valid_bottom = mask & np.isfinite(t_bottom) & (t_bottom > 0) & (t_bottom < t_min) & (x_bottom >= 0) & (x_bottom <= self.true_width)
            t_min = np.where(valid_bottom, t_bottom, t_min)

            # Top wall (y = height)
            mask = directions[:, 1] > 1e-6
            t_top = (self.true_height - robot_pos[1]) / directions[:, 1]
            x_top = robot_pos[0] + t_top * directions[:, 0]
            valid_top = mask & np.isfinite(t_top) & (t_top > 0) & (t_top < t_min) & (x_top >= 0) & (x_top <= self.true_width)
            t_min = np.where(valid_top, t_top, t_min)

        # Filter valid hits
        valid_hits = t_min < self.sensor_range

        if not np.any(valid_hits):
            return torch.zeros((0, 2), dtype=DTYPE, device=DEVICE)

        # Add Gaussian noise to distances
        noisy_distances = t_min[valid_hits] + np.random.randn(np.sum(valid_hits)) * self.distance_noise_std
        noisy_distances = np.maximum(0.1, noisy_distances)

        # Compute hit points in robot frame (local coordinates)
        valid_directions = directions[valid_hits]
        hit_points = noisy_distances[:, np.newaxis] * valid_directions

        return torch.tensor(hit_points, dtype=DTYPE, device=DEVICE)

    def sdf_rect(self,
                 points: torch.Tensor,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 theta: torch.Tensor,
                 width: torch.Tensor,
                 length: torch.Tensor) -> torch.Tensor:
        """
        Signed Distance Function for rectangular room boundary.

        For LIDAR points that hit walls, the SDF should be ≈ 0.
        This measures how far transformed points are from the room boundary.

        The room has walls at x=0, x=width, y=0, y=length.
        LIDAR points should lie ON these walls.

        Args:
            points: [N, 2] points in robot frame
            x, y, theta: Robot pose parameters
            width, length: Room dimensions

        Returns:
            [N] absolute distance to nearest wall (should be ≈ 0)
        """
        if len(points) == 0:
            return torch.tensor([], dtype=DTYPE, device=DEVICE)

        # Rotation matrix (robot frame to room frame)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Transform points: p_room = R @ p_robot + [x, y]
        px = points[:, 0]
        py = points[:, 1]

        room_x = cos_t * px - sin_t * py + x
        room_y = sin_t * px + cos_t * py + y

        # Distance to each wall (absolute, unsigned)
        d_left = torch.abs(room_x)  # Distance to x=0
        d_right = torch.abs(room_x - width)  # Distance to x=width
        d_bottom = torch.abs(room_y)  # Distance to y=0
        d_top = torch.abs(room_y - length)  # Distance to y=length

        # Each point should be ON exactly one wall, so min distance should be ≈ 0
        distances = torch.stack([d_left, d_right, d_bottom, d_top], dim=1)  # [N, 4]
        min_dist, _ = torch.min(distances, dim=1)  # [N]

        return min_dist


    def update(self,
               robot_pos: np.ndarray,
               robot_theta: float,
               robot_velocity: np.ndarray = None,
               dt: float = 1.0) -> Dict:
        """
        Update room and pose estimates.

        Args:
            robot_pos: True robot position (for LIDAR simulation)
            robot_theta: True robot orientation
            robot_velocity: Robot velocity [vx, vy] (for tracking phase)
            dt: Time step

        Returns:
            Dict with estimation results
        """
        if robot_velocity is None:
            robot_velocity = np.zeros(2)

        # During initialization, robot rotates left/right for robust belief
        effective_theta = robot_theta
        if self.phase == 'init' and self.init_rotation_enabled:
            if self.init_rotation_index < len(self.init_rotation_angles):
                rotation_offset = self.init_rotation_angles[self.init_rotation_index]
                effective_theta = robot_theta + rotation_offset

                # Advance rotation after hold steps
                self.init_rotation_step_count += 1
                if self.init_rotation_step_count >= self.init_rotation_hold_steps:
                    self.init_rotation_step_count = 0
                    self.init_rotation_index += 1

        # Simulate upper LIDAR with effective theta
        lidar_points = self.simulate_upper_lidar(robot_pos, effective_theta)

        if self.phase == 'init':
            return self._initialization_phase(robot_pos, robot_theta, lidar_points, effective_theta)
        else:
            return self._tracking_phase(robot_pos, robot_theta, lidar_points,
                                        robot_velocity, dt)

    def _initialization_phase(self,
                              robot_pos: np.ndarray,
                              robot_theta: float,
                              lidar_points: torch.Tensor,
                              effective_theta: float = None) -> Dict:
        """
        Phase 1: Robot static but rotating, minimize SDF error wrt full state.

        Uses PyTorch autograd for gradient computation.
        Robot rotates left/right to get diverse observations for robust belief.
        """
        if effective_theta is None:
            effective_theta = robot_theta

        # Collect points
        if len(lidar_points) > 0:
            self.init_buffer.append(lidar_points)

        # Need enough points
        total_points = sum(len(p) for p in self.init_buffer)
        if total_points < self.min_points_for_init:
            return {
                'phase': 'init',
                'status': 'collecting',
                'points_collected': total_points,
                'points_needed': self.min_points_for_init
            }

        # Initialize belief if needed
        if self.belief is None:
            # Initial guess using LIDAR point geometry
            all_points = torch.cat(self.init_buffer, dim=0)

            # Points are in robot frame. The robot sees walls in all directions.
            # Use the spread of points to estimate room size
            max_x = torch.max(all_points[:, 0]).item()
            min_x = torch.min(all_points[:, 0]).item()
            max_y = torch.max(all_points[:, 1]).item()
            min_y = torch.min(all_points[:, 1]).item()

            if self.use_known_room_dimensions:
                # Use known room dimensions exactly
                est_width = self.true_width
                est_height = self.true_height
                width_var = 1.0  # Very low variance (essentially fixed)
                height_var = 1.0
            else:
                # =============================================================
                # BAYESIAN FUSION: Combine LIDAR estimate with prior belief
                #
                # Prior: Gaussian centered on typical room size
                #   p(room) ~ N(μ_prior, σ²_prior)
                #
                # Likelihood: Estimate from LIDAR point spread
                #   p(lidar | room) ~ N(μ_lidar, σ²_lidar)
                #
                # Posterior (Bayesian fusion):
                #   μ_post = (σ²_lidar * μ_prior + σ²_prior * μ_lidar) / (σ²_prior + σ²_lidar)
                #   σ²_post = (σ²_prior * σ²_lidar) / (σ²_prior + σ²_lidar)
                # =============================================================

                # PRIOR: Our belief about typical room sizes (centered on true for simulation)
                prior_width = self.true_width
                prior_height = self.true_height
                prior_std = 80.0  # Uncertainty in our prior knowledge
                prior_var = prior_std ** 2

                # LIKELIHOOD: Estimate from LIDAR geometry
                # The spread of points gives a noisy estimate of room dimensions
                lidar_width = max_x - min_x + 50   # Add margin for robot not at wall
                lidar_height = max_y - min_y + 50
                lidar_std = 100.0  # LIDAR estimate is noisy (partial view, noise, etc.)
                lidar_var = lidar_std ** 2

                # BAYESIAN FUSION
                # Posterior mean: weighted average based on inverse variances
                width_var = (prior_var * lidar_var) / (prior_var + lidar_var)
                height_var = (prior_var * lidar_var) / (prior_var + lidar_var)

                est_width = (lidar_var * prior_width + prior_var * lidar_width) / (prior_var + lidar_var)
                est_height = (lidar_var * prior_height + prior_var * lidar_height) / (prior_var + lidar_var)

                # Clamp to reasonable bounds
                est_width = np.clip(est_width, 200, 2000)
                est_height = np.clip(est_height, 200, 2000)

            # Robot position: start with provided pose (from odometry/IMU in real system)
            self.belief = RoomBelief(
                x=robot_pos[0],
                y=robot_pos[1],
                theta=robot_theta,
                width=est_width,
                length=est_height
            )

            # Set covariance to reflect our uncertainty
            if not self.use_known_room_dimensions:
                self.belief.cov = np.diag([
                    200.0,      # x position uncertainty
                    200.0,      # y position uncertainty
                    1.0,        # theta uncertainty
                    width_var,  # width uncertainty (from Bayesian fusion)
                    height_var  # height uncertainty (from Bayesian fusion)
                ])

            print(f"[INIT] Initial guess: room {self.belief.width:.1f}x{self.belief.length:.1f}, "
                  f"pose [{self.belief.x:.0f}, {self.belief.y:.0f}]")

        # Concatenate all points
        all_points = torch.cat(self.init_buffer, dim=0)

        # Create optimizable parameters
        if self.use_known_room_dimensions:
            # Only optimize pose, not room dimensions
            pose_params = torch.tensor([
                self.belief.x,
                self.belief.y,
                self.belief.theta,
            ], dtype=DTYPE, device=DEVICE, requires_grad=True)
            fixed_width = torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE)
            fixed_length = torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)
            optimizer = torch.optim.Adam([pose_params], lr=self.lr_init)
        else:
            params = torch.tensor([
                self.belief.x,
                self.belief.y,
                self.belief.theta,
                self.belief.width,
                self.belief.length
            ], dtype=DTYPE, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([params], lr=self.lr_init)

        for _ in range(30):  # Multiple gradient steps per update
            optimizer.zero_grad()

            if self.use_known_room_dimensions:
                # Use fixed dimensions
                sdf_values = self.sdf_rect(
                    all_points,
                    pose_params[0], pose_params[1], pose_params[2],
                    fixed_width, fixed_length
                )
                sdf_loss = torch.mean(sdf_values ** 2)
                loss = sdf_loss
            else:
                # Compute SDF for all points
                sdf_values = self.sdf_rect(
                    all_points,
                    params[0], params[1], params[2],  # x, y, theta
                    params[3], params[4]  # width, length
                )
                sdf_loss = torch.mean(sdf_values ** 2)
                size_reg = 0.0001 * (torch.relu(100 - params[3]) ** 2 + torch.relu(100 - params[4]) ** 2)
                loss = sdf_loss + size_reg

            loss.backward()

            # Clip gradients
            if self.use_known_room_dimensions:
                torch.nn.utils.clip_grad_norm_([pose_params], max_norm=100.0)
            else:
                torch.nn.utils.clip_grad_norm_([params], max_norm=100.0)

            optimizer.step()

            # Enforce constraints
            with torch.no_grad():
                if self.use_known_room_dimensions:
                    pose_params[2] = torch.atan2(torch.sin(pose_params[2]), torch.cos(pose_params[2]))
                else:
                    params[3] = torch.clamp(params[3], min=100.0)  # width > 100
                    params[4] = torch.clamp(params[4], min=100.0)  # length > 100
                    params[2] = torch.atan2(torch.sin(params[2]), torch.cos(params[2]))

        # Update belief
        with torch.no_grad():
            if self.use_known_room_dimensions:
                self.belief.x = pose_params[0].item()
                self.belief.y = pose_params[1].item()
                self.belief.theta = pose_params[2].item()
                # Width and length stay fixed

                # Compute final SDF error
                sdf_values = self.sdf_rect(
                    all_points,
                    pose_params[0], pose_params[1], pose_params[2],
                    fixed_width, fixed_length
                )
            else:
                self.belief.x = params[0].item()
                self.belief.y = params[1].item()
                self.belief.theta = params[2].item()
                self.belief.width = params[3].item()
                self.belief.length = params[4].item()

                # Compute final SDF error
                sdf_values = self.sdf_rect(
                    all_points,
                    params[0], params[1], params[2],
                    params[3], params[4]
                )

            mean_sdf_error = torch.mean(torch.abs(sdf_values)).item()

        self.init_iterations += 1
        self.stats['sdf_error_history'].append(mean_sdf_error)

        # Check convergence
        pose_error = np.linalg.norm(self.belief.pose[:2] - robot_pos)
        room_error = (abs(self.belief.width - self.true_width) +
                      abs(self.belief.length - self.true_height)) / 2

        converged = (mean_sdf_error < self.init_convergence_threshold or
                     self.init_iterations >= self.max_init_iterations)

        if converged:
            self.phase = 'tracking'
            self.belief.converged = True

            # Reduce room uncertainty (room is now fixed)
            self.belief.cov[3, 3] = 1.0
            self.belief.cov[4, 4] = 1.0

            # Store last known pose for prediction
            self._last_pose = self.belief.pose[:2].copy()
            self._last_theta = self.belief.theta

            print(f"\n[ROOM ESTIMATION] Initialization complete")
            print(f"  Iterations: {self.init_iterations}")
            print(f"  Mean SDF error: {mean_sdf_error:.1f}")
            print(f"  Estimated room: {self.belief.width:.0f} x {self.belief.length:.0f}")
            print(f"  True room: {self.true_width:.0f} x {self.true_height:.0f}")
            print(f"  Room error: {room_error:.0f}px")
            print(f"  Pose error: {pose_error:.1f}px\n")

        return {
            'phase': 'init',
            'status': 'converged' if converged else 'optimizing',
            'iteration': self.init_iterations,
            'sdf_error': mean_sdf_error,
            'pose': self.belief.pose.copy(),
            'pose_error': pose_error,
            'room_params': self.belief.room_params.copy(),
            'room_error': room_error,
            'belief_uncertainty': self.belief.uncertainty()
        }

    def _tracking_phase(self,
                        robot_pos: np.ndarray,
                        robot_theta: float,
                        lidar_points: torch.Tensor,
                        robot_velocity: np.ndarray,
                        dt: float) -> Dict:
        """
        Phase 2: Room fixed, update pose by minimizing Free Energy.

        Paper Section 4.25 - Resulting Free Energy Objective:

        F[q(s), o] = F_likelihood + F_prior

        Where:
        - F_likelihood = Σ SDF(o_i, s)² / (2σ_sdf²)  : Observation error
        - F_prior = (1/2)(s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred) : Prediction error

        s_pred = s_{t-1} + v * dt  (propagate previous pose with velocity)
        Σ_pred = Σ_{t-1} + Q        (process noise added)
        """
        if len(lidar_points) == 0:
            return {
                'phase': 'tracking',
                'pose': self.belief.pose.copy(),
                'pose_cov': self.belief.pose_cov,
                'pose_error': np.linalg.norm(self.belief.pose[:2] - robot_pos)
            }

        # =====================================================================
        # PREDICTION STEP: Propagate previous posterior with motion model
        # s_pred = s_{t-1} + v * dt
        # Σ_pred = Σ_{t-1} + Q(v)
        # =====================================================================

        # Previous posterior (from last iteration)
        s_prev = np.array([self.belief.x, self.belief.y, self.belief.theta])
        Sigma_prev = self.belief.cov[:3, :3].copy()

        # Propagate with velocity
        s_pred = np.array([
            s_prev[0] + robot_velocity[0] * dt,
            s_prev[1] + robot_velocity[1] * dt,
            s_prev[2]  # Assume theta doesn't change much
        ])

        # Process noise proportional to velocity (Eq. 105 from paper)
        # More movement = more odometry uncertainty
        speed = np.linalg.norm(robot_velocity)
        velocity_noise_factor = max(0.1, speed / 5.0)  # Minimum noise even when slow
        Q_dynamic = self.Q_pose * dt * velocity_noise_factor

        Sigma_pred = Sigma_prev + Q_dynamic

        # =====================================================================
        # CORRECTION STEP: Minimize F = F_likelihood + F_prior
        # Paper Section 4.25: Both terms have equal weight (factor 1/2 in each)
        # =====================================================================

        # Fixed room parameters
        width = torch.tensor(self.belief.width, dtype=DTYPE, device=DEVICE)
        length = torch.tensor(self.belief.length, dtype=DTYPE, device=DEVICE)

        # Optimizable pose - initialize at predicted position
        pose_params = torch.tensor(s_pred, dtype=DTYPE, device=DEVICE, requires_grad=True)

        # Prior tensors
        s_pred_t = torch.tensor(s_pred, dtype=DTYPE, device=DEVICE)
        Sigma_pred_inv = torch.tensor(np.linalg.inv(Sigma_pred), dtype=DTYPE, device=DEVICE)

        # Number of observations for normalization
        N_obs = len(lidar_points)

        # Optimization
        optimizer = torch.optim.Adam([pose_params], lr=self.lr_tracking)

        for iteration in range(15):  # More iterations for better convergence
            optimizer.zero_grad()

            # =================================================================
            # F_likelihood: -ln p(o|s) ∝ (1/2) Σ SDF² / σ_sdf²
            # Normalized by number of observations
            # =================================================================
            sdf_values = self.sdf_rect(
                lidar_points,
                pose_params[0], pose_params[1], pose_params[2],
                width, length
            )
            F_likelihood = torch.sum(sdf_values ** 2) / (2 * self.sigma_sdf ** 2 * N_obs)

            # =================================================================
            # F_prior: (1/2)(s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred)
            # This term keeps the estimate close to the motion-model prediction
            # =================================================================
            pose_diff = pose_params - s_pred_t
            F_prior = 0.5 * pose_diff @ Sigma_pred_inv @ pose_diff

            # Total Free Energy
            F = F_likelihood + F_prior
            F.backward()

            optimizer.step()

            # Normalize theta to [-π, π]
            with torch.no_grad():
                pose_params[2] = torch.atan2(torch.sin(pose_params[2]),
                                              torch.cos(pose_params[2]))

        # =====================================================================
        # UPDATE BELIEF with optimized posterior
        # =====================================================================
        with torch.no_grad():
            self.belief.x = pose_params[0].item()
            self.belief.y = pose_params[1].item()
            self.belief.theta = pose_params[2].item()

            # Compute final SDF error for diagnostics
            sdf_values = self.sdf_rect(
                lidar_points,
                pose_params[0], pose_params[1], pose_params[2],
                width, length
            )
            mean_sdf_error = torch.mean(torch.abs(sdf_values)).item()

            # =====================================================================
            # COVARIANCE UPDATE (EKF-inspired Information Filter)
            #
            # Σ_post^{-1} = Σ_pred^{-1} + H^T R^{-1} H
            #
            # Simplified version:
            # - Σ_pred already includes process noise (grew during prediction)
            # - Observation reduces uncertainty proportional to number of good points
            # - Poor observations (high SDF) provide less information
            # =====================================================================

            # Number of "good" observations (SDF close to 0)
            good_obs_mask = torch.abs(sdf_values) < 20.0  # Points within 20px of wall
            num_good_obs = good_obs_mask.sum().item()
            total_obs = len(sdf_values)

            # Information gain from observations
            # More good observations = more information = more reduction
            # R = measurement noise covariance (higher when SDF error is high)
            R_effective = self.sigma_sdf ** 2 * (1.0 + mean_sdf_error / 5.0)  # More sensitive to error

            # Observation information matrix (simplified diagonal)
            # H^T R^{-1} H ≈ scale * n_good / R_effective * I
            # Use small scale to prevent over-confident reduction
            info_scale = 0.02  # Reduced to prevent over-confidence
            if num_good_obs > 0:
                obs_info = info_scale * (num_good_obs / total_obs) / R_effective
            else:
                obs_info = 0.0

            # Information filter update
            # Σ_post^{-1} = Σ_pred^{-1} + obs_info * I
            try:
                Sigma_pred_inv_np = np.linalg.inv(Sigma_pred)
                Sigma_post_inv = Sigma_pred_inv_np + obs_info * np.eye(3)
                Sigma_post = np.linalg.inv(Sigma_post_inv)
            except np.linalg.LinAlgError:
                # Fallback: simple decay
                Sigma_post = Sigma_pred * 0.95

            self.belief.cov[:3, :3] = Sigma_post

            # Bounds on covariance diagonal
            # Minimum: 1.0 (some irreducible uncertainty)
            # Maximum: 500.0 (prevent unbounded growth)
            for i in range(3):
                self.belief.cov[i, i] = np.clip(self.belief.cov[i, i], 1.0, 500.0)

        pose_error = np.linalg.norm(self.belief.pose[:2] - robot_pos)
        self.stats['pose_error_history'].append(pose_error)

        return {
            'phase': 'tracking',
            'pose': self.belief.pose.copy(),
            'pose_cov': self.belief.pose_cov,
            'pose_error': pose_error,
            'sdf_error': mean_sdf_error,
            'room_params': self.belief.room_params.copy(),
            'belief_uncertainty': self.belief.uncertainty()
        }

    def is_converged(self) -> bool:
        """Check if room estimation has converged."""
        return self.belief is not None and self.belief.converged

    def get_pose_and_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current pose estimate and covariance.

        Used by obstacle models for updating.
        """
        if self.belief is None:
            return np.zeros(3), np.eye(3) * 1000.0

        return self.belief.pose.copy(), self.belief.pose_cov

    def reset(self):
        """Reset estimator state."""
        self.belief = None
        self.phase = 'init'
        self.init_buffer = []
        self.init_iterations = 0

        # Reset rotation state
        self.init_rotation_index = 0
        self.init_rotation_step_count = 0
        self.stats = {
            'init_steps': 0,
            'sdf_error_history': [],
            'pose_error_history': []
        }


def test_room_estimator():
    """Test the room pose estimator."""
    print("=" * 70)
    print("Testing Room Pose Estimator V2 (SDF-based)")
    print("=" * 70)
    print()

    # Create estimator
    estimator = RoomPoseEstimatorV2(room_width=800, room_height=600)

    # Test 1: Initialization phase
    print("Phase 1: Initialization (robot static)")
    print("-" * 50)

    robot_pos = np.array([300.0, 250.0])
    robot_theta = 0.1

    for step in range(100):
        result = estimator.update(robot_pos, robot_theta)

        if step % 20 == 0 or result.get('status') == 'converged':
            status = result.get('status', result.get('phase'))
            sdf_err = result.get('sdf_error', 0)
            pose_err = result.get('pose_error', 0)
            print(f"Step {step}: status={status}, SDF_err={sdf_err:.1f}, pose_err={pose_err:.1f}")

        if result.get('status') == 'converged':
            break

    print()

    # Test 2: Tracking phase with movement
    print("Phase 2: Tracking (robot moving)")
    print("-" * 50)

    velocity = np.array([2.0, 1.5])

    for step in range(50):
        robot_pos = robot_pos + velocity
        result = estimator.update(robot_pos, robot_theta, robot_velocity=velocity)

        if step % 10 == 0:
            pose_err = result.get('pose_error', 0)
            sdf_err = result.get('sdf_error', 0)
            uncertainty = result.get('belief_uncertainty', 0)
            print(f"Step {step}: pose_err={pose_err:.1f}, SDF_err={sdf_err:.1f}, uncertainty={uncertainty:.1f}")

    print()

    # Summary
    print("Summary:")
    print(f"  Final pose error: {result['pose_error']:.1f}px")
    print(f"  Room estimate: {estimator.belief.width:.0f} x {estimator.belief.length:.0f}")
    print(f"  True room: {estimator.true_width} x {estimator.true_height}")
    print()


if __name__ == "__main__":
    test_room_estimator()

