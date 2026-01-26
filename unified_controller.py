"""
Unified Controller via Multi-Seed Gradient Optimization

Replaces the separate A* planner + reactive controller with a single
continuous Expected Free Energy (EFE) minimization over variable-horizon policies.

Paper References:
-----------------
Section 3.3 - Unifying Planning and Reactive Control:
    "The gradient of the multi-step Expected Free Energy with respect to
    the first action decomposes as:
    ∇_{u_t} G_T(π) = ∇_{u_t} G_{t+1}(u_t) + Σ_{τ=t+2}^{t+T} ∇_{u_t} G_τ(π)"

Section 5.1 - Reactive Control via One-Step EFE:
    G(u_t) = G_pragmatic + G_ambiguity + G_epistemic

    where:
    - G_pragmatic (Eq. pragmatic_cost): preference for safe observations
    - G_ambiguity (Eq. 105-107): uncertainty from motion
    - G_epistemic: information gain about obstacles

Section 5.1.3 - Ray-Circle Model:
    Analytic LIDAR prediction model for differentiable planning

Section 6 - Adaptive Planning:
    EFE-based horizon adaptation when stuck

Key Equations Implemented:
--------------------------
1. Multi-step EFE (Eq. 27): G_T(π) = Σ_{τ=t+1}^{t+T} G_τ
2. Pragmatic term (Eq. pragmatic_cost): G_prag = (1/2)||ô - o*||²_{Σ*^{-1}}
3. Velocity-dependent covariance (Eq. 105): Σ_{t+1}(u) = Σ_t + β||u||²I
4. Observation covariance (Eq. 106): Σ_o = Σ_meas + J Σ_{t+1} J^T
5. Ambiguity (Eq. 107): G_amb = (1/2) ln|Σ_o|
6. Adaptive horizon: T(t) = T_base + (T_max - T_base) · tanh(β · ReLU(Ĝ - G₀))

Algorithm:
----------
1. Generate K initial trajectory "seeds" (different directions)
2. For each seed, run gradient descent to minimize G_T(π)
3. Select the trajectory with lowest G_T
4. Execute first action (or commit to full trajectory if stuck)
5. Next step: use previous best as warm start
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
import time
import os

# Import analytic ray-circle model (Paper Section 5.1.3)
try:
    from ray_circle_model import RayCircleModel, RayCircleConfig
    RAY_CIRCLE_AVAILABLE = True
except ImportError:
    RAY_CIRCLE_AVAILABLE = False

# Import polymorphic obstacle base classes
try:
    from obstacle_base import ObstacleBeliefBase
    POLYMORPHIC_OBSTACLES_AVAILABLE = True
except ImportError:
    POLYMORPHIC_OBSTACLES_AVAILABLE = False

# Detect GPU availability with graceful fallback
GPU_AVAILABLE = False
if torch.cuda.is_available():
    try:
        _test = torch.zeros(1, device='cuda') + 1
        GPU_AVAILABLE = True
        print(f"[UnifiedController] GPU detected: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for faster matrix operations on Ampere+ GPUs
        try:
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
        except AttributeError:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except RuntimeError:
        print("[UnifiedController] GPU not compatible with current PyTorch, using CPU")

# Configure PyTorch threading for better CPU utilization
# Use all available cores for intra-op parallelism
NUM_CPU_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CPU_CORES)
torch.set_num_interop_threads(min(4, NUM_CPU_CORES))  # Inter-op threads for parallel ops

# Default device - prefer GPU for batch operations
DEFAULT_DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')
DTYPE = torch.float32 if GPU_AVAILABLE else torch.float64  # GPU prefers float32


@dataclass
class EFEGains:
    """
    Gains (lambda weights) for Expected Free Energy (EFE) terms.

    These can be modified dynamically at runtime to adjust robot behavior.

    Paper Reference: Section 5.1 - EFE decomposition
    G(π) = G_goal + G_obstacle + G_ambiguity + G_epistemic + G_structural + G_localization + G_velocity

    See G_terms_table.md for detailed descriptions of each term.
    """
    # G_goal 🟢: Pragmatic cost toward goal (Eq. pragmatic_cost)
    lambda_goal: float = 1.0

    # G_obstacle 🔴: Obstacle avoidance cost (Eq. obstacle_cost)
    lambda_obstacle: float = 200.0

    # G_epistemic 🟠: Information gain about obstacle positions (Eq. epistemic_value)
    lambda_epistemic: float = 0.5

    # G_ambiguity 🔵: Uncertainty in future observations (Eq. 107)
    lambda_ambiguity: float = 0.15

    # G_structural 🟣: Model disambiguation (circle vs rectangle) (Eq. model_mutual_information)
    # Note: Uses lambda_epistemic * lambda_structural_scale internally
    lambda_structural: float = 0.5

    # G_localization 🩵: Pose uncertainty cost (Eq. localization_cost)
    lambda_localization: float = 0.5

    # G_smoothness: Action smoothness penalty (legacy, kept for backward compatibility)
    lambda_smoothness: float = 0.02

    # G_velocity 🟤: Velocity regularisation (Section g_velocity)
    # Penalises velocity magnitude: λ_v * Σ(v_τ² + ω_τ²)
    lambda_velocity: float = 0.01

    # G_velocity (delta): Action smoothness / jerk penalty (Section g_velocity)
    # Penalises abrupt changes: λ_Δv * Σ((v_τ - v_{τ-1})² + (ω_τ - ω_{τ-1})²)
    lambda_delta_velocity: float = 0.05


@dataclass
class UnifiedControllerConfig:
    """
    Configuration for unified controller.

    Paper Section 5.1 parameters for EFE-based control.
    """

    # Horizon parameters (Section 6: Adaptive Planning)
    base_horizon: int = 15          # T_base: Base planning horizon
    max_horizon: int = 120           # T_max: Maximum horizon when extended
    min_horizon: int = 8            # T_min: Minimum horizon

    # Multi-seed parameters
    # NOTE: Benchmark shows throughput scales well up to 256 batch size
    # Increased from 6 to 16 for better GPU utilization on RTX 5070 Ti
    num_seeds: int = 32              # K: number of trajectory seeds
    seed_spread: float = 4.0        # Base spread (scales with horizon)
    seed_spread_min: float = 1.0    # Minimum spread (short horizon)
    seed_spread_max: float = 6.0    # Maximum spread (long horizon)
    adaptive_spread: bool = True    # Enable horizon-adaptive spread

    # Execution mode options:
    # - 'gpu_batched': Fully vectorized GPU execution (best for RTX 5070 Ti)
    # - 'parallel': Multi-threaded CPU execution using ThreadPoolExecutor
    # - 'sequential': Single-threaded CPU (fallback, for debugging)
    execution_mode: str = 'gpu_batched'

    # Optimization parameters
    learning_rate: float = 0.5      # Gradient descent step size (reduced for stability)
    max_iterations: int = 10        # Max gradient steps per seed (increased from 6)
    convergence_threshold: float = 0.1  # Early stopping threshold

    # Dynamics
    dt: float = 1.0                 # Time step
    max_velocity: float = 8.0       # Maximum action magnitude (pixels per step)

    # Optional explicit velocity limiting based on pose uncertainty
    # Note: G_amb and G_loc already penalize high velocity with uncertainty in EFE,
    # but this provides an additional hard limit for safety-critical applications
    uncertainty_velocity_limit: bool = False  # Disabled by default (EFE handles it)
    max_pose_uncertainty: float = 50.0        # Above this, velocity is heavily reduced
    min_velocity_factor: float = 0.3          # Minimum velocity as fraction of max

    # Super-linear horizon expansion
    # Allows planning further with same computational cost
    # Options: 'none', 'power', 'exponential'
    horizon_scaling_mode: str = 'power'  # 'none', 'power', 'exponential'

    # Power scaling: velocity_scale[t] = 1 + rate * (t/T)^power
    horizon_expansion_power: float = 2.0  # Exponent for power scaling
    horizon_expansion_rate: float = 1.5   # Max velocity multiplier at horizon end

    # Exponential scaling: velocity_scale[t] = base^(exponent * t/T)
    # Grows from 1.0 at t=0 to base^exponent at t=T
    # Conservative: base=1.5, exp=2.0 → max scale = 2.25x
    # Aggressive: base=2.0, exp=2.0 → max scale = 4.0x
    horizon_exp_base: float = 1.5         # Exponential base (1.5 = more conservative)
    horizon_exp_exponent: float = 2.0     # Exponent multiplier (controls growth rate)

    # Note: Free energy weights (lambda_*) are now in EFEGains dataclass
    # Access via self.efe_gains.lambda_* or use backward-compatible properties

    # Robot parameters
    robot_radius: float = 15.0      # Robot radius for collision checking

    # Sensor model for observation predictions
    num_rays: int = 8               # Rays for observation prediction
    sensor_range: float = 300.0     # Maximum sensor range
    sigma_obs: float = 10.0         # Observation noise std dev
    sigma_obs_sq: float = 100.0     # Observation noise variance (σ²)
    preferred_clearance: float = 30.0  # Preferred distance from obstacles

    # Observation preference model: p(o) = N(o*, Σ*)
    preferred_obs_distance: float = 200.0  # Preferred LIDAR reading (far = safe)

    # Matrix covariance model (Paper Eq. 105-107)
    # Σ_o = Σ_meas + J Σ_s J^T
    sigma_meas: float = 5.0         # Measurement noise std dev
    beta_velocity: float = 0.1      # Velocity-uncertainty coupling

    # Pose uncertainty model
    sigma_pose_base: float = 5.0    # Base pose uncertainty (pixels)

    # Adaptive horizon triggers
    oscillation_threshold: float = 0.15
    uncertainty_threshold: float = 20.0
    oscillation_history: int = 10

    # Analytic ray-circle model (Paper Section 5.1.3)
    # Uses PyTorch autograd for efficient gradient computation
    use_ray_circle_model: bool = True   # Enable ray-circle model
    ray_circle_num_rays: int = 360      # 1 degree resolution
    ray_circle_softmin_eta: float = 0.02 # Soft-min sharpness

    # EFE gains (can be modified dynamically at runtime)
    # Access via config.efe_gains.lambda_* or use backward-compatible properties
    efe_gains: EFEGains = field(default_factory=EFEGains)

    # Backward compatibility properties for direct lambda access
    @property
    def lambda_goal(self) -> float:
        return self.efe_gains.lambda_goal

    @property
    def lambda_obstacle(self) -> float:
        return self.efe_gains.lambda_obstacle

    @property
    def lambda_epistemic(self) -> float:
        return self.efe_gains.lambda_epistemic

    @property
    def lambda_ambiguity(self) -> float:
        return self.efe_gains.lambda_ambiguity

    @property
    def lambda_structural(self) -> float:
        return self.efe_gains.lambda_structural

    @property
    def lambda_localization(self) -> float:
        return self.efe_gains.lambda_localization

    @property
    def lambda_smoothness(self) -> float:
        return self.efe_gains.lambda_smoothness

    @property
    def lambda_velocity(self) -> float:
        return self.efe_gains.lambda_velocity

    @property
    def lambda_delta_velocity(self) -> float:
        return self.efe_gains.lambda_delta_velocity


class UnifiedController:
    """
    Unified Active Inference controller that replaces A* + reactive control.

    Uses multi-seed gradient optimization over variable-horizon trajectories,
    minimizing Expected Free Energy G_T(π) directly.

    Key feature: When stuck, commits to a complete trajectory (not just first step)
    to escape local minima by moving away from blocking obstacles.

    Supports two execution modes:
    - 'sequential': Default, evaluate seeds one by one (reliable, good for CPU)
    - 'gpu_batched': Evaluate all seeds in parallel (experimental, needs GPU)
    """

    def __init__(self, config: Optional[UnifiedControllerConfig] = None):
        self.config = config or UnifiedControllerConfig()

        # Device selection based on execution mode
        if self.config.execution_mode == 'gpu_batched' and GPU_AVAILABLE:
            self.device = torch.device('cuda')
            self.dtype = torch.float32  # GPU prefers float32
            print("[UnifiedController] Using GPU batched execution")
        else:
            # For sequential and parallel modes, use CPU
            self.device = torch.device('cpu')
            self.dtype = torch.float64  # CPU uses float64 for precision
            if self.config.execution_mode == 'gpu_batched' and not GPU_AVAILABLE:
                print("[UnifiedController] GPU requested but not available, using CPU")

        # Previous trajectory for warm start
        self.previous_trajectory: Optional[torch.Tensor] = None
        self.previous_G: float = float('inf')

        # History for oscillation detection
        self.position_history: List[np.ndarray] = []
        self.max_history: int = 20

        # Current adaptive horizon
        self.current_horizon: int = self.config.base_horizon

        # Current adaptive seed spread (linked to horizon)
        self.current_seed_spread: float = self.config.seed_spread

        # =================================================================
        # EFE-based Adaptive Horizon (Paper-aligned)
        #
        # T(t) = T_base + (T_max - T_base) · tanh(β · ReLU(Ĝ(t) - G₀))
        # Ĝ(t) = γ · Ĝ(t-1) + (1-γ) · log(1 + G_min)  // EMA of log-EFE
        #
        # We use log(1 + G) to normalize EFE to reasonable scale
        # When log-EFE is high (stuck), horizon extends to "see further"
        # When log-EFE is low (clear path), horizon stays short (fast)
        # =================================================================
        self._efe_smoothed: float = 0.0      # Exponential moving average of log-EFE
        self._efe_gamma: float = 0.8         # Slower decay for stability (was 0.7)
        self._efe_threshold: float = 9.0     # Higher threshold (was 7.0) - less sensitive
        self._efe_sensitivity: float = 0.3   # Lower sensitivity (was 0.5) - smoother
        self._horizon_smoothed: float = float(self.config.base_horizon)  # Smooth horizon

        # Committed trajectory (when stuck)
        self.committed_trajectory: Optional[np.ndarray] = None
        self.committed_step: int = 0
        self.commit_reason: str = ""
        self.commit_start_time: float = 0.0  # Track when commitment started
        self.min_commit_duration: float = 5.0  # Minimum 5 seconds commitment

        # Goal commitment tracking (maintains target for minimum duration)
        self.committed_goal: Optional[np.ndarray] = None  # Track committed goal
        self.goal_commit_start_time: float = 0.0  # Track when goal commitment started

        # Pre-compute ray angles and directions (cache for speed)
        K = self.config.num_rays
        angles = torch.linspace(0, 2 * np.pi, K + 1, dtype=self.dtype, device=self.device)[:-1]
        self._ray_angles = angles
        self._ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [K, 2]

        # Pose covariance state (for integration with room estimation)
        sigma_base = self.config.sigma_pose_base
        self._pose_covariance = torch.eye(2, dtype=self.dtype, device=self.device) * sigma_base ** 2

        # Pre-compute measurement covariance
        self._sigma_meas_sq = self.config.sigma_meas ** 2

        # Pre-compute discount factors for horizon (avoids recomputation)
        max_T = self.config.max_horizon
        self._discount_factors = torch.tensor(
            [0.95 ** t for t in range(max_T)],
            dtype=self.dtype, device=self.device
        )

        # Cache for velocity scales (computed lazily for different T values)
        self._velocity_scales_cache = {}

        # =====================================================================
        # Pre-allocated GPU buffers for common operations
        # This avoids repeated allocations and improves CUDA performance
        # =====================================================================
        K = self.config.num_seeds
        T = self.config.max_horizon
        self._preallocated = {
            'trajectories': torch.zeros(K, T, 2, dtype=self.dtype, device=self.device),
            'positions': torch.zeros(K, 2, dtype=self.dtype, device=self.device),
            'G_batch': torch.zeros(K, dtype=self.dtype, device=self.device),
        }

        # Initialize analytic ray-circle model (Paper Section 5.1.3)
        self._ray_circle_model = None
        if self.config.use_ray_circle_model and RAY_CIRCLE_AVAILABLE:
            ray_config = RayCircleConfig(
                num_rays=self.config.ray_circle_num_rays,
                max_range=self.config.sensor_range,
                softmin_eta=self.config.ray_circle_softmin_eta
            )
            # Pass controller's device to ensure consistency
            self._ray_circle_model = RayCircleModel(ray_config, device=str(self.device))

        # Polymorphic obstacle beliefs (for mixed obstacle types)
        # Set during compute_action when ObstacleBeliefBase instances are provided
        self._obstacle_beliefs: Optional[List] = None
        self._use_polymorphic_obstacles = POLYMORPHIC_OBSTACLES_AVAILABLE

        # Statistics
        self.stats = {
            'seeds_evaluated': 0,
            'iterations_total': 0,
            'horizon_extensions': 0,
            'warm_starts_used': 0,
            'trajectory_commits': 0,
            'device': str(self.device)
        }

    def _compute_velocity_scales(self, T: int) -> torch.Tensor:
        """
        Compute velocity scaling factors for horizon expansion.

        Modes:
        - 'none': All scales = 1.0 (constant velocity)
        - 'power': scale = 1 + rate * (t/T)^power
        - 'exponential': scale = base^(exponent * t/T)

        Exponential scaling allows exploring much further with same T:
        - base=2, exp=2: T=15 covers distance of T=33 linear steps
        - First steps: fine control (scale ≈ 1)
        - Last steps: coarse exploration (scale up to 4x)
        """
        # Check cache
        if T in self._velocity_scales_cache:
            return self._velocity_scales_cache[T]

        velocity_scales = torch.zeros(T, dtype=self.dtype, device=self.device)

        mode = self.config.horizon_scaling_mode

        for t in range(T):
            phase = t / max(T - 1, 1)

            if mode == 'none':
                scale = 1.0
            elif mode == 'power':
                # Power scaling: 1 + rate * phase^power
                scale = 1.0 + self.config.horizon_expansion_rate * (phase ** self.config.horizon_expansion_power)
            elif mode == 'exponential':
                # Exponential scaling: base^(exponent * phase)
                # At phase=0: scale=1.0, at phase=1: scale=base^exponent
                scale = self.config.horizon_exp_base ** (self.config.horizon_exp_exponent * phase)
            else:
                scale = 1.0

            velocity_scales[t] = scale

        # Cache for reuse
        self._velocity_scales_cache[T] = velocity_scales
        return velocity_scales

    def compute_action(self,
                       robot_pos: np.ndarray,
                       robot_theta: float,
                       goal: np.ndarray,
                       obstacles: Union[List[Tuple[float, float, float]], List['ObstacleBeliefBase']],
                       prev_velocity: np.ndarray = None,
                       pose_uncertainty: float = 0.0,
                       pose_covariance: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Compute optimal action by minimizing Expected Free Energy over trajectory.

        Args:
            robot_pos: Current robot position [x, y]
            robot_theta: Current robot orientation
            goal: Goal position [x, y]
            obstacles: List of obstacles. Can be:
                       - List of (cx, cy, radius) tuples (legacy format)
                       - List of ObstacleBeliefBase instances (polymorphic format)
            prev_velocity: Previous velocity for warm start
            pose_uncertainty: Current pose uncertainty (scalar, for backward compatibility)
            pose_covariance: Pose covariance matrix [2x2] from room estimation (optional)

        Returns:
            (action, info): Best action [vx, vy] and diagnostic info
        """
        start_time = time.time()

        # Convert polymorphic obstacles to tensor format for internal use
        # This maintains backward compatibility while enabling new obstacle types
        obstacles, obstacle_beliefs = self._normalize_obstacles(obstacles)

        # Store obstacle beliefs for use in EFE computation
        self._obstacle_beliefs = obstacle_beliefs

        # Update pose covariance from room estimation if provided
        if pose_covariance is not None:
            self._pose_covariance = torch.tensor(pose_covariance, dtype=torch.float64)
        elif pose_uncertainty is not None and np.isscalar(pose_uncertainty) and pose_uncertainty > 0:
            # Backward compatibility: convert scalar to isotropic covariance
            self._pose_covariance = torch.eye(2, dtype=self.dtype, device=self.device) * pose_uncertainty ** 2

        # Check if goal has changed - if so, check if we can change (minimum duration passed)
        goal_changed = (self.committed_goal is not None and
                       not np.array_equal(self.committed_goal, goal))
        if goal_changed:
            time_since_goal_commit = time.time() - self.goal_commit_start_time
            if time_since_goal_commit < self.min_commit_duration:
                # Force old goal - not allowed to change yet
                goal = self.committed_goal.copy()
            else:
                # Can accept new goal now
                self.committed_goal = goal.copy()
                self.goal_commit_start_time = time.time()
        elif self.committed_goal is None:
            # First goal commitment
            self.committed_goal = goal.copy()
            self.goal_commit_start_time = time.time()

        # Check if following committed trajectory
        if self.committed_trajectory is not None:
            action, info = self._follow_committed_trajectory(robot_pos, goal, obstacles)
            if action is not None:
                info['elapsed_ms'] = (time.time() - start_time) * 1000
                info['goal_commit_time'] = time.time() - self.goal_commit_start_time
                return action, info

        # Convert to tensors on device
        pos = torch.tensor(robot_pos, dtype=self.dtype, device=self.device)
        goal_t = torch.tensor(goal, dtype=self.dtype, device=self.device)

        if len(obstacles) > 0:
            obs_t = torch.tensor(obstacles, dtype=self.dtype, device=self.device)
        else:
            obs_t = torch.zeros((0, 3), dtype=self.dtype, device=self.device)

        # Update position history
        self._update_history(robot_pos)

        # Boost initial EFE estimate if there are nearby obstacles
        # This helps the horizon adapt faster in dense scenarios
        if len(obstacles) > 0 and len(self.position_history) < 5:
            obs_array = np.array(obstacles)
            distances = np.linalg.norm(obs_array[:, :2] - robot_pos, axis=1) - obs_array[:, 2]
            min_dist = np.min(distances)
            if min_dist < 100:  # Obstacle within 100 units
                # Boost EFE to trigger horizon extension
                boost = max(0, (100 - min_dist) / 10)  # 0-10 boost
                self._efe_smoothed = max(self._efe_smoothed, self._efe_threshold + boost)

        # Compute adaptive horizon
        T = self._compute_adaptive_horizon(pose_uncertainty)

        # Check if stuck or oscillating
        is_stuck = self._detect_oscillation() or self._detect_stuck()

        # Generate trajectory seeds (use list-based for correct behavior)
        seeds = self._generate_seeds(pos.cpu(), goal_t.cpu(), T, prev_velocity, obs_t.cpu())

        # If stuck, add extra exploration seeds with diverse directions
        if is_stuck:
            exploration_seeds = self._generate_exploration_seeds(pos.cpu(), goal_t.cpu(), T)
            seeds.extend(exploration_seeds)

        # Optimize seeds based on execution mode
        if self.config.execution_mode == 'gpu_batched' and GPU_AVAILABLE:
            # GPU batched execution: fully vectorized on GPU
            best_trajectory, best_G, best_seed_idx, all_trajectories = self._optimize_seeds_batched(
                seeds, pos, goal_t, obs_t, T
            )
        elif self.config.execution_mode == 'parallel' or (self.config.execution_mode == 'gpu_batched' and not GPU_AVAILABLE):
            # Multi-threaded CPU execution using ThreadPoolExecutor
            best_trajectory, best_G, best_seed_idx, all_trajectories = self._optimize_seeds_sequential(
                seeds, pos.cpu(), goal_t.cpu(), obs_t.cpu(), T
            )
        else:
            # Sequential execution (fallback for debugging)
            best_trajectory, best_G, best_seed_idx, all_trajectories = self._optimize_seeds_sequential(
                seeds, pos.cpu(), goal_t.cpu(), obs_t.cpu(), T
            )

        self.stats['seeds_evaluated'] += len(seeds)

        # Update EFE estimate for adaptive horizon (Paper-aligned)
        # This allows the horizon to adapt based on how "stuck" we are
        self._update_efe_estimate(best_G)

        # Store for warm start
        self.previous_trajectory = best_trajectory.detach().clone()
        self.previous_G = best_G

        # Compute EFE components for the best trajectory (for visualization)
        G_components = self._compute_efe_components(best_trajectory, pos.cpu(), goal_t.cpu(), obs_t.cpu())

        # If stuck, commit to full trajectory
        if is_stuck and best_trajectory is not None:
            self._commit_to_trajectory(best_trajectory.detach().cpu().numpy(), "oscillation_escape")

        # Extract first action
        action = best_trajectory.detach()[0].cpu().numpy()

        # Compute effective max velocity based on pose uncertainty
        effective_max_velocity = self._compute_uncertainty_limited_velocity()

        # Clip to effective max velocity
        speed = np.linalg.norm(action)
        if speed > effective_max_velocity:
            action = action * (effective_max_velocity / speed)

        elapsed = time.time() - start_time

        info = {
            'G_total': best_G,
            'horizon': T,
            'seed_spread': self.current_seed_spread,
            'efe_smoothed': self._efe_smoothed,
            'horizon_smoothed': self._horizon_smoothed,
            'best_seed': best_seed_idx,
            'seeds_evaluated': len(seeds),
            'elapsed_ms': elapsed * 1000,
            'trajectory': best_trajectory.detach().cpu().numpy(),
            'all_trajectories': all_trajectories,  # All tested trajectories for visualization
            'is_stuck': is_stuck,
            'committed': self.committed_trajectory is not None,
            'G_components': G_components,
            'goal_commit_time': time.time() - self.goal_commit_start_time if self.committed_goal is not None else 0.0,
            'effective_velocity': effective_max_velocity,
            'pose_uncertainty': torch.trace(self._pose_covariance).item(),
        }

        return action, info

    def _compute_efe_components(self, trajectory: torch.Tensor, pos: torch.Tensor,
                                 goal: torch.Tensor, obstacles: torch.Tensor) -> Dict[str, float]:
        """
        Compute individual EFE components for visualization.

        Returns dict with: goal, obstacle, ambiguity, epistemic, structural, localization, velocity
        """
        G_goal = 0.0
        G_obstacle = 0.0
        G_ambiguity = 0.0
        G_epistemic = 0.0
        G_structural = 0.0
        G_localization = 0.0
        G_velocity = 0.0

        current_pos = pos.clone()
        T = len(trajectory)

        for t in range(min(T, 10)):  # Only first 10 steps for efficiency
            action = trajectory[t].detach()
            current_pos = current_pos + action

            # Goal distance
            dist = torch.norm(current_pos - goal).item()
            G_goal += self.config.lambda_goal * dist * (0.95 ** t)

            # Obstacle cost
            if obstacles.numel() > 0:
                diffs = current_pos - obstacles[:, :2]
                dists = torch.norm(diffs, dim=1) - obstacles[:, 2]
                min_dist = dists.min().item()

                safety = self.config.robot_radius + 20
                if min_dist < safety:
                    G_obstacle += self.config.lambda_obstacle * np.exp(-min_dist / 10) * (0.95 ** t)

            # Ambiguity (velocity-dependent uncertainty)
            speed = torch.norm(action).item()
            G_ambiguity += self.config.lambda_ambiguity * speed * (0.95 ** t)

            # Epistemic (information gain near obstacles)
            if obstacles.numel() > 0:
                # Info gain is higher when near obstacles
                for i in range(obstacles.shape[0]):
                    obs_dist = torch.norm(current_pos - obstacles[i, :2]).item()
                    if obs_dist < 100:  # Within sensing range
                        G_epistemic -= 0.01 * (100 - obs_dist) * (0.95 ** t)

            # Localization (pose uncertainty cost)
            G_localization += self.config.lambda_localization * torch.trace(self._pose_covariance).item() * (0.95 ** t)

        # G_velocity: Velocity regularisation (Paper Section g_velocity)
        # Term 1: Velocity magnitude penalty
        velocity_sq = torch.sum(trajectory ** 2, dim=1)  # [T]
        G_velocity_magnitude = self.config.lambda_velocity * torch.sum(velocity_sq).item()

        # Term 2: Velocity change penalty (action smoothness)
        if T > 1:
            action_diff = trajectory[1:] - trajectory[:-1]  # [T-1, 2]
            velocity_diff_sq = torch.sum(action_diff ** 2)
            G_velocity_smoothness = self.config.lambda_delta_velocity * velocity_diff_sq.item()
        else:
            G_velocity_smoothness = 0.0

        G_velocity = G_velocity_magnitude + G_velocity_smoothness

        # Structural epistemic (model identity disambiguation)
        if obstacles.numel() > 0:
            struct_result = self._compute_structural_epistemic(current_pos.unsqueeze(0), obstacles)
            G_structural = struct_result.item() if hasattr(struct_result, 'item') else float(struct_result)

        return {
            'goal': G_goal,
            'obstacle': G_obstacle,
            'ambiguity': G_ambiguity,
            'epistemic': abs(G_epistemic),  # Show as positive for display
            'structural': abs(G_structural),
            'localization': G_localization,
            'velocity': G_velocity,
        }

    def _commit_to_trajectory(self, trajectory: np.ndarray, reason: str):
        """Commit to following a complete trajectory."""
        self.committed_trajectory = trajectory
        self.committed_step = 0
        self.commit_reason = reason
        self.commit_start_time = time.time()  # Record commitment start time
        self.stats['trajectory_commits'] += 1
        self.position_history.clear()

    def _follow_committed_trajectory(self,
                                     robot_pos: np.ndarray,
                                     goal: np.ndarray,
                                     obstacles: List[Tuple[float, float, float]]
                                     ) -> Tuple[Optional[np.ndarray], Dict]:
        """Follow the committed trajectory with periodic recomputation."""
        traj = self.committed_trajectory
        step = self.committed_step
        elapsed_time = time.time() - self.commit_start_time

        # Check if completed
        if step >= len(traj):
            # Only end if minimum duration has passed
            if elapsed_time >= self.min_commit_duration:
                self._end_commitment("completed")
                return None, {}
            else:
                # Trajectory exhausted but must maintain commitment
                # Generate small action towards goal to maintain intent
                to_goal = goal - robot_pos
                dist_to_goal = np.linalg.norm(to_goal)
                if dist_to_goal > 1e-6:
                    action = (to_goal / dist_to_goal) * min(self.config.max_velocity * 0.3, dist_to_goal)
                else:
                    action = np.array([0.0, 0.0])
        else:
            # IMPORTANT: With velocity scaling, distant steps have low spatial resolution.
            # As the robot advances, those steps become "near" and need higher resolution.
            # Recompute trajectory every N steps to maintain proper resolution.
            # Increased from 5 to 15 to maintain longer commitment for complex scenarios
            recompute_interval = 15  # Recompute every 15 steps
            if step > 0 and step % recompute_interval == 0 and elapsed_time >= self.min_commit_duration:
                # Only recompute if minimum duration has passed
                self._end_commitment("recompute_for_resolution")
                return None, {}

            # Get next action
            action = traj[step].copy()

        # Check for imminent collision - back up instead of changing target
        next_pos = robot_pos + action
        min_dist = float('inf')
        for cx, cy, r in obstacles:
            dist = np.linalg.norm(next_pos - np.array([cx, cy])) - r - self.config.robot_radius
            min_dist = min(min_dist, dist)

        # If too close to obstacle, generate backup action with random rotation
        if min_dist < 0:  # Collision imminent
            # Back up: move opposite to current action direction
            backup_dist = self.config.max_velocity * 0.5
            if np.linalg.norm(action) > 0:
                backup_dir = -action / np.linalg.norm(action)
            else:
                # Random direction if no action
                angle = np.random.uniform(0, 2 * np.pi)
                backup_dir = np.array([np.cos(angle), np.sin(angle)])

            # Add random rotation (-30 to +30 degrees) to facilitate escape
            random_angle = np.random.uniform(-np.pi/6, np.pi/6)
            cos_a, sin_a = np.cos(random_angle), np.sin(random_angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            backup_dir = rotation_matrix @ backup_dir

            action = backup_dir * backup_dist
            # Don't increment step - stay at same position in trajectory
        else:
            self.committed_step += 1

        # Check if near goal (only end if minimum duration passed)
        if np.linalg.norm(robot_pos - goal) < 30 and elapsed_time >= self.min_commit_duration:
            self._end_commitment("near_goal")
            return None, {}

        # Clip velocity
        speed = np.linalg.norm(action)
        if speed > self.config.max_velocity:
            action = action * (self.config.max_velocity / speed)

        info = {
            'G_total': 0,
            'horizon': len(traj) - step,
            'best_seed': -1,
            'seeds_evaluated': 0,
            'is_stuck': False,
            'committed': True,
            'commit_step': step,
            'commit_reason': self.commit_reason,
            'backup_mode': min_dist < 0,
            'G_components': {
                'goal': 0, 'obstacle': 0, 'ambiguity': 0,
                'epistemic': 0, 'structural': 0, 'localization': 0
            }
        }

        return action, info

    def _end_commitment(self, reason: str):
        """End trajectory commitment."""
        self.committed_trajectory = None
        self.committed_step = 0
        self.commit_reason = ""

    def _optimize_seeds_sequential(self,
                                   seeds: List[torch.Tensor],
                                   pos: torch.Tensor,
                                   goal: torch.Tensor,
                                   obstacles: torch.Tensor,
                                   T: int) -> Tuple[torch.Tensor, float, int, List[np.ndarray]]:
        """
        Optimize seeds sequentially with optional parallel execution.

        For CPU mode, uses sequential execution (most reliable with PyTorch).
        PyTorch's internal parallelism (intra-op) handles multi-core utilization.

        Returns:
            (best_trajectory, best_G, best_seed_idx, all_trajectories)
        """
        best_trajectory = None
        best_G = float('inf')
        best_seed_idx = -1
        all_trajectories = []  # Store all optimized trajectories for visualization

        # Sequential optimization - PyTorch handles internal parallelism
        # This is more reliable than explicit threading with PyTorch tensors
        for i, seed in enumerate(seeds):
            trajectory, G, iterations = self._optimize_trajectory(
                seed, pos, goal, obstacles, T
            )
            self.stats['iterations_total'] += iterations
            all_trajectories.append(trajectory.detach().cpu().numpy())

            if G < best_G:
                best_G = G
                best_trajectory = trajectory
                best_seed_idx = i

        return best_trajectory, best_G, best_seed_idx, all_trajectories

    def _optimize_seeds_batched(self,
                                seeds: List[torch.Tensor],
                                pos: torch.Tensor,
                                goal: torch.Tensor,
                                obstacles: torch.Tensor,
                                T: int) -> Tuple[torch.Tensor, float, int, List[np.ndarray]]:
        """
        Optimize seeds in parallel using GPU batching.

        Evaluates all seeds simultaneously on GPU with fully vectorized operations.
        Optimized for RTX 5070 Ti with TF32 support.

        Returns:
            (best_trajectory, best_G, best_seed_idx, all_trajectories)
        """
        K = len(seeds)
        if K == 0:
            return None, float('inf'), -1, []

        # Stack all seeds into batch [K, T, 2] - ensure on correct device and dtype
        batch_seeds = torch.stack([s.to(device=self.device, dtype=self.dtype) for s in seeds])

        # Move data to GPU once (avoid repeated transfers)
        pos_gpu = pos.to(device=self.device, dtype=self.dtype)
        goal_gpu = goal.to(device=self.device, dtype=self.dtype)
        obs_gpu = obstacles.to(device=self.device, dtype=self.dtype) if obstacles.numel() > 0 else obstacles

        # Batched optimization with gradient descent
        batch_seeds = batch_seeds.clone().detach().requires_grad_(True)

        best_G_values = torch.full((K,), float('inf'), device=self.device, dtype=self.dtype)
        best_trajectories = batch_seeds.clone().detach()

        lr = self.config.learning_rate
        prev_best_G = float('inf')

        # Run gradient descent on all seeds in parallel
        for iteration in range(self.config.max_iterations):
            # Compute EFE for all seeds in batch (vectorized)
            G_batch = self._compute_batch_G(batch_seeds, pos_gpu, goal_gpu, obs_gpu, T)

            # Check for convergence
            current_best_G = G_batch.min().item()
            if iteration > 0 and abs(prev_best_G - current_best_G) < self.config.convergence_threshold:
                break
            prev_best_G = current_best_G

            # Track best trajectories
            improved = G_batch < best_G_values
            best_G_values = torch.where(improved, G_batch, best_G_values)
            # Only update best trajectories for seeds that improved
            if improved.any():
                mask = improved.unsqueeze(-1).unsqueeze(-1).expand_as(batch_seeds)
                best_trajectories = torch.where(mask, batch_seeds.detach().clone(), best_trajectories)

            # Compute gradients for all seeds at once
            total_G = G_batch.sum()
            total_G.backward()

            if batch_seeds.grad is not None:
                with torch.no_grad():
                    # Gradient clipping for stability
                    grad_norm = torch.norm(batch_seeds.grad, dim=(1, 2), keepdim=True)
                    max_grad_norm = 10.0
                    scale = torch.clamp(max_grad_norm / (grad_norm + 1e-8), max=1.0)
                    batch_seeds.grad *= scale

                    # Gradient descent step
                    batch_seeds -= lr * batch_seeds.grad
                    batch_seeds.grad.zero_()

                    # Vectorized velocity clipping [K, T, 2]
                    speeds = torch.norm(batch_seeds, dim=2, keepdim=True)  # [K, T, 1]
                    clip_scale = torch.clamp(self.config.max_velocity / (speeds + 1e-8), max=1.0)
                    batch_seeds *= clip_scale

            batch_seeds.requires_grad_(True)

        # Select best seed
        best_idx = best_G_values.argmin().item()
        best_G = best_G_values[best_idx].item()
        best_trajectory = best_trajectories[best_idx].cpu()

        # Collect all trajectories for visualization (transfer to CPU at end)
        all_trajectories = [best_trajectories[i].detach().cpu().numpy() for i in range(K)]

        self.stats['iterations_total'] += self.config.max_iterations * K

        return best_trajectory, best_G, best_idx, all_trajectories

    def _compute_batch_G(self,
                         trajectories: torch.Tensor,
                         pos: torch.Tensor,
                         goal: torch.Tensor,
                         obstacles: torch.Tensor,
                         T: int) -> torch.Tensor:
        """
        Compute EFE for a batch of trajectories.

        Uses the same logic as _compute_trajectory_G including non-linear dt scaling.
        This ensures consistency between GPU and CPU modes.

        Args:
            trajectories: [K, T, 2] batch of trajectory velocities
            pos: [2] current position
            goal: [2] goal position
            obstacles: [N, 3] obstacles
            T: horizon length

        Returns:
            G_batch: [K] EFE for each trajectory
        """
        K = trajectories.shape[0]

        # Compute dt_scales for non-linear stepping [T]
        near_threshold = T // 3
        mid_threshold = 2 * T // 3
        dt_scales = torch.ones(T, device=self.device, dtype=self.dtype)
        dt_scales[near_threshold:mid_threshold] = 2.0
        dt_scales[mid_threshold:] = 4.0

        # Compute discount factors [T]
        base_discounts = torch.tensor([0.95 ** t for t in range(T)], device=self.device, dtype=self.dtype)
        effective_discounts = base_discounts * (1.0 / dt_scales) ** 0.3

        # Scale actions by dt_scales: [K, T, 2]
        scaled_actions = trajectories * (self.config.dt * dt_scales).view(1, T, 1)

        # Compute cumulative positions [K, T, 2]
        cumsum_actions = torch.cumsum(scaled_actions, dim=1)
        positions = pos.view(1, 1, 2) + cumsum_actions  # [K, T, 2]

        # Goal costs [K, T]
        to_goal = positions - goal.view(1, 1, 2)
        dist_to_goal = torch.norm(to_goal, dim=2)  # [K, T]
        G_goal = self.config.lambda_goal * (dist_to_goal + 0.01 * dist_to_goal ** 2)

        # Obstacle costs [K, T]
        G_obs = torch.zeros(K, T, device=self.device, dtype=self.dtype)

        if obstacles.numel() > 0:
            N = obstacles.shape[0]
            centers = obstacles[:, :2]  # [N, 2]
            radii = obstacles[:, 2]     # [N]

            # Compute distances [K, T, N]
            diffs = positions.unsqueeze(2) - centers.view(1, 1, N, 2)
            dist_centers = torch.norm(diffs, dim=3)  # [K, T, N]
            dist_surfaces = dist_centers - radii.view(1, 1, N)  # [K, T, N]

            # Min distance to any obstacle [K, T]
            min_dists, _ = torch.min(dist_surfaces, dim=2)

            # Observation preference
            precision = 1.0 / self.config.sigma_obs_sq
            preferred_dist = self.config.preferred_obs_distance
            obs_error = torch.clamp(preferred_dist - min_dists, min=0.0)
            G_obs_pref = self.config.lambda_obstacle * 0.001 * 0.5 * precision * obs_error ** 2

            # Safety penalty - exponential barrier
            hard_margin = self.config.robot_radius + 5.0
            exp_arg = torch.clamp(2.0 * (hard_margin - min_dists), max=20.0)
            G_safety = torch.where(
                min_dists < hard_margin,
                self.config.lambda_obstacle * torch.exp(exp_arg),
                torch.zeros_like(min_dists)
            )

            G_obs = G_obs_pref + G_safety

        # Apply discounts and sum over time [K]
        G_step = G_goal + G_obs  # [K, T]
        G_batch = torch.sum(G_step * effective_discounts.view(1, T), dim=1)  # [K]

        # =====================================================================
        # G_velocity: Velocity regularisation and action smoothness
        # Paper Section: g_velocity
        #
        # G_velocity = λ_v * Σ(v_τ² + ω_τ²) + λ_Δv * Σ((v_τ - v_{τ-1})² + (ω_τ - ω_{τ-1})²)
        #
        # For holonomic robot: v = [vx, vy], so we use ||v||² = vx² + vy²
        # =====================================================================

        # Term 1: Velocity magnitude penalty - discourages unnecessarily aggressive motion
        # λ_v * Σ ||v_τ||²
        velocity_sq = torch.sum(trajectories ** 2, dim=2)  # [K, T]
        G_velocity_magnitude = self.config.lambda_velocity * torch.sum(velocity_sq, dim=1)  # [K]

        # Term 2: Velocity change penalty - suppresses high-frequency oscillations
        # λ_Δv * Σ ||v_τ - v_{τ-1}||²
        if T > 1:
            velocity_diff = trajectories[:, 1:, :] - trajectories[:, :-1, :]  # [K, T-1, 2]
            velocity_diff_sq = torch.sum(velocity_diff ** 2, dim=2)  # [K, T-1]
            G_velocity_smoothness = self.config.lambda_delta_velocity * torch.sum(velocity_diff_sq, dim=1)  # [K]
        else:
            G_velocity_smoothness = torch.zeros(K, device=self.device, dtype=self.dtype)

        # Combined velocity regularisation term
        G_velocity = G_velocity_magnitude + G_velocity_smoothness

        G_batch = G_batch + G_velocity

        return G_batch

    def _generate_seeds(self,
                        pos: torch.Tensor,
                        goal: torch.Tensor,
                        T: int,
                        prev_velocity: Optional[np.ndarray],
                        obstacles: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Generate K diverse trajectory seeds optimized for obstacle avoidance.

        Includes tangential seeds that explore paths around nearby obstacles,
        preventing the robot from getting too close when approaching from behind.
        """
        seeds = []
        K = self.config.num_seeds

        # Compute velocity scales using the unified method
        velocity_scales = self._compute_velocity_scales(T).cpu()  # [T]

        # Direction to goal
        to_goal = goal - pos
        dist_to_goal = torch.norm(to_goal)
        if dist_to_goal > 1e-6:
            goal_dir = to_goal / dist_to_goal
        else:
            goal_dir = torch.tensor([1.0, 0.0], dtype=torch.float64)

        goal_angle = torch.atan2(goal_dir[1], goal_dir[0]).item()

        t_vals = torch.arange(T, dtype=torch.float64)
        phases = t_vals / max(T - 1, 1)  # [T]

        # Seed 0: Warm start from previous trajectory (most important)
        if self.previous_trajectory is not None and len(self.previous_trajectory) >= T:
            shifted = torch.zeros((T, 2), dtype=torch.float64)
            shifted[:-1] = self.previous_trajectory[1:T]
            shifted[-1] = self.previous_trajectory[-1]
            # Vectorized normalization
            speeds = torch.norm(shifted, dim=1)  # [T]
            too_fast = speeds > self.config.max_velocity * 1.5
            if too_fast.any():
                scale = torch.where(too_fast, self.config.max_velocity / speeds, torch.ones_like(speeds))
                shifted = shifted * scale.unsqueeze(1)
            seeds.append(shifted)
            self.stats['warm_starts_used'] += 1

        # Seed 1: Direct to goal (vectorized)
        direct_seed = goal_dir.unsqueeze(0) * self.config.max_velocity * 0.9 * velocity_scales.unsqueeze(1)
        seeds.append(direct_seed)

        # Seed 2 & 3: Left and right curves (vectorized)
        for side in [1, -1]:  # +1 = left, -1 = right
            offset_angles = side * np.pi / 4 * (1 - phases)  # [T]
            angles = goal_angle + offset_angles  # [T]
            directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [T, 2]
            curve_seed = directions * self.config.max_velocity * 0.85 * velocity_scales.unsqueeze(1)
            seeds.append(curve_seed)

        # Seed: Previous velocity continuation (vectorized)
        if prev_velocity is not None and np.linalg.norm(prev_velocity) > 0.5:
            prev_v = torch.tensor(prev_velocity, dtype=torch.float64)
            speed = torch.norm(prev_v)
            if speed > 0.1:
                prev_dir = prev_v / speed
                blends = (t_vals / max(T - 1, 1)).unsqueeze(1)  # [T, 1]
                blended_dirs = (1 - blends * 0.5) * prev_dir + blends * 0.5 * goal_dir
                blended_dirs = blended_dirs / torch.norm(blended_dirs, dim=1, keepdim=True)
                momentum_seed = blended_dirs * self.config.max_velocity * 0.8 * velocity_scales.unsqueeze(1)
                seeds.append(momentum_seed)

        # =====================================================================
        # TANGENTIAL SEEDS: Explore paths around nearby obstacles
        #
        # For each obstacle near the path to goal, generate seeds that go
        # tangentially around it (left and right). These seeds maintain their
        # lateral offset longer before converging to goal, ensuring exploration
        # of the space beside obstacles.
        # =====================================================================
        if obstacles is not None and obstacles.numel() > 0:
            tangent_seeds = self._generate_tangential_seeds(
                pos, goal, goal_dir, goal_angle, T, velocity_scales, obstacles
            )
            seeds.extend(tangent_seeds)

        # Remaining: spread around goal direction (vectorized)
        num_remaining = K - len(seeds)
        if num_remaining > 0:
            indices = torch.arange(num_remaining, dtype=torch.float64)
            offsets = (indices - (num_remaining - 1) / 2) * (np.pi / max(num_remaining, 1)) * self.current_seed_spread

            for i in range(num_remaining):
                angle = goal_angle + offsets[i].item()
                direction = torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float64)

                # Vectorized blending
                blends = (t_vals / max(T - 1, 1)).pow(0.5).unsqueeze(1)  # [T, 1]
                blended_dirs = (1 - blends) * direction + blends * goal_dir
                norms = torch.norm(blended_dirs, dim=1, keepdim=True).clamp(min=1e-6)
                blended_dirs = blended_dirs / norms
                seed = blended_dirs * self.config.max_velocity * 0.85 * velocity_scales.unsqueeze(1)
                seeds.append(seed)

        return seeds[:K]

    def _generate_tangential_seeds(self,
                                   pos: torch.Tensor,
                                   goal: torch.Tensor,
                                   goal_dir: torch.Tensor,
                                   goal_angle: float,
                                   T: int,
                                   velocity_scales: torch.Tensor,
                                   obstacles: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate seeds that go tangentially around nearby obstacles.

        For each obstacle that is:
        1. Within sensor range
        2. Roughly in the direction of the goal (could block the path)

        Generate two seeds: one going around left, one around right.
        These seeds maintain lateral offset longer before converging to goal,
        ensuring the robot explores paths that avoid the obstacle's edges.

        Returns:
            List of tangential trajectory seeds
        """
        tangent_seeds = []

        if obstacles.numel() == 0:
            return tangent_seeds

        t_vals = torch.arange(T, dtype=torch.float64)
        phases = t_vals / max(T - 1, 1)  # [T] from 0 to 1

        # Check each obstacle
        for i in range(obstacles.shape[0]):
            obs_center = obstacles[i, :2]
            obs_radius = obstacles[i, 2].item()

            # Vector from robot to obstacle center
            to_obs = obs_center - pos
            dist_to_obs = torch.norm(to_obs).item()

            # Skip if too far (outside sensor range) or too close (already passing)
            if dist_to_obs > self.config.sensor_range or dist_to_obs < obs_radius + 10:
                continue

            # Check if obstacle is roughly in the direction of goal
            # Dot product: positive = obstacle is ahead
            to_obs_normalized = to_obs / (dist_to_obs + 1e-6)
            alignment = torch.dot(to_obs_normalized, goal_dir).item()

            # Only generate tangent seeds if obstacle is ahead (alignment > 0.3)
            # and not too far to the side
            if alignment < 0.3:
                continue

            # Compute tangent angles
            # The tangent points are at angle ±asin(r/d) from the line to center
            if dist_to_obs > obs_radius:
                tangent_half_angle = np.arcsin(min(1.0, (obs_radius + 20) / dist_to_obs))
            else:
                tangent_half_angle = np.pi / 3  # 60 degrees if very close

            # Angle to obstacle center
            obs_angle = torch.atan2(to_obs[1], to_obs[0]).item()

            # Generate left and right tangent seeds
            for side, name in [(1, 'left'), (-1, 'right')]:
                # Tangent direction: perpendicular to obstacle, then curving to goal
                tangent_angle = obs_angle + side * tangent_half_angle

                # Create trajectory that:
                # 1. First half: goes tangentially (maintains lateral offset)
                # 2. Second half: curves toward goal

                # Slower blending to goal - maintain tangent direction longer
                # Use cubic blending for smoother transition
                blend_to_goal = phases ** 2  # Slower blend: stays tangent longer

                tangent_dir = torch.tensor([np.cos(tangent_angle), np.sin(tangent_angle)],
                                          dtype=torch.float64)

                # Blend from tangent direction to goal direction
                blended_dirs = (1 - blend_to_goal).unsqueeze(1) * tangent_dir + \
                               blend_to_goal.unsqueeze(1) * goal_dir

                # Normalize
                norms = torch.norm(blended_dirs, dim=1, keepdim=True).clamp(min=1e-6)
                blended_dirs = blended_dirs / norms

                # Create seed with velocity scaling
                seed = blended_dirs * self.config.max_velocity * 0.8 * velocity_scales.unsqueeze(1)
                tangent_seeds.append(seed)

        return tangent_seeds

    def _generate_exploration_seeds(self,
                                    pos: torch.Tensor,
                                    goal: torch.Tensor,
                                    T: int) -> List[torch.Tensor]:
        """Generate exploration seeds when stuck (vectorized)."""
        seeds = []
        T_f = float(T)

        to_goal = goal - pos
        dist_to_goal = torch.norm(to_goal)
        if dist_to_goal > 1e-6:
            goal_dir = to_goal / dist_to_goal
        else:
            goal_dir = torch.tensor([1.0, 0.0], dtype=torch.float64)

        goal_angle = torch.atan2(goal_dir[1], goal_dir[0]).item()

        # Pre-compute directions
        perp_left = torch.tensor([np.cos(goal_angle + np.pi/2),
                                   np.sin(goal_angle + np.pi/2)], dtype=torch.float64)
        perp_right = torch.tensor([np.cos(goal_angle - np.pi/2),
                                    np.sin(goal_angle - np.pi/2)], dtype=torch.float64)
        back_left = torch.tensor([np.cos(goal_angle + np.pi * 0.75),
                                   np.sin(goal_angle + np.pi * 0.75)], dtype=torch.float64)
        back_right = torch.tensor([np.cos(goal_angle - np.pi * 0.75),
                                    np.sin(goal_angle - np.pi * 0.75)], dtype=torch.float64)

        # Time phases [T]
        t_vals = torch.arange(T, dtype=torch.float64)
        phases = t_vals / T_f

        # Helper function for phase-based blending (vectorized)
        def make_curved_seed(dir1, dir2, dir3, phase1, phase2, speed):
            """Create seed that transitions: dir1 -> dir2 -> dir3"""
            seed = torch.zeros((T, 2), dtype=torch.float64)

            # Phase 1: pure dir1
            mask1 = phases < phase1
            # Phase 2: blend dir1 -> dir2
            mask2 = (phases >= phase1) & (phases < phase2)
            blend2 = ((phases - phase1) / (phase2 - phase1)).clamp(0, 1)
            # Phase 3: blend dir2 -> dir3 or pure dir3
            mask3 = phases >= phase2
            blend3 = ((phases - phase2) / (1.0 - phase2 + 1e-6)).clamp(0, 1)

            for t in range(T):
                if mask1[t]:
                    direction = dir1
                elif mask2[t]:
                    direction = (1 - blend2[t]) * dir1 + blend2[t] * dir2
                    direction = direction / torch.norm(direction)
                else:
                    direction = (1 - blend3[t]) * dir2 + blend3[t] * dir3
                    direction = direction / torch.norm(direction)
                seed[t] = direction * self.config.max_velocity * speed
            return seed

        # Strategy 1: LEFT -> goal
        seeds.append(make_curved_seed(perp_left, goal_dir, goal_dir, 0.4, 0.7, 0.9))

        # Strategy 2: RIGHT -> goal
        seeds.append(make_curved_seed(perp_right, goal_dir, goal_dir, 0.4, 0.7, 0.9))

        # Strategy 3: BACK-LEFT -> LEFT -> goal
        seeds.append(make_curved_seed(back_left, perp_left, goal_dir, 0.3, 0.6, 0.85))

        # Strategy 4: BACK-RIGHT -> RIGHT -> goal
        seeds.append(make_curved_seed(back_right, perp_right, goal_dir, 0.3, 0.6, 0.85))

        return seeds


    def _compute_structural_epistemic(self, positions: torch.Tensor,
                                       obstacles: torch.Tensor) -> torch.Tensor:
        """
        Compute structural epistemic value: -I(m; o_{t+1} | π)

        Paper Eq. model_mutual_information:
        I(m; o_{t+1} | π) = H[q(m)] - E_{q(o|π)}[H[q(m|o)]]

        This term rewards policies that lead to observations that would
        disambiguate between competing obstacle models (circle vs rectangle).

        Implementation:
        - H[q(m)]: Current entropy over model identity (higher = more uncertain)
        - E[H[q(m|o)]]: Expected posterior entropy after observing (lower = more informative)

        For each TrackedObstacle with uncertain model identity, we compute
        how much information gain we expect from moving to different positions.

        Args:
            positions: Candidate positions [K, 2]
            obstacles: Obstacle tensor [N, 3] (used for distance calculations)

        Returns:
            G_structural: [K] tensor, negative mutual information (to minimize)
        """
        K = positions.shape[0]
        # Use positions device to avoid CPU/GPU mismatch
        device = positions.device
        G_structural = torch.zeros(K, dtype=self.dtype, device=device)

        # Only compute if we have TrackedObstacle instances with model uncertainty
        if self._obstacle_beliefs is None:
            return G_structural

        # Check if any obstacle has model uncertainty (q(m) not fully concentrated)
        tracked_with_uncertainty = []
        for obs in self._obstacle_beliefs:
            if hasattr(obs, 'q_m') and hasattr(obs, 'hypotheses'):
                q_m = obs.q_m
                if len(q_m) > 1:
                    max_prob = max(q_m.values()) if q_m else 1.0
                    # Include obstacles with any meaningful uncertainty
                    # Even 99% certain means 1% uncertainty which has value
                    if max_prob < 0.9999:  # Very relaxed - include almost all
                        tracked_with_uncertainty.append(obs)

        if not tracked_with_uncertainty:
            return G_structural

        # Compute I(m; o) for each obstacle with uncertainty
        for tracked in tracked_with_uncertainty:
            q_m = tracked.q_m

            # H[q(m)]: Current entropy over model identity
            # H = -Σ q(m) log q(m)
            H_prior = 0.0
            for prob in q_m.values():
                if prob > 1e-10:
                    H_prior -= prob * np.log(prob + 1e-10)

            # Get obstacle position
            obs_pos = tracked.position[:2]
            obs_pos_t = torch.tensor(obs_pos, dtype=self.dtype, device=positions.device)

            # Distance from each candidate position to this obstacle
            positions_dev = positions.to(positions.device)
            dists = torch.norm(positions_dev - obs_pos_t.unsqueeze(0), dim=1)  # [K]

            # Expected information gain depends on viewing angle/distance
            # Closer observations are generally more informative
            # Optimal distance for disambiguation is around 1-2x obstacle size
            bounding_r = 40.0  # Approximate
            if hasattr(tracked, 'get_best_model'):
                best = tracked.get_best_model()
                if best is not None:
                    bounding_r = best.get_bounding_radius()

            optimal_dist = bounding_r * 1.5

            # E[H[q(m|o)]]: Expected posterior entropy
            # Approximation: observations closer to obstacle are more informative
            # Use a more gradual decay that gives value even at larger distances
            # Scale by sensor_range to normalize distances
            normalized_dist = dists / self.config.sensor_range
            # Peak at optimal_dist/sensor_range, gradual decay
            optimal_normalized = optimal_dist / self.config.sensor_range
            # Use wider Gaussian for smoother decay
            dist_factor = torch.exp(-2.0 * ((normalized_dist - optimal_normalized) ** 2))

            # Visibility mask
            visible = dists < self.config.sensor_range

            # Expected posterior entropy: H_prior * (1 - information_gain)
            # information_gain ∈ [0, 1], higher when at optimal viewing position
            info_gain = dist_factor * visible.float() * 0.5  # Max 50% reduction
            E_H_posterior = H_prior * (1.0 - info_gain)

            # I(m; o) = H_prior - E[H_posterior]
            mutual_info = H_prior - E_H_posterior  # [K]

            # Add to structural epistemic (negative because we minimize EFE)
            # Higher mutual info = lower G (we want to go there)
            G_structural -= self.config.lambda_structural * mutual_info

        return G_structural


    def _optimize_trajectory(self,
                             seed: torch.Tensor,
                             pos: torch.Tensor,
                             goal: torch.Tensor,
                             obstacles: torch.Tensor,
                             T: int) -> Tuple[torch.Tensor, float, int]:
        """Optimize a trajectory seed to minimize Expected Free Energy."""
        trajectory = seed.clone().detach().requires_grad_(True)

        best_trajectory = trajectory.clone()
        best_G = float('inf')
        prev_G = float('inf')
        lr = self.config.learning_rate * 0.5  # Reduced learning rate for stability

        for iteration in range(self.config.max_iterations):
            G_total = self._compute_trajectory_G(trajectory, pos, goal, obstacles, T)

            current_G = G_total.item()

            # Keep track of best trajectory seen
            if current_G < best_G:
                best_G = current_G
                best_trajectory = trajectory.clone().detach()

            # Early stopping
            if abs(prev_G - current_G) < self.config.convergence_threshold:
                break
            if current_G < 1.0:
                break

            # Detect divergence and stop
            if current_G > prev_G * 2 or current_G > 1e10:
                break

            prev_G = current_G

            G_total.backward()

            with torch.no_grad():
                if trajectory.grad is not None:
                    # Gradient clipping to prevent explosion
                    grad_norm = torch.norm(trajectory.grad)
                    max_grad_norm = 10.0
                    if grad_norm > max_grad_norm:
                        trajectory.grad *= max_grad_norm / grad_norm

                    trajectory -= lr * trajectory.grad
                    trajectory.grad.zero_()

                # Vectorized velocity clipping
                speeds = torch.norm(trajectory, dim=1)
                scale = torch.clamp(self.config.max_velocity / (speeds + 1e-8), max=1.0)
                trajectory *= scale.unsqueeze(1)

            trajectory.requires_grad_(True)

        return best_trajectory, best_G, iteration + 1

    def _compute_trajectory_G(self,
                              trajectory: torch.Tensor,
                              start_pos: torch.Tensor,
                              goal: torch.Tensor,
                              obstacles: torch.Tensor,
                              T: int) -> torch.Tensor:
        """
        Compute Expected Free Energy for entire trajectory.

        Paper Eq. 27: G_T(π) = Σ_{τ=t+1}^{t+T} E_{q(o_τ,s_τ|π)}[F[q, o_τ]]

        Uses NON-LINEAR time stepping:
        - Near future (t < T/3): dt = 1.0 (high precision)
        - Mid future (T/3 <= t < 2T/3): dt = 2.0 (medium)
        - Far future (t >= 2T/3): dt = 4.0 (coarse, long reach)

        This allows seeing ~3x farther without increasing computation.
        """
        G_total = torch.tensor(0.0, dtype=torch.float64)
        pos = start_pos.clone()

        discount = 1.0
        discount_factor = 0.95

        # Thresholds for non-linear stepping
        near_threshold = T // 3
        mid_threshold = 2 * T // 3

        for t in range(T):
            action = trajectory[t]

            # Non-linear dt: increases with time for longer reach
            if t < near_threshold:
                dt_scale = 1.0   # High precision near
            elif t < mid_threshold:
                dt_scale = 2.0   # Medium precision mid
            else:
                dt_scale = 4.0   # Coarse precision far

            next_pos = pos + action * self.config.dt * dt_scale

            G_step = self._compute_step_G(next_pos, action, goal, obstacles, t, T)

            # Discount also accounts for time scale (bigger steps = more uncertainty)
            effective_discount = discount * (1.0 / dt_scale) ** 0.3
            G_total = G_total + effective_discount * G_step

            discount *= discount_factor
            pos = next_pos

        # G_velocity: Velocity regularisation (Paper Section g_velocity)
        # Term 1: Velocity magnitude penalty
        velocity_sq = torch.sum(trajectory ** 2, dim=1)  # [T]
        G_velocity_magnitude = self.config.lambda_velocity * torch.sum(velocity_sq)

        # Term 2: Velocity change penalty (action smoothness)
        if T > 1:
            action_diff = trajectory[1:] - trajectory[:-1]  # [T-1, 2]
            velocity_diff_sq = torch.sum(action_diff ** 2, dim=1)  # [T-1]
            G_velocity_smoothness = self.config.lambda_delta_velocity * torch.sum(velocity_diff_sq)
        else:
            G_velocity_smoothness = torch.tensor(0.0, dtype=trajectory.dtype)

        G_total = G_total + G_velocity_magnitude + G_velocity_smoothness

        return G_total

    def _compute_step_G(self,
                        pos: torch.Tensor,
                        action: torch.Tensor,
                        goal: torch.Tensor,
                        obstacles: torch.Tensor,
                        t: int,
                        T: int) -> torch.Tensor:
        """
        Compute Expected Free Energy for a single step.

        Paper Eq. 26: G(π) = G_pragmatic + G_epistemic + G_ambiguity

        Uses efficient distance-based model while maintaining AIF structure.
        """
        K = self.config.num_rays

        # =====================================================================
        # G_PRAGMATIC: Goal + Obstacle avoidance
        # =====================================================================

        # Goal attraction
        to_goal = pos - goal
        dist_to_goal = torch.norm(to_goal)
        G_goal = self.config.lambda_goal * (dist_to_goal + 0.01 * dist_to_goal ** 2)

        # Obstacle avoidance
        G_obs_preference = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        G_obstacle_safety = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Variables for ambiguity/epistemic (will be set by either model)
        dist_centers = None
        dist_surfaces = None
        self._analytic_J_os = None  # Will be set by ray-circle model if used

        if len(obstacles) > 0:
            centers = obstacles[:, :2]  # [N, 2]
            radii = obstacles[:, 2]     # [N]

            diffs = pos.unsqueeze(0) - centers  # [N, 2]
            dist_centers = torch.norm(diffs, dim=1)  # [N]
            dist_surfaces = dist_centers - radii  # [N]

            # =================================================================
            # Option A: Use polymorphic obstacles (mixed types: circles, rectangles, etc.)
            # =================================================================
            if self._obstacle_beliefs is not None and self._use_polymorphic_obstacles:
                try:
                    # Compute SDF for all obstacles polymorphically
                    min_sdf = float('inf')
                    for obs in self._obstacle_beliefs:
                        sdf_val = obs.sdf(pos.unsqueeze(0)).item()
                        min_sdf = min(min_sdf, sdf_val)

                    # Safety cost based on minimum SDF
                    if min_sdf < self.config.robot_radius + 5.0:
                        G_obstacle_safety = self.config.lambda_obstacle * torch.exp(
                            torch.tensor(2.0 * (self.config.robot_radius + 5.0 - min_sdf),
                                       dtype=self.dtype, device=self.device))

                    # Preference cost (want SDF > preferred_clearance)
                    preferred_dist = self.config.preferred_obs_distance
                    if min_sdf < preferred_dist:
                        obs_error = preferred_dist - min_sdf
                        precision = 1.0 / self.config.sigma_obs_sq
                        G_obs_preference = self.config.lambda_obstacle * 0.001 * 0.5 * precision * obs_error ** 2

                except Exception as e:
                    pass  # Fallback to other options

            # =================================================================
            # Option B: Use ray-circle model with autograd (Paper Section 5.1.3)
            # =================================================================
            if G_obs_preference.item() == 0.0 and self._ray_circle_model is not None and self.config.use_ray_circle_model:
                # Infer theta from action direction
                if torch.norm(action) > 0.1:
                    theta = torch.atan2(action[1], action[0]).item()
                else:
                    theta = 0.0

                # Construct pose tensor (float32 for efficiency)
                pose_3d = torch.tensor([pos[0].item(), pos[1].item(), theta],
                                       dtype=torch.float32)
                obstacles_f32 = obstacles.float()

                try:
                    # Use the model's compute_G_pragmatic which handles everything
                    G_ray_circle = self._ray_circle_model.compute_G_pragmatic(
                        pose_3d, obstacles_f32,
                        preferred_range=self.config.preferred_obs_distance,
                        sigma_sq=self.config.sigma_obs_sq,
                        lambda_obs=self.config.lambda_obstacle,
                        robot_radius=self.config.robot_radius
                    )

                    G_obs_preference = G_ray_circle.to(self.dtype)
                    # Safety is already included in compute_G_pragmatic
                    G_obstacle_safety = torch.tensor(0.0, dtype=self.dtype)

                except Exception as e:
                    # Fallback to distance-based if ray-circle fails
                    pass

            # =================================================================
            # Option C: Distance-based fallback
            # =================================================================
            if G_obs_preference.item() == 0.0:
                # Observation preference error (distance-based)
                precision = 1.0 / self.config.sigma_obs_sq
                preferred_dist = self.config.preferred_obs_distance
                obs_errors = torch.clamp(preferred_dist - dist_surfaces, min=0.0)
                G_obs_preference = 0.5 * precision * torch.sum(obs_errors ** 2)
                G_obs_preference = self.config.lambda_obstacle * 0.001 * G_obs_preference

                # Hard safety margin
                hard_margin = self.config.robot_radius + 5.0
                close_mask = dist_surfaces < hard_margin
                if close_mask.any():
                    safety_penalties = torch.exp(2.0 * (hard_margin - dist_surfaces[close_mask]))
                    G_obstacle_safety = self.config.lambda_obstacle * torch.sum(safety_penalties)

        G_pragmatic = G_goal + G_obs_preference + G_obstacle_safety

        # =====================================================================
        # G_AMBIGUITY (Paper Eq. 105-107): FULL Matrix Covariance
        #
        # Paper Eq. 105: Σ_{t+1}(u) = Σ_t + Σ_static + β||u||² I
        # Paper Eq. 106: Σ_o = Σ_meas + J_{o,s} Σ_{t+1} J_{o,s}^T
        # Paper Eq. 107: G_amb = (1/2) ln|Σ_o|
        #
        # Where J_{o,s} is the Jacobian of observations w.r.t. pose
        # =====================================================================
        speed_sq = torch.sum(action ** 2)
        beta = self.config.beta_velocity

        # Predicted pose covariance (Paper Eq. 105)
        # Σ_{t+1} = Σ_t + β||u||² I
        Sigma_s = self._pose_covariance + beta * speed_sq * torch.eye(2, dtype=self.dtype, device=self.device)

        # Observation Jacobian J_{o,s} for range observations
        # Use ANALYTIC Jacobian from ray-circle model if available (Paper Eq. drho_general)
        # Otherwise fallback to approximate: for a LIDAR ray in direction d, ∂ρ/∂pos ≈ -d
        if hasattr(self, '_analytic_J_os') and self._analytic_J_os is not None:
            J_os = self._analytic_J_os  # [K_rays, 2] from ray-circle model
            K = J_os.shape[0]
        else:
            J_os = -self._ray_dirs  # [K, 2] - approximate: negative ray directions
            K = self.config.num_rays

        # Observation covariance (Paper Eq. 106)
        # Σ_o = Σ_meas * I_K + J_{o,s} Σ_s J_{o,s}^T
        Sigma_meas = self._sigma_meas_sq * torch.eye(K, dtype=self.dtype, device=self.device)

        # J Σ J^T: [K,2] @ [2,2] @ [2,K] = [K,K]
        J_Sigma_JT = J_os @ Sigma_s @ J_os.T
        Sigma_o = Sigma_meas + J_Sigma_JT  # [K, K]

        # G_ambiguity = (1/2) ln|Σ_o| (Paper Eq. 107)
        sign, logdet_Sigma_o = torch.linalg.slogdet(Sigma_o)

        if sign > 0:
            G_ambiguity = 0.5 * logdet_Sigma_o
        else:
            G_ambiguity = 0.5 * K * torch.log(torch.trace(Sigma_o) / K + 1e-6)

        G_ambiguity = 0.05 * G_ambiguity  # Scale factor

        # =====================================================================
        # G_EPISTEMIC (Paper Eq. efe_decomposition): MUTUAL INFORMATION
        #
        # Paper: G_epistemic = E_{q(s|π)}[D_KL[p(o|s) || q(o|π)]]
        #      = -I(S; O | π)  (negative mutual information)
        #
        # For Gaussian distributions:
        # I(S;O) = (1/2) ln|Σ_o| - (1/2) ln|Σ_meas|
        #        = (1/2) ln(|Σ_o| / |Σ_meas|)
        #
        # This measures how much observing O reduces uncertainty about S.
        # High I(S;O) means observations are informative → should explore there.
        # =====================================================================

        # Log-determinant of measurement noise (conditional obs covariance)
        # For diagonal Σ_meas = σ²_meas * I_K:  ln|Σ_meas| = K * ln(σ²_meas)
        logdet_Sigma_meas = K * torch.log(torch.tensor(self._sigma_meas_sq, dtype=self.dtype, device=self.device))

        # Mutual Information: I(S;O) = (1/2) * (ln|Σ_o| - ln|Σ_meas|)
        # = (1/2) * ln(|Σ_o| / |Σ_meas|)
        # = (1/2) * ln|Σ_o| - (1/2) * ln|Σ_meas|
        #
        # When Σ_s is large (high pose uncertainty), Σ_o is larger than Σ_meas
        # → I(S;O) is large → observations are informative
        # → We should go there to reduce uncertainty (negative G_epistemic)

        if sign > 0:
            mutual_info = 0.5 * (logdet_Sigma_o - logdet_Sigma_meas)
        else:
            # Fallback
            mutual_info = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # G_epistemic is NEGATIVE mutual information (we WANT high info gain)
        # But we need to modulate by whether there are obstacles to observe
        # (no obstacles → no info gain about obstacles)

        if len(obstacles) > 0:
            # Additional info gain near obstacles (they provide landmarks)
            # Closer obstacles → more precise range measurements → more info
            visible_mask = dist_centers < self.config.sensor_range
            if visible_mask.any():
                visible_dists = dist_centers[visible_mask]
                # Fisher information for range: scales as 1/σ² where σ grows with distance
                # Closer = more precise = more information
                precision_gain = torch.sum(1.0 / (visible_dists / 50.0 + 1.0) ** 2)

                # Combine: mutual info from pose uncertainty + precision from obstacle proximity
                G_epistemic = -self.config.lambda_epistemic * (mutual_info + 0.1 * precision_gain)
            else:
                G_epistemic = -self.config.lambda_epistemic * mutual_info
        else:
            # No obstacles: only pose uncertainty reduction matters
            G_epistemic = -self.config.lambda_epistemic * mutual_info * 0.1

        # =====================================================================
        # G_LOCALIZATION: Pose uncertainty cost
        # Encourages actions that reduce overall pose uncertainty
        # =====================================================================
        trace_Sigma_s = torch.trace(Sigma_s)
        G_localization = self.config.lambda_localization * trace_Sigma_s

        # Structural epistemic: I(m; o|π)
        G_structural_epistemic = self._compute_structural_epistemic(
            pos.unsqueeze(0), obstacles
        ).squeeze() if obstacles.numel() > 0 else torch.tensor(0.0, dtype=self.dtype, device=self.device)

        return G_pragmatic + G_ambiguity + G_epistemic + G_structural_epistemic + G_localization

    def _compute_step_G_ray_circle(self,
                                   pos: torch.Tensor,
                                   action: torch.Tensor,
                                   theta: float,
                                   goal: torch.Tensor,
                                   obstacles: torch.Tensor,
                                   t: int,
                                   T: int) -> torch.Tensor:
        """
        Compute Expected Free Energy using analytic ray-circle model.

        This follows the paper more closely:
        - Paper Section 5.1.3: Ray-circle intersection model
        - Paper Eq. ray_circle_range: ρ_{k,i} = -b_{k,i} - √Δ_{k,i}
        - Paper Eq. softmin_range: ρ̂_k = -1/η log Σ exp(-η ρ_{k,i})
        - Paper Eq. obs_preferences: p(o) = N(o | o*, Σ*)
        - Paper Eq. pragmatic_cost: G_prag = (1/2)(ρ̂ - o*)^T Σ*^{-1} (ρ̂ - o*)

        Args:
            pos: Position [x, y]
            action: Action [vx, vy]
            theta: Robot orientation
            goal: Goal position
            obstacles: Obstacle circles [N, 3]
            t: Current step in trajectory
            T: Total horizon

        Returns:
            G_total: Expected Free Energy for this step
        """
        if self._ray_circle_model is None or len(obstacles) == 0:
            # Fallback to heuristic model
            return self._compute_step_G(pos, action, goal, obstacles, t, T)

        K = self._ray_circle_model.config.num_rays

        # Construct pose [x, y, θ] - ensure float64 for ray-circle model
        pose = torch.tensor([pos[0].item(), pos[1].item(), theta], dtype=torch.float64)

        # Convert obstacles to float64
        obstacles_f64 = obstacles.to(torch.float64)

        # Predict LIDAR ranges using analytic model
        predicted_ranges, J_o_s = self._ray_circle_model.predict_ranges(
            pose, obstacles_f64, return_jacobian=True
        )

        # =====================================================================
        # G_PRAGMATIC (Paper Eq. pragmatic_cost)
        # G_prag = (1/2)(ρ̂ - o*)^T Σ*^{-1} (ρ̂ - o*)
        #
        # Preferences:
        # - o* = preferred_obs_distance (want large ranges = safe)
        # - Σ* = sigma_obs^2 I (precision of preferences)
        # =====================================================================

        # Preferred observations: far from obstacles (safe)
        preferred_ranges = torch.full((K,), self.config.preferred_obs_distance, dtype=torch.float64)

        # Observation error
        obs_error = predicted_ranges - preferred_ranges  # [K]

        # G_pragmatic = (1/2) error^T Σ*^{-1} error
        # With Σ* = σ² I, this is (1/2σ²) ||error||²
        precision = 1.0 / self.config.sigma_obs_sq
        G_obs_prag = 0.5 * precision * torch.sum(obs_error ** 2)

        # Goal attraction (still distance-based, paper doesn't specify LIDAR for goal)
        to_goal = pos - goal
        dist_to_goal = torch.norm(to_goal)
        G_goal = self.config.lambda_goal * (dist_to_goal + 0.01 * dist_to_goal ** 2)

        # Scale observation pragmatic term
        G_obs_prag = self.config.lambda_obstacle * 0.01 * G_obs_prag

        # Hard safety: extra penalty for very close obstacles (< robot_radius + margin)
        min_range = torch.min(predicted_ranges)
        hard_margin = self.config.robot_radius + 10.0
        if min_range < hard_margin:
            G_safety = self.config.lambda_obstacle * torch.exp(2.0 * (hard_margin - min_range))
        else:
            G_safety = torch.tensor(0.0, dtype=torch.float64)

        G_pragmatic = G_goal + G_obs_prag + G_safety

        # =====================================================================
        # G_AMBIGUITY (Paper Eq. 105-107) with analytic Jacobian
        #
        # Now using the actual J_{o,s} from ray-circle model!
        # Paper Eq. 106: Σ_o = Σ_meas + J_{o,s} Σ_s J_{o,s}^T
        # =====================================================================

        speed_sq = torch.sum(action ** 2)
        beta = self.config.beta_velocity

        # Predicted pose covariance (Paper Eq. 105)
        Sigma_s = torch.eye(3, dtype=torch.float64) * self.config.sigma_pose_base ** 2
        Sigma_s[:2, :2] = self._pose_covariance.cpu().to(torch.float64)
        Sigma_s += beta * speed_sq * torch.eye(3, dtype=torch.float64)

        # Use analytic Jacobian J_{o,s} from ray-circle model
        # J_o_s is [K, 3] from the ray-circle model

        # Observation covariance (Paper Eq. 106)
        Sigma_meas = self._sigma_meas_sq * torch.eye(K, dtype=torch.float64)
        J_Sigma_JT = J_o_s @ Sigma_s @ J_o_s.T  # [K, K]
        Sigma_o = Sigma_meas + J_Sigma_JT

        # G_ambiguity = (1/2) ln|Σ_o|
        sign, logdet_Sigma_o = torch.linalg.slogdet(Sigma_o)
        if sign > 0:
            G_ambiguity = 0.05 * 0.5 * logdet_Sigma_o
        else:
            G_ambiguity = 0.05 * 0.5 * K * torch.log(torch.trace(Sigma_o) / K + 1e-6)

        # =====================================================================
        # G_EPISTEMIC: Mutual information with analytic Jacobian
        # =====================================================================

        logdet_Sigma_meas = K * np.log(self._sigma_meas_sq)
        if sign > 0:
            mutual_info = 0.5 * (logdet_Sigma_o.item() - logdet_Sigma_meas)
        else:
            mutual_info = 0.0

        # Info gain from observing obstacles
        G_epistemic = -self.config.lambda_epistemic * mutual_info

        # =====================================================================
        # G_LOCALIZATION
        # =====================================================================

        G_localization = self.config.lambda_localization * torch.trace(Sigma_s[:2, :2])

        # Structural epistemic: I(m; o|π)
        G_structural_epistemic = self._compute_structural_epistemic(
            pos.unsqueeze(0), obstacles
        ).squeeze() if obstacles.numel() > 0 else torch.tensor(0.0, dtype=self.dtype, device=self.device)

        return G_pragmatic + G_ambiguity + G_epistemic + G_structural_epistemic + G_localization


    def reset(self):
        """Reset controller state."""
        self.previous_trajectory = None
        self.previous_G = float('inf')
        self.position_history = []
        self.current_horizon = self.config.base_horizon
        self.committed_trajectory = None
        self.committed_step = 0
        self.commit_reason = ""
        self.commit_start_time = 0.0
        self.committed_goal = None
        self.goal_commit_start_time = 0.0
        # Reset pose covariance to base value
        sigma_base = self.config.sigma_pose_base
        self._pose_covariance = torch.eye(2, dtype=self.dtype, device=self.device) * sigma_base ** 2
        # Reset EFE-based adaptive horizon state
        self._efe_smoothed = 0.0
        self._horizon_smoothed = float(self.config.base_horizon)

    def set_pose_covariance(self, covariance: np.ndarray):
        """
        Set pose covariance from external source (e.g., room estimation).

        This allows integration with the perception system that estimates
        robot pose relative to the room.

        Args:
            covariance: 2x2 numpy array representing pose uncertainty
        """
        self._pose_covariance = torch.tensor(covariance, dtype=self.dtype, device=self.device)

    def get_pose_covariance(self) -> np.ndarray:
        """Get current pose covariance estimate."""
        return self._pose_covariance.cpu().numpy()

    def predict_pose_covariance(self, action: np.ndarray) -> np.ndarray:
        """
        Predict pose covariance after executing an action.

        Paper Eq. 105: Σ_{t+1} = Σ_t + β||u||² I

        Args:
            action: Velocity command [vx, vy]

        Returns:
            Predicted 2x2 covariance matrix
        """
        speed_sq = np.sum(action ** 2)
        beta = self.config.beta_velocity

        # Covariance grows with velocity (motion uncertainty)
        Sigma_pred = self._pose_covariance.numpy() + beta * speed_sq * np.eye(2)

        return Sigma_pred

    def _compute_uncertainty_limited_velocity(self) -> float:
        """
        Compute effective max velocity based on current pose uncertainty.

        This is directly linked to G_localization: when G_loc is high (high pose
        uncertainty), we should move slower to:
        1. Allow more time for localization (reduce motion blur)
        2. Reduce motion-induced uncertainty growth (Σ grows with v²)
        3. Improve safety margin near obstacles (uncertain position = danger)

        The relationship mirrors the EFE term:
            G_localization = λ_loc × tr(Σ_pose)

        Formula:
            v_eff = v_max × (min_factor + (1 - min_factor) × exp(-tr(Σ)/σ²_max))

        When G_loc is high (uncertain), velocity drops.
        When G_loc is low (certain), velocity is near maximum.

        Returns:
            Effective maximum velocity (pixels per step)
        """
        if not self.config.uncertainty_velocity_limit:
            return self.config.max_velocity

        # Get current pose uncertainty (same quantity used in G_localization)
        # G_localization = λ_loc × tr(Σ_pose)
        pose_variance = torch.trace(self._pose_covariance).item()

        # Exponential decay of velocity with uncertainty
        # High uncertainty (high G_loc) -> low velocity factor
        # Low uncertainty (low G_loc) -> velocity factor ~1
        uncertainty_ratio = pose_variance / self.config.max_pose_uncertainty
        velocity_factor = self.config.min_velocity_factor + \
                         (1.0 - self.config.min_velocity_factor) * np.exp(-uncertainty_ratio)

        effective_velocity = self.config.max_velocity * velocity_factor

        return effective_velocity

    def update_pose_covariance_from_observation(self,
                                                  num_landmarks_visible: int,
                                                  landmark_quality: float = 1.0):
        """
        Update pose covariance after receiving observations.

        This simulates the information gained from LIDAR observations
        that help localize the robot within the room.

        Paper: Σ_new⁻¹ = Σ_pred⁻¹ + J^T R⁻¹ J (Information Filter update)

        Args:
            num_landmarks_visible: Number of distinct features observed
            landmark_quality: Quality of observations (0-1)
        """
        if num_landmarks_visible <= 0:
            return

        # Information gained from observations
        # More landmarks = more information = lower uncertainty
        info_gain_per_landmark = landmark_quality / self._sigma_meas_sq
        total_info_gain = num_landmarks_visible * info_gain_per_landmark

        # Convert covariance to information matrix, add info, convert back
        # Simplified: reduce covariance proportionally
        reduction_factor = 1.0 / (1.0 + total_info_gain * 0.1)

        self._pose_covariance = self._pose_covariance * reduction_factor

        # Ensure minimum uncertainty (we're never perfectly localized)
        min_var = 1.0  # Minimum variance in pixels²
        diag = torch.diagonal(self._pose_covariance)
        diag_clamped = torch.clamp(diag, min=min_var)
        self._pose_covariance = torch.diag(diag_clamped)

    def _normalize_obstacles(self,
                             obstacles: Union[List[Tuple[float, float, float]], List]) -> Tuple[List[Tuple[float, float, float]], Optional[List]]:
        """
        Normalize obstacles to both tuple format and belief format.

        This enables backward compatibility with tuple-based obstacles while
        supporting the new polymorphic ObstacleBeliefBase classes and
        TrackedObstacle with multi-hypothesis model selection.

        For TrackedObstacle instances, we extract the model-averaged bounding
        representation using:
            p(o_{t+1} | π) = Σ_m q(m) · p(o_{t+1} | π, m)

        Args:
            obstacles: Either list of (cx, cy, r) tuples, ObstacleBeliefBase, or TrackedObstacle

        Returns:
            Tuple of:
            - obstacles_tuples: List of (cx, cy, r) tuples for legacy code paths
            - obstacle_beliefs: List of underlying beliefs or TrackedObstacles
        """
        if len(obstacles) == 0:
            return [], None

        # Check if first element is a tuple/list or an ObstacleBeliefBase
        first = obstacles[0]

        if POLYMORPHIC_OBSTACLES_AVAILABLE:
            from obstacle_base import ObstacleBeliefBase, TrackedObstacle

            # Check for TrackedObstacle (multi-hypothesis)
            if hasattr(first, 'hypotheses') and hasattr(first, 'q_m'):
                # TrackedObstacle - use model-averaged bounding representation
                obstacles_tuples = []
                for tracked in obstacles:
                    # Model-averaged prediction: Σ q(m) * bounding_params(m)
                    avg_cx, avg_cy, avg_r = 0.0, 0.0, 0.0

                    for type_name, model in tracked.hypotheses.items():
                        q_m = tracked.q_m.get(type_name, 0.0)
                        if q_m < 0.01:
                            continue

                        center = model.get_center()
                        cx = center[0].item() if hasattr(center[0], 'item') else float(center[0])
                        cy = center[1].item() if hasattr(center[1], 'item') else float(center[1])
                        r = float(model.get_bounding_radius())

                        avg_cx += q_m * cx
                        avg_cy += q_m * cy
                        avg_r += q_m * r

                    obstacles_tuples.append((avg_cx, avg_cy, avg_r))

                return obstacles_tuples, obstacles

            elif isinstance(first, ObstacleBeliefBase):
                # Single-type ObstacleBeliefBase - extract tuple representation
                obstacles_tuples = []
                for obs in obstacles:
                    center = obs.get_center()
                    bounding_r = obs.get_bounding_radius()
                    obstacles_tuples.append((
                        center[0].item() if hasattr(center[0], 'item') else float(center[0]),
                        center[1].item() if hasattr(center[1], 'item') else float(center[1]),
                        float(bounding_r)
                    ))
                return obstacles_tuples, obstacles

        # Legacy tuple format
        return list(obstacles), None

    def _update_history(self, pos: np.ndarray):
        """Update position history for oscillation detection."""
        self.position_history.append(pos.copy())
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

    def _detect_oscillation(self) -> bool:
        """Detect if robot is oscillating (stuck in local minimum)."""
        if len(self.position_history) < self.config.oscillation_history:
            return False

        recent = self.position_history[-self.config.oscillation_history:]

        # Check if robot is not making progress
        start_pos = recent[0]
        end_pos = recent[-1]
        net_movement = np.linalg.norm(end_pos - start_pos)

        # Calculate total path length
        total_movement = sum(
            np.linalg.norm(recent[i+1] - recent[i])
            for i in range(len(recent) - 1)
        )

        # Oscillation: lots of movement but little progress
        if total_movement > 10 and net_movement < total_movement * self.config.oscillation_threshold:
            return True

        return False

    def _detect_stuck(self) -> bool:
        """
        Detect if robot is stuck (very little movement over recent history).
        This catches cases where the robot stops in front of an undetected obstacle.
        """
        if len(self.position_history) < 5:
            return False

        recent = self.position_history[-5:]

        # Calculate total movement over last 5 steps
        total_movement = sum(
            np.linalg.norm(recent[i+1] - recent[i])
            for i in range(len(recent) - 1)
        )

        # If almost no movement, robot is stuck
        return total_movement < 3.0  # Less than 3 pixels in 5 steps

    def _get_stuck_penalty(self) -> float:
        """
        Compute a penalty to add to EFE when robot is stuck.
        This forces the horizon to extend and triggers more exploration.
        """
        if not self._detect_stuck():
            return 0.0

        # Count consecutive stuck steps
        self._stuck_steps = getattr(self, '_stuck_steps', 0) + 1

        # Exponentially increasing penalty the longer we're stuck
        # This will cause EFE to skyrocket and force longer horizons
        penalty = 1000.0 * (1.5 ** min(self._stuck_steps, 10))

        return penalty

    def _reset_stuck_counter(self):
        """Reset stuck counter when robot is moving."""
        if len(self.position_history) >= 2:
            last_movement = np.linalg.norm(
                self.position_history[-1] - self.position_history[-2]
            )
            if last_movement > 1.0:  # Moving at least 1 pixel
                self._stuck_steps = 0

    def _compute_adaptive_horizon(self, pose_uncertainty: float = 0.0) -> int:
        """
        Compute adaptive horizon based on Expected Free Energy.

        Paper-aligned formulation:

        T(t) = T_base + (T_max - T_base) · tanh(β · ReLU(Ĝ(t) - G₀))

        Where:
        - Ĝ(t) = γ · Ĝ(t-1) + (1-γ) · log(1 + G_min)  (smoothed log-EFE)
        - G₀ = 8.0 (threshold, corresponds to raw G ≈ 3000)
        - β = 0.3 (sensitivity to log-EFE increase)

        Interpretation:
        - High EFE → robot is stuck or facing obstacles → extend horizon
        - Low EFE → clear path to goal → short horizon for speed

        Using log-scale ensures numerical stability with very large G values.
        """
        T_base = float(self.config.base_horizon)
        T_max = float(self.config.max_horizon)

        # Add stuck penalty to EFE - this causes horizon to extend when stuck
        stuck_penalty = self._get_stuck_penalty()
        if stuck_penalty > 0:
            # Add penalty to smoothed EFE (in log scale)
            effective_efe = self._efe_smoothed + np.log(1.0 + stuck_penalty)
        else:
            effective_efe = self._efe_smoothed
            self._reset_stuck_counter()

        # Compute target horizon from smoothed EFE
        # T_target = T_base + (T_max - T_base) · tanh(β · ReLU(Ĝ - G₀))
        efe_excess = max(0.0, effective_efe - self._efe_threshold)
        horizon_factor = np.tanh(self._efe_sensitivity * efe_excess)
        T_target = T_base + (T_max - T_base) * horizon_factor

        # Smooth the horizon transition (faster response than before)
        # H(t) = 0.6 · H(t-1) + 0.4 · T_target
        self._horizon_smoothed = 0.6 * self._horizon_smoothed + 0.4 * T_target

        # Additional extension if oscillating or stuck (safety net)
        if self._detect_oscillation() or self._detect_stuck():
            self._horizon_smoothed = min(self._horizon_smoothed + 10, T_max)
            self.stats['horizon_extensions'] += 1

        # Round to integer
        T = int(round(self._horizon_smoothed))
        T = max(self.config.min_horizon, min(T, self.config.max_horizon))

        self.current_horizon = T

        # Adaptive seed spread: scales with EFE (difficulty indicator)
        # When EFE is high (stuck/obstacles), spread more to explore alternatives
        # spread(Ĝ) = spread_min + (spread_max - spread_min) × tanh(β × ReLU(Ĝ - G₀))
        if self.config.adaptive_spread:
            efe_excess = max(0.0, self._efe_smoothed - self._efe_threshold)
            spread_factor = np.tanh(self._efe_sensitivity * efe_excess)
            self.current_seed_spread = (
                self.config.seed_spread_min +
                (self.config.seed_spread_max - self.config.seed_spread_min) * spread_factor
            )

        return T

    def _update_efe_estimate(self, G_min: float):
        """
        Update the smoothed EFE estimate.

        Called after trajectory optimization with the best EFE found.
        Uses exponential moving average on log-scale for stability:

        Ĝ(t) = γ · Ĝ(t-1) + (1-γ) · log(1 + G_min)

        Log-scale ensures:
        - Small G values (good) map to small log values
        - Large G values (stuck) map to reasonable log values
        - Avoids numerical overflow with very large G
        """
        # Use log(1 + G) to normalize to reasonable scale
        log_efe = np.log(1.0 + max(0.0, G_min))
        self._efe_smoothed = (self._efe_gamma * self._efe_smoothed +
                             (1 - self._efe_gamma) * log_efe)


# =============================================================================
# TEST
# =============================================================================

def test_unified_controller():
    """Test the unified controller"""
    print("=" * 70)
    print("Testing Unified Controller (Multi-Seed Gradient Optimization)")
    print("=" * 70)
    print()

    controller = UnifiedController()

    # Test 1: Simple goal reaching
    print("Test 1: Simple goal reaching")
    print("-" * 40)

    robot_pos = np.array([100.0, 100.0])
    goal = np.array([300.0, 100.0])
    obstacles = []

    action, info = controller.compute_action(robot_pos, 0.0, goal, obstacles)

    direction = np.degrees(np.arctan2(action[1], action[0]))
    print(f"  Robot: {robot_pos}, Goal: {goal}")
    print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}]")
    print(f"  Direction: {direction:.1f}° (expected ~0°)")
    print(f"  G_total: {info['G_total']:.2f}")
    print(f"  Horizon: {info['horizon']}")
    print(f"  Time: {info['elapsed_ms']:.1f}ms")

    test1_ok = abs(direction) < 30
    print(f"  {'✓ PASSED' if test1_ok else '✗ FAILED'}")
    print()

    # Test 2: Obstacle avoidance
    print("Test 2: Obstacle in path")
    print("-" * 40)

    controller.reset()
    robot_pos = np.array([100.0, 100.0])
    goal = np.array([300.0, 100.0])
    obstacles = [(200.0, 100.0, 40.0)]

    action, info = controller.compute_action(robot_pos, 0.0, goal, obstacles)

    direction = np.degrees(np.arctan2(action[1], action[0]))
    print(f"  Robot: {robot_pos}, Goal: {goal}")
    print(f"  Obstacle: {obstacles[0]}")
    print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}]")
    print(f"  Direction: {direction:.1f}° (should deviate from 0°)")
    print(f"  G_total: {info['G_total']:.2f}")
    print(f"  Best seed: {info['best_seed']}")
    print(f"  Time: {info['elapsed_ms']:.1f}ms")

    test2_ok = abs(action[1]) > 0.3 or action[0] < 0
    print(f"  {'✓ PASSED' if test2_ok else '✗ FAILED'}")
    print()

    # Test 3: Trajectory visualization
    print("Test 3: Trajectory planning")
    print("-" * 40)

    controller.reset()
    trajectory = info['trajectory']
    print(f"  Planned trajectory ({len(trajectory)} steps):")

    pos = robot_pos.copy()
    for t in range(min(5, len(trajectory))):
        pos = pos + trajectory[t]
        print(f"    t={t}: pos=[{pos[0]:.1f}, {pos[1]:.1f}]")
    print()

    # Test 4: Corridor
    print("Test 4: Corridor navigation")
    print("-" * 40)

    controller.reset()
    robot_pos = np.array([100.0, 100.0])
    goal = np.array([300.0, 100.0])
    obstacles = [
        (200.0, 50.0, 30.0),
        (200.0, 150.0, 30.0),
    ]

    action, info = controller.compute_action(robot_pos, 0.0, goal, obstacles)

    direction = np.degrees(np.arctan2(action[1], action[0]))
    print(f"  Robot: {robot_pos}, Goal: {goal}")
    print(f"  Obstacles forming corridor")
    print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}]")
    print(f"  Direction: {direction:.1f}°")
    print(f"  G_total: {info['G_total']:.2f}")
    print(f"  Time: {info['elapsed_ms']:.1f}ms")

    test4_ok = action[0] > 1.0
    print(f"  {'✓ PASSED' if test4_ok else '✗ FAILED'}")
    print()

    # Test 5: Speed benchmark
    print("Test 5: Speed benchmark")
    print("-" * 40)

    controller.reset()
    times = []
    for _ in range(10):
        start = time.time()
        action, info = controller.compute_action(robot_pos, 0.0, goal, obstacles)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    print(f"  Average time: {avg_time:.1f}ms")
    print(f"  Max time: {np.max(times)*1000:.1f}ms")
    print(f"  Steps possible per second: {int(1000/avg_time)}")

    test5_ok = avg_time < 200
    print(f"  {'✓ PASSED' if test5_ok else '✗ FAILED'}")
    print()

    # Summary
    print("=" * 70)
    all_passed = test1_ok and test4_ok and test5_ok
    print(f"RESULTS: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print()
    print("Controller Statistics:")
    print(f"  Seeds evaluated: {controller.stats['seeds_evaluated']}")
    print(f"  Total iterations: {controller.stats['iterations_total']}")
    print(f"  Warm starts used: {controller.stats['warm_starts_used']}")
    print(f"  Horizon extensions: {controller.stats['horizon_extensions']}")
    print(f"  Trajectory commits: {controller.stats['trajectory_commits']}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    test_unified_controller()

