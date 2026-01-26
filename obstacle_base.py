"""
Obstacle Base Classes - Abstract Interface for Polymorphic Obstacle Types

This module defines the abstract base class for obstacle beliefs and the
TrackedObstacle class that maintains multiple hypotheses for model selection.

Architecture:
    OBSTACLE_REGISTRY (global registry of obstacle types)
    │
    TrackedObstacle (maintains multiple hypotheses for one physical obstacle)
    │   ├── hypotheses: Dict[str, ObstacleBeliefBase]
    │   └── q_m: Dict[str, float]  (posterior over model identity)
    │
    ObstacleBeliefBase (ABC)
        ├── CircleObstacle  (from concept_circle.py)
        ├── RectangleObstacle  (from concept_rectangle.py)
        └── [Future types - just register them]

Paper Reference (Section: Model Identity as a Latent Variable):
- m ∈ M is a discrete latent variable for model identity
- q(m) ∝ p(m) · exp(-F_m) where F_m is the VFE for model m
- Commitment criteria: posterior concentration, behavioral irrelevance, control necessity

Key Design Principles:
1. Each obstacle type defines its own SDF and ray-intersection
2. TrackedObstacle maintains ALL registered types as hypotheses
3. New types are added by: create class + register (no other changes needed)
4. Model selection via Bayesian comparison of VFEs
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List, Type, Callable
from dataclasses import dataclass
from enum import Enum


# Device configuration with graceful fallback
DTYPE = torch.float64
DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    try:
        _test = torch.zeros(1, device='cuda') + 1
        DEVICE = torch.device('cuda')
    except RuntimeError:
        pass  # Use CPU fallback


# =============================================================================
# OBSTACLE TYPE REGISTRY
# New obstacle types register themselves here for automatic inclusion
# =============================================================================

OBSTACLE_REGISTRY: Dict[str, Dict] = {}
"""
Global registry of obstacle types.
Each entry contains:
- 'class': The ObstacleBeliefBase subclass
- 'config_class': The config dataclass for this type
- 'fit_func': Function to fit initial parameters from points
- 'prior': Prior probability p(m) for this type
"""


def register_obstacle_type(
    name: str,
    cls: Type['ObstacleBeliefBase'],
    config_class: Type = None,
    fit_func: Callable = None,
    prior: float = 1.0
):
    """
    Register a new obstacle type for automatic hypothesis creation.

    Args:
        name: Unique identifier (e.g., 'circle', 'rectangle', 'ellipse')
        cls: The ObstacleBeliefBase subclass
        config_class: Configuration dataclass (optional)
        fit_func: Function(points, config) -> instance or None
        prior: Prior probability p(m) for Bayesian model comparison
    """
    OBSTACLE_REGISTRY[name] = {
        'class': cls,
        'config_class': config_class,
        'fit_func': fit_func,
        'prior': prior,
    }


def get_registered_types() -> List[str]:
    """Get list of registered obstacle type names."""
    return list(OBSTACLE_REGISTRY.keys())


def get_obstacle_class(name: str) -> Type['ObstacleBeliefBase']:
    """Get the class for a registered obstacle type."""
    return OBSTACLE_REGISTRY[name]['class']


class ObstacleType(Enum):
    """Enumeration of supported obstacle types."""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


@dataclass
class ObstacleBeliefConfig:
    """
    Configuration for obstacle belief tracking.

    Common parameters shared by all obstacle types.
    Type-specific parameters should be in subclass configs.
    """
    # Optimization parameters
    num_iterations: int = 10
    learning_rate: float = 0.4

    # Prior strength (static obstacles)
    prior_position_std: float = 5.0

    # Likelihood (SDF-based observation model)
    sdf_sigma: float = 5.0

    # Initialization
    initial_position_std: float = 30.0

    # Matching thresholds
    match_threshold: float = 60.0

    # Belief lifecycle
    stale_threshold: int = 50
    min_observations: int = 5
    min_confidence: float = 0.1

    # Sensor parameters
    sensor_range: float = 350.0
    fov_angle: float = np.radians(240)

    # Constraints
    use_gpu: bool = True


class ObstacleManager:
    """
    Manages the lifecycle of obstacle belief instances.

    Each concept class (CircleObstacle, RectangleObstacle) has its own manager
    that handles:
    - Instance creation from point clusters
    - Matching clusters to existing instances
    - Updating instances with new observations
    - Removing instances based on visibility logic

    The manager is a class-level singleton for each obstacle type.

    NOTE: This is the legacy single-type manager. For multi-hypothesis tracking,
    use ObstacleTracker instead.
    """

    def __init__(self, config: ObstacleBeliefConfig = None):
        self.config = config or ObstacleBeliefConfig()
        self.instances: List['ObstacleBeliefBase'] = []
        self.frame_count = 0


class ObstacleTracker:
    """
    Manages TrackedObstacle instances with Bayesian model selection.

    This is the main tracker that should be used for obstacle perception.
    It creates TrackedObstacle instances that maintain hypotheses for ALL
    registered obstacle types, enabling proper Bayesian model comparison.

    Paper: "Model Identity as a Latent Variable and Commitment Criteria"

    Usage:
        tracker = ObstacleTracker()

        # Each frame:
        tracker.update(clusters, robot_pose)
        obstacles = tracker.get_obstacles()
    """

    def __init__(self, config: ObstacleBeliefConfig = None):
        self.config = config or ObstacleBeliefConfig()
        self.tracked: List['TrackedObstacle'] = []
        self.frame_count = 0

    def update(self, clusters: List[np.ndarray], robot_pose: np.ndarray) -> List['TrackedObstacle']:
        """
        Update all tracked obstacles with new observations.

        Args:
            clusters: List of point clusters from DBSCAN
            robot_pose: Robot pose [x, y, theta]

        Returns:
            List of active TrackedObstacle instances
        """
        self.frame_count += 1
        robot_pos = robot_pose[:2]
        robot_theta = robot_pose[2]

        # Match clusters to existing tracked obstacles
        matched_tracked = set()
        matched_clusters = set()

        for i, cluster in enumerate(clusters):
            if len(cluster) < 3:
                continue

            cluster_center = np.mean(cluster, axis=0)

            # Find closest existing tracked obstacle
            best_match = None
            best_dist = self.config.match_threshold

            for j, tracked in enumerate(self.tracked):
                if j in matched_tracked:
                    continue
                dist = np.linalg.norm(tracked.position - cluster_center)
                if dist < best_dist:
                    best_dist = dist
                    best_match = j

            if best_match is not None:
                # Update existing tracked obstacle
                self.tracked[best_match].update(cluster, robot_pose)
                matched_tracked.add(best_match)
                matched_clusters.add(i)
            else:
                # Create new tracked obstacle with all hypotheses
                if len(OBSTACLE_REGISTRY) > 0:
                    new_tracked = TrackedObstacle(cluster, robot_pose, self.config)
                    if new_tracked.hypotheses:  # Has at least one valid hypothesis
                        self.tracked.append(new_tracked)
                        matched_clusters.add(i)

        # Update visibility for unmatched obstacles
        for j, tracked in enumerate(self.tracked):
            was_detected = j in matched_tracked
            tracked.update_visibility(
                was_detected=was_detected,
                robot_pos=robot_pos,
                robot_theta=robot_theta,
                sensor_range=self.config.sensor_range,
                frame=self.frame_count
            )

        # Remove obstacles that should be removed
        self.tracked = [t for t in self.tracked
                       if not t.should_be_removed(self.config.min_confidence)]

        return self.tracked

    def get_obstacles(self) -> List['TrackedObstacle']:
        """Get all active tracked obstacles."""
        return self.tracked

    def clear(self):
        """Clear all tracked obstacles."""
        self.tracked = []
        self.frame_count = 0



# =============================================================================
# TRACKED OBSTACLE - Maintains multiple hypotheses for model selection
# Paper: "Model Identity as a Latent Variable and Commitment Criteria"
# =============================================================================

class TrackedObstacle:
    """
    Maintains multiple model hypotheses for a single physical obstacle.

    This implements the paper's approach where model identity m is a discrete
    latent variable with posterior q(m) computed via Bayesian model comparison.

    Paper equations:
    - Generative model: p(o,s,θ,m) = p(o|s,θ_m,m) p(θ_m|m) p(m) p(s)
    - Posterior: q(m) ∝ p(m) exp(-F_m)
    - Commitment: when max q(m) > threshold or behavioral irrelevance

    Attributes:
        hypotheses: Dict[str, ObstacleBeliefBase] - one model per registered type
        q_m: Dict[str, float] - posterior probability for each model type
        historical_points: Shared point accumulator for all hypotheses
        commitment_status: None (uncommitted) or str (committed type name)
    """

    def __init__(self, points: np.ndarray, robot_pose: np.ndarray,
                 config: ObstacleBeliefConfig = None):
        """
        Create a TrackedObstacle with hypotheses for all registered types.

        Args:
            points: Initial LIDAR points [N, 2]
            robot_pose: Robot pose [x, y, theta]
            config: Configuration (optional)
        """
        self.config = config or ObstacleBeliefConfig()
        self.id = id(self)
        self.age = 0
        self.last_seen = 0

        # Create hypotheses for ALL registered obstacle types
        self.hypotheses: Dict[str, 'ObstacleBeliefBase'] = {}
        self._create_all_hypotheses(points, robot_pose)

        # Initialize uniform prior over models
        n_types = len(self.hypotheses)
        if n_types > 0:
            uniform_prob = 1.0 / n_types
            self.q_m = {name: uniform_prob for name in self.hypotheses}
        else:
            self.q_m = {}

        # Shared historical points (in room frame)
        self.historical_points: Optional[torch.Tensor] = None
        self.max_historical_points = 500

        # Commitment status: None = uncommitted, str = committed type
        self.commitment_status: Optional[str] = None
        self.commitment_threshold = 0.95  # Posterior concentration threshold

        # Compute initial posterior
        self._compute_model_posterior()

    def _create_all_hypotheses(self, points: np.ndarray, robot_pose: np.ndarray):
        """Create hypothesis for each registered obstacle type."""
        for type_name, type_info in OBSTACLE_REGISTRY.items():
            fit_func = type_info.get('fit_func')
            if fit_func is not None:
                try:
                    instance = fit_func(points, robot_pose)
                    if instance is not None:
                        self.hypotheses[type_name] = instance
                except Exception as e:
                    # Skip types that fail to initialize
                    pass

    def _accumulate_historical_points(self, points: np.ndarray):
        """Add new points to historical buffer with angular redistribution around contour."""
        if len(points) == 0:
            return

        points_t = torch.tensor(points, dtype=DTYPE, device=DEVICE)

        if self.historical_points is None:
            self.historical_points = points_t
        else:
            self.historical_points = torch.cat([self.historical_points, points_t], dim=0)

        # Limit size with angular redistribution around the contour
        if len(self.historical_points) > self.max_historical_points:
            self._redistribute_historical_points_angular()

    def _redistribute_historical_points_angular(self):
        """
        Redistribute historical points evenly around the obstacle contour.

        Uses angular binning: divides 360° into bins and keeps one point per bin,
        prioritizing the most recent points. This ensures uniform coverage around
        the obstacle perimeter.
        """
        if self.historical_points is None or len(self.historical_points) <= self.max_historical_points:
            return

        # Compute center of points
        center = self.historical_points.mean(dim=0)

        # Compute angles from center to each point
        relative = self.historical_points - center
        angles = torch.atan2(relative[:, 1], relative[:, 0])  # [-π, π]

        # Normalize to [0, 2π)
        angles = (angles + 2 * 3.14159) % (2 * 3.14159)

        # Create angular bins
        n_bins = self.max_historical_points
        bin_size = 2 * 3.14159 / n_bins
        bin_indices = (angles / bin_size).long() % n_bins

        # Vectorized: for each bin, keep the last (most recent) point
        # Create a tensor of indices [0, 1, 2, ..., n-1]
        n_points = len(self.historical_points)
        point_indices = torch.arange(n_points, device=self.historical_points.device)

        # For each bin, we want the maximum index (most recent)
        # Use scatter_reduce with 'amax' to find max index per bin
        bin_max_indices = torch.full((n_bins,), -1, dtype=torch.long, device=self.historical_points.device)
        bin_max_indices.scatter_reduce_(0, bin_indices, point_indices, reduce='amax', include_self=False)

        # Filter out empty bins (still -1)
        valid_mask = bin_max_indices >= 0
        selected_indices = bin_max_indices[valid_mask]

        if len(selected_indices) > 0:
            self.historical_points = self.historical_points[selected_indices]

    def _get_all_points(self, current_points: np.ndarray) -> np.ndarray:
        """Combine current and historical points."""
        if self.historical_points is None:
            return current_points

        hist_np = self.historical_points.cpu().numpy()
        return np.vstack([current_points, hist_np])

    def _compute_model_posterior(self):
        """
        Compute posterior over model identity: q(m) ∝ p(m) exp(-F_m)

        Paper equation:
            q(m) ∝ p(m) · exp(-F_m)

        where F_m is the Variational Free Energy for model m.
        """
        if not self.hypotheses:
            return

        # Compute VFE for each model
        vfe = {}
        for name, model in self.hypotheses.items():
            # Use negative log-likelihood as VFE proxy
            # Lower VFE = better fit = higher probability
            if self.historical_points is not None and len(self.historical_points) > 0:
                points = self.historical_points.cpu().numpy()
                vfe[name] = model.compute_vfe(points, np.zeros(3))
            else:
                vfe[name] = 0.0

        # Get priors
        priors = {name: OBSTACLE_REGISTRY.get(name, {}).get('prior', 1.0)
                  for name in self.hypotheses}

        # Compute unnormalized posterior: p(m) * exp(-F_m)
        # Use softmax for numerical stability
        min_vfe = min(vfe.values()) if vfe else 0
        unnorm = {}
        for name in self.hypotheses:
            log_prior = np.log(priors[name] + 1e-10)
            unnorm[name] = np.exp(-(vfe[name] - min_vfe) + log_prior)

        # Normalize
        total = sum(unnorm.values()) + 1e-10
        self.q_m = {name: v / total for name, v in unnorm.items()}

    def _check_commitment(self, robot_pos: np.ndarray = None):
        """
        Check commitment criteria from the paper.

        Paper Section: "Commitment to a specific model"

        Three criteria (Eq. posterior_concentration, behavioral_irrelevance, risk_sensitive_commitment):

        1. Posterior concentration: max q(m) > 1 - ε  OR  H[q(m)] < H_min
           Structural uncertainty resolved by perception alone.

        2. Behavioral irrelevance: ∀π: |G(π|m₁) - G(π|m₂)| < δ
           Model identity has no material effect on predicted outcomes.

        3. Control necessity (risk-sensitive): m* = argmin_m G_risk(π|m)
           Near-field situations require commitment to conservative model.
        """
        if self.commitment_status is not None:
            return  # Already committed

        if not self.q_m or len(self.q_m) < 2:
            return

        # Require minimum age before commitment (allow perception to stabilize)
        min_age_for_commitment = 20  # frames - increased to allow more observations
        if self.age < min_age_for_commitment:
            return

        # =====================================================================
        # Criterion 1: Posterior concentration (Eq. posterior_concentration)
        # max_m q(m) > 1 - ε  OR  H[q(m)] < H_min
        #
        # Additional stability check: track best type over recent history
        # =====================================================================
        max_prob = max(self.q_m.values())
        current_best = max(self.q_m, key=self.q_m.get)

        # Track stability: only commit if same type has been best for N frames
        if not hasattr(self, '_best_type_history'):
            self._best_type_history = []

        self._best_type_history.append(current_best)
        if len(self._best_type_history) > 10:
            self._best_type_history = self._best_type_history[-10:]

        # Require stability: all recent frames must agree
        stable = len(self._best_type_history) >= 10 and len(set(self._best_type_history)) == 1

        if max_prob > self.commitment_threshold and stable:  # Default 0.95
            best_type = max(self.q_m, key=self.q_m.get)
            self.commitment_status = best_type
            return

        # Entropy threshold: H[q(m)] < H_min
        H_qm = -sum(p * np.log(p + 1e-10) for p in self.q_m.values() if p > 0)
        H_min = 0.1  # Low entropy threshold (bits)
        if H_qm < H_min:
            best_type = max(self.q_m, key=self.q_m.get)
            self.commitment_status = best_type
            return

        # =====================================================================
        # Criterion 2: Behavioral irrelevance (Eq. behavioral_irrelevance)
        # ∀π: |G(π|m₁) - G(π|m₂)| < δ
        #
        # Approximation: If bounding radii are similar, model choice doesn't
        # affect collision avoidance significantly.
        #
        # Only apply when there is GENUINE uncertainty (0.3 < max_prob < 0.7)
        # to avoid premature commitment before posterior stabilizes.
        # =====================================================================
        radii = []
        for name, model in self.hypotheses.items():
            radii.append(model.get_bounding_radius())

        if len(radii) >= 2:
            max_radius = max(radii)
            min_radius = min(radii)
            delta = 5.0  # Threshold in pixels

            # Only apply behavioral irrelevance when genuinely uncertain
            genuine_uncertainty = 0.3 < max_prob < 0.7

            if (max_radius - min_radius) < delta and genuine_uncertainty:
                # Models are behaviorally equivalent - commit to simpler one
                # Prefer circle (3 params) over rectangle (5 params)
                if 'circle' in self.hypotheses:
                    self.commitment_status = 'circle'
                else:
                    self.commitment_status = max(self.q_m, key=self.q_m.get)
                return

        # =====================================================================
        # Criterion 3: Control necessity / Risk-sensitive (Eq. risk_sensitive_commitment)
        # m* = argmin_m G_risk(π|m)
        #
        # When close to obstacle, commit to the MOST CONSERVATIVE model
        # (largest bounding radius) to ensure safety.
        # =====================================================================
        if robot_pos is not None:
            obs_pos = self.position[:2]
            dist_to_robot = np.linalg.norm(robot_pos - obs_pos)

            # Near-field threshold (robot radius + safety margin)
            near_field_threshold = 80.0  # pixels

            if dist_to_robot < near_field_threshold:
                # Risk-sensitive: select most conservative model
                # Conservative = largest bounding radius
                most_conservative = None
                max_bounding_r = 0.0

                for name, model in self.hypotheses.items():
                    r = model.get_bounding_radius()
                    if r > max_bounding_r:
                        max_bounding_r = r
                        most_conservative = name

                if most_conservative:
                    self.commitment_status = most_conservative
                    return

    def update(self, points: np.ndarray, robot_pose: np.ndarray):
        """
        Update all hypotheses with new observations.

        Args:
            points: LIDAR points [N, 2]
            robot_pose: Robot pose [x, y, theta]
        """
        self.age += 1
        self.last_seen = self.age

        # Accumulate historical points (shared by all hypotheses)
        self._accumulate_historical_points(points)

        # Get all constraint points
        all_points = self._get_all_points(points)

        # Update each hypothesis with the same points
        for model in self.hypotheses.values():
            model.optimize(all_points)
            # Share historical points with each model for visualization
            model.historical_points = self.historical_points

        # Recompute model posterior
        self._compute_model_posterior()

        # Check commitment criteria (pass robot position for risk-sensitive)
        robot_pos = robot_pose[:2] if robot_pose is not None else None
        self._check_commitment(robot_pos)

    # =========================================================================
    # INTERFACE METHODS - Compatible with existing code
    # =========================================================================

    @property
    def position(self) -> np.ndarray:
        """Get position from best hypothesis."""
        best = self.get_best_model()
        return best.position if best else np.zeros(2)

    @property
    def confidence(self) -> float:
        """Get confidence (max posterior probability)."""
        if not self.q_m:
            return 0.0
        return max(self.q_m.values())

    @confidence.setter
    def confidence(self, value: float):
        """Confidence is computed, not set directly."""
        pass

    def get_best_type(self) -> str:
        """Get the most probable model type."""
        if self.commitment_status:
            return self.commitment_status
        if not self.q_m:
            return 'circle'  # Default
        return max(self.q_m, key=self.q_m.get)

    def get_best_model(self) -> Optional['ObstacleBeliefBase']:
        """Get the most probable model instance."""
        best_type = self.get_best_type()
        return self.hypotheses.get(best_type)

    def get_type_probs(self) -> Dict[str, float]:
        """Get posterior probabilities over model types."""
        return self.q_m.copy()

    def get_type_confidence(self) -> float:
        """Get confidence in the best type."""
        return self.confidence

    def get_active_model(self) -> Optional['ObstacleBeliefBase']:
        """Get the active (best) model."""
        return self.get_best_model()

    def get_circle_params(self) -> Tuple[float, float, float]:
        """Get circle parameters (from circle hypothesis or bounding circle)."""
        if 'circle' in self.hypotheses:
            return self.hypotheses['circle'].get_params_tuple()
        # Fallback: bounding circle from best model
        best = self.get_best_model()
        if best:
            cx, cy = best.position
            r = best.get_bounding_radius()
            return (cx, cy, r)
        return (0, 0, 30)

    def get_rect_params(self) -> Tuple[float, float, float, float, float]:
        """Get rectangle parameters (from rect hypothesis or bounding box)."""
        if 'rectangle' in self.hypotheses:
            return self.hypotheses['rectangle'].get_params_tuple()
        # Fallback: bounding box from best model
        best = self.get_best_model()
        if best:
            cx, cy = best.position
            r = best.get_bounding_radius()
            return (cx, cy, r*2, r*2, 0.0)
        return (0, 0, 60, 60, 0)

    @property
    def circle_model(self) -> Optional['ObstacleBeliefBase']:
        """Get circle hypothesis."""
        return self.hypotheses.get('circle')

    @property
    def rect_model(self) -> Optional['ObstacleBeliefBase']:
        """Get rectangle hypothesis."""
        return self.hypotheses.get('rectangle')

    def should_be_removed(self, min_confidence: float = 0.1) -> bool:
        """Check if this tracked obstacle should be removed."""
        # Remove if all hypotheses have low confidence
        if not self.hypotheses:
            return True
        avg_conf = np.mean([m.confidence for m in self.hypotheses.values()])
        return avg_conf < min_confidence

    def update_visibility(self, was_detected: bool, robot_pos: np.ndarray,
                          robot_theta: float, sensor_range: float, frame: int):
        """Update visibility for all hypotheses."""
        for model in self.hypotheses.values():
            model.update_visibility(was_detected, robot_pos, robot_theta,
                                   sensor_range, frame)


class ObstacleBeliefBase(ABC):
    """
    Abstract base class for obstacle beliefs.

    All obstacle types (circles, rectangles, etc.) must inherit from this
    class and implement the abstract methods.

    The belief represents a Gaussian distribution over obstacle parameters:
        q(θ) = N(μ, Σ)

    where θ depends on the obstacle type:
        - Circle: θ = [cx, cy, r]
        - Rectangle: θ = [cx, cy, w, h, angle]
    """

    def __init__(self,
                 obstacle_type: ObstacleType,
                 config: ObstacleBeliefConfig,
                 device: torch.device = DEVICE):
        """
        Initialize base obstacle belief.

        Args:
            obstacle_type: Type of obstacle (CIRCLE, RECTANGLE, etc.)
            config: Configuration parameters
            device: PyTorch device for computation
        """
        self.obstacle_type = obstacle_type
        self.config = config
        self.device = device

        # These must be set by subclass
        self.mu: torch.Tensor = None  # Mean parameters
        self.Sigma: torch.Tensor = None  # Covariance matrix

        # Common metadata
        self.id = id(self)
        self.age = 0
        self.last_seen = 0
        self.observation_count = 1
        self.confidence = 0.4

        # Historical points for drift prevention
        self.historical_points: Optional[torch.Tensor] = None
        self.max_historical_points = 500

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute Signed Distance Function for given points.

        This is the core geometric function that defines the obstacle shape.
        Must be implemented using differentiable PyTorch operations.

        Args:
            points: Points to evaluate [N, 2]

        Returns:
            sdf_values: Signed distance for each point [N]
                       Negative inside, positive outside, zero on boundary
        """
        pass

    @abstractmethod
    def compute_vfe(self, points: np.ndarray, pose: np.ndarray) -> float:
        """
        Compute Variational Free Energy for this obstacle model.

        Used for Bayesian Model Selection (BMS) to compare different
        obstacle types (circle vs rectangle).

        VFE = -ln p(o|s) + D_KL[q(s)||p(s)]

        Approximation:
        - Likelihood term: based on SDF distance of points to obstacle surface
        - KL term: regularization toward prior (encodes model complexity)

        Args:
            points: LIDAR points in global frame [N, 2] (numpy array)
            pose: Robot pose [x, y, theta] (numpy array)

        Returns:
            VFE value (lower = better fit)
        """
        pass

    @abstractmethod
    def predict_range(self,
                      ray_origin: torch.Tensor,
                      ray_dirs: torch.Tensor) -> torch.Tensor:
        """
        Predict LIDAR range for rays intersecting this obstacle.

        Used by UnifiedController for observation prediction.
        Must be differentiable for gradient computation.

        Args:
            ray_origin: Origin of rays [2] (robot position)
            ray_dirs: Unit direction vectors [K, 2] for K rays

        Returns:
            ranges: Distance to obstacle for each ray [K]
                   Returns large value (e.g., 1e6) if no intersection
        """
        pass

    @abstractmethod
    def get_params_tuple(self) -> Tuple:
        """
        Get obstacle parameters as a tuple.

        Returns:
            For Circle: (cx, cy, r)
            For Rectangle: (cx, cy, w, h, angle)
        """
        pass

    @abstractmethod
    def get_center(self) -> torch.Tensor:
        """
        Get obstacle center position.

        Returns:
            center: [cx, cy] tensor
        """
        pass

    @abstractmethod
    def get_bounding_radius(self) -> float:
        """
        Get bounding circle radius (for broad-phase collision).

        Returns:
            radius: Maximum distance from center to any point on obstacle
        """
        pass

    @abstractmethod
    def clone(self) -> 'ObstacleBeliefBase':
        """Create a deep copy of this belief."""
        pass

    @abstractmethod
    def optimize(self, points: np.ndarray, num_iters: int = 15, lr: float = 0.1):
        """
        Optimize obstacle parameters via gradient descent on SDF loss.

        Each obstacle type implements its own optimization using its SDF.
        Uses PyTorch autograd for gradient computation.

        Args:
            points: LIDAR points [N, 2] numpy array
            num_iters: Number of gradient descent iterations (default: 15)
            lr: Learning rate
        """
        pass

    # =========================================================================
    # COMMON METHODS - Shared by all obstacle types
    # =========================================================================

    @property
    def position(self) -> np.ndarray:
        """Return center position as numpy array."""
        return self.get_center().detach().cpu().numpy()

    def get_covariance(self) -> np.ndarray:
        """Return covariance matrix as numpy array."""
        return self.Sigma.detach().cpu().numpy()

    def is_confirmed(self) -> bool:
        """Check if belief has enough observations to be stable."""
        return self.observation_count >= self.config.min_observations

    def is_stale(self, current_frame: int) -> bool:
        """Check if belief should be removed due to lack of observations."""
        return (current_frame - self.last_seen) > self.config.stale_threshold

    def update_metadata(self, frame: int, observed: bool = True):
        """Update belief metadata after an observation cycle."""
        self.age += 1
        if observed:
            self.last_seen = frame
            self.observation_count += 1
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.confidence = max(0.1, self.confidence - 0.02)

    # =========================================================================
    # LIFECYCLE METHODS - Visibility and removal logic
    # =========================================================================

    def update_with_points(self, points: np.ndarray, robot_pose: np.ndarray):
        """
        Update this instance with new observation points.

        This method:
        1. Accumulates historical points
        2. Gets all constraint points (current + historical)
        3. Runs SDF optimization

        Args:
            points: LIDAR points [N, 2] in room frame
            robot_pose: Robot pose [x, y, theta]
        """
        if len(points) < 3:
            return

        # Convert to tensor
        points_t = torch.tensor(points, dtype=DTYPE, device=self.device)

        # Accumulate historical points
        self.add_historical_points(points_t)

        # Get all constraint points
        all_points = self.get_all_constraint_points(points_t)

        # Optimize with all points
        if len(all_points) >= 3:
            self.optimize(all_points.cpu().numpy())

        # Update observation count
        self.observation_count += 1

    # =========================================================================
    # UNIFIED INTERFACE - Methods for compatibility with tracking system
    # =========================================================================

    def get_type_probs(self) -> Dict[str, float]:
        """Get type probabilities. Single-type models return 100% for their type."""
        if self.obstacle_type == ObstacleType.CIRCLE:
            return {'circle': 1.0, 'rectangle': 0.0}
        else:
            return {'circle': 0.0, 'rectangle': 1.0}

    def get_best_type(self) -> str:
        """Get the obstacle type as string."""
        return self.obstacle_type.value

    def get_type_confidence(self) -> float:
        """Get confidence in the type (same as overall confidence for single-type)."""
        return self.confidence

    def get_active_model(self) -> 'ObstacleBeliefBase':
        """Get the active model (returns self for single-type obstacles)."""
        return self

    def get_circle_params(self) -> Tuple[float, float, float]:
        """
        Get circle parameters (cx, cy, r).
        For circles: returns actual params.
        For rectangles: returns bounding circle.
        """
        if self.obstacle_type == ObstacleType.CIRCLE:
            return self.get_params_tuple()
        else:
            cx, cy = self.position
            r = self.get_bounding_radius()
            return (cx, cy, r)

    def get_rect_params(self) -> Tuple[float, float, float, float, float]:
        """
        Get rectangle parameters (cx, cy, w, h, angle).
        For rectangles: returns actual params.
        For circles: returns bounding box.
        """
        if self.obstacle_type == ObstacleType.RECTANGLE:
            return self.get_params_tuple()
        else:
            cx, cy, r = self.get_params_tuple()
            return (cx, cy, r * 2, r * 2, 0.0)

    @property
    def circle_model(self) -> Optional['ObstacleBeliefBase']:
        """Get circle model (self if circle, None otherwise)."""
        return self if self.obstacle_type == ObstacleType.CIRCLE else None

    @property
    def rect_model(self) -> Optional['ObstacleBeliefBase']:
        """Get rectangle model (self if rectangle, None otherwise)."""
        return self if self.obstacle_type == ObstacleType.RECTANGLE else None

    def is_in_sensor_range(self, robot_pos: np.ndarray, sensor_range: float) -> bool:
        """
        Check if obstacle is within sensor range.

        Args:
            robot_pos: Robot position [x, y]
            sensor_range: Maximum sensor range

        Returns:
            True if obstacle center is within sensor range
        """
        center = self.get_center().cpu().numpy()
        dist = np.linalg.norm(center - robot_pos[:2])
        # Account for obstacle size - it might be partially visible
        return dist < (sensor_range + self.get_bounding_radius())

    def is_in_field_of_view(self, robot_pos: np.ndarray, robot_theta: float,
                            fov_angle: float = np.radians(240)) -> bool:
        """
        Check if obstacle is within robot's field of view.

        Args:
            robot_pos: Robot position [x, y]
            robot_theta: Robot heading angle (radians)
            fov_angle: Total field of view angle (default 240° for typical LIDAR)

        Returns:
            True if obstacle is within FOV
        """
        center = self.get_center().cpu().numpy()

        # Angle from robot to obstacle
        diff = center - robot_pos[:2]
        angle_to_obs = np.arctan2(diff[1], diff[0])

        # Angle difference from robot heading
        angle_diff = angle_to_obs - robot_theta
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        return np.abs(angle_diff) < (fov_angle / 2)

    def should_be_visible(self, robot_pos: np.ndarray, robot_theta: float,
                          sensor_range: float) -> bool:
        """
        Check if this obstacle SHOULD be visible to the robot.

        An obstacle should be visible if:
        1. It is within sensor range
        2. It is within field of view

        Note: This does NOT account for occlusion by other obstacles.
        Occlusion should be handled at a higher level if needed.

        Args:
            robot_pos: Robot position [x, y]
            robot_theta: Robot heading (radians)
            sensor_range: Maximum sensor range

        Returns:
            True if obstacle should be visible (and thus if not detected, it's evidence of removal)
        """
        return (self.is_in_sensor_range(robot_pos, sensor_range) and
                self.is_in_field_of_view(robot_pos, robot_theta))

    def update_visibility(self, was_detected: bool, robot_pos: np.ndarray,
                          robot_theta: float, sensor_range: float, frame: int):
        """
        Update obstacle confidence based on visibility.

        This implements the lifecycle logic:
        - If detected: increase confidence
        - If should be visible but not detected: decrease confidence (evidence of removal)
        - If out of range or FOV: maintain confidence (no evidence either way)

        Args:
            was_detected: Whether this obstacle was matched to a cluster
            robot_pos: Robot position [x, y]
            robot_theta: Robot heading (radians)
            sensor_range: Maximum sensor range
            frame: Current frame number
        """
        self.age += 1

        if was_detected:
            # Obstacle was seen - increase confidence
            self.last_seen = frame
            self.observation_count += 1
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            # Obstacle was NOT detected
            if self.should_be_visible(robot_pos, robot_theta, sensor_range):
                # Should be visible but wasn't - evidence of removal
                # Decay confidence slowly
                self.confidence *= 0.98
            # else: Out of range or FOV - no change to confidence

    def should_be_removed(self, min_confidence: float = 0.1) -> bool:
        """
        Check if this obstacle belief should be removed.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            True if obstacle should be removed from tracking
        """
        return self.confidence < min_confidence

    def add_historical_points(self, points: torch.Tensor, sdf_threshold: float = 8.0):
        """
        Add points with good SDF fit to historical buffer.

        Points are redistributed along the model surface (by angle from center)
        to maintain spatial diversity instead of just keeping recent points.

        Args:
            points: LIDAR points [N, 2]
            sdf_threshold: Max SDF error to consider "on surface"
        """
        if len(points) == 0:
            return

        # Ensure points are on same device as model
        points = points.to(self.device)

        # Compute SDF for each point
        sdf_values = self.sdf(points)
        sdf_errors = torch.abs(sdf_values)

        # Select points with good fit
        good_mask = sdf_errors < sdf_threshold
        good_points = points[good_mask]

        if len(good_points) == 0:
            return

        # Add to historical buffer
        if self.historical_points is None:
            self.historical_points = good_points.clone()
        else:
            self.historical_points = torch.cat(
                [self.historical_points, good_points], dim=0
            )

        # Limit buffer size with angular redistribution
        if len(self.historical_points) > self.max_historical_points:
            self._redistribute_historical_points()

    def _redistribute_historical_points(self):
        """
        Redistribute historical points to maintain spatial diversity.

        Uses angular binning from obstacle center to keep points from
        all viewing angles, not just the most recent ones.
        """
        if self.historical_points is None or len(self.historical_points) <= self.max_historical_points:
            return

        # Get obstacle center
        center = self.get_center()

        # Compute angle from center for each point
        relative = self.historical_points - center
        angles = torch.atan2(relative[:, 1], relative[:, 0])

        # Bin points by angle
        n_bins = self.max_historical_points
        bin_size = 2 * np.pi / n_bins
        bins = ((angles + np.pi) / bin_size).long() % n_bins

        # Vectorized: for each bin, keep the last (most recent) point
        n_points = len(self.historical_points)
        point_indices = torch.arange(n_points, device=self.historical_points.device)

        # Use scatter_reduce with 'amax' to find max index per bin
        bin_max_indices = torch.full((n_bins,), -1, dtype=torch.long, device=self.historical_points.device)
        bin_max_indices.scatter_reduce_(0, bins, point_indices, reduce='amax', include_self=False)

        # Filter out empty bins
        valid_mask = bin_max_indices >= 0
        kept_indices = bin_max_indices[valid_mask]

        if len(kept_indices) > 0:
            self.historical_points = self.historical_points[kept_indices]

    def get_all_constraint_points(self, current_points: torch.Tensor) -> torch.Tensor:
        """
        Get all points that constrain the obstacle: current + historical.

        Historical points are stored in room frame and should ALL be used
        for optimization - they provide evidence from multiple viewing angles.
        No visibility filtering is applied.
        """
        if self.historical_points is None or len(self.historical_points) == 0:
            return current_points

        if len(current_points) == 0:
            return self.historical_points

        # Ensure both tensors are on same device
        current_points = current_points.to(self.device)
        return torch.cat([current_points, self.historical_points], dim=0)


    def to_dict(self) -> Dict:
        """Serialize belief to dictionary."""
        return {
            'type': self.obstacle_type.value,
            'params': self.get_params_tuple(),
            'covariance': self.get_covariance().tolist(),
            'confidence': self.confidence,
            'age': self.age,
            'observations': self.observation_count,
        }

    def __repr__(self) -> str:
        params = self.get_params_tuple()
        return (f"{self.obstacle_type.value}("
                f"params={params}, "
                f"conf={self.confidence:.2f}, "
                f"obs={self.observation_count})")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def softmin_ranges(ranges_list: List[torch.Tensor],
                   eta: float = 0.1) -> torch.Tensor:
    """
    Compute soft-min over ranges from multiple obstacles.

    Used to aggregate LIDAR predictions from all obstacles into a single
    predicted observation vector. Soft-min is differentiable, allowing
    gradients to flow through all obstacles.

    Paper Eq. softmin_range:
        ρ̂_k = -1/η * log(Σ_i exp(-η * ρ_{k,i}))

    Args:
        ranges_list: List of range tensors [K] from each obstacle
        eta: Sharpness parameter (higher = closer to hard min)

    Returns:
        aggregated_ranges: Soft-min ranges [K]
    """
    if len(ranges_list) == 0:
        raise ValueError("ranges_list cannot be empty")

    if len(ranges_list) == 1:
        return ranges_list[0]

    # Stack: [num_obstacles, K]
    stacked = torch.stack(ranges_list, dim=0)

    # Soft-min: -1/η * log(Σ exp(-η * ρ))
    # Numerically stable using logsumexp
    neg_scaled = -eta * stacked
    logsumexp = torch.logsumexp(neg_scaled, dim=0)
    softmin = -logsumexp / eta

    return softmin


def predict_all_ranges(obstacles: List[ObstacleBeliefBase],
                       ray_origin: torch.Tensor,
                       ray_dirs: torch.Tensor,
                       max_range: float = 300.0,
                       softmin_eta: float = 0.1) -> torch.Tensor:
    """
    Predict LIDAR ranges for all obstacles combined.

    This is the main interface for UnifiedController to get predicted
    observations. It iterates over all obstacles polymorphically.

    Args:
        obstacles: List of obstacle beliefs (any type)
        ray_origin: Robot position [2]
        ray_dirs: Ray direction unit vectors [K, 2]
        max_range: Maximum sensor range
        softmin_eta: Soft-min sharpness

    Returns:
        predicted_ranges: Aggregated ranges [K]
    """
    K = ray_dirs.shape[0]
    device = ray_dirs.device
    dtype = ray_dirs.dtype

    if len(obstacles) == 0:
        return torch.full((K,), max_range, dtype=dtype, device=device)

    # Collect ranges from all obstacles
    ranges_list = []
    for obs in obstacles:
        ranges = obs.predict_range(ray_origin, ray_dirs)
        # Clamp to max range
        ranges = torch.clamp(ranges, max=max_range)
        ranges_list.append(ranges)

    # Aggregate with soft-min
    return softmin_ranges(ranges_list, eta=softmin_eta)

