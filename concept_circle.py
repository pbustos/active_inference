"""
Concept: Circle - Circular Obstacle Belief Implementation

This module implements circular obstacle beliefs and their lifecycle management.
The CircleManager class handles all instance creation, update, and removal.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from obstacle_base import (
    ObstacleBeliefBase,
    ObstacleBeliefConfig,
    ObstacleManager,
    ObstacleType,
    DEVICE,
    DTYPE
)


@dataclass
class CircleObstacleConfig(ObstacleBeliefConfig):
    """Configuration specific to circular obstacles."""
    prior_radius_std: float = 3.0
    radius_prior_mean: float = 35.0
    radius_prior_std: float = 6.0
    min_radius: float = 12.0
    max_radius: float = 45.0
    initial_radius_std: float = 15.0


class CircleManager(ObstacleManager):
    """
    Manages the lifecycle of CircleObstacle instances.

    Handles:
    - Creating circles from point clusters (algebraic circle fit)
    - Matching clusters to existing circles
    - Updating circles with new observations
    - Removing circles based on visibility logic
    """

    def __init__(self, config: CircleObstacleConfig = None):
        super().__init__(config or CircleObstacleConfig())
        self.circle_config = config or CircleObstacleConfig()

    def _create_from_points(self, points: np.ndarray, robot_pose: np.ndarray) -> Optional['CircleObstacle']:
        """
        Create a new CircleObstacle from a point cluster using algebraic circle fit.
        """
        if len(points) < 3:
            return None

        cx, cy, radius = self._fit_circle(points)

        # Validate radius
        if radius < self.circle_config.min_radius or radius > self.circle_config.max_radius:
            return None

        circle = CircleObstacle(cx, cy, radius, self.circle_config)
        circle.confidence = 0.4
        return circle

    @staticmethod
    def _fit_circle(points: np.ndarray) -> Tuple[float, float, float]:
        """
        Algebraic circle fit using least squares.

        Fits circle: (x - cx)² + (y - cy)² = r²

        Returns:
            (cx, cy, radius)
        """
        if len(points) < 3:
            centroid = np.mean(points, axis=0)
            return centroid[0], centroid[1], 30.0

        centroid = np.mean(points, axis=0)

        # Set up least squares: x² + y² = 2*cx*x + 2*cy*y + (r² - cx² - cy²)
        A = np.column_stack([
            2 * points[:, 0],
            2 * points[:, 1],
            np.ones(len(points))
        ])
        b = points[:, 0]**2 + points[:, 1]**2

        try:
            result, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            cx = result[0]
            cy = result[1]
            radius_sq = result[2] + cx**2 + cy**2

            if radius_sq > 0:
                radius = np.sqrt(radius_sq)
            else:
                radius = np.mean(np.linalg.norm(points - centroid, axis=1))

            radius = np.clip(radius, 10, 100)
        except:
            cx, cy = centroid
            radius = 30.0

        return cx, cy, radius


class CircleObstacle(ObstacleBeliefBase):
    """Circular obstacle belief. State: [cx, cy, r]"""
    def __init__(self, cx: float, cy: float, r: float,
                 config: Optional[CircleObstacleConfig] = None,
                 device: torch.device = DEVICE):
        config = config or CircleObstacleConfig()
        super().__init__(ObstacleType.CIRCLE, config, device)
        self.circle_config = config
        self.mu = torch.tensor([cx, cy, r], dtype=DTYPE, device=device)
        self.Sigma = torch.diag(torch.tensor([
            config.initial_position_std ** 2,
            config.initial_position_std ** 2,
            config.initial_radius_std ** 2
        ], dtype=DTYPE, device=device))
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """SDF_circle(p, c, r) = ||p - c|| - r"""
        center = self.mu[:2]
        radius = self.mu[2]
        return torch.norm(points - center, dim=1) - radius
    def predict_range(self, ray_origin: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
        """Ray-circle intersection."""
        center, radius = self.mu[:2], self.mu[2]
        K = ray_dirs.shape[0]
        q = ray_origin - center
        b = torch.sum(ray_dirs * q.unsqueeze(0), dim=1)
        c = torch.sum(q ** 2) - radius ** 2
        discriminant = b ** 2 - c
        ranges = torch.full((K,), 1e6, dtype=DTYPE, device=self.device)
        valid = discriminant > 0
        if valid.any():
            sqrt_disc = torch.sqrt(discriminant[valid])
            rho = -b[valid] - sqrt_disc
            positive = rho > 0
            valid_indices = torch.where(valid)[0]
            positive_indices = valid_indices[positive]
            if len(positive_indices) > 0:
                ranges[positive_indices] = rho[positive]
        return ranges
    def get_params_tuple(self) -> Tuple[float, float, float]:
        return (self.mu[0].item(), self.mu[1].item(), self.mu[2].item())
    def get_center(self) -> torch.Tensor:
        return self.mu[:2]
    def get_bounding_radius(self) -> float:
        return self.mu[2].item()

    def compute_vfe(self, points: np.ndarray, pose: np.ndarray) -> float:
        """
        Compute Variational Free Energy for circular obstacle model.

        From paper Eq. obstacle_objective:

        F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)

        where:
        - dᵢ(s) = ||pᵢ - c|| - r  (SDF for circle, Eq. circle_sdf)
        - First term: prediction error (negative accuracy)
        - Second term: complexity (prior regulariser / KL divergence)

        For model selection, we use a simplified form that captures
        the essential trade-off between fit quality and model complexity.

        Args:
            points: LIDAR points [N, 2] numpy array
            pose: Robot pose [x, y, theta] numpy array

        Returns:
            VFE value (lower = better fit)
        """
        if len(points) == 0:
            return 1e6

        # Convert to tensor
        points_t = torch.tensor(points, dtype=DTYPE, device=self.device)

        # =====================================================================
        # PREDICTION ERROR TERM (Eq. cluster_likelihood)
        # -ln p(C|s) ∝ (1/2σ²_o) Σᵢ dᵢ(s)²
        # =====================================================================
        sdf_values = self.sdf(points_t)  # dᵢ(s) = ||pᵢ - c|| - r
        sigma_o = 5.0  # Observation noise σ_o (matches paper's SDF residual model)
        prediction_error = torch.sum(sdf_values ** 2) / (2.0 * sigma_o ** 2)

        # =====================================================================
        # COMPLEXITY TERM (Prior regulariser / KL divergence)
        # (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ) + (1/2)ln|Σₚ|
        #
        # For model comparison, we approximate this as:
        # - Number of parameters (Occam factor): 3 for circle
        # - Uncertainty penalty: log-determinant of covariance
        # =====================================================================
        n_params = 3  # Circle: (cx, cy, r)

        # Log-determinant of covariance (uncertainty measure)
        # Higher uncertainty = higher complexity cost
        log_det = torch.logdet(self.Sigma + 1e-6 * torch.eye(n_params, device=self.device))

        # Complexity = Occam factor + uncertainty penalty
        # The Occam factor penalizes models with more parameters
        complexity = 0.1 * n_params + 0.05 * torch.clamp(log_det, min=0.0)

        vfe = prediction_error.item() + complexity.item()
        return vfe
    def clone(self) -> 'CircleObstacle':
        cx, cy, r = self.get_params_tuple()
        new_obs = CircleObstacle(cx, cy, r, self.circle_config, self.device)
        new_obs.mu = self.mu.clone()
        new_obs.Sigma = self.Sigma.clone()
        new_obs.confidence = self.confidence
        new_obs.age = self.age
        new_obs.last_seen = self.last_seen
        new_obs.observation_count = self.observation_count
        if self.historical_points is not None:
            new_obs.historical_points = self.historical_points.clone()
        return new_obs

    def optimize(self, points: np.ndarray, num_iters: int = 15, lr: float = 0.15):
        """
        Optimize circle parameters via gradient descent on SDF loss.

        Uses PyTorch autograd for gradient computation.

        F = (1/2σ²) Σ (||p - c|| - r)² + (1/2σ_r²)(r - r₀)²

        Args:
            points: LIDAR points [N, 2] numpy array
            num_iters: Number of gradient descent iterations
            lr: Learning rate
        """
        if len(points) < 3:
            return

        points_t = torch.tensor(points, dtype=DTYPE, device=self.device)

        # Radius prior (expected size for typical obstacles)
        RADIUS_PRIOR_MEAN = self.circle_config.radius_prior_mean
        RADIUS_PRIOR_STD = self.circle_config.radius_prior_std
        radius_precision = 1.0 / (RADIUS_PRIOR_STD ** 2)

        # Make parameters require gradients
        mu = self.mu.clone().detach().requires_grad_(True)

        sigma_sq = 25.0  # Observation noise variance

        for _ in range(num_iters):
            # Compute SDF with current parameters
            cx, cy, r = mu[0], mu[1], mu[2]

            # Circle SDF: ||p - c|| - r
            diffs = points_t - torch.stack([cx, cy])
            dists = torch.norm(diffs, dim=1)
            sdf = dists - r

            # =====================================================================
            # LOSS = Prediction Error + Radius Prior
            # =====================================================================

            # Prediction error (SDF loss)
            prediction_error = torch.sum(sdf ** 2) / (2.0 * sigma_sq)

            # Radius prior - pull toward expected radius
            radius_prior = 0.5 * radius_precision * (r - RADIUS_PRIOR_MEAN) ** 2

            # Total loss
            loss = prediction_error + 0.1 * radius_prior

            # Gradient via autograd
            if mu.grad is not None:
                mu.grad.zero_()
            loss.backward()

            # Update with gradient clipping
            with torch.no_grad():
                grad = mu.grad
                if grad is not None and not torch.isnan(grad).any():
                    grad = torch.clamp(grad, -10.0, 10.0)
                    mu -= lr * grad
                # Clamp radius
                mu[2] = torch.clamp(mu[2], self.circle_config.min_radius, self.circle_config.max_radius)
                # Check for NaN and reset if needed
                if torch.isnan(mu).any():
                    mu = self.mu.clone().detach()
                    break

            mu.requires_grad_(True)

        # Apply updates
        self.mu = mu.detach()

    @property
    def radius(self) -> float:
        return self.mu[2].item()


# =============================================================================
# REGISTRATION - Register circle as an obstacle type
# =============================================================================

def fit_circle_from_points(points: np.ndarray, robot_pose: np.ndarray = None) -> Optional[CircleObstacle]:
    """
    Fit a CircleObstacle from points using algebraic circle fit.

    This is the factory function used by TrackedObstacle to create circle hypotheses.
    Always returns a hypothesis (even if poor fit) to enable model comparison.
    """
    if len(points) < 3:
        return None

    config = CircleObstacleConfig()
    cx, cy, radius = CircleManager._fit_circle(points)

    # If radius is out of bounds, use bounding circle instead
    if radius < config.min_radius or radius > config.max_radius:
        # Fallback: use centroid and max distance as radius
        centroid = np.mean(points, axis=0)
        cx, cy = centroid[0], centroid[1]
        distances = np.linalg.norm(points - centroid, axis=1)
        radius = np.clip(np.max(distances) * 1.1, config.min_radius, config.max_radius)

    circle = CircleObstacle(cx, cy, radius, config)
    circle.confidence = 0.4
    return circle


# Register circle type in the global registry
from obstacle_base import register_obstacle_type
register_obstacle_type(
    name='circle',
    cls=CircleObstacle,
    config_class=CircleObstacleConfig,
    fit_func=fit_circle_from_points,
    prior=1.0  # Equal prior with rectangles
)


if __name__ == "__main__":
    print("Testing CircleObstacle...")
    circle = CircleObstacle(100.0, 150.0, 30.0)
    print(f"Created: {circle}")
    test_points = torch.tensor([[100.0, 150.0], [130.0, 150.0]], dtype=DTYPE, device=circle.device)
    print(f"SDF: {circle.sdf(test_points).tolist()}")
    ray_origin = torch.tensor([0.0, 150.0], dtype=DTYPE, device=circle.device)
    ray_dirs = torch.tensor([[1.0, 0.0]], dtype=DTYPE, device=circle.device)
    print(f"Range: {circle.predict_range(ray_origin, ray_dirs).tolist()}")
    print("OK!")
