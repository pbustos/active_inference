"""
Analytic Ray-Circle Intersection Model for Active Inference

Implements the differentiable LIDAR observation model from the paper:
- Section 5.1.3: "Analytic generative observation model: ray-circle ranges"

Key Equations Implemented:
--------------------------
1. Ray-circle intersection (Eq. ray_circle_range):
   For ray k and obstacle i with center c_i and radius r_i:

   q_i = o - c_i                    (vector from obstacle to robot origin)
   b_{k,i} = d_k^T q_i              (projection of q onto ray direction)
   c_i = ||q_i||² - r_i²            (squared distance minus radius squared)
   Δ_{k,i} = b_{k,i}² - c_i         (discriminant)

   ρ_{k,i} = -b_{k,i} - √Δ_{k,i}    (intersection range, valid if Δ > 0, ρ > 0)

2. Soft-min aggregation (Eq. softmin_range):
   To maintain differentiability, we use soft-min instead of hard min:

   ρ̂_k = -1/η * log(Σ_i exp(-η * ρ_{k,i}))

   where η controls sharpness (η→∞ gives hard min).

3. Predicted observation vector (Eq. predicted_lidar_observation):
   ô_{t+1}(u_t) = ρ̂(u_t) = (ρ̂_1, ..., ρ̂_K)^T

4. Pragmatic term (Eq. pragmatic_cost):
   G_prag = (1/2)(ô - o*)^T Σ*^{-1} (ô - o*)

   where o* is the preferred observation (large ranges = safe).

Uses PyTorch autograd for efficient gradient computation with many rays (360).
All operations are vectorized and differentiable.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RayCircleConfig:
    """
    Configuration for ray-circle model.

    Paper Section 5.1.3 parameters:
    - K (num_rays): Number of LIDAR rays, typically 360 for 1° resolution
    - η (softmin_eta): Soft-min sharpness parameter (Eq. softmin_range)
    """
    num_rays: int = 360             # K: number of LIDAR rays (1 degree resolution)
    max_range: float = 300.0        # Maximum sensor range (ρ_max)
    softmin_eta: float = 0.02       # η: soft-min sharpness (higher = sharper)
    min_range: float = 1.0          # Minimum valid range (ρ_min)


class RayCircleModel:
    """
    Differentiable ray-circle intersection model using PyTorch autograd.

    All operations are vectorized for GPU acceleration.
    Gradients are computed automatically via autograd.

    Includes caching for repeated obstacle configurations to speed up
    multi-seed trajectory optimization where obstacles don't change.

    Paper equations:
    - ρ_{k,i} = -b_{k,i} - √Δ_{k,i}  (Eq. ray_circle_range)
    - ρ̂_k = -1/η log Σ exp(-η ρ_{k,i})  (Eq. softmin_range)
    """

    def __init__(self, config: RayCircleConfig = None, device: str = None):
        self.config = config or RayCircleConfig()
        # Auto-select device: prefer GPU if available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Pre-compute ray directions in robot frame [K, 2]
        K = self.config.num_rays
        angles = torch.linspace(0, 2 * np.pi, K + 1, dtype=torch.float32, device=self.device)[:-1]
        self.ray_angles_rob = angles
        self.ray_dirs_rob = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1)  # [K, 2]

        # Cache for obstacle-dependent computations
        # When obstacles don't change, we can reuse c_i and radii
        self._cache_enabled = True
        self._cached_obstacles_hash = None
        self._cached_centers = None
        self._cached_radii = None
        self._cached_c = None  # c_i = ||q_i||² - r_i² (obstacle-dependent part)

    def _compute_obstacles_hash(self, obstacles: torch.Tensor) -> int:
        """Compute hash of obstacles tensor for cache invalidation."""
        if obstacles.numel() == 0:
            return 0
        # Use sum of all values as simple hash (fast but not collision-free)
        return int(obstacles.sum().item() * 1000) + obstacles.shape[0]

    def _update_obstacle_cache(self, obstacles: torch.Tensor):
        """Update cached obstacle data if obstacles changed."""
        if not self._cache_enabled:
            return

        obs_hash = self._compute_obstacles_hash(obstacles)
        if obs_hash != self._cached_obstacles_hash:
            # Cache miss - update cached values
            self._cached_obstacles_hash = obs_hash
            self._cached_centers = obstacles[:, :2].clone()  # [N, 2]
            self._cached_radii = obstacles[:, 2].clone()      # [N]
            # Note: c_i depends on robot position, so can't cache it fully
            # But we can cache radii² for reuse
            self._cached_radii_sq = self._cached_radii ** 2

    def predict_ranges(self,
                       pose: torch.Tensor,
                       obstacles: torch.Tensor,
                       ) -> torch.Tensor:
        """
        Predict LIDAR ranges for given pose and obstacles.

        Implements Paper Section 5.1.3 ray-circle intersection model.
        Fully differentiable via PyTorch autograd.

        Paper Equations:
        ----------------
        For each ray k and obstacle i:

        1. q_i = o - c_i                    (Eq. before ray_circle_range)
        2. b_{k,i} = d_k^T q_i
        3. c_i = ||q_i||² - r_i²
        4. Δ_{k,i} = b_{k,i}² - c_i
        5. ρ_{k,i} = -b_{k,i} - √Δ_{k,i}    (Eq. ray_circle_range)
        6. ρ̂_k = softmin_i(ρ_{k,i})         (Eq. softmin_range)

        Args:
            pose: Robot pose [x, y, θ] (can have requires_grad=True)
            obstacles: Obstacle circles [N, 3] as (cx, cy, r)

        Returns:
            ranges: Predicted ranges [K], differentiable
        """
        # Ensure correct dtype and device
        pose = pose.to(dtype=torch.float32, device=self.device)
        if obstacles.numel() == 0:
            return torch.full((self.config.num_rays,), self.config.max_range,
                            dtype=torch.float32, device=self.device)
        obstacles = obstacles.to(dtype=torch.float32, device=self.device)

        # Update obstacle cache if needed
        self._update_obstacle_cache(obstacles)

        x, y, theta = pose[0], pose[1], pose[2]
        K = self.config.num_rays
        N = obstacles.shape[0]

        # Rotation matrix R(θ) - differentiable
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R = torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t, cos_t])
        ])  # [2, 2]

        # Ray directions in room frame: d_k = R(θ) @ d_k^rob
        ray_dirs = self.ray_dirs_rob @ R.T  # [K, 2]

        # Robot origin
        origin = pose[:2]  # [2]

        # Use cached centers and radii if available
        if self._cache_enabled and self._cached_centers is not None:
            centers = self._cached_centers
            radii_sq = self._cached_radii_sq
        else:
            centers = obstacles[:, :2]
            radii_sq = obstacles[:, 2] ** 2

        # Vectorized ray-circle intersection for all K rays and N obstacles
        # q_i = origin - center_i  [N, 2]
        q = origin.unsqueeze(0) - centers  # [N, 2]

        # c_i = ||q_i||² - r_i²  [N] - use cached radii_sq
        c = torch.sum(q ** 2, dim=1) - radii_sq  # [N]

        # b_{k,i} = d_k · q_i  [K, N]
        b = ray_dirs @ q.T  # [K, N]

        # Δ_{k,i} = b_{k,i}² - c_i  [K, N]
        delta = b ** 2 - c.unsqueeze(0)  # [K, N]

        # ρ_{k,i} = -b_{k,i} - √Δ_{k,i} (only valid if Δ > 0 and ρ > 0)
        # Use smooth approximation for sqrt to maintain gradients
        delta_safe = torch.clamp(delta, min=1e-6)
        sqrt_delta = torch.sqrt(delta_safe)
        rho = -b - sqrt_delta  # [K, N]

        # Mask invalid intersections (Δ <= 0 or ρ <= min_range)
        valid = (delta > 0) & (rho > self.config.min_range)

        # Replace invalid with large value for soft-min
        large_val = self.config.max_range * 2
        rho_masked = torch.where(valid, rho, torch.full_like(rho, large_val))

        # Add max_range as fallback when no hits [K, N+1]
        max_range_col = torch.full((K, 1), self.config.max_range,
                                   dtype=torch.float32, device=self.device)
        rho_with_max = torch.cat([rho_masked, max_range_col], dim=1)

        # Soft-min aggregation (Eq. softmin_range) - differentiable
        # ρ̂_k = -1/η log Σ exp(-η ρ_{k,i})
        eta = self.config.softmin_eta

        # Use logsumexp for numerical stability
        # soft_min = -logsumexp(-η * ρ) / η
        ranges = -torch.logsumexp(-eta * rho_with_max, dim=1) / eta

        # Clamp to valid range
        ranges = torch.clamp(ranges, min=self.config.min_range, max=self.config.max_range)

        return ranges

    def compute_G_pragmatic(self,
                            pose: torch.Tensor,
                            obstacles: torch.Tensor,
                            preferred_range: float = 80.0,
                            sigma_sq: float = 400.0,
                            lambda_obs: float = 50.0,
                            robot_radius: float = 10.0
                            ) -> torch.Tensor:
        """
        Compute pragmatic free energy from predicted ranges.

        Paper Eq. pragmatic_cost:
        G_prag = (1/2σ²) ||ρ̂ - o*||²

        Plus safety penalty for very close obstacles.

        Args:
            pose: Robot pose [x, y, θ]
            obstacles: Obstacle circles [N, 3]
            preferred_range: o* preferred observation distance
            sigma_sq: σ² observation variance
            lambda_obs: Obstacle avoidance weight
            robot_radius: Robot collision radius

        Returns:
            G_prag: Scalar pragmatic cost (differentiable)
        """
        ranges = self.predict_ranges(pose, obstacles)
        K = len(ranges)

        # Observation error: penalize ranges closer than preferred
        obs_error = torch.clamp(preferred_range - ranges, min=0.0)

        # G_prag = (1/2σ²) Σ (o* - ρ̂_k)²  (Paper Eq. pragmatic_cost)
        precision = 1.0 / sigma_sq
        G_obs = 0.5 * precision * torch.sum(obs_error ** 2) / K
        G_obs = lambda_obs * 0.005 * G_obs

        # Safety penalty: exponential for very close obstacles
        soft_margin = robot_radius + 25.0
        hard_margin = robot_radius + 12.0

        # Soft zone penalty
        close_mask = ranges < soft_margin
        if close_mask.any():
            close_ranges = ranges[close_mask]
            soft_penalty = torch.sum((soft_margin - close_ranges) ** 2) / 100.0
            G_safety = lambda_obs * 0.2 * soft_penalty
        else:
            G_safety = torch.tensor(0.0, device=self.device)

        # Hard zone exponential penalty
        very_close_mask = ranges < hard_margin
        if very_close_mask.any():
            very_close_ranges = ranges[very_close_mask]
            hard_penalty = torch.sum(torch.exp(2.0 * (hard_margin - very_close_ranges)))
            G_safety = G_safety + lambda_obs * hard_penalty

        return G_obs + G_safety


def test_ray_circle_model():
    """Test the ray-circle model with autograd"""
    print("=" * 70)
    print("Testing Ray-Circle Model with Autograd (256 rays)")
    print("=" * 70)
    print()

    import time

    config = RayCircleConfig(num_rays=256, max_range=300.0, softmin_eta=0.02)
    model = RayCircleModel(config)

    # Test 1: Basic prediction
    print("Test 1: Range prediction with 256 rays")
    pose = torch.tensor([100.0, 100.0, 0.5], requires_grad=True)
    obstacles = torch.tensor([
        [200.0, 120.0, 40.0],
        [150.0, 200.0, 30.0],
        [300.0, 100.0, 25.0]
    ])

    t0 = time.time()
    ranges = model.predict_ranges(pose, obstacles)
    t1 = time.time()

    print(f"  Prediction time: {(t1-t0)*1000:.2f} ms")
    print(f"  Ranges: min={ranges.min().item():.1f}, max={ranges.max().item():.1f}")
    print(f"  Close rays (<50): {(ranges < 50).sum().item()}")
    print()

    # Test 2: Gradient via autograd
    print("Test 2: Gradient computation via autograd")
    pose = torch.tensor([100.0, 100.0, 0.0], requires_grad=True)

    t0 = time.time()
    ranges = model.predict_ranges(pose, obstacles)
    loss = ranges.sum()  # Simple loss for testing
    loss.backward()
    t1 = time.time()

    print(f"  Gradient time: {(t1-t0)*1000:.2f} ms")
    print(f"  ∂loss/∂x = {pose.grad[0].item():.4f}")
    print(f"  ∂loss/∂y = {pose.grad[1].item():.4f}")
    print(f"  ∂loss/∂θ = {pose.grad[2].item():.4f}")
    print()

    # Test 3: G_pragmatic computation
    print("Test 3: G_pragmatic computation")
    pose = torch.tensor([100.0, 100.0, 0.0], requires_grad=True)

    t0 = time.time()
    G = model.compute_G_pragmatic(pose, obstacles)
    G.backward()
    t1 = time.time()

    print(f"  G_prag = {G.item():.2f}")
    print(f"  Computation time: {(t1-t0)*1000:.2f} ms")
    print(f"  ∇G = [{pose.grad[0].item():.2f}, {pose.grad[1].item():.2f}, {pose.grad[2].item():.2f}]")
    print()

    # Test 4: Batch performance
    print("Test 4: Batch performance (100 iterations)")
    times = []
    for _ in range(100):
        pose = torch.tensor([100.0 + np.random.randn()*10,
                            100.0 + np.random.randn()*10,
                            np.random.randn()*0.5], requires_grad=True)
        t0 = time.time()
        G = model.compute_G_pragmatic(pose, obstacles)
        G.backward()
        times.append(time.time() - t0)

    print(f"  Avg time: {np.mean(times)*1000:.2f} ms")
    print(f"  Std time: {np.std(times)*1000:.2f} ms")
    print()

    print("=" * 70)
    print("Ray-Circle Model with Autograd: OK")
    print("=" * 70)



if __name__ == "__main__":
    test_ray_circle_model()
