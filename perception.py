"""Perception module: ray-casting, clustering, circle fitting"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.cluster import DBSCAN

def ray_cast_vectorized_full(robot_pos: np.ndarray, robot_theta: float,
                              obstacles: np.ndarray, num_rays: int,
                              sensor_range: float) -> Dict:
    """
    VECTORIZED ray-casting with FULL Jacobian for Active Inference.
    
    Args:
        robot_pos: [x, y] robot position
        robot_theta: robot orientation
        obstacles: Nx3 array [x, y, r] for each obstacle
        num_rays: number of rays to cast
        sensor_range: maximum detection range
    
    Returns:
        Dict with:
        - distances: [num_rays] distance per ray
        - jacobian_dx: [num_rays] âˆ‚distance/âˆ‚x
        - jacobian_dy: [num_rays] âˆ‚distance/âˆ‚y
        - hit_obstacle_ids: [num_rays] which obstacle was hit
    """
    # Generate all ray angles
    ray_angles = robot_theta + np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    ray_dirs = np.column_stack([np.cos(ray_angles), np.sin(ray_angles)])
    
    # Initialize outputs
    distances = np.full(num_rays, sensor_range, dtype=np.float64)
    jacobian_dx = np.zeros(num_rays, dtype=np.float64)
    jacobian_dy = np.zeros(num_rays, dtype=np.float64)
    hit_ids = np.full(num_rays, -1, dtype=np.int32)
    
    if len(obstacles) == 0:
        return {
            'distances': distances,
            'jacobian_dx': jacobian_dx,
            'jacobian_dy': jacobian_dy,
            'hit_obstacle_ids': hit_ids
        }
    
    # For each obstacle, compute ALL ray intersections (vectorized)
    for obs_idx, (ox, oy, r) in enumerate(obstacles):
        delta = robot_pos - np.array([ox, oy])
        
        # Ray-circle intersection: solve t^2 - 2*b*t + c = 0
        b = np.dot(ray_dirs, delta)  # [num_rays]
        c = np.dot(delta, delta) - r**2
        discriminant = b**2 - c
        
        # Valid intersections
        valid = (discriminant >= 0) & (b <= 0)
        
        if not np.any(valid):
            continue
        
        # Compute distances
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
        t = -b - sqrt_disc
        t = np.maximum(t, 0)
        
        # Update closest intersections
        closer = valid & (t < distances)
        
        distances[closer] = t[closer]
        hit_ids[closer] = obs_idx
        
        # Compute Jacobian for updated rays
        if np.any(closer):
            mask_idx = np.where(closer)[0]
            for idx in mask_idx:
                dx = ray_dirs[idx, 0]
                dy = ray_dirs[idx, 1]
                sqrt_d = sqrt_disc[idx]
                
                if sqrt_d > 1e-10:
                    jacobian_dx[idx] = -dx - (b[idx] * dx - delta[0]) / sqrt_d
                    jacobian_dy[idx] = -dy - (b[idx] * dy - delta[1]) / sqrt_d
    
    return {
        'distances': distances,
        'jacobian_dx': jacobian_dx,
        'jacobian_dy': jacobian_dy,
        'hit_obstacle_ids': hit_ids
    }


def cluster_points_dbscan(points: np.ndarray, epsilon: float, 
                          min_samples: int, min_cluster_size: int) -> List[np.ndarray]:
    """
    Cluster LiDAR points using DBSCAN.
    
    Uses GPU-accelerated cuML DBSCAN when available, falls back to sklearn.

    Args:
        points: Nx2 array of [x, y] coordinates
        epsilon: DBSCAN epsilon parameter
        min_samples: minimum samples for core point
        min_cluster_size: minimum cluster size to keep
    
    Returns:
        List of point arrays, one per cluster
    """
    if len(points) < min_samples:
        return []
    
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    clusters = []
    for label in set(labels):
        if label == -1:  # Noise
            continue
        cluster_points = points[labels == label]
        if len(cluster_points) >= min_cluster_size:
            clusters.append(cluster_points)

    return clusters



def fit_circle_least_squares(points: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit a circle to a cluster of points using Kåsa algebraic least squares method.
    
    Reference: I. Kåsa, "A circle fitting procedure and its error analysis," 
    IEEE Transactions on Instrumentation and Measurement, 1976.
    
    Solves the linear system:
        x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
    
    Which can be written as:
        x^2 + y^2 = A*x + B*y + C
    
    Where A = 2*cx, B = 2*cy, C = r^2 - cx^2 - cy^2
    
    Args:
        points: Nx2 array of [x, y] coordinates
    
    Returns:
        (cx, cy, r, fit_variance) or None if fit fails
    """
    if len(points) < 3:
        return None
    
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Build design matrix for least squares
    # We solve: x^2 + y^2 = A*x + B*y + C
    n = len(points)
    
    # Right-hand side
    b = x**2 + y**2
    
    # Design matrix [x, y, 1]
    A_matrix = np.column_stack([x, y, np.ones(n)])
    
    try:
        # Solve least squares: A_matrix @ [A, B, C]^T = b
        # Using normal equations: (A^T A)^-1 A^T b
        params, residuals, rank, s = np.linalg.lstsq(A_matrix, b, rcond=None)
        
        if rank < 3:
            # Singular matrix, points are collinear
            return None
        
        A, B, C = params
        
        # Extract circle parameters
        cx = A / 2.0
        cy = B / 2.0
        r = np.sqrt(C + cx**2 + cy**2)
        
        # Validate
        if not np.isfinite([cx, cy, r]).all():
            return None
        
        # Reject unrealistic radii - typical obstacles are 20-50 pixels
        if r < 10 or r > 60:
            return None
        
        # Compute fit quality (RMS error)
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        errors = distances - r
        fit_variance = np.mean(errors**2)
        rms_error = np.sqrt(fit_variance)
        
        # Reject poor fits
        if rms_error > r * 0.5:  # Error more than 50% of radius
            return None
        
        # Ensure fit_variance is at least 1.0 for numerical stability
        fit_variance = max(1.0, fit_variance)
        
        return cx, cy, r, fit_variance
        
    except np.linalg.LinAlgError:
        return None
