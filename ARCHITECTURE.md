# Active Inference Navigation - Complete Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ACTIVE INFERENCE NAVIGATION                            │
│                                                                                  │
│  Paper: "Active Inference for Robot Navigation" (main.tex)                      │
│  Core Principle: Minimize Expected Free Energy G(π) for action selection        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
         ┌──────────────────┐                ┌──────────────────┐
         │    PERCEPTION    │                │      ACTION      │
         │  (VFE Minimization)│              │ (EFE Minimization)│
         └────────┬─────────┘                └────────┬─────────┘
                  │                                   │
     ┌────────────┴────────────┐                     │
     ▼                         ▼                     ▼
┌─────────┐              ┌─────────┐          ┌─────────────────┐
│ ROOM    │              │OBSTACLE │          │    UNIFIED      │
│ CONCEPT │              │ CONCEPT │          │   CONTROLLER    │
└─────────┘              └─────────┘          └─────────────────┘
```

---

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              test_viewer.py                                      │
│                           (Main Entry Point)                                     │
│                                                                                  │
│  - PyGame visualization                                                          │
│  - User interaction (goal setting, obstacles)                                   │
│  - Real-time display of beliefs, trajectories, EFE components                   │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          integrated_navigator.py                                 │
│                        (System Orchestrator)                                     │
│                                                                                  │
│  - Coordinates perception and action                                            │
│  - Manages belief lifecycle                                                      │
│  - Interfaces with simulated sensors                                            │
└───────────┬─────────────────────────────────────┬───────────────────────────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐               ┌─────────────────────────────────────────┐
│   PERCEPTION LAYER    │               │           ACTION LAYER                   │
│                       │               │                                          │
│  ┌─────────────────┐  │               │  ┌───────────────────────────────────┐  │
│  │  concept_room   │  │               │  │      unified_controller.py        │  │
│  │     .py         │  │               │  │                                   │  │
│  │                 │  │               │  │  Paper Section 5: EFE Control     │  │
│  │ RoomPoseEstimator│ │               │  │                                   │  │
│  │ - SDF_rect      │  │               │  │  G(π) = G_pragmatic              │  │
│  │ - VFE minimization│◄───────────────┼──┤       + G_ambiguity               │  │
│  │ - Laplace approx │ │  pose_cov     │  │       + G_epistemic               │  │
│  └─────────────────┘  │               │  │       + G_localization            │  │
│                       │               │  │                                   │  │
│  ┌─────────────────┐  │               │  │  - Multi-seed optimization       │  │
│  │concept_circle   │  │               │  │  - Adaptive horizon              │  │
│  │concept_rectangle│  │               │  │  - Trajectory commitment         │  │
│  │                 │  │               │  └───────────────────────────────────┘  │
│  │ CircleManager   │  │               │                   │                     │
│  │ RectangleManager│  │               │                   ▼                     │
│  │ - SDF_circle    │  │               │  ┌───────────────────────────────────┐  │
│  │ - SDF_rect      │◄─┼───────────────┼──┤      ray_circle_model.py          │  │
│  │ - DBSCAN cluster│  │  obstacles    │  │                                   │  │
│  │ - Lifecycle mgmt│  │               │  │  Paper Section 5.1.3:             │  │
│  └─────────────────┘  │               │  │  Analytic LIDAR prediction        │  │
│                       │               │  │  Analytic LIDAR prediction        │  │
└───────────────────────┘               │  │  ρ = -b - √Δ  (ray-circle)       │  │
                                        │  │  Soft-min aggregation             │  │
                                        │  └───────────────────────────────────┘  │
                                        └─────────────────────────────────────────┘
```

---

## Polymorphic Obstacle System (NEW)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            obstacle_base.py                                      │
│                     (Abstract Base Class + Utilities)                            │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ ObstacleBeliefBase│   │  CircleObstacle  │    │RectangleObstacle │
│     (ABC)        │    │                  │    │                  │
│                  │    │ concept_circle.py│    │concept_rectangle │
│ Abstract Methods:│    │                  │    │      .py         │
│ ├─ sdf()         │    │ State: [cx,cy,r] │    │                  │
│ ├─ predict_range()│   │                  │    │ State:           │
│ ├─ get_center()  │    │ SDF:             │    │ [cx,cy,w,h,θ]   │
│ ├─ get_params()  │    │ ‖p-c‖ - r       │    │                  │
│ └─ clone()       │    │                  │    │ SDF:             │
│                  │    │ Ray-Circle:      │    │ max(|x'|-w/2,    │
│ Common Methods:  │    │ ρ = -b - √Δ     │    │     |y'|-h/2)   │
│ ├─ is_stale()    │    │                  │    │                  │
│ ├─ add_history() │    │                  │    │ Ray-Slab:        │
│ └─ to_dict()     │    │                  │    │ 4-edge intersect │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                   │                        │
                                   └────────────┬───────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────┐
                              │     predict_all_ranges()        │
                              │                                 │
                              │  Polymorphic aggregation:       │
                              │  ρ̂_k = softmin_i(obs_i.range()) │
                              │                                 │
                              │  → Used by UnifiedController    │
                              └─────────────────────────────────┘
```

---

## Bayesian Model Selection (BMS) for Obstacle Types

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     OBSTACLE TYPE INFERENCE                                      │
│                                                                                  │
│  Given LIDAR observations o, infer obstacle type m ∈ {circle, rectangle, ...}  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STEP 1: Compute VFE for each model                          │
│                                                                                  │
│  F_circle = -ln p(o|s, circle) + D_KL[q(s|circle) || p(s|circle)]              │
│  F_rect   = -ln p(o|s, rect)   + D_KL[q(s|rect)   || p(s|rect)]                │
│                                                                                  │
│  Likelihood via SDF:  p(o|s,m) ∝ exp(-Σ SDF_m(p_k)² / 2σ²)                     │
│  KL regularization:   Penalizes model complexity (Occam's razor)                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STEP 2: Bayesian Model Selection                            │
│                                                                                  │
│  P(m|o) = P(o|m) · P(m) / Σ_m' P(o|m') · P(m')                                 │
│                                                                                  │
│  where: P(o|m) ≈ exp(-F_m)  (VFE approximates log model evidence)              │
│                                                                                  │
│  Implementation: Log-space for numerical stability                              │
│  log P(m|o) ∝ -F_m + log P(m)                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
         ┌──────────────────┐                ┌──────────────────┐
         │    MAP Decision  │                │   Soft Mixture   │
         │                  │                │                  │
         │ m* = argmax P(m|o)│               │ Use P(m|o) as    │
         │                  │                │ weights in EFE   │
         │ For perception   │                │                  │
         └──────────────────┘                │ G = Σ P(m|o)·G_m │
                                             │                  │
                                             │ For control      │
                                             └──────────────────┘
```

---

## Data Flow

```
┌─────────────┐     Upper LIDAR      ┌─────────────────┐
│   SENSORS   │─────(room walls)────▶│  concept_room   │
│             │                      │                 │
│  - Upper    │                      │  Output:        │
│    LIDAR    │     Lower LIDAR      │  - pose (x,y,θ) │
│  - Lower    │────(obstacles)──────▶│  - pose_cov [3×3]│
│    LIDAR    │                      │  - room (W,L)   │
└─────────────┘                      └────────┬────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
                    ▼                                                   ▼
          ┌─────────────────┐                               ┌───────────────────┐
          │concept_circle   │                               │unified_controller │
          │concept_rectangle│                               │                   │
          │                 │                               │ Input:            │
          │ Input:          │                               │ - robot_pos       │
          │ - LIDAR points  │                               │ - goal            │
          │ - pose from room│                               │ - obstacles       │
          │                 │                               │ - pose_covariance │
          │ Output:         │                               │                   │
          │ - List[Obstacle │──────obstacles────────────────▶                   │
          │   BeliefBase]   │                               │ Output:           │
          │ - μ, Σ per obs  │                               │ - action [vx,vy]  │
          └─────────────────┘                               │ - diagnostics     │
                                                            └─────────┬─────────┘
                                                                      │
                                                                      ▼
                                                            ┌───────────────────┐
                                                            │   Robot Motion    │
                                                            │                   │
                                                            │ x' = x + v·Δt    │
                                                            └───────────────────┘
```

---

## Expected Free Energy Decomposition

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              G(π) = Σ G_τ                                        │
│                                                                                  │
│              Paper Eq. 26: G(π) = G_pragmatic + G_epistemic + G_ambiguity       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│   G_PRAGMATIC     │      │    G_AMBIGUITY    │      │   G_EPISTEMIC     │
│                   │      │                   │      │                   │
│ "Goal achievement │      │ "Motion causes    │      │ "Information      │
│  + Safety"        │      │  uncertainty"     │      │  seeking"         │
│                   │      │                   │      │                   │
│ G_goal:           │      │ Paper Eq. 105-107:│      │ Mutual info:      │
│ λ_g·‖pos-goal‖   │      │                   │      │                   │
│                   │      │ Σ_{t+1} = Σ_t +  │      │ I(S;O) =          │
│ G_obs:            │      │   β‖u‖²I         │      │  ½ln|Σ_o| -       │
│ Via SDF or LIDAR  │      │                   │      │  ½ln|Σ_meas|     │
│ prediction error  │      │ Σ_o = Σ_meas +   │      │                   │
│                   │      │   J·Σ_{t+1}·J^T  │      │ G_epi = -λ·I(S;O) │
│ G_safety:         │      │                   │      │                   │
│ exp(margin-dist)  │      │ G_amb = ½ln|Σ_o| │      │ (negative = seek  │
│                   │      │                   │      │  information)     │
└───────────────────┘      └───────────────────┘      └───────────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │    + G_LOCALIZATION   │
                          │                       │
                          │  λ_loc·trace(Σ_pose)  │
                          │                       │
                          │  (Practical extension │
                          │   to reduce pose      │
                          │   uncertainty)        │
                          └───────────────────────┘
```

---

## Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-SEED TRAJECTORY OPTIMIZATION                       │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1: Initialize Seeds
┌─────────────────────────────────────────────────────────────────────────────────┐
│  seed_0: toward goal                                                             │
│  seed_1: goal + 30° rotation                                                     │
│  seed_2: goal - 30° rotation                                                     │
│  seed_3: goal + 60° rotation                                                     │
│  ...                                                                             │
│  seed_k: warm start (previous best trajectory)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
Step 2: Parallel Optimization (GPU Batched)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  for each seed in parallel:                                                     │
│      trajectory = initialize_trajectory(seed, horizon=T)                        │
│                                                                                  │
│      for iter in range(num_iterations):                                         │
│          G_T = compute_trajectory_G(trajectory)    ◄── Forward pass            │
│          ∇G_T = autograd(G_T, trajectory)          ◄── Backward pass           │
│          trajectory = trajectory - α·∇G_T          ◄── Gradient descent        │
│                                                                                  │
│      G_final[seed] = G_T                                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
Step 3: Select Best
┌─────────────────────────────────────────────────────────────────────────────────┐
│  best_seed = argmin(G_final)                                                    │
│  best_trajectory = trajectories[best_seed]                                      │
│  action = best_trajectory[0]  ◄── Execute first action (receding horizon)      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
Step 4: Adaptive Horizon
┌─────────────────────────────────────────────────────────────────────────────────┐
│  if stuck_detected():                                                           │
│      horizon = min(horizon + 5, max_horizon)                                    │
│      maybe_commit_trajectory()   ◄── Execute multiple steps without replanning │
│  else:                                                                          │
│      horizon = max(horizon - 1, base_horizon)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Summary

```
active_inference/
│
├── ENTRY POINT
│   └── test_viewer.py          # Main program with PyGame visualization
│
├── ORCHESTRATION  
│   └── integrated_navigator.py # Coordinates perception + action
│
├── PERCEPTION (VFE Minimization)
│   ├── concept_room.py         # Room + Pose estimation (Paper §4.1)
│   ├── concept_circle.py        # Circle obstacle tracking (Paper §4.2)
│   ├── concept_rectangle.py     # Rectangle obstacle tracking (Paper §4.2)
│   └── perception.py           # LIDAR processing, clustering
│
├── OBSTACLE ABSTRACTION (Polymorphic)
│   ├── obstacle_base.py        # Abstract base class
│   ├── concept_circle.py       # Circular obstacles
│   ├── concept_rectangle.py    # Rectangular obstacles
│   └── obstacle_type_selector.py # Bayesian Model Selection (BMS)
│
├── ACTION (EFE Minimization)
│   ├── unified_controller.py   # Main controller (Paper §5)
│   └── ray_circle_model.py     # Analytic LIDAR model (Paper §5.1.3)
│
├── UTILITIES
│   ├── belief.py               # Gaussian belief representation
│   ├── variational_inference.py # VFE computation helpers
│   └── viewer.py               # Visualization components
│
├── TESTS
│   ├── test_unified_controller.py
│   ├── test_belief_evolution.py
│   └── test_*.py               # Various test suites
│
└── DOCUMENTATION
    ├── main.tex                # Paper source
    ├── PAPER_ALIGNMENT.md      # Code-paper mapping
    └── ARCHITECTURE.md         # This file
```

---

## Key Equations Reference

| Paper Eq. | Name | Implementation |
|-----------|------|----------------|
| VFE | F[q,o] = D_KL[q‖p] - E[ln p(o\|s)] | `concept_room.py`, `concept_circle.py`, `concept_rectangle.py` |
| EFE | G(π) = G_prag + G_amb + G_epi | `unified_controller._compute_step_G()` |
| Eq. 105 | Σ_{t+1} = Σ_t + β‖u‖²I | Velocity-dependent covariance |
| Eq. 106 | Σ_o = Σ_meas + J·Σ·J^T | Observation covariance |
| Eq. 107 | G_amb = ½ln\|Σ_o\| | Log-determinant |
| ray_circle | ρ = -b - √Δ | `ray_circle_model.py`, `concept_circle.py` |
| softmin | ρ̂ = -1/η log Σexp(-ηρ) | `obstacle_base.py` |
| SDF_circle | ‖p-c‖ - r | `concept_circle.py` |
| SDF_rect | max(\|x'\|-w/2, \|y'\|-h/2) | `concept_rectangle.py` |
| VFE_obs | F_m = -ln p(o\|s,m) + D_KL[q\|\|p] | `concept_circle.compute_vfe()`, `concept_rectangle.compute_vfe()` |
| BMS | P(m\|o) ∝ exp(-F_m) · P(m) | `obstacle_type_selector.py` |

---

## Adding New Obstacle Types

To add a new obstacle type (e.g., Ellipse, Polygon):

1. Create `concept_<type>.py` inheriting from `ObstacleBeliefBase`
2. Implement required abstract methods:
   - `sdf(points)` - Signed distance function
   - `predict_range(origin, dirs)` - Ray intersection
   - `compute_vfe(points, pose)` - Variational Free Energy for BMS
   - `get_params_tuple()` - Parameter extraction
   - `get_center()` - Center position
   - `get_bounding_radius()` - For broad-phase collision
   - `clone()` - Deep copy
3. Use in `integrated_navigator.py` or pass directly to `UnifiedController.compute_action()`
4. Gradients are computed automatically via PyTorch autograd
5. Use `ObstacleTypeSelector` for Bayesian Model Selection between types

