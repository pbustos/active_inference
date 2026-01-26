# Distributed Active Inference Architecture - Migration Guide

**Date:** 2025-01-25  
**Status:** Ready for implementation  
**Branch:** distributed (to be created from current clean state)

---

## 1. Executive Summary

This document describes the migration from a monolithic Active Inference navigation system to a distributed architecture where Concept-X classes (CircleFitter, RectFitter, DoorFitter, etc.) run as autonomous processes communicating through a shared Working Memory (WM).

**Key Insight:** The Working Memory is structured as a **Scene Graph** where:
- **Nodes** = Concept instances (Robot, Room, Obstacles, Doors, etc.)
- **Edges** = Relations (geometric transforms, constraints, visibility)

This graph structure maps directly to GTSAM factor graphs for joint optimization.
Conceptually, the Working Memory represents a factorised approximation to a joint posterior 
q(S), where each SceneNode corresponds to a marginal belief and each SceneEdge encodes a probabilistic relation between beliefs.

### Key Goals:
1. **Separation of concerns**: Each Concept-X process only knows its own type
2. **Extensibility**: Adding new obstacle types = new process + register
3. **Constraint support**: Enable structural constraints (e.g., DOOR on WALL) via GTSAM
4. **Pub/Sub communication**: Avoid polling, use event-driven updates. Events from WM are delivered in order per node/edge, but not globally ordered.
5. **Scene Graph structure**: Native graph representation for GTSAM translation

---

## 2. Current Architecture (Monolithic)

```
integrated_navigator.py (orchestrator)
├── unified_controller.py      # Planning + control via EFE minimization
├── concept_room.py            # Room/pose estimation (SDF-based)
├── obstacle_base.py           # TrackedObstacle with model selection
│   ├── concept_circle.py      # Circle belief + fitting
│   └── concept_rectangle.py   # Rectangle belief + fitting
├── perception.py              # DBSCAN clustering
├── lidar_gpu.py              # GPU LIDAR simulation
└── viewer.py                  # DearPyGui visualization
```

### Current Model Selection (in TrackedObstacle):
```python
class TrackedObstacle:
    hypotheses: Dict[str, ObstacleBeliefBase]  # {circle: CircleObstacle, rect: RectObstacle}
    q_m: Dict[str, float]                       # Posterior over model types
    
    def _compute_model_posterior(self):
        # q(m) ∝ p(m) · exp(-VFE_m)
        # Compares VFEs of ALL types to compute posterior
```

**Problem:** Model selection requires access to all hypotheses simultaneously, creating tight coupling.

---

## 3. Target Architecture (Distributed)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAW SENSOR DATA                                   │
│                    (LIDAR points - accessed directly)                       │
└─────────────────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ CircleFitter  │       │  RectFitter   │       │  DoorFitter   │
│   Process     │       │   Process     │       │   Process     │
│               │       │               │       │               │
│ - Reads raw   │       │ - Reads raw   │       │ - Reads raw   │
│   sensors     │       │   sensors     │       │   sensors     │
│ - Fits model  │       │ - Fits model  │       │ - Fits model  │
│ - Computes VFE│       │ - Computes VFE│       │ - Detects     │
│ - CRUD on WM  │       │ - CRUD on WM  │       │   constraints │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        │    PUBLISH            │    PUBLISH            │    PUBLISH
        │    SUBSCRIBE          │    SUBSCRIBE          │    SUBSCRIBE
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WORKING MEMORY                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      BELIEFS (elements)                             │   │
│  │                                                                     │   │
│  │  belief_001: {                                                      │   │
│  │      owner: 'circle',              ← Concept-X that created it     │   │
│  │      state: STABLE,                ← lifecycle state               │   │
│  │      hypothesis: {type, VFE, params, covariance},                   │   │
│  │      competing: [{type:'rect', VFE:...}, ...],  ← other Concept-X  │   │
│  │      constraints: [ON_WALL(wall_id='north')]                        │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SUBSCRIPTION SYSTEM                              │   │
│  │                                                                     │   │
│  │  Events: belief_created, belief_updated, belief_competing,          │   │
│  │          belief_stable, belief_removed                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ BeliefIntegrator  │  │  GTSAM Process  │  │   Controller    │
│                   │  │                 │  │                 │
│ SUBSCRIBE:        │  │ SUBSCRIBE:      │  │ SUBSCRIBE:      │
│ - belief_updated  │  │ - belief_stable │  │ - belief_stable │
│                   │  │                 │  │                 │
│ - Computes q(m)   │  │ - Factor graph  │  │ - Navigation    │
│ - Marks STABLE    │  │ - Optimizes     │  │ - EFE control   │
└───────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 4. Belief Lifecycle States

```
    ┌─────────┐
    │   NEW   │◀────── Concept-X creates belief
    └────┬────┘
         │ Other Concept-X publish competing hypotheses
         ▼
  ┌──────────────┐
  │  HYPOTHESES  │◀──── Multiple Concept-X have published
  │   PENDING    │
  └──────┬───────┘
         │ BeliefIntegrator calculates q(m)
         ▼
  ┌──────────────┐
  │   EVALUATED  │◀──── q(m) computed, best_type identified
  │              │
  └──────┬───────┘
         │ q(m) > 0.75 for 5 consecutive frames
         ▼
    ┌─────────┐
    │ STABLE  │◀────── Committed, ready for GTSAM/Controller
    └────┬────┘
         │ Not seen for 30 frames
         ▼
    ┌─────────┐
    │ STALE   │◀────── Owner should remove
    └─────────┘
```

---

## 5. Working Memory as Scene Graph

The Working Memory is structured as a **Scene Graph** - a directed graph where:
- **Nodes** = Concept instances (beliefs about entities in the world)
- **Edges** = Relations between nodes (geometric transforms, constraints)

This structure directly maps to GTSAM factor graphs for joint optimization.

### 5.1 Scene Graph Structure

```
                            ┌─────────────┐
                            │    WORLD    │  (root node)
                            │   origin    │
                            └──────┬──────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             ┌───────────┐  ┌───────────┐  ┌───────────┐
             │   ROOM    │  │   ROBOT   │  │  AMBIENT  │
             │  W,L,pose │  │  x,y,θ,Σ  │  │  (future) │
             └─────┬─────┘  └─────┬─────┘  └───────────┘
                   │              │
        ┌──────────┼──────────┐   │
        │          │          │   │
        ▼          ▼          ▼   │
   ┌────────┐ ┌────────┐ ┌────────┐
   │ WALL_N │ │ WALL_S │ │ WALL_E │ ...
   │        │ │        │ │        │
   └────┬───┘ └────────┘ └────────┘
        │
        │ ON_WALL constraint
        ▼
   ┌─────────┐
   │  DOOR   │
   │ cx,cy,w │
   └─────────┘

   ┌─────────────────────────────────────────┐
   │           OBSTACLES (in ROOM)           │
   │                                         │
   │  ┌────────┐  ┌────────┐  ┌────────┐    │
   │  │CIRCLE_1│  │ RECT_1 │  │CIRCLE_2│    │
   │  │cx,cy,r │  │cx,cy,  │  │cx,cy,r │    │
   │  │        │  │w,h,θ   │  │        │    │
   │  └────────┘  └────────┘  └────────┘    │
   └─────────────────────────────────────────┘
```

### 5.2 Node Types

```python
class NodeType(Enum):
    """Types of nodes in the scene graph."""
    WORLD = "world"           # Root node, global origin
    ROOM = "room"             # Room geometry (W, L) and pose
    ROBOT = "robot"           # Robot pose (x, y, θ) and covariance
    WALL = "wall"             # Wall segment (part of room)
    OBSTACLE = "obstacle"     # Generic obstacle (circle, rect, etc.)
    DOOR = "door"             # Door (constrained to wall)
    LANDMARK = "landmark"     # Fixed reference point
    # Future extensions:
    # FURNITURE = "furniture"
    # PERSON = "person"
```

### 5.3 SceneNode (Base class for all nodes)

```python
@dataclass
class SceneNode:
    """A node in the scene graph representing a concept instance."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # IDENTITY
    # ─────────────────────────────────────────────────────────────────────────
    node_id: str                         # Unique ID (e.g., "obstacle_001")
    node_type: NodeType                  # Type of node
    
    # ─────────────────────────────────────────────────────────────────────────
    # OWNERSHIP & LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────
    owner: str                           # Concept-X process that created it
    state: NodeState                     # NEW, PENDING, EVALUATED, STABLE, STALE
    created_at: float
    last_updated: float
    last_seen_frame: int
    
    # ─────────────────────────────────────────────────────────────────────────
    # BELIEF STATE (Type-specific parameters as tensors for GTSAM)
    # ─────────────────────────────────────────────────────────────────────────
    params: Dict[str, float]             # e.g., {'cx': 100, 'cy': 200, 'r': 30}
    covariance: np.ndarray               # Uncertainty over params
    
    Covariances are propagated locally by the owning Concept-X process and may be globally refined by the GTSAM process.
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODEL SELECTION (for polymorphic nodes like obstacles)
    # ─────────────────────────────────────────────────────────────────────────
    hypothesis: Optional[Hypothesis]     # Current best hypothesis
    competing_hypotheses: Dict[str, Hypothesis]  # From other Concept-X
    q_m: Optional[Dict[str, float]]      # Posterior over model types
    committed_type: Optional[str]        # Committed model type (or None)
    
    # ─────────────────────────────────────────────────────────────────────────
    # FREE ENERGY
    # ─────────────────────────────────────────────────────────────────────────
    vfe: float = 0.0                     # Variational Free Energy of this node
    
    # ─────────────────────────────────────────────────────────────────────────
    # GTSAM KEY (for factor graph mapping)
    # ─────────────────────────────────────────────────────────────────────────
    gtsam_key: Optional[int] = None      # Key in GTSAM factor graph
```

### 5.4 Edge Types

```python
class EdgeType(Enum):
    """Types of edges (relations) in the scene graph."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # GEOMETRIC RELATIONS (transforms)
    # ─────────────────────────────────────────────────────────────────────────
    TRANSFORM = "transform"       # SE2/SE3 transform between frames
    CONTAINS = "contains"         # Parent contains child (room contains obstacles)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRUCTURAL CONSTRAINTS
    # ─────────────────────────────────────────────────────────────────────────
    ON_WALL = "on_wall"           # Object is on a wall (door, window)
    NEAR = "near"                 # Proximity constraint
    ALIGNED = "aligned"           # Alignment constraint
    INSIDE = "inside"             # Object is inside another
    
    # ─────────────────────────────────────────────────────────────────────────
    # PERCEPTUAL RELATIONS
    # ─────────────────────────────────────────────────────────────────────────
    VISIBLE_FROM = "visible_from"  # Node A is visible from node B
    OCCLUDES = "occludes"          # Node A occludes node B
    
    # ─────────────────────────────────────────────────────────────────────────
    # SEMANTIC RELATIONS (future)
    # ─────────────────────────────────────────────────────────────────────────
    # SUPPORTS = "supports"        # Table supports objects
    # PART_OF = "part_of"          # Handle is part of door
```

### 5.5 SceneEdge

```python
@dataclass
class SceneEdge:
    """An edge in the scene graph representing a relation between nodes."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # IDENTITY
    # ─────────────────────────────────────────────────────────────────────────
    edge_id: str                         # Unique ID
    edge_type: EdgeType
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENDPOINTS
    # ─────────────────────────────────────────────────────────────────────────
    from_node: str                       # Source node ID
    to_node: str                         # Target node ID
    
    # ─────────────────────────────────────────────────────────────────────────
    # RELATION DATA
    # ─────────────────────────────────────────────────────────────────────────
    params: Dict[str, Any]               # Relation-specific parameters
    # For TRANSFORM: {'dx': 10, 'dy': 20, 'dtheta': 0.1}
    # For ON_WALL: {'wall_segment': 'north', 'offset': 50}
    # For NEAR: {'max_distance': 30}
    
    covariance: Optional[np.ndarray]     # Uncertainty in relation
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRAINT STRENGTH
    # ─────────────────────────────────────────────────────────────────────────
    is_hard: bool = False                # Hard constraint vs soft
    noise_model: Optional[str] = None    # GTSAM noise model type
    sigma: float = 1.0                   # Noise standard deviation
    
    # ─────────────────────────────────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────────────────────────────────
    created_at: float
    created_by: str                      # Process that created this edge
```

### 5.6 SceneGraph (Working Memory)

```python
class SceneGraph:
    """
    Working Memory structured as a Scene Graph.
    
    Provides:
    - Node CRUD operations
    - Edge CRUD operations  
    - Pub/Sub for changes
    - Translation to GTSAM factor graph
    """
    
    def __init__(self):
        # ─────────────────────────────────────────────────────────────────────
        # GRAPH STORAGE
        # ─────────────────────────────────────────────────────────────────────
        self.nodes: Dict[str, SceneNode] = {}
        self.edges: Dict[str, SceneEdge] = {}
        
        # Adjacency lists for fast traversal
        self.outgoing: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        self.incoming: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        
        # ─────────────────────────────────────────────────────────────────────
        # INDICES (for fast queries)
        # ─────────────────────────────────────────────────────────────────────
        self.nodes_by_type: Dict[NodeType, Set[str]] = {}
        self.nodes_by_owner: Dict[str, Set[str]] = {}
        self.nodes_by_state: Dict[NodeState, Set[str]] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # PUB/SUB
        # ─────────────────────────────────────────────────────────────────────
        self.subscribers: Dict[str, List[Callable]] = {
            'node_created': [],
            'node_updated': [],
            'node_stable': [],
            'node_removed': [],
            'edge_created': [],
            'edge_removed': [],
            'graph_optimized': [],  # After GTSAM optimization
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # NODE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_node(self, node: SceneNode) -> str:
        """Add a node to the graph."""
        ...
    
    def update_node(self, node_id: str, updates: Dict, by: str):
        """Update node parameters."""
        ...
    
    def remove_node(self, node_id: str, by: str):
        """Remove node and all connected edges."""
        ...
    
    def get_node(self, node_id: str) -> SceneNode:
        """Get node by ID."""
        ...
    
    # ─────────────────────────────────────────────────────────────────────────
    # EDGE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_edge(self, edge: SceneEdge) -> str:
        """Add an edge (relation) between nodes."""
        ...
    
    def remove_edge(self, edge_id: str):
        """Remove an edge."""
        ...
    
    def get_edges_from(self, node_id: str) -> List[SceneEdge]:
        """Get all outgoing edges from a node."""
        ...
    
    def get_edges_to(self, node_id: str) -> List[SceneEdge]:
        """Get all incoming edges to a node."""
        ...
    
    # ─────────────────────────────────────────────────────────────────────────
    # QUERIES
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[SceneNode]:
        """Get all nodes of a specific type."""
        ...
    
    def get_stable_nodes(self) -> List[SceneNode]:
        """Get all nodes in STABLE state."""
        ...
    
    def get_children(self, node_id: str) -> List[SceneNode]:
        """Get child nodes (via CONTAINS edges)."""
        ...
    
    def find_nodes_near(self, position: np.ndarray, radius: float) -> List[SceneNode]:
        """Spatial query for nearby nodes."""
        ...
    
    # ─────────────────────────────────────────────────────────────────────────
    # GTSAM TRANSLATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_gtsam_graph(self) -> 'gtsam.NonlinearFactorGraph':
        """
        Translate scene graph to GTSAM factor graph.
        
        - Each STABLE node becomes a variable
        - Each edge becomes a factor
        - Returns factor graph + initial values
        """
        ...
    
    def update_from_gtsam(self, result: 'gtsam.Values'):
        """
        Update node parameters from GTSAM optimization result.
        
        - Updates params and covariance
        - Emits 'graph_optimized' event
        """
        ...
```

### 5.7 GTSAM Factor Mapping

| Edge Type | GTSAM Factor |
|-----------|--------------|
| TRANSFORM | BetweenFactorPose2 |
| ON_WALL | CustomOnWallFactor (point on line) |
| NEAR | RangeFactor or BoundedFactor |
| ALIGNED | OrientationFactor |
| INSIDE | CustomInsideFactor |
| CONTAINS | (implicit, for hierarchy only) |

```python
def edge_to_gtsam_factor(edge: SceneEdge, 
                          from_node: SceneNode, 
                          to_node: SceneNode) -> gtsam.NonlinearFactor:
    """Convert a SceneEdge to the corresponding GTSAM factor."""
    
    if edge.edge_type == EdgeType.TRANSFORM:
        return gtsam.BetweenFactorPose2(
            from_node.gtsam_key,
            to_node.gtsam_key,
            gtsam.Pose2(edge.params['dx'], edge.params['dy'], edge.params['dtheta']),
            gtsam.noiseModel.Diagonal.Sigmas([edge.sigma] * 3)
        )
    
    elif edge.edge_type == EdgeType.ON_WALL:
        # Custom factor: door position must be on wall line
        return OnWallFactor(
            to_node.gtsam_key,      # door
            from_node.gtsam_key,    # wall
            edge.params['offset'],
            gtsam.noiseModel.Isotropic.Sigma(1, edge.sigma)
        )
    
    elif edge.edge_type == EdgeType.NEAR:
        return gtsam.RangeFactor(
            from_node.gtsam_key,
            to_node.gtsam_key,
            edge.params['max_distance'],
            gtsam.noiseModel.Isotropic.Sigma(1, edge.sigma)
        )
    
    # ... more factor types
```

---

## 6. Hypothesis (for Model Selection)

For polymorphic nodes (obstacles that could be circle, rect, or door), we still need hypotheses:

```python
@dataclass
class Hypothesis:
    """A hypothesis about a node's type and parameters."""
    type_name: str                 # 'circle', 'rect', 'door'
    params: Dict[str, float]       # Type-specific parameters
    covariance: np.ndarray         # Parameter uncertainty
    vfe: float                     # Variational Free Energy
    timestamp: float
    owner: str                     # Which Concept-X proposed this
```

The node stores:
- `hypothesis`: Current best (from owner)
- `competing_hypotheses`: Alternatives from other Concept-X
- `q_m`: Posterior over types (computed by BeliefIntegrator)
- `committed_type`: Final decision (when q(m) > threshold)

---

## 7. Process Responsibilities

### 7.1 Concept-X Fitter Process (CircleFitter, RectFitter, DoorFitter)

| Action | Description |
|--------|-------------|
| **Read** | Raw sensor data (LIDAR clusters) |
| **Create** | New nodes when detecting unmatched clusters |
| **Update** | Own nodes with new observations |
| **Compete** | Add competing hypotheses to other nodes |
| **Remove** | Own nodes when stale |
| **Detect** | Constraints and create edges (DoorFitter detects ON_WALL) |

### 7.2 BeliefIntegrator Process

| Action | Description |
|--------|-------------|
| **Subscribe** | node_created, node_updated |
| **Compute** | q(m) = softmax(-VFE) for all hypotheses |
| **Mark** | STABLE when q(m) > threshold for N frames |
| **Transfer** | Ownership if different type dominates |

### 7.3 GTSAM Process

| Action | Description |
|--------|-------------|
| **Subscribe** | node_stable, edge_created |
| **Build** | Factor graph from SceneGraph.to_gtsam_graph() |
| **Optimize** | Minimize joint Free Energy |
| **Update** | SceneGraph.update_from_gtsam(result) |

### 7.4 Controller Process

| Action | Description |
|--------|-------------|
| **Subscribe** | node_stable, graph_optimized |
| **Read** | Stable nodes for obstacle avoidance |
| **Compute** | EFE minimization for action selection |

---


## 8. Launcher Protocol

### Startup Sequence:
```
1. Main Process starts
2. Start Working Memory (as Manager or Process)
3. Wait for WM ready
4. Start worker processes with Barrier synchronization:
   - CircleFitterProcess
   - RectFitterProcess  
   - BeliefIntegratorProcess
   - ControllerProcess
5. Workers subscribe to WM events
6. All workers signal ready via Barrier
7. System running
```

### Shutdown Sequence:
```
1. Ctrl-C / SIGTERM received
2. Set shutdown_event
3. WM notifies all subscribers
4. Workers cleanup and exit
5. Join all processes (with timeout)
6. Stop WM
7. Exit Main
```

---

## 9. File Structure

```
active_inference/
├── distributed/                    # NEW: Distributed package
│   ├── __init__.py
│   ├── launcher.py                 # System startup/shutdown
│   ├── working_memory.py           # WM with pub/sub
│   ├── base_process.py             # Base class for workers
│   ├── protocols.py                # Dataclasses (Hypothesis, WMBelief, Constraint)
│   └── processes/
│       ├── __init__.py
│       ├── circle_fitter.py        # Uses concept_circle.py logic
│       ├── rect_fitter.py          # Uses concept_rectangle.py logic
│       ├── belief_integrator.py    # Model selection + GTSAM prep
│       └── controller.py           # Uses unified_controller.py logic
│
├── concept_circle.py               # KEEP: Circle fitting logic
├── concept_rectangle.py            # KEEP: Rectangle fitting logic
├── concept_room.py                 # KEEP: Room estimation
├── unified_controller.py           # KEEP: EFE control logic
├── obstacle_base.py                # REFACTOR: Extract TrackedObstacle logic
├── viewer.py                       # KEEP: Visualization
├── perception.py                   # KEEP: Clustering
├── lidar_gpu.py                   # KEEP: GPU LIDAR
├── observer.py                     # KEEP: Observer pattern
│
└── run_distributed.py              # NEW: Entry point
```

---

## 10. Migration Steps

### Phase 1: Infrastructure (Start here)
1. Create `distributed/` package structure
2. Implement `protocols.py` with dataclasses
3. Implement `working_memory.py` with mp.Manager
4. Implement `base_process.py` with lifecycle
5. Implement `launcher.py` with startup/shutdown

### Phase 2: First Process
6. Implement `circle_fitter.py` using `concept_circle.py` logic
7. Test with WM (create/update beliefs)

### Phase 3: Model Selection
8. Implement `belief_integrator.py`
9. Extract `_compute_model_posterior` from TrackedObstacle
10. Test model selection with CircleFitter

### Phase 4: Additional Processes
11. Implement `rect_fitter.py`
12. Implement `controller.py`
13. Full integration test

### Phase 5: GTSAM Integration
14. Add GTSAM process for constraint optimization
15. Implement DoorFitter with ON_WALL constraint
16. Test constraint propagation

---

## 11. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| WM implementation | mp.Manager (start), Process (optimize later) | Simpler to implement |
| Sensor access | Direct by Fitters | Keep simple, no extra process |
| Model selection | Centralized in BeliefIntegrator | Correct q(m) calculation |
| Ownership | First Concept-X to create | Clear responsibility |
| Ownership transfer | By BeliefIntegrator when q(m) shifts | Handles type changes |
| GTSAM | Separate process | Decouples optimization |

---

## 12. Current Code State

### Cleaned and Ready:
- `unified_controller.py`: 2,428 lines, no dead code
- `obstacle_base.py`: 1,291 lines, TrackedObstacle ready to refactor
- `concept_circle.py`: 350 lines, clean
- `concept_rectangle.py`: 500 lines, clean
- `integrated_navigator.py`: 1,031 lines, orchestrator to be replaced

### Key Functions to Extract:
- `TrackedObstacle._compute_model_posterior()` → BeliefIntegrator
- `TrackedObstacle._check_commitment()` → BeliefIntegrator
- `CircleManager._create_from_points()` → CircleFitterProcess
- `RectangleManager._create_from_points()` → RectFitterProcess

---

## 13. Testing Strategy

### Unit Tests:
- WM CRUD operations
- Pub/sub event delivery
- Hypothesis/WMBelief serialization

### Integration Tests:
- Single Fitter + WM
- Two Fitters competing
- BeliefIntegrator model selection
- Full pipeline with Controller

### System Tests:
- Startup/shutdown sequence
- Ctrl-C handling
- Worker crash recovery
- Performance benchmarks

---

## 14. Commands to Start

```bash
# Create new branch
git checkout -b distributed

# Run distributed system (after implementation)
python run_distributed.py

# Run with test mode
python run_distributed.py --test

# Run single test
python -m pytest distributed/tests/test_working_memory.py
```

---

## 15. References

- Current `obstacle_base.py`: TrackedObstacle implementation
- Current `concept_circle.py`: CircleManager and CircleObstacle
- Current `unified_controller.py`: EFE computation and optimization
- Paper: "Using Active Inference for perception, planning and control (AIPPC)"

---

**End of Migration Guide**

