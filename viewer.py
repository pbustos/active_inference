"""
DearPyGui Viewer for Active Inference Navigation

Real-time visualization using Observer pattern.
Shows robot, obstacles, beliefs, path, and metrics without
interfering with navigation logic.

Active Inference Visualization:
- EFE components (G_goal, G_obs, G_amb, G_epi)
- Adaptive horizon display
- Trajectory candidates
- Pose and room estimation
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Dict, Any, List
from observer import Observer, NavigationEvent


class NavigationViewer(Observer):
    """
    Real-time visualization of navigation system.
    
    Uses Observer pattern to receive updates without coupling
    to the navigation logic.
    """
    
    def __init__(self, width: int = 800, height: int = 600, scale: float = 1.0):
        """
        Initialize viewer.
        
        Args:
            width: Room width in pixels
            height: Room height in pixels
            scale: Display scale factor
        """
        self.room_width = width
        self.room_height = height
        self.scale = scale
        
        # Display dimensions
        self.display_width = int(width * scale)
        self.display_height = int(height * scale)
        
        # State from navigation system
        self.robot_pos = np.array([100.0, 100.0])
        self.robot_theta = 0.0
        self.target = None
        self.true_obstacles = []  # Circles: (cx, cy, r)
        self.true_rectangles = []  # Rectangles: (cx, cy, w, h, angle)
        self.obstacle_beliefs = []
        self.provisional_beliefs = []
        self.tracked_obstacles = []  # TrackedObstacle instances with BMS
        self.global_path = []
        
        # Room estimation
        self.room_phase = "detection"
        self.room_estimate = None  # [width, height]
        self.pose_estimate = None
        self.pose_error = 0.0
        self.room_covariance = None  # Room parameter uncertainty

        # Active Inference state
        self.current_horizon = 15
        self.num_seeds = 16  # Number of trajectory seeds
        self.efe_smoothed = 0.0
        self.G_total = 0.0
        self.G_components = {'goal': 0, 'obstacle': 0, 'ambiguity': 0, 'epistemic': 0, 'structural': 0, 'localization': 0, 'velocity': 0}
        self.trajectory = []  # Current planned trajectory (best)
        self.all_trajectories = []  # All tested trajectories
        self.best_seed_idx = 0  # Index of best trajectory

        # LIDAR data
        self.lidar_points = []  # List of (x, y) points from LIDAR
        self.lidar_upper_points = []  # Upper LIDAR for room estimation

        # Room/Pose uncertainty
        self.pose_covariance = None  # 2x2 or 3x3 covariance matrix
        self.room_covariance = None  # Room parameter uncertainty

        # Current velocity
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]

        # Metrics
        self.steps = 0
        self.target_num = 0
        self.emergencies = 0
        self.replans = 0
        self.success_count = 0
        self.emergency_active = False
        self.recovery_active = False
        self.elapsed_ms = 0.0

        # UI state - start running immediately
        self.running = True
        self.show_trajectories = True
        self.show_beliefs = True
        self.show_lidar = False
        self.show_uncertainty = True
        self.show_historical_points = False

        # Initialize DearPyGui
        self._init_dpg()
    
    def _init_dpg(self):
        """Initialize DearPyGui context and windows"""
        dpg.create_context()
        
        # Layout dimensions
        right_panel_width = 280
        bottom_panel_height = 200
        total_width = self.display_width + right_panel_width + 40
        total_height = self.display_height + bottom_panel_height + 60
        
        # Create main window
        with dpg.window(label="Active Inference Navigation", 
                       tag="main_window",
                       width=total_width,
                       height=total_height,
                       no_resize=True,
                       no_move=True,
                       no_close=True):
            
            # Main horizontal layout: Left column (canvas + controls) + Right panel
            with dpg.group(horizontal=True):
                # Left column: Canvas on top, controls below
                with dpg.group():
                    # Canvas for visualization
                    with dpg.child_window(width=self.display_width + 20,
                                         height=self.display_height + 20):
                        with dpg.drawlist(width=self.display_width,
                                         height=self.display_height,
                                         tag="canvas"):
                            pass

                    # Bottom row: Controls, Options, Legend spread horizontally
                    with dpg.group(horizontal=True):
                        # Bottom-left: Controls
                        with dpg.child_window(width=220, height=bottom_panel_height):
                            self._create_control_panel()

                        # Bottom-center: Options
                        with dpg.child_window(width=220, height=bottom_panel_height):
                            self._create_options_panel()

                        # Bottom-center-right: Legend
                        with dpg.child_window(width=220, height=bottom_panel_height):
                            self._create_legend_panel()

                # Right panel: Metrics and info (full height)
                with dpg.child_window(width=right_panel_width,
                                     height=self.display_height + bottom_panel_height + 40):
                    self._create_metrics_panel()
                    self._create_efe_panel()
                    self._create_room_panel()

        # Setup and show
        dpg.create_viewport(title="Active Inference Navigation Viewer",
                           width=total_width + 20,
                           height=total_height + 20)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def _create_control_panel(self):
        """Create control buttons"""
        dpg.add_text("CONTROLS", color=(255, 255, 100))
        dpg.add_separator()
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="▶ Start", tag="btn_start", 
                          callback=lambda: self._on_start())
            dpg.add_button(label="⏸ Pause", tag="btn_pause",
                          callback=lambda: self._on_pause())
        
        dpg.add_separator()
    
    def _create_metrics_panel(self):
        """Create metrics display"""
        dpg.add_text("METRICS", color=(255, 255, 100))
        dpg.add_separator()
        
        dpg.add_text("Target: 0/0", tag="txt_target")
        dpg.add_text("Steps: 0", tag="txt_steps")
        dpg.add_text("Success: 0/0", tag="txt_success")
        dpg.add_text("Emergencies: 0", tag="txt_emergencies")
        dpg.add_text("Replans: 0", tag="txt_replans")
        dpg.add_text("Status: Waiting", tag="txt_status", color=(200, 200, 200))
        
        dpg.add_separator()
    
    def _create_room_panel(self):
        """Create room estimation display"""
        dpg.add_text("ROOM ESTIMATION", color=(255, 255, 100))
        dpg.add_separator()
        
        dpg.add_text("Phase: Detection", tag="txt_room_phase")

        # Room dimensions
        dpg.add_text("True Room: -- x --", tag="txt_room_true", color=(150, 150, 150))
        dpg.add_text("Est. Room: -- x --", tag="txt_room_est", color=(0, 255, 255))
        dpg.add_text("Room Error: --", tag="txt_room_error")

        dpg.add_separator()

        # Robot pose in room
        dpg.add_text("ROBOT POSE", color=(200, 200, 200))
        dpg.add_text("True: (---, ---)", tag="txt_pose_true", color=(150, 150, 150))
        dpg.add_text("Est:  (---, ---)", tag="txt_pose_est", color=(100, 200, 255))
        dpg.add_text("Pose Error: --", tag="txt_pose_error")
        
        # Pose covariance display (diagonal elements)
        dpg.add_text("Cov: σx=-- σy=-- σθ=--", tag="txt_pose_cov", color=(180, 180, 180))
        # Velocity display
        dpg.add_text("Vel: vx=-- vy=-- |v|=--", tag="txt_velocity", color=(180, 255, 180))

        dpg.add_separator()

    def _create_efe_panel(self):
        """Create Active Inference EFE diagnostic panel"""
        dpg.add_text("ACTIVE INFERENCE", color=(255, 200, 100))
        dpg.add_separator()

        # Horizon display with progress bar
        dpg.add_text("Horizon", tag="txt_horizon_label")
        dpg.add_progress_bar(default_value=0.2, tag="bar_horizon",
                            overlay="15", width=200)

        # Number of trajectory seeds
        dpg.add_text("Seeds: 16", tag="txt_num_seeds", color=(180, 180, 180))

        # EFE smoothed
        dpg.add_text("EFE (smoothed): 0.0", tag="txt_efe_smoothed")

        # EFE Components
        dpg.add_text("EFE Components:", color=(180, 180, 180))

        with dpg.group(horizontal=True):
            dpg.add_text("  G_goal:", color=(100, 255, 100))
            dpg.add_text("0.0", tag="txt_g_goal")

        with dpg.group(horizontal=True):
            dpg.add_text("  G_obs: ", color=(255, 100, 100))
            dpg.add_text("0.0", tag="txt_g_obs")

        with dpg.group(horizontal=True):
            dpg.add_text("  G_amb: ", color=(150, 150, 255))
            dpg.add_text("0.0", tag="txt_g_amb")

        with dpg.group(horizontal=True):
            dpg.add_text("  G_epi: ", color=(255, 200, 100))
            dpg.add_text("0.0", tag="txt_g_epi")

        with dpg.group(horizontal=True):
            dpg.add_text("  G_struct:", color=(200, 100, 255))
            dpg.add_text("0.0", tag="txt_g_struct")

        with dpg.group(horizontal=True):
            dpg.add_text("  G_loc: ", color=(100, 200, 200))
            dpg.add_text("0.0", tag="txt_g_loc")

        with dpg.group(horizontal=True):
            dpg.add_text("  G_vel: ", color=(180, 140, 100))
            dpg.add_text("0.0", tag="txt_g_vel")

        # Performance
        dpg.add_text("Time: 0.0 ms", tag="txt_elapsed_ms")

        dpg.add_separator()

    def _create_options_panel(self):
        """Create visualization options"""
        dpg.add_text("OPTIONS", color=(255, 255, 100))
        dpg.add_separator()

        dpg.add_checkbox(label="Show Trajectories", default_value=True,
                        tag="chk_trajectories",
                        callback=lambda s, d: setattr(self, 'show_trajectories', d))
        dpg.add_checkbox(label="Show Beliefs", default_value=True,
                        tag="chk_beliefs",
                        callback=lambda s, d: setattr(self, 'show_beliefs', d))
        dpg.add_checkbox(label="Show LIDAR", default_value=False,
                        tag="chk_lidar",
                        callback=lambda s, d: setattr(self, 'show_lidar', d))
        dpg.add_checkbox(label="Show Uncertainty", default_value=True,
                        tag="chk_uncertainty",
                        callback=lambda s, d: setattr(self, 'show_uncertainty', d))
        dpg.add_checkbox(label="Show Historical Pts", default_value=False,
                        tag="chk_historical",
                        callback=lambda s, d: setattr(self, 'show_historical_points', d))

        dpg.add_separator()
    
    def _create_legend_panel(self):
        """Create legend"""
        dpg.add_text("LEGEND", color=(255, 255, 100))
        dpg.add_separator()
        
        with dpg.group(horizontal=True):
            dpg.add_text("●", color=(50, 150, 255))
            dpg.add_text("Robot")
        
        with dpg.group(horizontal=True):
            dpg.add_text("★", color=(255, 255, 0))
            dpg.add_text("Target")
        
        with dpg.group(horizontal=True):
            dpg.add_text("●", color=(255, 100, 100))
            dpg.add_text("True Obstacles")
        
        with dpg.group(horizontal=True):
            dpg.add_text("●", color=(100, 255, 100))
            dpg.add_text("Beliefs (Stable)")
        
        with dpg.group(horizontal=True):
            dpg.add_text("●", color=(150, 150, 255))
            dpg.add_text("Beliefs (Provisional)")
        
        with dpg.group(horizontal=True):
            dpg.add_text("━", color=(255, 180, 50))
            dpg.add_text("Planned Trajectory")

        with dpg.group(horizontal=True):
            dpg.add_text("━", color=(100, 200, 255))
            dpg.add_text("Global Path")

        with dpg.group(horizontal=True):
            dpg.add_text("□", color=(0, 255, 255))
            dpg.add_text("Estimated Room")

        with dpg.group(horizontal=True):
            dpg.add_text("◠", color=(100, 255, 100))
            dpg.add_text("Horizon Arc")


    def _on_start(self):
        """Start continuous simulation"""
        self.running = True
        self.step_mode = False
        dpg.configure_item("txt_status", default_value="Running", color=(100, 255, 100))
    
    def _on_pause(self):
        """Pause simulation"""
        self.running = False
        dpg.configure_item("txt_status", default_value="Paused", color=(255, 200, 100))
    
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """
        Observer update method.
        
        Called by navigation system when state changes.
        """
        if event == NavigationEvent.STEP:
            self._handle_step(data)
        elif event == NavigationEvent.RESET:
            self._handle_reset(data)
        elif event == NavigationEvent.TARGET_REACHED:
            self._handle_target_reached(data)
        elif event == NavigationEvent.PERCEPTION_UPDATE:
            self._handle_perception(data)
        elif event == NavigationEvent.PLANNING_COMPLETE:
            self._handle_planning(data)
        elif event == NavigationEvent.EMERGENCY_START:
            self._handle_emergency_start(data)
        elif event == NavigationEvent.EMERGENCY_END:
            self._handle_emergency_end(data)
        elif event == NavigationEvent.RECOVERY_START:
            self._handle_recovery_start(data)
        elif event == NavigationEvent.RECOVERY_END:
            self._handle_recovery_end(data)
        elif event == NavigationEvent.ROOM_PHASE_CHANGE:
            self._handle_room_phase(data)
        
        # Redraw
        self._render()
    
    def _handle_step(self, data: Dict[str, Any]):
        """Handle step event"""
        self.robot_pos = data.get('robot_pos', self.robot_pos)
        self.robot_theta = data.get('robot_theta', self.robot_theta)
        self.steps = data.get('steps', self.steps)
        
        # Update metrics display
        dpg.configure_item("txt_steps", default_value=f"Steps: {self.steps}")
        
        # Update target and success counters
        if 'target_num' in data:
            self.target_num = data['target_num']
            total = data.get('total_targets', 0)
            if total > 0:
                dpg.configure_item("txt_target", default_value=f"Target: {self.target_num}/{total}")
            else:
                dpg.configure_item("txt_target", default_value=f"Target: {self.target_num}")
        
        if 'success_count' in data:
            self.success_count = data['success_count']
            total = data.get('total_targets', 0)
            if total > 0:
                dpg.configure_item("txt_success", default_value=f"Success: {self.success_count}/{total}")
            else:
                dpg.configure_item("txt_success", default_value=f"Success: {self.success_count}")
        
        # Update pose error if available
        if 'pose_error' in data:
            pose_error = data['pose_error']
            # Color code: green if <10px, yellow if <30px, red if >30px
            if pose_error < 10:
                color = (100, 255, 100)
            elif pose_error < 30:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            
            dpg.configure_item("txt_pose_error", 
                             default_value=f"Pose Error: {pose_error:.1f}px",
                             color=color)

        # Active Inference data
        if 'controller_info' in data:
            self._update_efe_display(data['controller_info'])

        # Trajectory data
        if 'trajectory' in data:
            self.trajectory = data['trajectory']
        if 'all_trajectories' in data:
            self.all_trajectories = data['all_trajectories']
            # Update number of seeds display
            self.num_seeds = len(self.all_trajectories)
            dpg.configure_item("txt_num_seeds", default_value=f"Seeds: {self.num_seeds}")
        if 'best_seed' in data:
            self.best_seed_idx = data['best_seed']

        # LIDAR data
        if 'lidar_points' in data:
            self.lidar_points = data['lidar_points']
        if 'lidar_upper_points' in data:
            self.lidar_upper_points = data['lidar_upper_points']

        # Pose covariance (uncertainty ellipse)
        if 'pose_covariance' in data:
            self.pose_covariance = data['pose_covariance']
            # Update sigma display
            if self.pose_covariance is not None:
                try:
                    sigma_x = np.sqrt(self.pose_covariance[0, 0])
                    sigma_y = np.sqrt(self.pose_covariance[1, 1])
                    # Try to get theta uncertainty if available (3x3 matrix)
                    if self.pose_covariance.shape[0] >= 3:
                        sigma_theta = np.sqrt(self.pose_covariance[2, 2])
                        dpg.configure_item("txt_pose_cov",
                                          default_value=f"Cov: σx={sigma_x:.1f} σy={sigma_y:.1f} σθ={sigma_theta:.2f}")
                    else:
                        dpg.configure_item("txt_pose_cov",
                                          default_value=f"Cov: σx={sigma_x:.1f} σy={sigma_y:.1f}")
                except:
                    pass

        # Velocity update
        if 'velocity' in data:
            self.velocity = np.array(data['velocity'])
            vx, vy = self.velocity[0], self.velocity[1]
            speed = np.linalg.norm(self.velocity)
            dpg.configure_item("txt_velocity",
                              default_value=f"Vel: vx={vx:.1f} vy={vy:.1f} |v|={speed:.1f}")

        # Room covariance - stored for potential future use
        if 'room_covariance' in data:
            self.room_covariance = data['room_covariance']

        # Obstacle beliefs with uncertainty
        if 'obstacle_beliefs' in data:
            self.obstacle_beliefs = data['obstacle_beliefs']

        # Tracked obstacles with BMS type inference
        if 'tracked_obstacles' in data:
            self.tracked_obstacles = data['tracked_obstacles']

        # Room estimation data
        if 'room_estimate' in data and data['room_estimate'] is not None:
            self.room_estimate = data['room_estimate']

            # Update true room display
            dpg.configure_item("txt_room_true",
                              default_value=f"True Room: {self.room_width} x {self.room_height}")

            # Update estimated room display
            dpg.configure_item("txt_room_est",
                              default_value=f"Est. Room: {self.room_estimate[0]:.0f} x {self.room_estimate[1]:.0f}")

            # Calculate error percentage
            width_err = abs(self.room_estimate[0] - self.room_width)
            height_err = abs(self.room_estimate[1] - self.room_height)
            err_pct = 100 * (width_err + height_err) / (self.room_width + self.room_height)

            # Color based on error
            if err_pct < 3:
                color = (100, 255, 100)
            elif err_pct < 10:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            dpg.configure_item("txt_room_error", default_value=f"Room Error: {err_pct:.1f}%", color=color)

        if 'room_phase' in data:
            self.room_phase = data['room_phase']
            phase_colors = {
                'detection': (255, 100, 100),
                'initialization': (255, 255, 100),
                'tracking': (100, 255, 100),
            }
            color = phase_colors.get(self.room_phase, (200, 200, 200))
            dpg.configure_item("txt_room_phase",
                              default_value=f"Phase: {self.room_phase.capitalize()}",
                              color=color)

        # Robot pose - true position
        if 'robot_pos' in data:
            pos = data['robot_pos']
            dpg.configure_item("txt_pose_true",
                              default_value=f"True: ({pos[0]:.0f}, {pos[1]:.0f})")

        # Robot pose - estimated position
        if 'estimated_pos' in data:
            est_pos = data['estimated_pos']
            dpg.configure_item("txt_pose_est",
                              default_value=f"Est:  ({est_pos[0]:.0f}, {est_pos[1]:.0f})")

        if 'pose_error' in data:
            self.pose_error = data['pose_error']
            if self.pose_error < 10:
                color = (100, 255, 100)
            elif self.pose_error < 30:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            dpg.configure_item("txt_pose_error",
                              default_value=f"Pose Error: {self.pose_error:.1f}px",
                              color=color)

    def _update_efe_display(self, info: Dict[str, Any]):
        """Update EFE diagnostic display"""
        # Horizon
        horizon = info.get('horizon', 15)
        self.current_horizon = horizon
        max_horizon = 80
        progress = (horizon - 8) / (max_horizon - 8)  # Normalize 8-80 to 0-1
        dpg.configure_item("bar_horizon", default_value=progress, overlay=f"{horizon}")

        # Color the horizon bar based on value
        # Low horizon = fast/green, High horizon = thorough/orange

        # EFE smoothed
        efe_smoothed = info.get('efe_smoothed', 0.0)
        self.efe_smoothed = efe_smoothed
        dpg.configure_item("txt_efe_smoothed",
                          default_value=f"EFE (smoothed): {efe_smoothed:.1f}")

        # G total
        g_total = info.get('G_total', 0.0)
        self.G_total = g_total

        # Elapsed time
        elapsed_ms = info.get('elapsed_ms', 0.0)
        self.elapsed_ms = elapsed_ms

        # Color code time: green if <100ms, yellow if <200ms, red if >200ms
        if elapsed_ms < 100:
            time_color = (100, 255, 100)
        elif elapsed_ms < 200:
            time_color = (255, 255, 100)
        else:
            time_color = (255, 100, 100)

        dpg.configure_item("txt_elapsed_ms",
                          default_value=f"Time: {elapsed_ms:.0f} ms",
                          color=time_color)

        # EFE components (if provided)
        if 'G_components' in info:
            comps = info['G_components']
            self.G_components = comps
            dpg.configure_item("txt_g_goal", default_value=f"{comps.get('goal', 0):.1f}")
            dpg.configure_item("txt_g_obs", default_value=f"{comps.get('obstacle', 0):.1f}")
            dpg.configure_item("txt_g_amb", default_value=f"{comps.get('ambiguity', 0):.1f}")
            dpg.configure_item("txt_g_epi", default_value=f"{comps.get('epistemic', 0):.1f}")
            dpg.configure_item("txt_g_struct", default_value=f"{comps.get('structural', 0):.1f}")
            dpg.configure_item("txt_g_loc", default_value=f"{comps.get('localization', 0):.1f}")
            dpg.configure_item("txt_g_vel", default_value=f"{comps.get('velocity', 0):.1f}")

    def _handle_reset(self, data: Dict[str, Any]):
        """Handle reset event"""
        self.robot_pos = data.get('robot_pos', np.array([100.0, 100.0]))
        self.robot_theta = data.get('robot_theta', 0.0)
        self.target = data.get('target', None)
        self.true_obstacles = data.get('obstacles', [])
        self.target_num = data.get('target_num', 0)
        self.steps = 0
        self.emergencies = data.get('emergencies_this_target', 0)
        self.replans = 0
        self.global_path = []
        self.emergency_active = False
        self.recovery_active = False
        
        dpg.configure_item("txt_target", 
                          default_value=f"Target: {self.target_num}/{data.get('total_targets', 0)}")
        dpg.configure_item("txt_steps", default_value="Steps: 0")
        dpg.configure_item("txt_emergencies", 
                          default_value=f"Emergencies: {self.emergencies}")
        dpg.configure_item("txt_replans", default_value="Replans: 0")
    
    def _handle_target_reached(self, data: Dict[str, Any]):
        """Handle target reached event"""
        self.success_count = data.get('success_count', self.success_count)
        total = data.get('total_targets', 0)
        
        dpg.configure_item("txt_success", 
                          default_value=f"Success: {self.success_count}/{total}")
        dpg.configure_item("txt_status", 
                          default_value="Target Reached!", 
                          color=(100, 255, 100))
    
    def _handle_perception(self, data: Dict[str, Any]):
        """Handle perception update"""
        self.obstacle_beliefs = data.get('beliefs', [])
        self.provisional_beliefs = data.get('provisional', [])
    
    def _handle_planning(self, data: Dict[str, Any]):
        """Handle planning complete"""
        self.global_path = data.get('path', [])
        self.replans = data.get('planning_count', self.replans)
        
        dpg.configure_item("txt_replans", default_value=f"Replans: {self.replans}")
    
    def _handle_emergency_start(self, data: Dict[str, Any]):
        """Handle emergency start"""
        self.emergency_active = True
        self.emergencies = data.get('emergencies', self.emergencies)
        
        dpg.configure_item("txt_emergencies", 
                          default_value=f"Emergencies: {self.emergencies}")
        dpg.configure_item("txt_status", 
                          default_value="EMERGENCY!", 
                          color=(255, 50, 50))
    
    def _handle_emergency_end(self, data: Dict[str, Any]):
        """Handle emergency end"""
        self.emergency_active = False
        
        if self.running:
            dpg.configure_item("txt_status", 
                              default_value="Running", 
                              color=(100, 255, 100))
    
    def _handle_recovery_start(self, data: Dict[str, Any]):
        """Handle recovery mode start"""
        self.recovery_active = True
        dpg.configure_item("txt_status", 
                          default_value="RECOVERY MODE", 
                          color=(255, 150, 50))
    
    def _handle_recovery_end(self, data: Dict[str, Any]):
        """Handle recovery mode end"""
        self.recovery_active = False
        
        if self.running and not self.emergency_active:
            dpg.configure_item("txt_status", 
                              default_value="Running", 
                              color=(100, 255, 100))
    
    def _handle_room_phase(self, data: Dict[str, Any]):
        """Handle room estimation phase change"""
        self.room_phase = data.get('phase', 'detection')
        self.room_estimate = data.get('room_params', None)
        self.pose_estimate = data.get('pose', None)
        self.pose_error = data.get('pose_error', 0.0)
        
        dpg.configure_item("txt_room_phase", 
                          default_value=f"Phase: {self.room_phase.capitalize()}")
        
        if self.room_estimate is not None:
            dpg.configure_item("txt_room_size",
                              default_value=f"Room: {self.room_estimate[0]:.0f} x {self.room_estimate[1]:.0f}")
            
            width_err = abs(self.room_estimate[0] - self.room_width)
            length_err = abs(self.room_estimate[1] - self.room_height)
            err_pct = 100 * np.mean([width_err/self.room_width, length_err/self.room_height])
            
            dpg.configure_item("txt_room_error",
                              default_value=f"Error: {err_pct:.1f}%")
        
        if self.pose_error > 0:
            dpg.configure_item("txt_pose_error",
                              default_value=f"Pose Error: {self.pose_error:.1f}px")
    
    def _render(self):
        """Render all graphics"""
        dpg.delete_item("canvas", children_only=True)
        
        with dpg.draw_node(parent="canvas"):
            # Background
            dpg.draw_rectangle((0, 0), 
                              (self.display_width, self.display_height),
                              fill=(20, 20, 30))
            
            # True room boundary (gray)
            dpg.draw_rectangle((0, 0),
                              (self.display_width, self.display_height),
                              color=(80, 80, 100),
                              thickness=2)
            
            # Estimated room contour (cyan, dashed effect with multiple rectangles)
            if self.room_estimate is not None and self.show_uncertainty:
                self._draw_estimated_room()

            # LIDAR points (if enabled)
            if self.show_lidar:
                self._draw_lidar_points()

            # True obstacles (red circles)
            for obs in self.true_obstacles:
                x, y, r = obs
                self._draw_circle(x, y, r, (255, 100, 100), fill=True)
            
            # True rectangles (red rectangles)
            for rect in self.true_rectangles:
                cx, cy, w, h, angle = rect
                self._draw_rectangle(cx, cy, w, h, angle, (255, 100, 100), fill=True)

            # Provisional beliefs (light blue, transparent)
            if self.show_beliefs:
                for belief in self.provisional_beliefs:
                    x, y = belief.position * self.scale
                    r = belief.radius * self.scale
                    dpg.draw_circle((x, y), r,
                                   color=(150, 150, 255, 100),
                                   fill=(150, 150, 255, 50),
                                   thickness=1)

                # Tracked obstacles with multi-hypothesis model selection
                for tracked in self.tracked_obstacles:
                    self._draw_tracked_obstacle_new(tracked)

            # Global path (blue line)
            if len(self.global_path) > 1:
                points = [(p[0] * self.scale, p[1] * self.scale) 
                         for p in self.global_path]
                dpg.draw_polyline(points,
                                 color=(100, 200, 255),
                                 thickness=2)
            
            # All tested trajectories (gray, thin) and best one (orange, thick)
            if self.show_trajectories:
                self._draw_all_trajectories()

            # Target (yellow star)
            if self.target is not None:
                tx, ty = self.target * self.scale
                self._draw_star(tx, ty, 15, (255, 255, 0))
            
            # Robot (blue circle with direction indicator)
            rx, ry = self.robot_pos * self.scale
            robot_r = 10 * self.scale
            
            # Robot body
            color = (255, 100, 100) if self.emergency_active else \
                   (255, 200, 100) if self.recovery_active else \
                   (50, 150, 255)
            
            dpg.draw_circle((rx, ry), robot_r,
                           color=color,
                           fill=color,
                           thickness=2)
            
            # Direction indicator
            end_x = rx + np.cos(self.robot_theta) * robot_r * 1.5
            end_y = ry + np.sin(self.robot_theta) * robot_r * 1.5
            dpg.draw_line((rx, ry), (end_x, end_y),
                         color=(255, 255, 255),
                         thickness=3)

            # Pose uncertainty ellipse
            if self.show_uncertainty and self.pose_covariance is not None:
                self._draw_uncertainty_ellipse(rx, ry, self.pose_covariance, (100, 200, 255, 100))

            # Horizon indicator (arc around robot)
            self._draw_horizon_indicator(rx, ry, robot_r)

    def _draw_trajectory(self, trajectory: np.ndarray, color: tuple, thickness: int):
        """Draw a trajectory as a polyline from robot position"""
        if len(trajectory) == 0:
            return

        # Build path from robot position + cumulative velocities
        points = [self.robot_pos.copy()]
        pos = self.robot_pos.copy()

        for vel in trajectory:
            pos = pos + vel
            points.append(pos.copy())

        # Scale and draw
        scaled_points = [(p[0] * self.scale, p[1] * self.scale) for p in points]
        if len(scaled_points) > 1:
            dpg.draw_polyline(scaled_points, color=color, thickness=thickness)

    def _draw_horizon_indicator(self, rx: float, ry: float, robot_r: float):
        """Draw horizon indicator around robot"""
        # Draw arc showing current horizon (more segments = longer horizon)
        if self.current_horizon > 8:
            # Normalize horizon to arc angle (15 = small arc, 80 = full circle)
            arc_fraction = (self.current_horizon - 8) / (80 - 8)
            arc_angle = arc_fraction * 2 * np.pi

            # Color: green for short, orange for long
            if self.current_horizon < 25:
                arc_color = (100, 255, 100, 150)
            elif self.current_horizon < 50:
                arc_color = (255, 255, 100, 150)
            else:
                arc_color = (255, 150, 50, 150)

            # Draw arc segments
            arc_r = robot_r * 2.5
            n_segments = int(arc_angle / (np.pi / 8)) + 1
            points = []
            for i in range(n_segments + 1):
                angle = self.robot_theta + (i / n_segments) * arc_angle - arc_angle / 2
                px = rx + arc_r * np.cos(angle)
                py = ry + arc_r * np.sin(angle)
                points.append((px, py))

            if len(points) > 1:
                dpg.draw_polyline(points, color=arc_color, thickness=3)

    def _draw_all_trajectories(self):
        """
        Draw all tested trajectories with the best one highlighted.

        - Non-selected trajectories: thin colored lines
        - Selected (best) trajectory: thick orange line
        """
        # Extended color palette for many trajectories
        colors = [
            (150, 150, 180),  # Gray-blue
            (100, 150, 200),  # Light blue
            (150, 100, 200),  # Purple
            (200, 150, 100),  # Tan
            (100, 200, 150),  # Teal
            (200, 100, 150),  # Pink
            (150, 200, 100),  # Lime
            (180, 180, 100),  # Olive
            (120, 180, 220),  # Sky blue
            (220, 120, 180),  # Magenta
            (180, 220, 120),  # Light green
            (220, 180, 120),  # Peach
            (120, 220, 180),  # Aqua
            (180, 120, 220),  # Violet
            (220, 220, 100),  # Yellow
            (100, 220, 220),  # Cyan
            (170, 130, 190),  # Lavender
            (190, 170, 130),  # Beige
            (130, 190, 170),  # Seafoam
            (190, 130, 170),  # Rose
            (140, 160, 200),  # Periwinkle
            (200, 160, 140),  # Salmon
            (160, 200, 140),  # Pistachio
            (200, 140, 160),  # Blush
            (140, 200, 160),  # Mint
            (160, 140, 200),  # Iris
            (180, 200, 160),  # Sage
            (200, 180, 160),  # Apricot
            (160, 180, 200),  # Cornflower
            (200, 160, 180),  # Coral
            (160, 200, 180),  # Seafoam light
            (180, 160, 200),  # Wisteria
        ]

        # Draw all non-best trajectories first (thin, colored)
        for i, traj in enumerate(self.all_trajectories):
            if i == self.best_seed_idx:
                continue  # Skip best, draw it last
            if len(traj) == 0:
                continue

            color = colors[i % len(colors)]
            # Use thickness 1 for thin trajectory lines
            self._draw_trajectory(traj, color, 1)

        # Draw best trajectory last (thick orange)
        if len(self.trajectory) > 0:
            self._draw_trajectory(self.trajectory, (255, 180, 50), 3)

    def _draw_circle(self, x: float, y: float, r: float, color: tuple, fill: bool = False):
        """Draw a circle (helper)"""
        x_scaled = x * self.scale
        y_scaled = y * self.scale
        r_scaled = r * self.scale
        
        if fill:
            dpg.draw_circle((x_scaled, y_scaled), r_scaled,
                           color=color, fill=color, thickness=1)
        else:
            dpg.draw_circle((x_scaled, y_scaled), r_scaled,
                           color=color, thickness=2)
    
    def _draw_rectangle(self, cx: float, cy: float, w: float, h: float,
                        angle: float, color: tuple, fill: bool = False):
        """Draw a rotated rectangle (helper)"""
        # Compute corners in local frame then rotate
        half_w, half_h = w / 2, h / 2
        local_corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        corners = []
        for lx, ly in local_corners:
            # Rotate and translate
            rx = lx * cos_a - ly * sin_a + cx
            ry = lx * sin_a + ly * cos_a + cy
            corners.append((rx * self.scale, ry * self.scale))

        if fill:
            dpg.draw_polygon(corners, color=color, fill=color)
        else:
            dpg.draw_polygon(corners, color=color, thickness=2)

    def _draw_star(self, x: float, y: float, size: float, color: tuple):
        """Draw a star (target indicator)"""
        # 5-pointed star
        points = []
        for i in range(10):
            angle = (i * np.pi / 5) - np.pi / 2
            r = size if i % 2 == 0 else size / 2.5
            px = x + r * np.cos(angle)
            py = y + r * np.sin(angle)
            points.append((px, py))
        
        dpg.draw_polygon(points, color=color, fill=color)
    
    def _draw_lidar_points(self):
        """Draw LIDAR points as small dots"""
        # Lower LIDAR (obstacle detection) - cyan
        for point in self.lidar_points:
            x, y = point[0] * self.scale, point[1] * self.scale
            dpg.draw_circle((x, y), 2, color=(0, 255, 255), fill=(0, 255, 255))

        # Upper LIDAR (room estimation) - magenta
        for point in self.lidar_upper_points:
            x, y = point[0] * self.scale, point[1] * self.scale
            dpg.draw_circle((x, y), 2, color=(255, 0, 255), fill=(255, 0, 255))

    def _draw_estimated_room(self):
        """
        Draw the estimated room contour as a solid cyan rectangle.

        The estimated room dimensions come from the room shape parameters
        (width, height) estimated by the room estimator.
        """
        if self.room_estimate is None:
            return

        est_w = self.room_estimate[0] * self.scale
        est_h = self.room_estimate[1] * self.scale

        # Draw estimated room as solid cyan rectangle
        dpg.draw_rectangle((0, 0), (est_w, est_h),
                          color=(0, 255, 255),
                          thickness=3)

        # Draw corner markers for visibility
        marker_size = 10
        corner_color = (0, 255, 255)

        # Top-right corner
        dpg.draw_line((est_w - marker_size, 0), (est_w, 0), color=corner_color, thickness=4)
        dpg.draw_line((est_w, 0), (est_w, marker_size), color=corner_color, thickness=4)

        # Bottom-left corner
        dpg.draw_line((0, est_h - marker_size), (0, est_h), color=corner_color, thickness=4)
        dpg.draw_line((0, est_h), (marker_size, est_h), color=corner_color, thickness=4)

        # Bottom-right corner
        dpg.draw_line((est_w - marker_size, est_h), (est_w, est_h), color=corner_color, thickness=4)
        dpg.draw_line((est_w, est_h - marker_size), (est_w, est_h), color=corner_color, thickness=4)

        # Label showing estimated dimensions
        label_x = est_w / 2 - 40
        label_y = est_h + 8
        if label_y < self.display_height - 20:
            dpg.draw_text((label_x, label_y),
                         f"Est: {self.room_estimate[0]:.0f} x {self.room_estimate[1]:.0f}",
                         color=(0, 255, 255), size=14)

    def _draw_uncertainty_ellipse(self, cx: float, cy: float, cov: np.ndarray, color: tuple):
        """
        Draw uncertainty ellipse from covariance matrix.

        Uses eigenvalue decomposition to get ellipse axes.
        """
        if cov is None or cov.shape[0] < 2:
            return

        try:
            # Extract 2x2 position covariance
            cov_2d = cov[:2, :2] if cov.shape[0] > 2 else cov

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)

            # Sort by eigenvalue (descending)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Ellipse parameters (2-sigma = 95% confidence)
            sigma_scale = 2.0
            width = sigma_scale * np.sqrt(eigenvalues[0]) * self.scale
            height = sigma_scale * np.sqrt(eigenvalues[1]) * self.scale
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

            # Draw ellipse as polygon approximation
            n_points = 32
            points = []
            for i in range(n_points):
                theta = 2 * np.pi * i / n_points
                # Point on unit circle
                px = width * np.cos(theta)
                py = height * np.sin(theta)
                # Rotate
                rx = px * np.cos(angle) - py * np.sin(angle)
                ry = px * np.sin(angle) + py * np.cos(angle)
                # Translate
                points.append((cx + rx, cy + ry))

            dpg.draw_polygon(points, color=color, fill=(color[0], color[1], color[2], 30))
        except:
            pass  # Skip if eigenvalue decomposition fails


    def _draw_tracked_obstacle_new(self, tracked):
        """
        Draw a TrackedObstacle with all its hypotheses.

        The TrackedObstacle contains multiple hypotheses (circle, rectangle, etc.)
        with posterior probabilities q(m). We draw each hypothesis with
        transparency proportional to its probability.
        """
        import numpy as np

        q_m = tracked.get_type_probs()

        # Draw each hypothesis with transparency based on probability
        for type_name, model in tracked.hypotheses.items():
            p = q_m.get(type_name, 0.0)
            if p < 0.01:  # Skip very unlikely hypotheses
                continue

            # Alpha based on probability
            alpha = int(p * 200) + 30

            if type_name == 'circle':
                cx, cy, r = model.get_params_tuple()
                x, y = cx * self.scale, cy * self.scale
                r_scaled = r * self.scale

                # Green for circle
                fill_color = (100, 255, 100, alpha // 2)
                edge_color = (100, 255, 100, alpha)

                dpg.draw_circle((x, y), r_scaled, color=edge_color, fill=fill_color, thickness=2)

                # Label above
                text_x = x - 30
                text_y = y - r_scaled - 25
                type_text = f"● {p:.0%}"
                text_color = (100, 255, 100, 255)

            elif type_name == 'rectangle':
                cx, cy, w, h, angle = model.get_params_tuple()
                x, y = cx * self.scale, cy * self.scale
                r_scaled = max(w, h) / 2 * self.scale

                # Blue for rectangle
                fill_color = (100, 150, 255, alpha // 2)
                edge_color = (100, 150, 255, alpha)

                self._draw_rectangle_filled(cx, cy, w, h, angle, edge_color, fill_color)

                # Label below
                text_x = x - 30
                text_y = y + r_scaled + 8
                type_text = f"■ {p:.0%}"
                text_color = (100, 180, 255, 255)
            else:
                continue  # Unknown type

            # Draw label with background
            if p > 0.1:  # Only show label for significant hypotheses
                dpg.draw_rectangle(
                    (text_x - 3, text_y - 2),
                    (text_x + 55, text_y + 16),
                    color=(0, 0, 0, 180),
                    fill=(0, 0, 0, 150),
                    rounding=3
                )
                dpg.draw_text((text_x, text_y), type_text, color=text_color, size=14)

            # Draw historical points if enabled
            if self.show_historical_points and hasattr(model, 'historical_points'):
                self._draw_model_historical_points(model, type_name)

    def _draw_model_historical_points(self, model, type_name: str):
        """
        Draw historical points accumulated by an obstacle model.

        Points are drawn as small dots with color based on model type.
        """
        if model.historical_points is None or len(model.historical_points) == 0:
            return

        # Color based on type
        if type_name == 'circle':
            point_color = (100, 255, 100, 200)  # Green for circle
        elif type_name == 'rectangle':
            point_color = (100, 150, 255, 200)  # Blue for rectangle
        else:
            point_color = (255, 255, 100, 200)  # Yellow for unknown

        # Get points (convert from torch if needed)
        hist_points = model.historical_points
        if hasattr(hist_points, 'cpu'):
            hist_points = hist_points.cpu().numpy()

        # Draw each point as a small dot
        for pt in hist_points:
            px, py = pt[0] * self.scale, pt[1] * self.scale
            dpg.draw_circle((px, py), 2, color=point_color, fill=point_color)


    def _draw_rectangle_filled(self, cx: float, cy: float, w: float, h: float,
                               angle: float, edge_color: tuple, fill_color: tuple):
        """Draw a filled rotated rectangle."""
        half_w, half_h = w / 2, h / 2
        local_corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        corners = []
        for lx, ly in local_corners:
            rx = lx * cos_a - ly * sin_a + cx
            ry = lx * sin_a + ly * cos_a + cy
            corners.append((rx * self.scale, ry * self.scale))

        dpg.draw_polygon(corners, color=edge_color, fill=fill_color, thickness=2)

    def render_frame(self):
        """Render one frame (call this in main loop)"""
        dpg.render_dearpygui_frame()
    
    def should_continue(self) -> bool:
        """Check if viewer should continue running"""
        return dpg.is_dearpygui_running()
    
    def cleanup(self):
        """Cleanup DearPyGui"""
        dpg.destroy_context()
    
    def is_running(self) -> bool:
        """Check if simulation is running"""
        return self.running
