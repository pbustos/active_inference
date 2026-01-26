#!/usr/bin/env python3
"""
Test script to measure timing between emergencies and target changes
"""

import numpy as np
import time
import sys

from integrated_navigator import IntegratedNavigator, IntegratedNavConfig

class TimingTest:
    """Test wrapper to measure commitment timing"""

    def __init__(self, scenario: str = "super_dense"):
        # Create navigator
        self.nav = IntegratedNavigator(IntegratedNavConfig(
            room_width=800,
            room_height=600,
            use_estimated_pose=False,
            use_estimated_obstacles=False
        ))

        # Setup scenario
        self._setup_scenario(scenario)

        # Timing tracking
        self.last_commit_time = None
        self.last_commit_reason = None
        self.commit_durations = []
        self.backup_count = 0
        self.step_count = 0
        self.last_goal_commit_time = 0.0

    def _setup_scenario(self, scenario: str):
        """Setup super_dense navigation scenario"""
        if scenario == "super_dense":
            start = [50, 300]
            goal = [750, 300]
            obstacles = [
                # Left region
                (150, 150, 30),
                (150, 350, 35),
                (150, 500, 30),
                # Center-left
                (250, 100, 35),
                (250, 300, 40),
                (250, 450, 30),
                # Center
                (350, 200, 35),
                (350, 400, 35),
                # Center-right
                (500, 150, 30),
                (500, 300, 40),
                (500, 500, 35),
                # Right region
                (650, 100, 30),
                (650, 250, 35),
                (650, 400, 30),
                (650, 550, 35),
            ]
        else:
            start = [100, 100]
            goal = [600, 400]
            obstacles = []

        self.nav.set_obstacles(obstacles)
        self.nav.reset()
        self.nav.robot_pos = np.array(start)
        self.nav.set_goal(np.array(goal))

    def run_test(self, max_steps: int = 500):
        """Run test and collect timing data"""
        print(f"Running timing test for super_dense scenario...")
        print(f"Target: maintain commitment for at least 5 seconds")
        print(f"=" * 70)

        for step in range(max_steps):
            self.step_count = step

            # Execute navigation step
            step_info = self.nav.step(dt=1.0)

            # Get control info
            control_info = step_info.get('control', {})

            # Always track goal commit time
            goal_commit_time = control_info.get('goal_commit_time', 0.0)

            # Show progress every 10 steps with goal commitment time
            if step % 10 == 0:
                print(f"[Step {step:3d}] Goal commit time: {goal_commit_time:.2f}s", end='')
                if control_info.get('committed', False):
                    print(f" | Trajectory committed: {control_info.get('commit_reason', 'unknown')}", end='')
                if control_info.get('backup_mode', False):
                    print(f" | 🔄 BACKUP", end='')
                print()

            # Check if committed
            if control_info.get('committed', False):
                commit_reason = control_info.get('commit_reason', 'unknown')

                # Track new commitment
                if self.last_commit_reason != commit_reason or self.last_commit_time is None:
                    current_time = time.time()

                    # Calculate duration of previous commitment
                    if self.last_commit_time is not None:
                        duration = current_time - self.last_commit_time
                        self.commit_durations.append(duration)
                        print(f"\n[Step {step}] Commitment ended after {duration:.2f}s")
                        print(f"  Previous reason: {self.last_commit_reason}")
                        print(f"  New reason: {commit_reason}")

                        if duration < 5.0:
                            print(f"  ⚠️  WARNING: Commitment too short! (< 5s)")

                    self.last_commit_time = current_time
                    self.last_commit_reason = commit_reason
                    print(f"\n[Step {step}] New commitment started: {commit_reason}")
                    print(f"  Goal commit time: {goal_commit_time:.2f}s")

                # Check for backup mode
                if control_info.get('backup_mode', False):
                    self.backup_count += 1
                    print(f"  🔄 Backup mode activated (total: {self.backup_count})")

            else:
                # Not committed anymore
                if self.last_commit_time is not None:
                    duration = time.time() - self.last_commit_time
                    self.commit_durations.append(duration)
                    print(f"\n[Step {step}] Commitment ended after {duration:.2f}s")
                    print(f"  Reason: {self.last_commit_reason}")

                    if duration < 5.0:
                        print(f"  ⚠️  WARNING: Commitment too short! (< 5s)")

                    self.last_commit_time = None
                    self.last_commit_reason = None

            # Test goal change blocking - try to change goal at step 25 and 100
            if step == 25:
                print(f"\n🔄 [TEST] Attempting to change goal at step {step} (commit time: {goal_commit_time:.2f}s)")
                print(f"  Current goal: {self.nav.goal}")
                print(f"  Trying to set new goal: [400.0, 400.0]")
                old_goal = self.nav.goal.copy()
                self.nav.set_goal(np.array([400.0, 400.0]))  # Try to change goal
                print(f"  Goal after set_goal: {self.nav.goal}")
                # Goal should be forced back in next compute_action call

            if step == 100:
                print(f"\n🔄 [TEST] Attempting to change goal at step {step} (commit time: {goal_commit_time:.2f}s)")
                print(f"  Current goal: {self.nav.goal}")
                print(f"  Trying to set new goal: [750.0, 250.0]")
                old_goal = self.nav.goal.copy()
                self.nav.set_goal(np.array([750.0, 250.0]))  # Try to change goal
                print(f"  Goal after set_goal: {self.nav.goal}")

            # Detect goal changes by checking timer resets
            if step > 0 and goal_commit_time < 1.0 and self.last_goal_commit_time > 4.0:
                print(f"\n⚠️  [DETECTED] Goal changed at step {step}! Timer reset from {self.last_goal_commit_time:.2f}s to {goal_commit_time:.2f}s")
                print(f"  New goal: {self.nav.goal}")

            self.last_goal_commit_time = goal_commit_time

            # Check if reached goal
            if np.linalg.norm(self.nav.robot_pos - self.nav.goal) < 30:
                print(f"\n✅ Goal reached at step {step}!")
                break

            # Progress indicator every 50 steps
            if step % 50 == 0 and step > 0:
                print(f"\n--- Progress: {step} steps ---")
                if control_info.get('goal_commit_time'):
                    print(f"    Current goal commit time: {control_info['goal_commit_time']:.2f}s")

        # Final statistics
        print(f"\n" + "=" * 70)
        print(f"TIMING TEST RESULTS")
        print(f"=" * 70)
        print(f"Total steps: {self.step_count + 1}")
        print(f"Total commitments: {len(self.commit_durations)}")
        print(f"Backup maneuvers: {self.backup_count}")

        if self.commit_durations:
            print(f"\nCommitment durations:")
            print(f"  Min: {min(self.commit_durations):.2f}s")
            print(f"  Max: {max(self.commit_durations):.2f}s")
            print(f"  Avg: {np.mean(self.commit_durations):.2f}s")
            print(f"  Median: {np.median(self.commit_durations):.2f}s")

            # Count violations
            violations = sum(1 for d in self.commit_durations if d < 5.0)
            print(f"\n⚠️  Commitments < 5s: {violations}/{len(self.commit_durations)}")

            # Show all durations
            print(f"\nAll durations (seconds):")
            for i, duration in enumerate(self.commit_durations, 1):
                status = "❌" if duration < 5.0 else "✅"
                print(f"  {status} Commitment {i}: {duration:.2f}s")
        else:
            print("\nNo commitments detected")


def main():
    """Main entry point"""
    test = TimingTest(scenario="super_dense")
    test.run_test(max_steps=500)


if __name__ == "__main__":
    main()
