"""
Observer Pattern for Navigation System

Provides a clean way to monitor navigation state without coupling
visualization to the core logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class Observer(ABC):
    """Abstract observer interface"""
    
    @abstractmethod
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """
        Called when observable state changes.
        
        Args:
            event: Type of event (e.g., 'step', 'plan', 'emergency')
            data: Event-specific data
        """
        pass


class Observable:
    """
    Mixin class to make objects observable.
    
    Usage:
        class MyClass(Observable):
            def __init__(self):
                super().__init__()
                self._init_observable()
            
            def do_something(self):
                self._notify_observers('action', {'value': 42})
    """
    
    def _init_observable(self):
        """Initialize observer list"""
        self._observers: List[Observer] = []
    
    def attach_observer(self, observer: Observer) -> None:
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach_observer(self, observer: Observer) -> None:
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, event: str, data: Dict[str, Any]) -> None:
        """Notify all observers of an event"""
        for observer in self._observers:
            try:
                observer.update(event, data)
            except Exception as e:
                print(f"Observer notification error: {e}")


class NavigationEvent:
    """Event types for navigation system"""
    
    # Simulation events
    STEP = "step"
    RESET = "reset"
    TARGET_REACHED = "target_reached"
    
    # Perception events
    PERCEPTION_UPDATE = "perception_update"
    BELIEF_CREATED = "belief_created"
    BELIEF_UPDATED = "belief_updated"
    BELIEF_REMOVED = "belief_removed"
    
    # Planning events
    PLANNING_START = "planning_start"
    PLANNING_COMPLETE = "planning_complete"
    PATH_INVALID = "path_invalid"
    
    # Control events
    EMERGENCY_START = "emergency_start"
    EMERGENCY_END = "emergency_end"
    STALL_DETECTED = "stall_detected"
    RECOVERY_START = "recovery_start"
    RECOVERY_END = "recovery_end"
    
    # Room estimation events
    ROOM_DETECTION = "room_detection"
    ROOM_PHASE_CHANGE = "room_phase_change"
    ROOM_INITIALIZED = "room_initialized"
