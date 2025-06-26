"""
Safety Checking and Collision Avoidance for Robot Control.

This module provides safety mechanisms to ensure that planned actions
are safe to execute, including collision avoidance, joint limit checking,
and emergency stopping capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from opencontrol.cli.commands import OpenControlConfig


class SafetyChecker:
    """
    Comprehensive safety checker for robot actions and trajectories.
    
    This class implements multiple safety checks including:
    - Joint limit violations
    - Velocity and acceleration limits
    - Collision detection
    - Workspace boundaries
    - Emergency stop conditions
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config.control
        self.model_config = config.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Safety parameters
        self.action_bounds = torch.tensor(self.config.action_bounds, device=self.device)
        self.max_velocity = getattr(self.config, 'max_velocity', 1.0)
        self.max_acceleration = getattr(self.config, 'max_acceleration', 2.0)
        
        # Emergency action (typically all zeros - stop)
        self.emergency_action = torch.zeros(self.model_config.action_dim, device=self.device)
        
        # Safety violation tracking
        self.violation_history = []
        
        # Initialize collision checker
        self.collision_checker = CollisionAvoidance(config)
        
    async def check_action_sequence(
        self,
        action_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Check if an action sequence is safe to execute.
        
        Args:
            action_sequence: [horizon, action_dim] tensor of actions
            observation: Current observation dictionary
            
        Returns:
            Dictionary containing safety information
        """
        violations = []
        is_safe = True
        
        # 1. Check action bounds
        action_violations = self._check_action_bounds(action_sequence)
        if action_violations:
            violations.extend(action_violations)
            is_safe = False
        
        # 2. Check velocity limits
        velocity_violations = self._check_velocity_limits(action_sequence)
        if velocity_violations:
            violations.extend(velocity_violations)
            is_safe = False
        
        # 3. Check acceleration limits
        accel_violations = self._check_acceleration_limits(action_sequence)
        if accel_violations:
            violations.extend(accel_violations)
            is_safe = False
        
        # 4. Check for collisions
        collision_violations = await self.collision_checker.check_trajectory(
            action_sequence, observation
        )
        if collision_violations:
            violations.extend(collision_violations)
            is_safe = False
        
        # 5. Check workspace boundaries
        workspace_violations = self._check_workspace_bounds(action_sequence, observation)
        if workspace_violations:
            violations.extend(workspace_violations)
            is_safe = False
        
        # Store violation history
        self.violation_history.append({
            'timestamp': torch.tensor(time.time()),
            'violations': violations,
            'is_safe': is_safe
        })
        
        # Keep only recent history
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-1000:]
        
        return {
            'is_safe': is_safe,
            'violations': violations,
            'severity': self._compute_severity(violations),
            'recommended_action': 'emergency_stop' if not is_safe else 'continue'
        }
    
    def _check_action_bounds(self, action_sequence: torch.Tensor) -> List[str]:
        """Check if actions are within valid bounds."""
        violations = []
        
        min_bound, max_bound = self.action_bounds[0], self.action_bounds[1]
        
        # Check if any action exceeds bounds
        if torch.any(action_sequence < min_bound) or torch.any(action_sequence > max_bound):
            violations.append("action_bounds_violation")
        
        return violations
    
    def _check_velocity_limits(self, action_sequence: torch.Tensor) -> List[str]:
        """Check if action changes exceed velocity limits."""
        violations = []
        
        if action_sequence.shape[0] > 1:
            # Compute action differences (proxy for velocity)
            action_diffs = action_sequence[1:] - action_sequence[:-1]
            max_diff = torch.max(torch.abs(action_diffs))
            
            if max_diff > self.max_velocity:
                violations.append("velocity_limit_violation")
        
        return violations
    
    def _check_acceleration_limits(self, action_sequence: torch.Tensor) -> List[str]:
        """Check if action accelerations exceed limits."""
        violations = []
        
        if action_sequence.shape[0] > 2:
            # Compute second differences (proxy for acceleration)
            action_diffs = action_sequence[1:] - action_sequence[:-1]
            action_accel = action_diffs[1:] - action_diffs[:-1]
            max_accel = torch.max(torch.abs(action_accel))
            
            if max_accel > self.max_acceleration:
                violations.append("acceleration_limit_violation")
        
        return violations
    
    def _check_workspace_bounds(
        self,
        action_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Check if actions would move robot outside workspace."""
        violations = []
        
        # This is a simplified check - in practice, you'd use forward kinematics
        if 'proprioception' in observation:
            current_pos = observation['proprioception'][:3]  # Assume first 3 dims are position
            
            # Simple workspace bounds (cube)
            workspace_bounds = torch.tensor([[-2, -2, 0], [2, 2, 2]], device=self.device)
            
            # Check if current position is within bounds
            if torch.any(current_pos < workspace_bounds[0]) or torch.any(current_pos > workspace_bounds[1]):
                violations.append("workspace_bounds_violation")
        
        return violations
    
    def _compute_severity(self, violations: List[str]) -> str:
        """Compute severity level of violations."""
        if not violations:
            return "none"
        
        critical_violations = ["collision_detected", "workspace_bounds_violation"]
        warning_violations = ["velocity_limit_violation", "acceleration_limit_violation"]
        
        if any(v in critical_violations for v in violations):
            return "critical"
        elif any(v in warning_violations for v in violations):
            return "warning"
        else:
            return "minor"
    
    async def get_safe_action_sequence(
        self,
        unsafe_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate a safe alternative to an unsafe action sequence.
        
        Args:
            unsafe_sequence: The original unsafe action sequence
            observation: Current observation
            
        Returns:
            Safe action sequence
        """
        # Strategy 1: Clip to bounds
        safe_sequence = torch.clamp(
            unsafe_sequence,
            self.action_bounds[0],
            self.action_bounds[1]
        )
        
        # Strategy 2: Smooth velocity changes
        if safe_sequence.shape[0] > 1:
            for i in range(1, safe_sequence.shape[0]):
                action_diff = safe_sequence[i] - safe_sequence[i-1]
                if torch.norm(action_diff) > self.max_velocity:
                    # Scale down the action difference
                    scale = self.max_velocity / torch.norm(action_diff)
                    safe_sequence[i] = safe_sequence[i-1] + action_diff * scale
        
        # Strategy 3: If still unsafe, use emergency action
        safety_check = await self.check_action_sequence(safe_sequence, observation)
        if not safety_check['is_safe']:
            # Return sequence of emergency actions
            safe_sequence = self.emergency_action.unsqueeze(0).repeat(
                unsafe_sequence.shape[0], 1
            )
        
        return safe_sequence
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update safety configuration."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get statistics about safety violations."""
        if not self.violation_history:
            return {'no_data': True}
        
        total_checks = len(self.violation_history)
        unsafe_checks = sum(1 for h in self.violation_history if not h['is_safe'])
        
        violation_types = {}
        for history in self.violation_history:
            for violation in history['violations']:
                violation_types[violation] = violation_types.get(violation, 0) + 1
        
        return {
            'total_checks': total_checks,
            'unsafe_rate': unsafe_checks / total_checks,
            'violation_types': violation_types,
            'recent_violations': self.violation_history[-10:]
        }


class CollisionAvoidance:
    """
    Collision detection and avoidance system.
    
    This class handles collision checking between the robot and its environment,
    including self-collision detection and obstacle avoidance.
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config.control
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Collision parameters
        self.safety_margin = getattr(self.config, 'safety_margin', 0.1)  # meters
        self.check_self_collision = getattr(self.config, 'check_self_collision', True)
        
        # Simple obstacle representation (spheres)
        self.static_obstacles = self._initialize_static_obstacles()
        
    def _initialize_static_obstacles(self) -> List[Dict]:
        """Initialize static obstacles in the environment."""
        # Example: table, walls, etc.
        obstacles = [
            {'center': torch.tensor([1.0, 0.0, 0.5], device=self.device), 'radius': 0.3},
            {'center': torch.tensor([0.0, 1.5, 0.8], device=self.device), 'radius': 0.2},
        ]
        return obstacles
    
    async def check_trajectory(
        self,
        action_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> List[str]:
        """
        Check trajectory for collisions.
        
        Args:
            action_sequence: Planned action sequence
            observation: Current observation
            
        Returns:
            List of collision violations
        """
        violations = []
        
        # 1. Check static obstacle collisions
        static_collisions = self._check_static_obstacles(action_sequence, observation)
        violations.extend(static_collisions)
        
        # 2. Check self-collisions
        if self.check_self_collision:
            self_collisions = self._check_self_collision(action_sequence, observation)
            violations.extend(self_collisions)
        
        # 3. Check dynamic obstacles (if available)
        dynamic_collisions = await self._check_dynamic_obstacles(action_sequence, observation)
        violations.extend(dynamic_collisions)
        
        return violations
    
    def _check_static_obstacles(
        self,
        action_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Check for collisions with static obstacles."""
        violations = []
        
        if 'proprioception' in observation:
            # Simplified: assume first 3 dimensions are end-effector position
            current_pos = observation['proprioception'][:3]
            
            # Check current position against all obstacles
            for obstacle in self.static_obstacles:
                distance = torch.norm(current_pos - obstacle['center'])
                if distance < (obstacle['radius'] + self.safety_margin):
                    violations.append("static_collision_detected")
                    break
        
        return violations
    
    def _check_self_collision(
        self,
        action_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Check for self-collisions between robot links."""
        violations = []
        
        # Simplified self-collision check
        # In practice, this would use detailed robot geometry
        if 'proprioception' in observation:
            joint_positions = observation['proprioception']
            
            # Simple heuristic: check if any joint angles are extreme
            extreme_threshold = 2.5  # radians
            if torch.any(torch.abs(joint_positions) > extreme_threshold):
                violations.append("self_collision_risk")
        
        return violations
    
    async def _check_dynamic_obstacles(
        self,
        action_sequence: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Check for collisions with dynamic obstacles (humans, other robots)."""
        violations = []
        
        # This would typically use perception data to detect dynamic obstacles
        # For now, we'll implement a placeholder
        
        if 'video' in observation:
            # In practice, you would run object detection on the video
            # to identify dynamic obstacles and predict their motion
            pass
        
        return violations
    
    def add_obstacle(self, center: torch.Tensor, radius: float):
        """Add a new static obstacle."""
        obstacle = {'center': center.to(self.device), 'radius': radius}
        self.static_obstacles.append(obstacle)
    
    def remove_obstacle(self, index: int):
        """Remove a static obstacle by index."""
        if 0 <= index < len(self.static_obstacles):
            del self.static_obstacles[index]
    
    def update_obstacle(self, index: int, center: torch.Tensor, radius: float):
        """Update an existing obstacle."""
        if 0 <= index < len(self.static_obstacles):
            self.static_obstacles[index]['center'] = center.to(self.device)
            self.static_obstacles[index]['radius'] = radius


class EmergencyStop:
    """
    Emergency stop system for immediate robot shutdown.
    
    This class provides mechanisms for safely stopping the robot
    in emergency situations.
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config.control
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_emergency_active = False
        self.emergency_triggers = []
        
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop with reason."""
        self.is_emergency_active = True
        self.emergency_triggers.append({
            'timestamp': time.time(),
            'reason': reason
        })
        
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        self.is_emergency_active = False
        
    def get_emergency_action(self, action_dim: int) -> torch.Tensor:
        """Get emergency action (typically all zeros)."""
        return torch.zeros(action_dim, device=self.device)
    
    def check_emergency_conditions(
        self,
        observation: Dict[str, torch.Tensor],
        system_state: Dict[str, Any]
    ) -> bool:
        """Check if emergency conditions are met."""
        # Check for various emergency conditions
        
        # 1. Hardware failures (would come from system_state)
        if system_state.get('hardware_failure', False):
            self.trigger_emergency_stop("Hardware failure detected")
            return True
        
        # 2. Communication loss
        if system_state.get('communication_timeout', False):
            self.trigger_emergency_stop("Communication timeout")
            return True
        
        # 3. Extreme sensor readings
        if 'proprioception' in observation:
            joint_positions = observation['proprioception']
            if torch.any(torch.isnan(joint_positions)) or torch.any(torch.isinf(joint_positions)):
                self.trigger_emergency_stop("Invalid sensor readings")
                return True
        
        return self.is_emergency_active 