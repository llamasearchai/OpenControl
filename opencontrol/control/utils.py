"""
Utility Classes for Control System.

This module provides utility classes for action processing, state estimation,
and other control-related functionality.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import time
from typing import Dict, Optional, Any, List
from collections import deque
import asyncio

from opencontrol.cli.commands import OpenControlConfig


class StateEstimator:
    """
    State estimation and observation preprocessing for control.
    
    This class handles:
    - Sensor fusion from multiple modalities
    - State filtering and smoothing
    - Observation preprocessing and normalization
    - Latency compensation
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config.control
        self.model_config = config.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State history for filtering
        self.state_history = deque(maxlen=100)
        self.observation_history = deque(maxlen=50)
        
        # Kalman filter parameters (simplified)
        self.process_noise = 0.01
        self.measurement_noise = 0.1
        
        # Normalization parameters
        self.obs_mean = {}
        self.obs_std = {}
        self._initialize_normalization_params()
        
    def _initialize_normalization_params(self):
        """Initialize observation normalization parameters."""
        # These would typically be computed from training data
        self.obs_mean = {
            'proprioception': torch.zeros(self.model_config.proprioception_dim, device=self.device),
            'video': torch.tensor(0.5, device=self.device),  # Assuming normalized images
            'audio': torch.tensor(0.0, device=self.device)
        }
        
        self.obs_std = {
            'proprioception': torch.ones(self.model_config.proprioception_dim, device=self.device),
            'video': torch.tensor(0.5, device=self.device),
            'audio': torch.tensor(1.0, device=self.device)
        }
    
    async def process_observation(
        self, 
        raw_observation: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw observation into clean, normalized state.
        
        Args:
            raw_observation: Raw sensor observations
            
        Returns:
            Processed observation dictionary
        """
        processed_obs = {}
        
        # Store raw observation
        self.observation_history.append({
            'timestamp': time.time(),
            'observation': raw_observation.copy()
        })
        
        # Process each modality
        for modality, data in raw_observation.items():
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
                
                # 1. Outlier detection and rejection
                filtered_data = self._filter_outliers(data, modality)
                
                # 2. Noise reduction
                smoothed_data = self._apply_smoothing(filtered_data, modality)
                
                # 3. Normalization
                normalized_data = self._normalize_observation(smoothed_data, modality)
                
                processed_obs[modality] = normalized_data
            else:
                processed_obs[modality] = data
        
        # 4. State estimation (sensor fusion)
        estimated_state = await self._estimate_state(processed_obs)
        
        return estimated_state
    
    def _filter_outliers(self, data: torch.Tensor, modality: str) -> torch.Tensor:
        """Remove outliers from sensor data."""
        if modality == 'proprioception':
            # Simple outlier detection for joint positions
            # Check if values are within reasonable bounds
            reasonable_bounds = torch.tensor([-3.14, 3.14], device=self.device)
            data = torch.clamp(data, reasonable_bounds[0], reasonable_bounds[1])
        
        elif modality == 'video':
            # Ensure pixel values are in valid range
            data = torch.clamp(data, 0.0, 1.0)
        
        return data
    
    def _apply_smoothing(self, data: torch.Tensor, modality: str) -> torch.Tensor:
        """Apply temporal smoothing to reduce noise."""
        if modality == 'proprioception' and len(self.observation_history) > 1:
            # Simple exponential smoothing
            alpha = 0.7  # Smoothing factor
            prev_obs = self.observation_history[-2]['observation'].get(modality)
            if prev_obs is not None:
                prev_obs = prev_obs.to(self.device)
                data = alpha * data + (1 - alpha) * prev_obs
        
        return data
    
    def _normalize_observation(self, data: torch.Tensor, modality: str) -> torch.Tensor:
        """Normalize observation data."""
        if modality in self.obs_mean:
            mean = self.obs_mean[modality]
            std = self.obs_std[modality]
            
            # Handle different tensor shapes
            if data.shape != mean.shape:
                # Broadcast or adapt as needed
                if modality == 'video':
                    # For video, normalize per-pixel
                    data = (data - mean) / std
                else:
                    # For other modalities, adapt shapes
                    if len(mean.shape) == 1 and len(data.shape) > 1:
                        # data might be batched
                        data = (data - mean.unsqueeze(0)) / std.unsqueeze(0)
                    else:
                        data = (data - mean) / std
            else:
                data = (data - mean) / std
        
        return data
    
    async def _estimate_state(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform sensor fusion to estimate true state."""
        # For now, we'll just return the processed observations
        # In practice, this would implement sophisticated sensor fusion
        
        # Store state estimate
        state_estimate = observations.copy()
        self.state_history.append({
            'timestamp': time.time(),
            'state': state_estimate
        })
        
        return state_estimate
    
    def get_state_velocity(self, modality: str = 'proprioception') -> Optional[torch.Tensor]:
        """Estimate state velocity from recent history."""
        if len(self.state_history) < 2:
            return None
        
        current_state = self.state_history[-1]['state'].get(modality)
        prev_state = self.state_history[-2]['state'].get(modality)
        
        if current_state is not None and prev_state is not None:
            dt = self.state_history[-1]['timestamp'] - self.state_history[-2]['timestamp']
            if dt > 0:
                velocity = (current_state - prev_state) / dt
                return velocity
        
        return None
    
    def predict_future_state(
        self, 
        steps_ahead: int = 1, 
        modality: str = 'proprioception'
    ) -> Optional[torch.Tensor]:
        """Predict future state based on current trend."""
        if len(self.state_history) < 2:
            return None
        
        velocity = self.get_state_velocity(modality)
        if velocity is not None:
            current_state = self.state_history[-1]['state'].get(modality)
            dt = 1.0 / self.config.control_frequency  # Assume constant time step
            predicted_state = current_state + velocity * dt * steps_ahead
            return predicted_state
        
        return None


class ActionProcessor:
    """
    Action post-processing and filtering.
    
    This class handles:
    - Action smoothing and filtering
    - Rate limiting
    - Safety constraints
    - Action scaling and transformation
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config.control
        self.model_config = config.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Action history for filtering
        self.action_history = deque(maxlen=50)
        
        # Rate limiting parameters
        self.max_action_change = getattr(self.config, 'max_action_change', 0.1)
        
        # Action bounds
        self.action_bounds = torch.tensor(self.config.action_bounds, device=self.device)
        
    async def process_action(
        self, 
        raw_action: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Process raw action from planner into safe, executable action.
        
        Args:
            raw_action: Raw action from planning algorithm
            observation: Current observation for context
            
        Returns:
            Processed action ready for execution
        """
        action = raw_action.to(self.device)
        
        # 1. Clip to action bounds
        action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
        
        # 2. Apply rate limiting
        action = self._apply_rate_limiting(action)
        
        # 3. Apply smoothing
        action = self._apply_smoothing(action)
        
        # 4. Apply safety constraints
        action = await self._apply_safety_constraints(action, observation)
        
        # 5. Store in history
        self.action_history.append({
            'timestamp': time.time(),
            'action': action.clone()
        })
        
        return action
    
    def _apply_rate_limiting(self, action: torch.Tensor) -> torch.Tensor:
        """Limit rate of change of actions."""
        if len(self.action_history) > 0:
            prev_action = self.action_history[-1]['action']
            action_diff = action - prev_action
            
            # Clip action difference to maximum allowed change
            max_change = torch.full_like(action_diff, self.max_action_change)
            action_diff = torch.clamp(action_diff, -max_change, max_change)
            
            action = prev_action + action_diff
        
        return action
    
    def _apply_smoothing(self, action: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to actions."""
        if len(self.action_history) > 0:
            # Exponential smoothing
            alpha = 0.8  # Smoothing factor
            prev_action = self.action_history[-1]['action']
            action = alpha * action + (1 - alpha) * prev_action
        
        return action
    
    async def _apply_safety_constraints(
        self, 
        action: torch.Tensor,
        observation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply additional safety constraints based on current state."""
        # Example: reduce action magnitude if robot is near obstacles
        if 'proprioception' in observation:
            joint_positions = observation['proprioception']
            
            # If any joint is near its limit, reduce action magnitude
            joint_limits = torch.tensor([-2.5, 2.5], device=self.device)
            near_limit_mask = (torch.abs(joint_positions) > 2.0)
            
            if torch.any(near_limit_mask):
                # Reduce action magnitude for joints near limits
                safety_factor = 0.5
                action = action * safety_factor
        
        return action
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent actions."""
        if not self.action_history:
            return {'no_data': True}
        
        recent_actions = torch.stack([h['action'] for h in self.action_history])
        
        return {
            'mean_action': torch.mean(recent_actions, dim=0),
            'std_action': torch.std(recent_actions, dim=0),
            'max_action': torch.max(recent_actions, dim=0)[0],
            'min_action': torch.min(recent_actions, dim=0)[0],
            'action_range': torch.max(recent_actions, dim=0)[0] - torch.min(recent_actions, dim=0)[0]
        }


class PerformanceMonitor:
    """
    Monitor and analyze control system performance.
    
    This class tracks various performance metrics and provides
    diagnostics for the control system.
    """
    
    def __init__(self, config: OpenControlConfig):
        self.config = config.control
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance metrics
        self.metrics_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        
        # Timing statistics
        self.timing_stats = {
            'planning_times': deque(maxlen=1000),
            'processing_times': deque(maxlen=1000),
            'total_cycle_times': deque(maxlen=1000)
        }
        
    def record_cycle_metrics(
        self,
        planning_time: float,
        processing_time: float,
        total_time: float,
        cost: float,
        safety_violations: List[str]
    ):
        """Record metrics for a single control cycle."""
        metrics = {
            'timestamp': time.time(),
            'planning_time': planning_time,
            'processing_time': processing_time,
            'total_time': total_time,
            'cost': cost,
            'safety_violations': safety_violations,
            'real_time_factor': total_time * self.config.control_frequency
        }
        
        self.metrics_history.append(metrics)
        self.timing_stats['planning_times'].append(planning_time)
        self.timing_stats['processing_times'].append(processing_time)
        self.timing_stats['total_cycle_times'].append(total_time)
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error occurrence."""
        error_record = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message
        }
        self.error_history.append(error_record)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'no_data': True}
        
        # Timing statistics
        planning_times = list(self.timing_stats['planning_times'])
        total_times = list(self.timing_stats['total_cycle_times'])
        
        # Safety statistics
        total_cycles = len(self.metrics_history)
        cycles_with_violations = sum(
            1 for m in self.metrics_history if m['safety_violations']
        )
        
        # Real-time performance
        real_time_factors = [m['real_time_factor'] for m in self.metrics_history]
        real_time_violations = sum(1 for rtf in real_time_factors if rtf > 1.0)
        
        return {
            'timing': {
                'avg_planning_time': np.mean(planning_times),
                'max_planning_time': np.max(planning_times),
                'avg_total_time': np.mean(total_times),
                'max_total_time': np.max(total_times),
                'real_time_violation_rate': real_time_violations / total_cycles
            },
            'safety': {
                'total_cycles': total_cycles,
                'cycles_with_violations': cycles_with_violations,
                'safety_violation_rate': cycles_with_violations / total_cycles,
                'recent_errors': list(self.error_history)[-5:]
            },
            'costs': {
                'avg_cost': np.mean([m['cost'] for m in self.metrics_history]),
                'cost_trend': [m['cost'] for m in self.metrics_history][-20:]
            }
        }
    
    def check_performance_alerts(self) -> List[str]:
        """Check for performance issues that need attention."""
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        # Check real-time performance
        recent_rtf = [m['real_time_factor'] for m in self.metrics_history[-10:]]
        if np.mean(recent_rtf) > 0.8:  # Getting close to real-time limit
            alerts.append("Real-time performance degrading")
        
        # Check for frequent safety violations
        recent_violations = [m['safety_violations'] for m in self.metrics_history[-20:]]
        violation_rate = sum(1 for v in recent_violations if v) / len(recent_violations)
        if violation_rate > 0.1:  # More than 10% violation rate
            alerts.append("High safety violation rate")
        
        # Check for increasing costs
        recent_costs = [m['cost'] for m in self.metrics_history[-10:]]
        if len(recent_costs) >= 5:
            cost_trend = np.polyfit(range(len(recent_costs)), recent_costs, 1)[0]
            if cost_trend > 0.1:  # Costs increasing
                alerts.append("Control costs increasing")
        
        return alerts 