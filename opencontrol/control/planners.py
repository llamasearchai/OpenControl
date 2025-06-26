"""
Advanced Planning Algorithms for Model Predictive Control.

This module implements various sampling-based and gradient-based planning
algorithms for real-time control with learned world models.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from opencontrol.core.world_model import OpenControlWorldModel

from opencontrol.cli.commands import OpenControlConfig


class BasePlanner(ABC):
    """Base class for all planning algorithms."""
    
    def __init__(
        self,
        world_model: "OpenControlWorldModel",
        config: OpenControlConfig,
        logger: logging.Logger
    ):
        self.world_model = world_model
        self.config = config.control
        self.model_config = config.model
        self.logger = logger
        self.device = next(world_model.parameters()).device
        
        # Action space
        self.action_dim = self.model_config.action_dim
        self.action_bounds = torch.tensor(
            self.config.action_bounds, device=self.device
        ).float()
        
        # Planning state
        self.last_action_sequence = None
        
    @abstractmethod
    async def plan(
        self,
        observation: Dict[str, torch.Tensor],
        goal: Optional[Dict[str, torch.Tensor]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan an action sequence given current observation and goal."""
        pass
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip actions to valid bounds."""
        return torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
    
    def _rollout_trajectories(
        self,
        initial_obs: Dict[str, torch.Tensor],
        action_sequences: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform parallel trajectory rollouts using the world model.
        
        Args:
            initial_obs: Initial observation
            action_sequences: [num_samples, horizon, action_dim]
            
        Returns:
            Dictionary of predicted trajectories
        """
        num_samples, horizon, _ = action_sequences.shape
        
        # Prepare initial observation for batch processing
        batch_obs = {}
        for key, value in initial_obs.items():
            if isinstance(value, torch.Tensor):
                # Expand to batch size
                batch_obs[key] = value.unsqueeze(0).repeat(num_samples, *[1] * len(value.shape))
        
        # For simplicity, we'll use the world model to predict the full trajectory
        # In practice, this might involve more sophisticated state management
        with torch.no_grad():
            outputs = self.world_model(batch_obs, prediction_horizon=horizon)
            
        return outputs.predictions
    
    def _compute_cost(
        self,
        trajectories: Dict[str, torch.Tensor],
        action_sequences: torch.Tensor,
        goal: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute cost for each trajectory.
        
        Args:
            trajectories: Predicted trajectories
            action_sequences: Action sequences [num_samples, horizon, action_dim]
            goal: Optional goal specification
            
        Returns:
            Costs for each trajectory [num_samples]
        """
        num_samples = action_sequences.shape[0]
        total_cost = torch.zeros(num_samples, device=self.device)
        
        # Action magnitude cost
        action_cost = torch.norm(action_sequences, dim=-1).mean(dim=-1)
        total_cost += self.config.cost_weights.get('action', 0.01) * action_cost
        
        # Action smoothness cost
        if action_sequences.shape[1] > 1:
            action_diff = action_sequences[:, 1:] - action_sequences[:, :-1]
            smoothness_cost = torch.norm(action_diff, dim=-1).mean(dim=-1)
            total_cost += self.config.cost_weights.get('smoothness', 0.1) * smoothness_cost
        
        # Goal-reaching cost
        if goal is not None and 'target_position' in goal:
            if 'proprioception' in trajectories:
                # Assume first 3 dimensions of proprioception are position
                pred_positions = trajectories['proprioception'][:, :, :3]  # [samples, horizon, 3]
                target_pos = goal['target_position'].to(self.device)
                
                # Distance to goal at final timestep
                final_positions = pred_positions[:, -1, :]  # [samples, 3]
                goal_distances = torch.norm(final_positions - target_pos, dim=-1)
                total_cost += self.config.cost_weights.get('goal', 1.0) * goal_distances
        
        # State cost (can be customized based on application)
        if 'proprioception' in trajectories:
            # Penalize extreme joint positions
            joint_positions = trajectories['proprioception']
            state_cost = torch.norm(joint_positions, dim=-1).mean(dim=-1)
            total_cost += self.config.cost_weights.get('state', 0.1) * state_cost
        
        return total_cost


class CEMPlanner(BasePlanner):
    """
    Cross-Entropy Method (CEM) planner.
    
    CEM is a sampling-based optimization algorithm that iteratively refines
    a probability distribution over action sequences to find the optimal plan.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # CEM-specific parameters
        self.num_samples = self.config.num_samples
        self.num_iterations = self.config.num_iterations
        self.elite_fraction = self.config.elite_fraction
        self.num_elites = int(self.num_samples * self.elite_fraction)
        
        # Initialize distribution parameters
        self.mean = torch.zeros(self.config.horizon, self.action_dim, device=self.device)
        self.std = torch.ones(self.config.horizon, self.action_dim, device=self.device) * 0.5
        
        # Momentum for distribution updates
        self.momentum = 0.1
    
    async def plan(
        self,
        observation: Dict[str, torch.Tensor],
        goal: Optional[Dict[str, torch.Tensor]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan using Cross-Entropy Method."""
        start_time = time.time()
        
        # Warm start from previous solution
        if self.last_action_sequence is not None:
            # Shift previous solution and add noise for last timestep
            self.mean[:-1] = self.last_action_sequence[1:]
            self.mean[-1] = self.last_action_sequence[-1]  # Repeat last action
        
        best_cost = float('inf')
        best_sequence = None
        
        for iteration in range(self.num_iterations):
            # Sample action sequences from current distribution
            noise = torch.randn(
                self.num_samples, self.config.horizon, self.action_dim,
                device=self.device
            )
            action_sequences = self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise
            action_sequences = self._clip_actions(action_sequences)
            
            # Evaluate trajectories
            trajectories = self._rollout_trajectories(observation, action_sequences)
            costs = self._compute_cost(trajectories, action_sequences, goal)
            
            # Select elite samples
            elite_indices = torch.topk(costs, self.num_elites, largest=False)[1]
            elite_sequences = action_sequences[elite_indices]
            
            # Update distribution
            new_mean = elite_sequences.mean(dim=0)
            new_std = elite_sequences.std(dim=0) + 1e-6  # Add small epsilon for numerical stability
            
            # Apply momentum
            self.mean = (1 - self.momentum) * self.mean + self.momentum * new_mean
            self.std = (1 - self.momentum) * self.std + self.momentum * new_std
            
            # Track best solution
            current_best_cost = costs[elite_indices[0]]
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_sequence = elite_sequences[0]
        
        # Store for warm start
        self.last_action_sequence = best_sequence.clone()
        
        solve_time = time.time() - start_time
        info = {
            'cost': best_cost.item(),
            'solve_time': solve_time,
            'iterations': self.num_iterations,
            'converged': True,  # CEM always runs full iterations
            'num_samples': self.num_samples
        }
        
        return best_sequence, info


class MPPIPlanner(BasePlanner):
    """
    Model Predictive Path Integral (MPPI) planner.
    
    MPPI uses importance sampling with exponential weighting to find optimal
    action sequences. It's particularly effective for stochastic systems.
    """
    
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_samples = self.config.num_samples
        self.temperature = temperature
        
        # Initialize action distribution (Gaussian)
        self.mean = torch.zeros(self.config.horizon, self.action_dim, device=self.device)
        self.cov = torch.eye(self.action_dim, device=self.device) * 0.25
    
    async def plan(
        self,
        observation: Dict[str, torch.Tensor],
        goal: Optional[Dict[str, torch.Tensor]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan using Model Predictive Path Integral."""
        start_time = time.time()
        
        # Sample noise trajectories
        noise = torch.randn(
            self.num_samples, self.config.horizon, self.action_dim,
            device=self.device
        )
        
        # Apply covariance (simplified - using diagonal covariance)
        cov_sqrt = torch.sqrt(torch.diag(self.cov)).unsqueeze(0).unsqueeze(0)
        action_sequences = self.mean.unsqueeze(0) + noise * cov_sqrt
        action_sequences = self._clip_actions(action_sequences)
        
        # Evaluate trajectories
        trajectories = self._rollout_trajectories(observation, action_sequences)
        costs = self._compute_cost(trajectories, action_sequences, goal)
        
        # Compute importance weights
        min_cost = torch.min(costs)
        weights = torch.exp(-(costs - min_cost) / self.temperature)
        weights = weights / torch.sum(weights)
        
        # Compute weighted average of action sequences
        optimal_sequence = torch.sum(
            weights.unsqueeze(-1).unsqueeze(-1) * action_sequences, dim=0
        )
        
        # Update mean for next iteration
        self.mean = optimal_sequence.clone()
        
        solve_time = time.time() - start_time
        info = {
            'cost': torch.sum(weights * costs).item(),
            'solve_time': solve_time,
            'iterations': 1,  # MPPI is single iteration
            'converged': True,
            'num_samples': self.num_samples,
            'temperature': self.temperature
        }
        
        return optimal_sequence, info


class RandomShootingPlanner(BasePlanner):
    """
    Random Shooting planner.
    
    Simple baseline that samples random action sequences and selects the best one.
    Useful for comparison and as a fallback method.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = self.config.num_samples
    
    async def plan(
        self,
        observation: Dict[str, torch.Tensor],
        goal: Optional[Dict[str, torch.Tensor]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan using random shooting."""
        start_time = time.time()
        
        # Sample random action sequences
        action_sequences = torch.rand(
            self.num_samples, self.config.horizon, self.action_dim,
            device=self.device
        )
        
        # Scale to action bounds
        action_range = self.action_bounds[1] - self.action_bounds[0]
        action_sequences = action_sequences * action_range + self.action_bounds[0]
        
        # Evaluate trajectories
        trajectories = self._rollout_trajectories(observation, action_sequences)
        costs = self._compute_cost(trajectories, action_sequences, goal)
        
        # Select best sequence
        best_idx = torch.argmin(costs)
        best_sequence = action_sequences[best_idx]
        best_cost = costs[best_idx]
        
        solve_time = time.time() - start_time
        info = {
            'cost': best_cost.item(),
            'solve_time': solve_time,
            'iterations': 1,
            'converged': True,
            'num_samples': self.num_samples
        }
        
        return best_sequence, info


class GradientBasedPlanner(BasePlanner):
    """
    Gradient-based planner using backpropagation through the world model.
    
    This planner directly optimizes action sequences using gradient descent,
    which can be more sample-efficient than sampling-based methods.
    """
    
    def __init__(self, *args, learning_rate: float = 0.01, num_iterations: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    async def plan(
        self,
        observation: Dict[str, torch.Tensor],
        goal: Optional[Dict[str, torch.Tensor]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan using gradient-based optimization."""
        start_time = time.time()
        
        # Initialize action sequence
        action_sequence = torch.zeros(
            self.config.horizon, self.action_dim,
            device=self.device, requires_grad=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam([action_sequence], lr=self.learning_rate)
        
        best_cost = float('inf')
        best_sequence = None
        
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Clip actions to bounds
            clipped_actions = self._clip_actions(action_sequence.unsqueeze(0))
            
            # Forward pass through world model
            trajectories = self._rollout_trajectories(observation, clipped_actions)
            cost = self._compute_cost(trajectories, clipped_actions, goal)[0]
            
            # Backward pass
            cost.backward()
            optimizer.step()
            
            # Track best solution
            if cost.item() < best_cost:
                best_cost = cost.item()
                best_sequence = clipped_actions[0].detach().clone()
        
        solve_time = time.time() - start_time
        info = {
            'cost': best_cost,
            'solve_time': solve_time,
            'iterations': self.num_iterations,
            'converged': True,
            'method': 'gradient_based'
        }
        
        return best_sequence, info 