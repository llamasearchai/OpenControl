"""
Production-Ready Visual Model Predictive Control (MPC).

This module implements a highly optimized, sampling-based MPC controller (CEM)
that leverages the learned world model for real-time planning and control.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
import asyncio

if TYPE_CHECKING:
    from opencontrol.core.world_model import OpenControlWorldModel

from opencontrol.cli.commands import OpenControlConfig
from .planners import CEMPlanner
from .safety import SafetyChecker
from .utils import ActionProcessor, StateEstimator


class ProductionVisualMPC:
    """
    Production-ready Visual Model Predictive Control system.
    
    This class implements a complete MPC pipeline including:
    - State estimation and preprocessing
    - Multiple planning algorithms (CEM, MPPI, etc.)
    - Safety checking and constraint enforcement
    - Action post-processing and filtering
    - Performance monitoring and diagnostics
    """

    def __init__(
        self,
        world_model: "OpenControlWorldModel",
        config: OpenControlConfig,
        logger: logging.Logger,
    ):
        self.world_model = world_model
        self.config = config.control
        self.model_config = config.model
        self.logger = logger
        self.device = next(world_model.parameters()).device

        # Initialize components
        self.state_estimator = StateEstimator(config)
        self.planner = CEMPlanner(world_model, config, logger)
        self.safety_checker = SafetyChecker(config)
        self.action_processor = ActionProcessor(config)
        
        # Performance tracking
        self.performance_stats = {
            'solve_times': [],
            'costs': [],
            'safety_violations': 0,
            'total_calls': 0
        }
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.world_model.eval()

    async def compute_action(
        self,
        observation: Dict[str, torch.Tensor],
        goal: Optional[Dict[str, torch.Tensor]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the optimal action for the current observation.
        
        Args:
            observation: Multi-modal observation dictionary
            goal: Optional goal specification
            constraints: Optional additional constraints
            
        Returns:
            Tuple of (action, info_dict)
        """
        start_time = time.time()
        self.performance_stats['total_calls'] += 1
        
        try:
            # 1. State estimation and preprocessing
            processed_obs = await self.state_estimator.process_observation(observation)
            
            # 2. Plan using the selected planner
            action_sequence, plan_info = await self.planner.plan(
                processed_obs, goal, constraints
            )
            
            # 3. Safety checking
            safety_info = await self.safety_checker.check_action_sequence(
                action_sequence, processed_obs
            )
            
            if not safety_info['is_safe']:
                self.logger.warning(f"Unsafe action detected: {safety_info['violations']}")
                self.performance_stats['safety_violations'] += 1
                # Use emergency action or modify the plan
                action_sequence = await self.safety_checker.get_safe_action_sequence(
                    action_sequence, processed_obs
                )
            
            # 4. Action post-processing
            final_action = await self.action_processor.process_action(
                action_sequence[0], processed_obs
            )
            
            # 5. Update performance statistics
            solve_time = time.time() - start_time
            self.performance_stats['solve_times'].append(solve_time)
            self.performance_stats['costs'].append(plan_info.get('cost', 0.0))
            
            # Keep only recent statistics
            if len(self.performance_stats['solve_times']) > 1000:
                self.performance_stats['solve_times'] = self.performance_stats['solve_times'][-1000:]
                self.performance_stats['costs'] = self.performance_stats['costs'][-1000:]
            
            info = {
                'solve_time': solve_time,
                'cost': plan_info.get('cost', 0.0),
                'safety_info': safety_info,
                'plan_info': plan_info,
                'iterations': plan_info.get('iterations', 0),
                'convergence': plan_info.get('converged', False)
            }
            
            self.logger.debug(
                f"MPC solved in {solve_time:.4f}s, cost: {info['cost']:.4f}, "
                f"safe: {safety_info['is_safe']}"
            )
            
            return final_action, info
            
        except Exception as e:
            self.logger.error(f"MPC computation failed: {e}", exc_info=True)
            # Return emergency action
            emergency_action = torch.zeros(self.model_config.action_dim, device=self.device)
            return emergency_action, {'error': str(e), 'solve_time': time.time() - start_time}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_stats['solve_times']:
            return {'no_data': True}
            
        solve_times = self.performance_stats['solve_times']
        costs = self.performance_stats['costs']
        
        return {
            'avg_solve_time': np.mean(solve_times),
            'max_solve_time': np.max(solve_times),
            'min_solve_time': np.min(solve_times),
            'std_solve_time': np.std(solve_times),
            'avg_cost': np.mean(costs),
            'safety_violation_rate': self.performance_stats['safety_violations'] / max(1, self.performance_stats['total_calls']),
            'total_calls': self.performance_stats['total_calls'],
            'real_time_factor': np.mean(solve_times) * self.config.control_frequency
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'solve_times': [],
            'costs': [],
            'safety_violations': 0,
            'total_calls': 0
        }

    def set_planner(self, planner_type: str, **kwargs):
        """Change the planning algorithm."""
        from .planners import CEMPlanner, MPPIPlanner, RandomShootingPlanner
        
        if planner_type.lower() == 'cem':
            self.planner = CEMPlanner(self.world_model, self.config, self.logger, **kwargs)
        elif planner_type.lower() == 'mppi':
            self.planner = MPPIPlanner(self.world_model, self.config, self.logger, **kwargs)
        elif planner_type.lower() == 'random_shooting':
            self.planner = RandomShootingPlanner(self.world_model, self.config, self.logger, **kwargs)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
            
        self.logger.info(f"Switched to {planner_type} planner")

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters dynamically."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
        
        # Reinitialize components that depend on config
        self.planner.update_config(new_config)
        self.safety_checker.update_config(new_config)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True) 