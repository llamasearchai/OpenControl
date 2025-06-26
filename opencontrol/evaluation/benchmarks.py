"""
Benchmarking Suite for World Models and Control Systems.

This module provides standardized benchmarks for evaluating world model
performance and control capabilities across various tasks and environments.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import json
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from opencontrol.core.world_model import OpenControlWorldModel
    from opencontrol.control.visual_mpc import ProductionVisualMPC

from opencontrol.cli.commands import OpenControlConfig


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: OpenControlConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        pass
    
    def _save_results(self, results: Dict[str, Any], benchmark_name: str):
        """Save benchmark results to disk."""
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"{benchmark_name}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {results_file}")


class WorldModelBenchmark(BaseBenchmark):
    """Comprehensive benchmark suite for world model evaluation."""
    
    def __init__(
        self,
        world_model: "OpenControlWorldModel",
        config: OpenControlConfig,
        logger: logging.Logger
    ):
        super().__init__(config, logger)
        self.world_model = world_model
        
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive world model benchmark."""
        self.logger.info("Starting World Model Benchmark")
        start_time = time.time()
        
        results = {
            'benchmark_name': 'world_model',
            'timestamp': time.time(),
            'config': self.config.__dict__
        }
        
        # Test 1: Prediction Accuracy
        self.logger.info("Running prediction accuracy tests")
        results['prediction_accuracy'] = await self._test_prediction_accuracy()
        
        # Test 2: Temporal Consistency
        self.logger.info("Running temporal consistency tests")
        results['temporal_consistency'] = await self._test_temporal_consistency()
        
        # Test 3: Multi-modal Alignment
        self.logger.info("Running multi-modal alignment tests")
        results['multimodal_alignment'] = await self._test_multimodal_alignment()
        
        # Test 4: Computational Efficiency
        self.logger.info("Running efficiency tests")
        results['efficiency'] = await self._test_computational_efficiency()
        
        # Test 5: Robustness
        self.logger.info("Running robustness tests")
        results['robustness'] = await self._test_robustness()
        
        # Test 6: Long-term Prediction
        self.logger.info("Running long-term prediction tests")
        results['long_term_prediction'] = await self._test_long_term_prediction()
        
        # Overall score
        results['overall_score'] = self._compute_overall_score(results)
        results['benchmark_time'] = time.time() - start_time
        
        self._save_results(results, 'world_model_benchmark')
        return results
    
    async def _test_prediction_accuracy(self) -> Dict[str, float]:
        """Test prediction accuracy across different horizons."""
        results = {}
        
        # Generate test data
        test_obs = self._generate_test_observations(batch_size=32)
        
        self.world_model.eval()
        with torch.no_grad():
            # Test different prediction horizons
            for horizon in [1, 5, 10, 20]:
                outputs = self.world_model(test_obs, prediction_horizon=horizon)
                
                # Compute accuracy metrics for each modality
                for modality, predictions in outputs.predictions.items():
                    if modality in test_obs:
                        target = test_obs[modality]
                        if predictions.shape[1] >= horizon:
                            mse = torch.nn.functional.mse_loss(
                                predictions[:, :horizon], target[:, :horizon]
                            )
                            results[f'{modality}_mse_h{horizon}'] = mse.item()
        
        return results
    
    async def _test_temporal_consistency(self) -> Dict[str, float]:
        """Test temporal consistency of predictions."""
        results = {}
        
        test_obs = self._generate_test_observations(batch_size=16)
        
        self.world_model.eval()
        with torch.no_grad():
            outputs = self.world_model(test_obs, prediction_horizon=10)
            
            for modality, predictions in outputs.predictions.items():
                if predictions.shape[1] > 1:
                    # Compute temporal smoothness
                    temporal_diff = predictions[:, 1:] - predictions[:, :-1]
                    smoothness = torch.norm(temporal_diff.view(temporal_diff.shape[0], temporal_diff.shape[1], -1), dim=-1).mean()
                    results[f'{modality}_temporal_smoothness'] = smoothness.item()
                    
                    # Compute temporal consistency score
                    consistency_score = 1.0 / (1.0 + smoothness.item())
                    results[f'{modality}_consistency_score'] = consistency_score
        
        return results
    
    async def _test_multimodal_alignment(self) -> Dict[str, float]:
        """Test alignment between different modalities."""
        results = {}
        
        test_obs = self._generate_test_observations(batch_size=16)
        
        self.world_model.eval()
        with torch.no_grad():
            outputs = self.world_model(test_obs, prediction_horizon=5)
            
            modalities = list(outputs.predictions.keys())
            
            # Test pairwise alignment
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    mod1, mod2 = modalities[i], modalities[j]
                    
                    # Simplified alignment metric using correlation
                    pred1 = outputs.predictions[mod1].view(outputs.predictions[mod1].shape[0], -1)
                    pred2 = outputs.predictions[mod2].view(outputs.predictions[mod2].shape[0], -1)
                    
                    # Normalize
                    pred1_norm = torch.nn.functional.normalize(pred1, p=2, dim=1)
                    pred2_norm = torch.nn.functional.normalize(pred2, p=2, dim=1)
                    
                    # Compute cosine similarity
                    alignment = torch.nn.functional.cosine_similarity(pred1_norm, pred2_norm, dim=1).mean()
                    results[f'{mod1}_{mod2}_alignment'] = alignment.item()
        
        return results
    
    async def _test_computational_efficiency(self) -> Dict[str, float]:
        """Test computational efficiency and throughput."""
        results = {}
        
        # Test inference speed
        test_obs = self._generate_test_observations(batch_size=1)
        
        # Warmup
        self.world_model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = self.world_model(test_obs, prediction_horizon=1)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = self.world_model(test_obs, prediction_horizon=1)
                times.append(time.time() - start)
        
        results['avg_inference_time'] = np.mean(times)
        results['std_inference_time'] = np.std(times)
        results['throughput_fps'] = 1.0 / np.mean(times)
        
        # Test memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = self.world_model(test_obs, prediction_horizon=10)
            results['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Test batch processing efficiency
        batch_sizes = [1, 4, 8, 16]
        batch_times = []
        
        for batch_size in batch_sizes:
            test_batch = self._generate_test_observations(batch_size=batch_size)
            
            start = time.time()
            with torch.no_grad():
                _ = self.world_model(test_batch, prediction_horizon=1)
            batch_times.append(time.time() - start)
        
        results['batch_efficiency'] = {
            'batch_sizes': batch_sizes,
            'batch_times': batch_times,
            'samples_per_second': [bs / bt for bs, bt in zip(batch_sizes, batch_times)]
        }
        
        return results
    
    async def _test_robustness(self) -> Dict[str, float]:
        """Test robustness to noise and perturbations."""
        results = {}
        
        test_obs = self._generate_test_observations(batch_size=16)
        
        # Get baseline predictions
        self.world_model.eval()
        with torch.no_grad():
            baseline_outputs = self.world_model(test_obs, prediction_horizon=5)
        
        # Test noise robustness
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        for noise_level in noise_levels:
            # Add noise to observations
            noisy_obs = {}
            for modality, data in test_obs.items():
                noise = torch.randn_like(data) * noise_level
                noisy_obs[modality] = data + noise
            
            with torch.no_grad():
                noisy_outputs = self.world_model(noisy_obs, prediction_horizon=5)
            
            # Compute prediction stability
            for modality in baseline_outputs.predictions:
                if modality in noisy_outputs.predictions:
                    baseline_pred = baseline_outputs.predictions[modality]
                    noisy_pred = noisy_outputs.predictions[modality]
                    
                    stability = 1.0 - torch.nn.functional.mse_loss(baseline_pred, noisy_pred).item()
                    results[f'{modality}_stability_noise_{noise_level}'] = max(0.0, stability)
        
        return results
    
    async def _test_long_term_prediction(self) -> Dict[str, float]:
        """Test long-term prediction capabilities."""
        results = {}
        
        test_obs = self._generate_test_observations(batch_size=8)
        
        # Test different long-term horizons
        horizons = [10, 20, 50, 100]
        
        self.world_model.eval()
        with torch.no_grad():
            for horizon in horizons:
                try:
                    outputs = self.world_model(test_obs, prediction_horizon=horizon)
                    
                    for modality, predictions in outputs.predictions.items():
                        # Compute prediction quality degradation over time
                        if predictions.shape[1] >= horizon:
                            early_pred = predictions[:, :5]  # First 5 steps
                            late_pred = predictions[:, -5:]  # Last 5 steps
                            
                            # Measure variance as proxy for prediction quality
                            early_var = torch.var(early_pred.view(early_pred.shape[0], -1), dim=1).mean()
                            late_var = torch.var(late_pred.view(late_pred.shape[0], -1), dim=1).mean()
                            
                            degradation = (late_var - early_var) / (early_var + 1e-8)
                            results[f'{modality}_degradation_h{horizon}'] = degradation.item()
                            
                except Exception as e:
                    self.logger.warning(f"Long-term prediction failed at horizon {horizon}: {e}")
                    results[f'failed_horizon_{horizon}'] = True
        
        return results
    
    def _generate_test_observations(self, batch_size: int = 16) -> Dict[str, torch.Tensor]:
        """Generate synthetic test observations."""
        obs = {}
        
        # Video observations
        if hasattr(self.config.model, 'video_height'):
            obs['video'] = torch.rand(
                batch_size, 1, 3, 
                self.config.model.video_height, 
                self.config.model.video_width,
                device=self.device
            )
        
        # Audio observations
        if hasattr(self.config.model, 'audio_length'):
            obs['audio'] = torch.rand(
                batch_size, 1, self.config.model.audio_length,
                device=self.device
            )
        
        # Action observations
        obs['actions'] = torch.rand(
            batch_size, 1, self.config.model.action_dim,
            device=self.device
        )
        
        # Proprioception observations
        obs['proprioception'] = torch.rand(
            batch_size, 1, self.config.model.proprioception_dim,
            device=self.device
        )
        
        return obs
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute overall benchmark score."""
        scores = []
        
        # Prediction accuracy score
        if 'prediction_accuracy' in results:
            acc_scores = [1.0 / (1.0 + v) for k, v in results['prediction_accuracy'].items() 
                         if 'mse' in k]
            if acc_scores:
                scores.append(np.mean(acc_scores))
        
        # Temporal consistency score
        if 'temporal_consistency' in results:
            cons_scores = [v for k, v in results['temporal_consistency'].items() 
                          if 'consistency_score' in k]
            if cons_scores:
                scores.append(np.mean(cons_scores))
        
        # Efficiency score (higher throughput = better)
        if 'efficiency' in results and 'throughput_fps' in results['efficiency']:
            # Normalize to 0-1 scale (assuming 30 FPS is good)
            efficiency_score = min(1.0, results['efficiency']['throughput_fps'] / 30.0)
            scores.append(efficiency_score)
        
        # Robustness score
        if 'robustness' in results:
            rob_scores = [v for k, v in results['robustness'].items() 
                         if 'stability' in k]
            if rob_scores:
                scores.append(np.mean(rob_scores))
        
        return np.mean(scores) if scores else 0.0


class ControlBenchmark(BaseBenchmark):
    """Comprehensive benchmark suite for control system evaluation."""
    
    def __init__(
        self,
        mpc_controller: "ProductionVisualMPC",
        config: OpenControlConfig,
        logger: logging.Logger
    ):
        super().__init__(config, logger)
        self.mpc_controller = mpc_controller
        
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive control benchmark."""
        self.logger.info("Starting Control System Benchmark")
        start_time = time.time()
        
        results = {
            'benchmark_name': 'control_system',
            'timestamp': time.time(),
            'config': self.config.__dict__
        }
        
        # Test 1: Tracking Performance
        self.logger.info("Running tracking performance tests")
        results['tracking_performance'] = await self._test_tracking_performance()
        
        # Test 2: Disturbance Rejection
        self.logger.info("Running disturbance rejection tests")
        results['disturbance_rejection'] = await self._test_disturbance_rejection()
        
        # Test 3: Control Efficiency
        self.logger.info("Running control efficiency tests")
        results['control_efficiency'] = await self._test_control_efficiency()
        
        # Test 4: Safety Compliance
        self.logger.info("Running safety compliance tests")
        results['safety_compliance'] = await self._test_safety_compliance()
        
        # Test 5: Robustness
        self.logger.info("Running control robustness tests")
        results['control_robustness'] = await self._test_control_robustness()
        
        # Overall score
        results['overall_score'] = self._compute_control_score(results)
        results['benchmark_time'] = time.time() - start_time
        
        self._save_results(results, 'control_benchmark')
        return results
    
    async def _test_tracking_performance(self) -> Dict[str, float]:
        """Test trajectory tracking performance."""
        results = {}
        
        # Generate reference trajectories
        trajectories = self._generate_reference_trajectories()
        
        for traj_name, trajectory in trajectories.items():
            tracking_errors = []
            
            for step, target_state in enumerate(trajectory):
                # Simulate observation
                observation = self._generate_observation_from_state(target_state)
                
                # Compute control action
                action, info = await self.mpc_controller.compute_action(
                    observation, goal={'target_position': target_state[:3]}
                )
                
                # Simulate system response (simplified)
                predicted_state = self._simulate_system_response(observation, action)
                
                # Compute tracking error
                error = torch.norm(predicted_state[:3] - target_state[:3])
                tracking_errors.append(error.item())
            
            results[f'{traj_name}_avg_error'] = np.mean(tracking_errors)
            results[f'{traj_name}_max_error'] = np.max(tracking_errors)
            results[f'{traj_name}_std_error'] = np.std(tracking_errors)
        
        return results
    
    async def _test_disturbance_rejection(self) -> Dict[str, float]:
        """Test ability to reject disturbances."""
        results = {}
        
        # Test different disturbance types
        disturbances = {
            'step': lambda t: 0.1 if t > 10 else 0.0,
            'sine': lambda t: 0.05 * np.sin(0.5 * t),
            'impulse': lambda t: 0.2 if t == 15 else 0.0
        }
        
        for dist_name, disturbance_fn in disturbances.items():
            recovery_times = []
            max_deviations = []
            
            # Simulate trajectory with disturbance
            state = torch.zeros(self.config.model.proprioception_dim, device=self.device)
            target = torch.tensor([1.0, 0.5, 0.3], device=self.device)
            
            for t in range(50):
                observation = self._generate_observation_from_state(state)
                
                # Add disturbance
                disturbance = disturbance_fn(t)
                observation['proprioception'][:3] += disturbance
                
                # Compute control
                action, _ = await self.mpc_controller.compute_action(
                    observation, goal={'target_position': target}
                )
                
                # Update state
                state = self._simulate_system_response(observation, action)
                
                # Track deviation from target
                deviation = torch.norm(state[:3] - target)
                max_deviations.append(deviation.item())
            
            results[f'{dist_name}_max_deviation'] = np.max(max_deviations)
            results[f'{dist_name}_settling_time'] = self._compute_settling_time(max_deviations)
        
        return results
    
    async def _test_control_efficiency(self) -> Dict[str, float]:
        """Test computational efficiency of control system."""
        results = {}
        
        # Test solve times
        solve_times = []
        observation = self._generate_test_observation()
        
        for _ in range(100):
            start = time.time()
            action, info = await self.mpc_controller.compute_action(observation)
            solve_time = time.time() - start
            solve_times.append(solve_time)
        
        results['avg_solve_time'] = np.mean(solve_times)
        results['max_solve_time'] = np.max(solve_times)
        results['std_solve_time'] = np.std(solve_times)
        results['real_time_factor'] = np.mean(solve_times) * self.config.control.control_frequency
        
        # Test action smoothness
        actions = []
        for _ in range(20):
            action, _ = await self.mpc_controller.compute_action(observation)
            actions.append(action)
        
        actions_tensor = torch.stack(actions)
        action_smoothness = torch.norm(actions_tensor[1:] - actions_tensor[:-1], dim=1).mean()
        results['action_smoothness'] = action_smoothness.item()
        
        return results
    
    async def _test_safety_compliance(self) -> Dict[str, float]:
        """Test safety constraint compliance."""
        results = {}
        
        # Test action bound compliance
        violations = 0
        total_tests = 100
        
        for _ in range(total_tests):
            observation = self._generate_test_observation()
            action, info = await self.mpc_controller.compute_action(observation)
            
            # Check if action violates bounds
            action_bounds = torch.tensor(self.config.control.action_bounds, device=self.device)
            if torch.any(action < action_bounds[0]) or torch.any(action > action_bounds[1]):
                violations += 1
        
        results['action_bound_violation_rate'] = violations / total_tests
        
        # Test safety system response time
        emergency_response_times = []
        for _ in range(10):
            observation = self._generate_unsafe_observation()
            start = time.time()
            action, info = await self.mpc_controller.compute_action(observation)
            response_time = time.time() - start
            emergency_response_times.append(response_time)
        
        results['emergency_response_time'] = np.mean(emergency_response_times)
        
        return results
    
    async def _test_control_robustness(self) -> Dict[str, float]:
        """Test robustness to model uncertainties."""
        results = {}
        
        # Test with noisy observations
        noise_levels = [0.01, 0.05, 0.1]
        
        for noise_level in noise_levels:
            performance_degradation = []
            
            for _ in range(20):
                # Clean observation
                clean_obs = self._generate_test_observation()
                clean_action, _ = await self.mpc_controller.compute_action(clean_obs)
                
                # Noisy observation
                noisy_obs = {}
                for key, value in clean_obs.items():
                    noise = torch.randn_like(value) * noise_level
                    noisy_obs[key] = value + noise
                
                noisy_action, _ = await self.mpc_controller.compute_action(noisy_obs)
                
                # Compute action difference
                action_diff = torch.norm(clean_action - noisy_action)
                performance_degradation.append(action_diff.item())
            
            results[f'noise_robustness_{noise_level}'] = np.mean(performance_degradation)
        
        return results
    
    def _generate_reference_trajectories(self) -> Dict[str, List[torch.Tensor]]:
        """Generate reference trajectories for tracking tests."""
        trajectories = {}
        
        # Linear trajectory
        linear_traj = []
        for t in range(30):
            state = torch.tensor([t * 0.1, 0.5, 0.3], device=self.device)
            linear_traj.append(state)
        trajectories['linear'] = linear_traj
        
        # Circular trajectory
        circular_traj = []
        for t in range(60):
            angle = t * 0.1
            state = torch.tensor([
                0.5 * np.cos(angle),
                0.5 * np.sin(angle),
                0.3
            ], device=self.device)
            circular_traj.append(state)
        trajectories['circular'] = circular_traj
        
        return trajectories
    
    def _generate_observation_from_state(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate observation from state."""
        obs = {}
        obs['proprioception'] = state
        obs['video'] = torch.rand(1, 3, 64, 64, device=self.device)
        obs['actions'] = torch.zeros(self.config.model.action_dim, device=self.device)
        return obs
    
    def _generate_test_observation(self) -> Dict[str, torch.Tensor]:
        """Generate test observation."""
        return self._generate_observation_from_state(
            torch.rand(self.config.model.proprioception_dim, device=self.device)
        )
    
    def _generate_unsafe_observation(self) -> Dict[str, torch.Tensor]:
        """Generate observation that should trigger safety responses."""
        obs = self._generate_test_observation()
        # Set joint positions near limits
        obs['proprioception'][:3] = torch.tensor([2.8, -2.8, 2.5], device=self.device)
        return obs
    
    def _simulate_system_response(self, observation: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Simulate system response to action (simplified)."""
        current_state = observation['proprioception']
        # Simple integration
        new_state = current_state + action * 0.1
        return new_state
    
    def _compute_settling_time(self, deviations: List[float], threshold: float = 0.05) -> float:
        """Compute settling time for disturbance rejection."""
        for i, dev in enumerate(deviations):
            if dev < threshold:
                return i * 0.1  # Assuming 0.1s time steps
        return len(deviations) * 0.1  # Never settled
    
    def _compute_control_score(self, results: Dict[str, Any]) -> float:
        """Compute overall control benchmark score."""
        scores = []
        
        # Tracking performance score
        if 'tracking_performance' in results:
            tracking_errors = [v for k, v in results['tracking_performance'].items() 
                             if 'avg_error' in k]
            if tracking_errors:
                tracking_score = 1.0 / (1.0 + np.mean(tracking_errors))
                scores.append(tracking_score)
        
        # Efficiency score
        if 'control_efficiency' in results:
            rtf = results['control_efficiency'].get('real_time_factor', 1.0)
            efficiency_score = max(0.0, 1.0 - rtf)  # Lower RTF is better
            scores.append(efficiency_score)
        
        # Safety score
        if 'safety_compliance' in results:
            violation_rate = results['safety_compliance'].get('action_bound_violation_rate', 1.0)
            safety_score = 1.0 - violation_rate
            scores.append(safety_score)
        
        return np.mean(scores) if scores else 0.0 