"""
Model Optimization for OpenControl Deployment.

This module provides model optimization techniques for production deployment
including quantization, pruning, and compilation optimizations.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import logging
from typing import Optional, Dict, Any
from opencontrol.cli.commands import OpenControlConfig


class ModelOptimizer:
    """
    Model optimization utilities for production deployment.
    
    This class provides various optimization techniques to improve
    inference speed and reduce memory usage in production.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def optimize_for_inference(
        self, 
        model: torch.nn.Module, 
        config: OpenControlConfig
    ) -> torch.nn.Module:
        """Apply comprehensive optimizations for inference."""
        
        self.logger.info("Starting model optimization for inference")
        
        # Set model to evaluation mode
        model.eval()
        
        # Apply torch.jit.script if possible
        try:
            if config.infrastructure.compile_model:
                self.logger.info("Applying TorchScript compilation")
                model = torch.jit.script(model)
        except Exception as e:
            self.logger.warning(f"TorchScript compilation failed: {e}")
        
        # Apply PyTorch 2.0 compilation if available
        try:
            if hasattr(torch, 'compile') and config.infrastructure.compile_model:
                self.logger.info("Applying PyTorch 2.0 compilation")
                model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            self.logger.warning(f"PyTorch 2.0 compilation failed: {e}")
        
        self.logger.info("Model optimization completed")
        return model
    
    def quantize_model(
        self, 
        model: torch.nn.Module, 
        quantization_type: str = "dynamic"
    ) -> torch.nn.Module:
        """Apply quantization to reduce model size and improve speed."""
        
        self.logger.info(f"Applying {quantization_type} quantization")
        
        if quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            # Static quantization (requires calibration data)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Note: Would need calibration data here
            quantized_model = torch.quantization.convert(model, inplace=True)
        else:
            self.logger.warning(f"Unknown quantization type: {quantization_type}")
            return model
        
        self.logger.info("Quantization completed")
        return quantized_model
    
    def optimize_memory_usage(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for reduced memory usage."""
        
        self.logger.info("Optimizing model memory usage")
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Optimize for inference
        with torch.no_grad():
            for param in model.parameters():
                param.requires_grad = False
        
        self.logger.info("Memory optimization completed")
        return model
    
    def benchmark_model(
        self, 
        model: torch.nn.Module, 
        sample_input: Dict[str, torch.Tensor],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        
        self.logger.info(f"Benchmarking model performance over {num_runs} runs")
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(sample_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate statistics
        import numpy as np
        results = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'throughput_fps': 1.0 / np.mean(times)
        }
        
        self.logger.info(f"Benchmark results: {results}")
        return results 