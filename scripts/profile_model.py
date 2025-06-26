#!/usr/bin/env python3
"""
Model Profiling Script for OpenControl.

This script profiles the performance of the OpenControl world model
to identify bottlenecks and optimization opportunities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import time
import torch
import torch.profiler
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opencontrol.cli.commands import get_dev_config, get_test_config
from opencontrol.core.world_model import OpenControlWorldModel
from opencontrol.data.dataset_manager import MultiModalDatasetManager


def create_dummy_batch(config, batch_size: int = 4, sequence_length: int = 32):
    """Create a dummy batch for profiling."""
    return {
        'video': torch.randn(batch_size, sequence_length, 3, 64, 64),
        'audio': torch.randn(batch_size, sequence_length, 1600), 
        'actions': torch.randn(batch_size, sequence_length, config.model.action_dim),
        'proprioception': torch.randn(batch_size, sequence_length, config.model.proprioception_dim)
    }


def profile_forward_pass(model, batch, device, num_iterations: int = 100):
    """Profile the forward pass of the model."""
    model.eval()
    
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch, prediction_horizon=1)
    
    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time the forward pass
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(batch, prediction_horizon=1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time, output


def profile_with_pytorch_profiler(model, batch, device):
    """Profile using PyTorch's built-in profiler."""
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if device.type == 'cuda' else None
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                output = model(batch, prediction_horizon=1)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
    
    return prof


def analyze_memory_usage(model, batch, device):
    """Analyze memory usage during forward pass."""
    if device.type != 'cuda':
        print("Memory analysis only available for CUDA devices")
        return
    
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory before
    mem_before = torch.cuda.memory_allocated()
    peak_mem_before = torch.cuda.max_memory_allocated()
    
    with torch.no_grad():
        output = model(batch, prediction_horizon=1)
        torch.cuda.synchronize()
    
    # Measure memory after
    mem_after = torch.cuda.memory_allocated()
    peak_mem_after = torch.cuda.max_memory_allocated()
    
    print(f"Memory before: {mem_before / 1024**3:.2f} GB")
    print(f"Memory after: {mem_after / 1024**3:.2f} GB")
    print(f"Memory used: {(mem_after - mem_before) / 1024**3:.2f} GB")
    print(f"Peak memory: {peak_mem_after / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Profile OpenControl model performance")
    parser.add_argument("--config", choices=["dev", "test"], default="test",
                        help="Configuration to use")
    parser.add_argument("--device", default="auto", 
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for profiling")
    parser.add_argument("--sequence-length", type=int, default=32,
                        help="Sequence length for profiling")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for timing")
    parser.add_argument("--save-trace", action="store_true",
                        help="Save PyTorch profiler trace")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config == "dev":
        config = get_dev_config()
    else:
        config = get_test_config()
    
    print(f"Using {args.config} configuration")
    
    # Create model
    print("Creating model...")
    model = OpenControlWorldModel(config)
    model = model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Create dummy batch
    print(f"Creating dummy batch (batch_size={args.batch_size}, seq_len={args.sequence_length})...")
    batch = create_dummy_batch(config, args.batch_size, args.sequence_length)
    
    # Profile forward pass timing
    print(f"\nProfiling forward pass ({args.iterations} iterations)...")
    avg_time, output = profile_forward_pass(model, batch, device, args.iterations)
    
    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {args.batch_size / avg_time:.2f} samples/sec")
    
    # Analyze memory usage
    print(f"\nAnalyzing memory usage...")
    analyze_memory_usage(model, batch, device)
    
    # PyTorch profiler
    print(f"\nRunning PyTorch profiler...")
    prof = profile_with_pytorch_profiler(model, batch, device)
    
    # Print top operations by CPU time
    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    if device.type == 'cuda':
        print("\nTop operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Save trace if requested
    if args.save_trace:
        trace_path = "profile_trace.json"
        prof.export_chrome_trace(trace_path)
        print(f"\nProfiler trace saved to {trace_path}")
        print("View in Chrome at chrome://tracing/")
    
    # Print output shapes for verification
    print(f"\nOutput verification:")
    print(f"Predictions: {len(output.predictions)} modalities")
    for modality, pred in output.predictions.items():
        print(f"  {modality}: {pred.shape}")
    print(f"Latent states: {output.latent_states.shape}")
    
    if output.uncertainty_estimates:
        print(f"Uncertainty estimates: {len(output.uncertainty_estimates)} modalities")


if __name__ == "__main__":
    main() 