#!/usr/bin/env python3
"""
Simple runner script for OpenControl.
This demonstrates the complete system working.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import sys
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from opencontrol.cli.main import OpenControlCLI
from opencontrol.cli.commands import get_dev_config


async def demo_opencontrol():
    """Run a simple demonstration of OpenControl."""
    print("OpenControl: Advanced Multi-Modal World Model Platform")
    print("=" * 60)
    print("This is a demonstration of the complete system working.")
    print()
    
    # Initialize the CLI with development config
    cli = OpenControlCLI()
    
    # Initialize the system
    print("Initializing system...")
    success = await cli.initialize_system()
    
    if not success:
        print("Failed to initialize system")
        return
    
    print()
    print("System initialized successfully!")
    print()
    
    # Show model information
    print("Model Information:")
    print("-" * 30)
    if cli.world_model:
        total_params = sum(p.numel() for p in cli.world_model.parameters())
        trainable_params = sum(p.numel() for p in cli.world_model.parameters() if p.requires_grad)
        
        print(f"  • Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  • Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  • Model Dimension: {cli.config.model.model_dim}")
        print(f"  • Number of Layers: {cli.config.model.num_layers}")
        print(f"  • Number of Heads: {cli.config.model.num_heads}")
    
    print()
    
    # Demo different functionalities
    print("Available Functionalities:")
    print("-" * 40)
    print("  • Multi-modal encoder (video, audio, actions, proprioception)")
    print("  • Transformer with RoPE positional encoding")
    print("  • Distributed training support")
    print("  • Real-time MPC control")
    print("  • Interactive CLI interface")
    print("  • Production deployment ready")
    
    print()
    print("Technical Features:")
    print("-" * 30)
    print("  • State-of-the-art transformer architecture")
    print("  • Rotary Positional Embeddings (RoPE)")
    print("  • Cross-Entropy Method (CEM) for control")
    print("  • Uncertainty estimation")
    print("  • Mixed precision training")
    print("  • Gradient checkpointing")
    print("  • Dynamic batching")
    
    print()
    print("Quick Model Test:")
    print("-" * 25)
    
    # Create some dummy input
    import torch
    dummy_input = {
        'video': torch.randn(1, 5, 3, 64, 64),  # 1 batch, 5 frames, 3 channels, 64x64
        'actions': torch.randn(1, 5, 8),        # 1 batch, 5 timesteps, 8 action dims
        'proprioception': torch.randn(1, 5, 16) # 1 batch, 5 timesteps, 16 proprio dims
    }
    
    try:
        print("  Running forward pass...")
        with torch.no_grad():
            output = cli.world_model(dummy_input, prediction_horizon=3)
        
        print(f"  Forward pass successful!")
        print(f"  Predictions shape: {len(output.predictions)} modalities")
        print(f"  Latent states shape: {output.latent_states.shape}")
        
        if output.predictions:
            for modality, pred in output.predictions.items():
                print(f"    • {modality}: {pred.shape}")
        
    except Exception as e:
        print(f"  Forward pass failed: {e}")
    
    print()
    print("CLI Commands Available:")
    print("-" * 35)
    print("  • opencontrol interactive  - Launch interactive dashboard")
    print("  • opencontrol train       - Train the model")
    print("  • opencontrol evaluate    - Evaluate performance")
    print("  • opencontrol serve       - Start model server")
    print("  • opencontrol info        - Show system information")
    
    print()
    print("To get started:")
    print("   source .venv/bin/activate")
    print("   opencontrol interactive")
    
    print()
    print("OpenControl demonstration complete!")
    
    # Cleanup
    await cli.shutdown_system()


if __name__ == "__main__":
    asyncio.run(demo_opencontrol()) 