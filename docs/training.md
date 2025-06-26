# Training Guide: World Models for Robotics

This comprehensive guide covers training multi-modal world models for robotics applications using OpenControl.

## Overview

World models are predictive models that learn to simulate the dynamics of robotic environments. They enable robots to:

- **Plan ahead**: Predict future states given current observations and actions
- **Learn efficiently**: Reduce real-world data requirements through simulation
- **Handle uncertainty**: Model stochastic environments and sensor noise
- **Enable model-based control**: Support advanced control algorithms like MPC

## Prerequisites

- Completed [Installation Guide](installation.md)
- Basic understanding of deep learning and robotics
- Access to robot demonstration data or simulation environment
- GPU with at least 8GB VRAM (recommended)

## Data Requirements

### Data Collection

Before training, you need robot demonstration data. OpenControl supports multiple data sources:

#### Robot Demonstrations

```python
from opencontrol.data import DataCollector
from opencontrol.robots import UniversalRobot
from opencontrol.sensors import RealSenseCamera, ForceTorqueSensor

# Initialize components
robot = UniversalRobot("192.168.1.100")
camera = RealSenseCamera(width=640, height=480, fps=30)
ft_sensor = ForceTorqueSensor("robotiq_ft300", "192.168.1.101")

# Setup data collector
collector = DataCollector(
    robot=robot,
    sensors=[camera, ft_sensor],
    save_path="data/demonstrations",
    recording_frequency=30,  # Hz
    modalities={
        "rgb": True,
        "depth": True,
        "joints": True,
        "force": True,
        "gripper": True
    }
)

# Collect demonstrations
for i in range(100):  # Collect 100 demonstrations
    collector.start_recording(f"demo_{i:03d}")
    
    # Perform demonstration (manual or scripted)
    # ...
    
    collector.stop_recording()
    print(f"Collected demonstration {i+1}/100")
```

#### Simulation Data

```python
from opencontrol.simulation import RobotSimulator
from opencontrol.data import SimulationDataCollector

# Initialize simulator
simulator = RobotSimulator(
    robot_type="ur5",
    environment="manipulation_table",
    physics_engine="pybullet"
)

# Setup data collector
sim_collector = SimulationDataCollector(
    simulator=simulator,
    save_path="data/simulation",
    episodes_per_task=1000
)

# Generate simulation data
tasks = ["pick_place", "stacking", "insertion"]
for task in tasks:
    sim_collector.collect_task_data(
        task_name=task,
        randomization=True,
        domain_randomization={
            "lighting": True,
            "textures": True,
            "object_properties": True,
            "camera_noise": True
        }
    )
```

### Data Format

OpenControl uses a standardized data format:

```
data/
├── demonstrations/
│   ├── demo_000/
│   │   ├── rgb/           # RGB images (PNG)
│   │   ├── depth/         # Depth images (NPZ)
│   │   ├── joints.csv     # Joint positions/velocities
│   │   ├── force.csv      # Force/torque readings
│   │   ├── gripper.csv    # Gripper state
│   │   └── metadata.json  # Episode metadata
│   ├── demo_001/
│   └── ...
└── metadata.json          # Dataset metadata
```

### Data Preprocessing

```python
from opencontrol.data import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    data_path="data/demonstrations",
    output_path="data/processed",
    config={
        "image_size": (224, 224),
        "sequence_length": 32,
        "prediction_horizon": 16,
        "normalization": {
            "joints": "minmax",
            "force": "zscore",
            "images": "imagenet"
        },
        "augmentation": {
            "rotation": 15,  # degrees
            "brightness": 0.2,
            "contrast": 0.2,
            "noise": 0.01
        }
    }
)

# Process data
preprocessor.process()
```

## Model Architecture

### Multi-Modal World Model

OpenControl's world model architecture combines multiple modalities:

```python
from opencontrol.models import MultiModalWorldModel

model = MultiModalWorldModel(
    # Vision encoder
    vision_encoder="resnet50",  # or "resnet18", "efficientnet", "vit"
    vision_features=512,
    
    # Sequence modeling
    sequence_length=32,
    prediction_horizon=16,
    hidden_dim=512,
    num_layers=8,
    
    # Multi-modal fusion
    fusion_type="cross_attention",  # or "concat", "film"
    
    # Output heads
    predict_rgb=True,
    predict_depth=True,
    predict_joints=True,
    predict_force=True,
    predict_rewards=True,
    
    # Architecture details
    attention_heads=8,
    dropout=0.1,
    layer_norm=True,
    residual_connections=True
)
```

### Architecture Components

#### Vision Encoder

```python
# Different vision encoders for different use cases
encoders = {
    "resnet18": {"params": "11M", "speed": "fast", "accuracy": "good"},
    "resnet50": {"params": "25M", "speed": "medium", "accuracy": "better"},
    "efficientnet_b0": {"params": "5M", "speed": "fast", "accuracy": "good"},
    "vit_small": {"params": "22M", "speed": "slow", "accuracy": "best"}
}

# Custom encoder configuration
vision_config = {
    "encoder": "resnet50",
    "pretrained": True,
    "freeze_backbone": False,
    "feature_dim": 512,
    "spatial_features": True  # Keep spatial information
}
```

#### Temporal Modeling

```python
# Transformer-based sequence modeling
temporal_config = {
    "architecture": "transformer",  # or "lstm", "gru"
    "num_layers": 8,
    "hidden_dim": 512,
    "attention_heads": 8,
    "positional_encoding": "rope",  # Rotary Position Embeddings
    "causal_attention": True
}
```

#### Multi-Modal Fusion

```python
# Different fusion strategies
fusion_strategies = {
    "early_fusion": "Concatenate features before sequence modeling",
    "late_fusion": "Separate processing, combine predictions",
    "cross_attention": "Attention-based fusion (recommended)",
    "film": "Feature-wise Linear Modulation"
}
```

## Training Configuration

### Basic Training Setup

```python
from opencontrol.training import WorldModelTrainer
from opencontrol.data import RobotDataset

# Load dataset
dataset = RobotDataset(
    data_path="data/processed",
    sequence_length=32,
    prediction_horizon=16,
    modalities=["rgb", "depth", "joints", "force"],
    split_ratio={"train": 0.8, "val": 0.1, "test": 0.1}
)

# Initialize trainer
trainer = WorldModelTrainer(
    model=model,
    dataset=dataset,
    config={
        # Training hyperparameters
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_epochs": 100,
        
        # Optimization
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 10,
        "gradient_clip": 1.0,
        
        # Hardware
        "device": "cuda",
        "mixed_precision": True,
        "compile_model": True,  # PyTorch 2.0 compilation
        
        # Logging and checkpointing
        "log_interval": 100,
        "save_interval": 10,
        "checkpoint_dir": "checkpoints/",
        "wandb_project": "opencontrol-world-models"
    }
)
```

### Advanced Training Configuration

```python
# Advanced training configuration
advanced_config = {
    # Loss configuration
    "losses": {
        "rgb_loss": {"weight": 1.0, "type": "mse"},
        "depth_loss": {"weight": 0.5, "type": "l1"},
        "joints_loss": {"weight": 2.0, "type": "mse"},
        "force_loss": {"weight": 1.0, "type": "mse"},
        "reward_loss": {"weight": 0.1, "type": "bce"}
    },
    
    # Regularization
    "regularization": {
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2
    },
    
    # Data loading
    "dataloader": {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2
    },
    
    # Distributed training
    "distributed": {
        "backend": "nccl",
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True
    }
}
```

## Training Process

### Single GPU Training

```python
# Start training
print("Starting world model training...")
trainer.train()

# Training will automatically:
# 1. Initialize model and optimizer
# 2. Load data with proper batching
# 3. Run training loop with validation
# 4. Save checkpoints and logs
# 5. Generate evaluation metrics
```

### Multi-GPU Training

```python
import torch.distributed as dist
from opencontrol.training import DistributedWorldModelTrainer

# Initialize distributed training
def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Distributed trainer
distributed_trainer = DistributedWorldModelTrainer(
    model=model,
    dataset=dataset,
    config=advanced_config
)

# Launch training
if __name__ == "__main__":
    setup_distributed()
    distributed_trainer.train()
```

### Training Script Example

```bash
#!/bin/bash
# train_world_model.sh

# Single GPU
python -m opencontrol.training.train_world_model \
    --config configs/training/world_model.yaml \
    --data-path data/processed \
    --output-dir checkpoints/

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 \
    -m opencontrol.training.train_world_model \
    --config configs/training/world_model.yaml \
    --data-path data/processed \
    --output-dir checkpoints/ \
    --distributed
```

## Loss Functions

### Multi-Modal Losses

```python
class MultiModalLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        
    def forward(self, predictions, targets):
        losses = {}
        
        # RGB reconstruction loss
        if "rgb" in predictions:
            losses["rgb"] = F.mse_loss(
                predictions["rgb"], targets["rgb"]
            )
        
        # Depth reconstruction loss
        if "depth" in predictions:
            losses["depth"] = F.l1_loss(
                predictions["depth"], targets["depth"]
            )
        
        # Joint prediction loss
        if "joints" in predictions:
            losses["joints"] = F.mse_loss(
                predictions["joints"], targets["joints"]
            )
        
        # Force prediction loss
        if "force" in predictions:
            losses["force"] = F.mse_loss(
                predictions["force"], targets["force"]
            )
        
        # Combine losses
        total_loss = sum(
            self.loss_weights[key] * loss 
            for key, loss in losses.items()
        )
        
        return total_loss, losses
```

### Perceptual Losses

```python
from opencontrol.losses import PerceptualLoss

# Perceptual loss for better image quality
perceptual_loss = PerceptualLoss(
    network="vgg16",
    layers=["conv1_2", "conv2_2", "conv3_3"],
    weights=[1.0, 1.0, 1.0]
)

# Usage in training loop
rgb_perceptual = perceptual_loss(
    predictions["rgb"], targets["rgb"]
)
```

## Evaluation

### Model Evaluation

```python
from opencontrol.evaluation import WorldModelEvaluator

# Initialize evaluator
evaluator = WorldModelEvaluator(
    model=model,
    test_dataset=dataset.test_split(),
    metrics=[
        "mse", "psnr", "ssim",  # Image metrics
        "mae", "rmse",          # Regression metrics
        "fvd", "lpips"          # Video metrics
    ]
)

# Run evaluation
results = evaluator.evaluate()
print(f"Results: {results}")
```

### Evaluation Metrics

```python
# Comprehensive evaluation metrics
evaluation_metrics = {
    # Image quality metrics
    "image_metrics": {
        "mse": "Mean Squared Error",
        "psnr": "Peak Signal-to-Noise Ratio",
        "ssim": "Structural Similarity Index",
        "lpips": "Learned Perceptual Image Patch Similarity"
    },
    
    # Video quality metrics
    "video_metrics": {
        "fvd": "Frechet Video Distance",
        "temporal_consistency": "Frame-to-frame consistency"
    },
    
    # Robotics-specific metrics
    "robotics_metrics": {
        "joint_accuracy": "Joint prediction accuracy",
        "force_prediction": "Force/torque prediction error",
        "task_success": "Downstream task success rate"
    }
}
```

## Hyperparameter Tuning

### Automated Hyperparameter Search

```python
from opencontrol.tuning import HyperparameterTuner

# Define search space
search_space = {
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "batch_size": [8, 16, 32],
    "hidden_dim": [256, 512, 768],
    "num_layers": [4, 6, 8],
    "dropout": [0.0, 0.1, 0.2]
}

# Initialize tuner
tuner = HyperparameterTuner(
    model_class=MultiModalWorldModel,
    dataset=dataset,
    search_space=search_space,
    strategy="bayesian",  # or "random", "grid"
    max_trials=50,
    objective="validation_loss"
)

# Run hyperparameter search
best_config = tuner.search()
print(f"Best configuration: {best_config}")
```

### Manual Tuning Guidelines

#### Learning Rate

```python
# Learning rate scheduling
lr_schedules = {
    "constant": "Fixed learning rate",
    "step": "Step decay at fixed intervals",
    "cosine": "Cosine annealing (recommended)",
    "warmup_cosine": "Warmup + cosine annealing"
}

# Typical learning rates
lr_ranges = {
    "small_models": (1e-4, 5e-4),
    "large_models": (1e-5, 1e-4),
    "fine_tuning": (1e-6, 1e-5)
}
```

#### Batch Size

```python
# Batch size considerations
batch_size_guidelines = {
    "memory_limited": 4,      # 8GB VRAM
    "standard": 16,           # 16GB VRAM
    "large_scale": 32,        # 24GB+ VRAM
    "distributed": 64         # Multi-GPU
}
```

## Advanced Training Techniques

### Curriculum Learning

```python
from opencontrol.training import CurriculumLearner

# Curriculum learning for complex tasks
curriculum = CurriculumLearner(
    stages=[
        {"name": "basic", "prediction_horizon": 4, "epochs": 20},
        {"name": "medium", "prediction_horizon": 8, "epochs": 30},
        {"name": "advanced", "prediction_horizon": 16, "epochs": 50}
    ]
)

# Apply curriculum to training
trainer.set_curriculum(curriculum)
```

### Domain Adaptation

```python
from opencontrol.training import DomainAdapter

# Sim-to-real domain adaptation
domain_adapter = DomainAdapter(
    source_domain="simulation",
    target_domain="real_robot",
    adaptation_method="dann",  # Domain Adversarial Neural Networks
    lambda_domain=0.1
)

# Add to training loop
trainer.add_domain_adaptation(domain_adapter)
```

### Few-Shot Learning

```python
from opencontrol.training import FewShotLearner

# Few-shot adaptation to new tasks
few_shot_learner = FewShotLearner(
    base_model=model,
    adaptation_method="maml",  # Model-Agnostic Meta-Learning
    inner_lr=0.01,
    num_inner_steps=5
)

# Adapt to new task with few examples
adapted_model = few_shot_learner.adapt(
    support_data=new_task_data,
    num_shots=5
)
```

## Monitoring and Debugging

### Training Monitoring

```python
# Weights & Biases integration
import wandb

wandb.init(
    project="opencontrol-world-models",
    config={
        "model": "multimodal_world_model",
        "dataset": "robot_demonstrations",
        "batch_size": 16,
        "learning_rate": 1e-4
    }
)

# Log training metrics
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "learning_rate": lr,
    "epoch": epoch
})
```

### Debugging Tools

```python
from opencontrol.debugging import ModelDebugger

# Debug model training
debugger = ModelDebugger(model, dataset)

# Check for common issues
debugger.check_gradients()      # Gradient flow
debugger.check_activations()    # Activation statistics
debugger.check_data_loading()   # Data loading bottlenecks
debugger.profile_training()     # Training performance
```

## Model Deployment

### Model Export

```python
# Export trained model
trainer.export_model(
    path="models/world_model.onnx",
    format="onnx",
    optimize=True,
    quantize=True
)

# PyTorch JIT export
torch.jit.save(
    torch.jit.script(model),
    "models/world_model_jit.pt"
)
```

### Inference Optimization

```python
from opencontrol.deployment import ModelOptimizer

# Optimize for inference
optimizer = ModelOptimizer(model)

# Apply optimizations
optimized_model = optimizer.optimize(
    techniques=[
        "quantization",      # INT8 quantization
        "pruning",          # Weight pruning
        "distillation",     # Knowledge distillation
        "tensorrt"          # TensorRT optimization
    ]
)
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```python
# Solutions for OOM errors
memory_solutions = {
    "reduce_batch_size": "Decrease batch_size",
    "gradient_checkpointing": "Enable gradient checkpointing",
    "mixed_precision": "Use automatic mixed precision",
    "gradient_accumulation": "Accumulate gradients over steps"
}

# Example: Gradient accumulation
trainer.config.update({
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size: 16
    "mixed_precision": True
})
```

#### Slow Training

```python
# Performance optimization
performance_tips = {
    "dataloader_workers": "Increase num_workers",
    "pin_memory": "Enable pin_memory=True",
    "compile_model": "Use torch.compile (PyTorch 2.0+)",
    "persistent_workers": "Enable persistent_workers=True"
}
```

#### Poor Convergence

```python
# Convergence issues
convergence_solutions = {
    "learning_rate": "Adjust learning rate",
    "gradient_clipping": "Add gradient clipping",
    "warmup": "Add learning rate warmup",
    "regularization": "Reduce regularization"
}
```

## Best Practices

### Data Best Practices

1. **Diverse Data**: Collect data from various scenarios and conditions
2. **Quality Control**: Remove corrupted or low-quality samples
3. **Balanced Dataset**: Ensure balanced representation of different tasks
4. **Data Augmentation**: Use appropriate augmentation techniques
5. **Validation Split**: Keep validation data completely separate

### Training Best Practices

1. **Start Small**: Begin with smaller models and scale up
2. **Monitor Overfitting**: Use validation metrics to detect overfitting
3. **Save Checkpoints**: Regular checkpointing for recovery
4. **Reproducibility**: Set random seeds for reproducible results
5. **Documentation**: Document experiments and configurations

### Model Best Practices

1. **Architecture Selection**: Choose appropriate model size for your data
2. **Regularization**: Apply appropriate regularization techniques
3. **Loss Weighting**: Balance multi-modal losses carefully
4. **Evaluation**: Use comprehensive evaluation metrics
5. **Ablation Studies**: Understand component contributions

## Next Steps

After training your world model:

1. **Evaluate Performance**: Run comprehensive evaluation on test data
2. **Deploy for Control**: Integrate with MPC controller
3. **Fine-tune**: Adapt model for specific tasks or robots
4. **Scale Up**: Train larger models with more data
5. **Contribute**: Share your trained models with the community

## Resources

- [Model Architecture Documentation](models.md)
- [MPC Controller Integration](visual_mpc.md)
- [Evaluation Metrics Guide](evaluation.md)
- [Deployment Guide](deployment.md)

---

**Author**: Nik Jois (nikjois@llamasearch.ai)  
**Last Updated**: December 2024 