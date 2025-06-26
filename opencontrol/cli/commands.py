"""
OpenControl Configuration Management
Defines the master configuration object for the entire system, loaded from YAML.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the world model architecture."""
    
    model_type: str = "opencontrol_production"
    model_dim: int = 2048
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32768
    max_sequence_length: int = 2048
    
    # Encoder configurations
    video_encoder: str = "vit_huge_patch14_224"
    audio_encoder: str = "wav2vec2_large_960h"
    text_encoder: str = "clip_vit_large_patch14"
    
    # Dimensions
    action_dim: int = 24
    proprioception_dim: int = 48
    
    # Training settings
    dropout: float = 0.1
    use_gradient_checkpointing: bool = True
    uncertainty_estimation: bool = True
    
    # Input specifications
    video_resolution: List[int] = field(default_factory=lambda: [224, 224])
    audio_sample_rate: int = 16000


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Optimizer settings
    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    
    # Scheduler settings
    lr_schedule: str = "cosine_with_warmup"
    warmup_steps: int = 5000
    
    # Batch and epoch settings
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    num_epochs: int = 100
    steps_per_epoch: int = 5000
    
    # Performance optimizations
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Checkpointing
    save_interval: int = 1000
    max_checkpoints: int = 5


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    data_path: str = "/data/episodes"
    sequence_length: int = 256
    video_resolution: List[int] = field(default_factory=lambda: [224, 224])
    audio_sample_rate: int = 16000
    
    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Data augmentation
    enable_augmentation: bool = True
    video_jitter: float = 0.1
    audio_noise: float = 0.01


@dataclass
class ControlConfig:
    """Configuration for MPC control system."""
    
    # MPC parameters
    horizon: int = 50
    num_samples: int = 10000
    num_iterations: int = 5
    elite_fraction: float = 0.1
    
    # Real-time constraints
    control_frequency: float = 30.0
    action_bounds: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    
    # Cost function weights
    cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "state": 1.0,
        "action": 0.01,
        "smoothness": 0.5,
        "goal": 10.0,
        "safety": 100.0
    })


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure and deployment."""
    
    # Distributed training
    world_size: int = 1
    distributed_backend: str = "nccl"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_steps: int = 5000
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # Monitoring
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class DeploymentConfig:
    """Configuration for deployment and serving."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_batch_size: int = 32
    timeout: int = 30
    
    # Model serving
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    
    # API settings
    enable_docs: bool = True
    enable_metrics: bool = True


@dataclass
class OpenControlConfig:
    """Master configuration for the OpenControl system."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "OpenControlConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenControlConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'control' in data:
            config.control = ControlConfig(**data['control'])
        if 'infrastructure' in data:
            config.infrastructure = InfrastructureConfig(**data['infrastructure'])
        if 'deployment' in data:
            config.deployment = DeploymentConfig(**data['deployment'])
            
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'control': self.control.__dict__,
            'infrastructure': self.infrastructure.__dict__,
            'deployment': self.deployment.__dict__,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        if self.model.model_dim % self.model.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        
        # Training validation
        if self.training.batch_size % self.infrastructure.world_size != 0:
            raise ValueError("batch_size must be divisible by world_size")
        
        # Control validation
        if not 0 < self.control.elite_fraction < 1:
            raise ValueError("elite_fraction must be between 0 and 1")
        
        # Data validation
        if not Path(self.data.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data.data_path}")


# Default configurations for different use cases
def get_default_config() -> OpenControlConfig:
    """Get the default configuration."""
    return OpenControlConfig()


def get_dev_config() -> OpenControlConfig:
    """Get a development configuration with smaller models and faster training."""
    config = OpenControlConfig()
    
    # Smaller model for development
    config.model.model_dim = 512
    config.model.num_layers = 8
    config.model.num_heads = 8
    config.model.video_resolution = [64, 64]
    
    # Faster training
    config.training.batch_size = 32
    config.training.num_epochs = 10
    config.training.steps_per_epoch = 100
    
    # Smaller control settings
    config.control.horizon = 10
    config.control.num_samples = 1000
    config.control.num_iterations = 3
    
    return config


def get_test_config() -> OpenControlConfig:
    """Get a minimal configuration for testing."""
    config = OpenControlConfig()
    
    # Minimal model
    config.model.model_dim = 64
    config.model.num_layers = 2
    config.model.num_heads = 2
    config.model.video_resolution = [224, 224]  # Use full resolution for vision encoder
    config.model.action_dim = 8  # Match test dimensions
    config.model.proprioception_dim = 16  # Match test dimensions
    config.model.video_encoder = "mock"  # Use mock encoder for testing
    config.model.audio_encoder = "mock"  # Use mock encoder for testing
    config.model.text_encoder = "mock"   # Use mock encoder for testing
    
    # Fast training
    config.training.batch_size = 2
    config.training.num_epochs = 1
    config.training.steps_per_epoch = 5
    
    # Test data path
    config.data.data_path = "data/test_episodes"
    config.data.video_resolution = [224, 224]
    
    # Minimal control
    config.control.horizon = 5
    config.control.num_samples = 100
    config.control.num_iterations = 2
    
    return config 