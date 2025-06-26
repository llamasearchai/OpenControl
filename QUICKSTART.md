# OpenControl Quick Start Guide

Welcome to **OpenControl**, a state-of-the-art multi-modal world model platform for embodied AI!

## 🚀 Quick Installation

```bash
# Clone the repository
git clone https://github.com/nikjois/opencontrol.git
cd opencontrol

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install OpenControl
pip install -e ".[dev]"

# Verify installation
python test_complete_system.py
```

## 🎯 Core Features

- **Advanced World Model**: 3B+ parameter transformer with temporal dynamics
- **Real-time MPC Control**: Sub-30ms inference for robotics applications
- **Multi-Modal Processing**: Video, audio, text, actions, and proprioception
- **Production Ready**: Docker, monitoring, and deployment infrastructure
- **Interactive CLI**: Rich terminal interface with live dashboards

## 🏃‍♂️ Quick Examples

### 1. Interactive Dashboard
```bash
opencontrol interactive
```

### 2. Train a Model
```bash
# Development training (fast)
opencontrol train --config configs/models/development.yaml

# Production training (full scale)
opencontrol train --config configs/models/production.yaml
```

### 3. Run Evaluation
```bash
opencontrol evaluate --checkpoint checkpoints/best_model.pt
```

### 4. Start Model Server
```bash
opencontrol serve --port 8000
```

### 5. System Information
```bash
opencontrol info
```

## 📁 Project Structure

```
opencontrol/
├── opencontrol/           # Core package
│   ├── core/             # World model components
│   ├── control/          # MPC and control systems
│   ├── training/         # Training infrastructure
│   ├── evaluation/       # Metrics and benchmarks
│   ├── data/            # Data loading and processing
│   ├── deployment/      # Production deployment
│   └── cli/             # Command-line interface
├── configs/             # Configuration files
├── scripts/             # Utility scripts
├── tests/               # Test suites
└── docs/                # Documentation
```

## 🔧 Configuration

OpenControl uses YAML configuration files. Key configurations:

- `configs/models/development.yaml` - Fast development setup
- `configs/models/production.yaml` - Full-scale production
- `configs/models/test.yaml` - Minimal testing setup

### Example Configuration
```yaml
model:
  model_dim: 2048
  num_layers: 32
  video_encoder: "vit_huge_patch14_224"
  action_dim: 24

training:
  batch_size: 256
  learning_rate: 3.0e-4
  num_epochs: 100

control:
  horizon: 50
  num_samples: 10000
```

## 🧪 Development

### Running Tests
```bash
# Full test suite
pytest tests/

# System integration test
python test_complete_system.py

# Specific test categories
pytest tests/test_core.py
pytest tests/test_training.py
```

### Code Quality
```bash
# Format code
black opencontrol/
isort opencontrol/

# Type checking
mypy opencontrol/

# Linting
ruff opencontrol/
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t opencontrol:latest .

# Run training
docker run --gpus all -v $(pwd)/data:/data opencontrol:latest \
  opencontrol train --config /app/configs/models/production.yaml

# Run server
docker run -p 8000:8000 opencontrol:latest \
  opencontrol serve --host 0.0.0.0 --port 8000
```

## 📊 Monitoring

OpenControl includes comprehensive monitoring:

- **Weights & Biases**: Training metrics and visualization
- **Production Metrics**: Real-time performance monitoring
- **Health Checks**: System status and diagnostics

## 🔗 API Usage

### Python API
```python
from opencontrol.core.world_model import OpenControlWorldModel
from opencontrol.cli.commands import get_dev_config

# Load model
config = get_dev_config()
model = OpenControlWorldModel(config)

# Make predictions
outputs = model(inputs, prediction_horizon=10)
```

### REST API
```bash
# Start server
opencontrol serve --port 8000

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"video": [...], "prediction_horizon": 5}'
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Data Loading Errors**
   - Check data path in configuration
   - Verify episode file format (.npz)
   - Ensure sufficient disk space

3. **Import Errors**
   - Install with `pip install -e ".[dev]"`
   - Check Python version (3.9+)
   - Verify all dependencies installed

### Performance Tips

- Use GPU for training and inference
- Enable mixed precision for faster training
- Adjust num_workers for data loading
- Use distributed training for large models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/nikjois/opencontrol/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nikjois/opencontrol/discussions)
- **Email**: nikjois@llamasearch.ai

---

**Built with ❤️ by Nik Jois**

*OpenControl: Advancing the frontier of embodied AI* 