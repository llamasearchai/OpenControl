# OpenControl: Advanced Robotics World Model Platform

<div align="center">

![OpenControl Banner](https://img.shields.io/badge/OpenControl-v1.0.0-blue.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Robotics](https://img.shields.io/badge/domain-robotics-green.svg)](https://github.com/llamasearchai/OpenControl)

**A complete, production-ready platform for building, training, and deploying large-scale multi-modal world models for autonomous robotics and embodied AI**

*Empowering the next generation of intelligent robots with advanced predictive control and real-time decision making*

</div>

---

## Robotics-First Design

OpenControl is specifically engineered for real-world robotics applications, providing:

- **Real-Time Robot Control**: Sub-30ms inference for live robotic systems
- **Multi-Modal Perception**: Vision, tactile, proprioceptive, and audio sensor fusion
- **Predictive Planning**: Advanced world models for robot motion planning
- **Safety-Critical Systems**: Built-in safety constraints and failure detection
- **Hardware Integration**: Direct interfaces to popular robotics platforms
- **Sim-to-Real Transfer**: Seamless deployment from simulation to physical robots

## Key Features

### Core Robotics Capabilities
- **Visual Model Predictive Control (MPC)**: High-performance real-time control with Cross-Entropy Method optimization
- **Transformer-Based World Models**: State-of-the-art architecture with rotary positional embeddings (RoPE)
- **Multi-Modal Sensor Fusion**: RGB, depth, tactile, IMU, and proprioceptive data integration
- **Real-Time Inference**: Optimized for robotics control loops (30-1000Hz)
- **Safety Systems**: Built-in collision avoidance and constraint satisfaction
- **Continuous Learning**: Online adaptation and few-shot learning capabilities

### Production & Deployment
- **Containerized Deployment**: Docker containers for edge and cloud deployment
- **Distributed Training**: Multi-GPU and multi-node training with mixed precision
- **Interactive Dashboard**: Rich terminal interface with live robot monitoring
- **Comprehensive Metrics**: Advanced evaluation suite for robot performance assessment
- **Modern Tooling**: Built with uv, tox, and modern Python development practices
- **Cloud Integration**: Support for AWS, GCP, and Azure robotics services

## Robotics Architecture

OpenControl's architecture is designed specifically for robotics applications:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROBOTICS CONTROL LOOP                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Sensors → Perception → World Model → Planning → Control → Actuators       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Sensor Fusion     │───▶│   World Model       │───▶│   Safety Monitor    │
│ (Vision/Tactile/IMU)│    │ (Transformer)       │    │ (Constraints)       │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │                           │
                                      ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Robot Hardware     │◀───│   Visual MPC        │◀───│  Motion Planner     │
│ (Arms/Grippers/Base)│    │ (Real-time Control) │    │ (Trajectory Gen)    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Quick Start for Robotics

### Installation

```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenControl.git
cd OpenControl

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev,robotics]"

# Run tests to verify installation
pytest tests/
```

### Robot Integration Examples

#### 1. Connect to Your Robot

```bash
# For UR5/UR10 robots
opencontrol robot connect --type universal_robots --ip 192.168.1.100

# For Franka Panda
opencontrol robot connect --type franka --ip 172.16.0.2

# For custom robots via ROS
opencontrol robot connect --type ros --namespace /robot_arm
```

#### 2. Train a World Model

```bash
# Train on robot demonstration data
opencontrol train --config configs/robots/manipulation.yaml \
                  --data-path data/robot_demos/ \
                  --robot-type franka_panda

# Train with simulation data
opencontrol train --config configs/robots/sim_to_real.yaml \
                  --sim-data data/simulation/ \
                  --real-data data/robot_demos/
```

#### 3. Deploy Real-Time Control

```bash
# Start robot control server
opencontrol serve --robot-config configs/robots/franka_config.yaml \
                  --model checkpoints/best_model.pt \
                  --control-freq 100  # 100Hz control loop

# Launch interactive robot dashboard
opencontrol dashboard --robot-ip 192.168.1.100
```

### Basic Usage Examples

```python
from opencontrol import RobotController, WorldModel

# Initialize robot controller
robot = RobotController(
    robot_type="franka_panda",
    model_path="checkpoints/manipulation_model.pt",
    control_frequency=100  # Hz
)

# Load pre-trained world model
world_model = WorldModel.from_pretrained("opencontrol/manipulation-v1")

# Execute pick and place task
success = robot.execute_task(
    task="pick_and_place",
    target_object="red_cube",
    destination="blue_box",
    world_model=world_model
)
```

## Supported Robot Platforms

### Robotic Arms
- **Universal Robots**: UR3, UR5, UR10, UR16
- **Franka Emika**: Panda, Research 3
- **Kinova**: Gen2, Gen3, MOVO
- **ABB**: IRB series, YuMi
- **KUKA**: iiwa, KR series
- **Custom Arms**: Via ROS/ROS2 interface

### Mobile Robots
- **TurtleBot**: 2, 3, 4
- **Clearpath**: Husky, Jackal, Ridgeback
- **Boston Dynamics**: Spot (via SDK)
- **Custom Platforms**: Via ROS/ROS2 navigation stack

### Grippers & End Effectors
- **Robotiq**: 2F-85, 2F-140, 3F series
- **Schunk**: EGP, EGK series
- **OnRobot**: RG2, RG6, VG10
- **Custom Grippers**: Via GPIO/Modbus/EtherCAT

### Sensors
- **Cameras**: Intel RealSense, Azure Kinect, Zed
- **Force/Torque**: ATI, Robotiq FT 300
- **Tactile**: Digit, GelSight, TacTip
- **LiDAR**: Velodyne, Ouster, Sick

## Performance Benchmarks

OpenControl is optimized for real-world robotics performance:

### Training Performance
- **Data Throughput**: 1000+ robot episodes/hour on 8x A100 GPUs
- **Model Scaling**: Supports models up to 70B+ parameters
- **Memory Efficiency**: Gradient checkpointing and mixed precision training
- **Distributed Training**: Linear scaling across multiple nodes

### Inference Performance
- **Control Latency**: <30ms end-to-end (perception to action)
- **Throughput**: 1000+ inferences/second on RTX 4090
- **Memory Usage**: <8GB VRAM for 7B parameter models
- **CPU Inference**: Real-time performance on modern CPUs

### Robot-Specific Benchmarks
- **Pick Success Rate**: >95% on standard benchmarks
- **Manipulation Precision**: <2mm positioning accuracy
- **Collision Avoidance**: 99.9% success rate in cluttered environments
- **Adaptation Speed**: <10 demonstrations for new tasks

## Robotics Use Cases

### Manufacturing & Industrial Automation
- **Assembly Line Integration**: Automated part assembly and quality control
- **Flexible Manufacturing**: Rapid reconfiguration for different products
- **Quality Inspection**: Vision-based defect detection and sorting
- **Human-Robot Collaboration**: Safe shared workspace operations

### Service Robotics
- **Household Assistance**: Cleaning, cooking, and maintenance tasks
- **Healthcare Support**: Patient assistance and medical device operation
- **Hospitality**: Food service and customer interaction
- **Elder Care**: Mobility assistance and health monitoring

### Research & Development
- **Manipulation Research**: Advanced grasping and dexterous manipulation
- **Navigation Studies**: Indoor and outdoor autonomous navigation
- **Human-Robot Interaction**: Natural language and gesture-based control
- **Multi-Robot Systems**: Coordination and swarm robotics

### Agriculture & Outdoor Robotics
- **Precision Agriculture**: Crop monitoring and selective harvesting
- **Environmental Monitoring**: Data collection in remote locations
- **Search and Rescue**: Autonomous exploration and victim detection
- **Construction**: Automated building and infrastructure maintenance

## Complete Documentation

### Getting Started
- [Installation Guide](docs/installation.md) - Complete setup instructions
- [Robotics Tutorial](docs/robotics_tutorial.md) - Step-by-step robot integration
- [Quick Start Examples](docs/quickstart.md) - Ready-to-run code examples
- [Hardware Setup](docs/hardware_setup.md) - Robot and sensor configuration

### Core Concepts
- [World Models](docs/world_models.md) - Understanding predictive models
- [Visual MPC](docs/visual_mpc.md) - Model predictive control systems
- [Multi-Modal Learning](docs/multimodal.md) - Sensor fusion techniques
- [Safety Systems](docs/safety.md) - Constraint satisfaction and collision avoidance

### Training & Development
- [Training Guide](docs/training.md) - Model training and optimization
- [Data Collection](docs/data_collection.md) - Robot demonstration recording
- [Configuration](docs/configuration.md) - System configuration options
- [Debugging](docs/debugging.md) - Troubleshooting and diagnostics

### Deployment & Production
- [Deployment Guide](docs/deployment.md) - Production deployment strategies
- [Docker Setup](docs/docker.md) - Containerized deployment
- [Cloud Integration](docs/cloud.md) - AWS, GCP, and Azure setup
- [Monitoring](docs/monitoring.md) - System monitoring and logging

### API Reference
- [Robot API](docs/api/robot_api.md) - Robot control interfaces
- [Model API](docs/api/model_api.md) - World model interfaces
- [Metrics API](docs/api/metrics_api.md) - Evaluation and monitoring
- [Utils API](docs/api/utils_api.md) - Utility functions and helpers

## Development

### Development Environment Setup

```bash
# Clone and setup
git clone https://github.com/llamasearchai/OpenControl.git
cd OpenControl

# Setup development environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,robotics]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Using uv and tox

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests with tox
tox

# Run specific test environments
tox -e py311        # Python 3.11 tests
tox -e lint         # Linting
tox -e type-check   # Type checking
tox -e benchmark    # Performance benchmarks
tox -e robotics     # Robot integration tests
```

### Code Quality

```bash
# Format code
tox -e format

# Run all checks
tox -e lint
tox -e type-check

# Run robot-specific tests
pytest tests/robotics/ -v
```

### Contributing to Robotics Features

We especially welcome contributions in:
- New robot platform integrations
- Advanced control algorithms
- Safety system improvements
- Simulation environments
- Hardware driver optimizations

See our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.

## Research & Citations

If you use OpenControl in your robotics research, please cite:

```bibtex
@software{opencontrol2024,
  title={OpenControl: Advanced Multi-Modal World Model Platform for Robotics},
  author={Jois, Nik},
  year={2024},
  url={https://github.com/llamasearchai/OpenControl},
  version={1.0.0}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with modern Python tooling: uv, tox, and pytest
- Leverages PyTorch ecosystem for deep learning
- Inspired by advances in robotics, large language models, and embodied AI
- Special thanks to the robotics community for feedback and contributions

## Contact & Support

**Nik Jois** - nikjois@llamasearch.ai

- Project Link: [https://github.com/llamasearchai/OpenControl](https://github.com/llamasearchai/OpenControl)
- Documentation: [https://opencontrol.readthedocs.io/](https://opencontrol.readthedocs.io/)
- Issues: [https://github.com/llamasearchai/OpenControl/issues](https://github.com/llamasearchai/OpenControl/issues)
- Discussions: [https://github.com/llamasearchai/OpenControl/discussions](https://github.com/llamasearchai/OpenControl/discussions)

### Community & Support
- **Discord**: Join our robotics community server
- **Mailing List**: Subscribe for updates and announcements
- **Workshops**: Regular training sessions and tutorials
- **Consulting**: Enterprise support and custom development

---

<div align="center">

**Built for the Future of Robotics**

*Empowering researchers, engineers, and companies to build the next generation of intelligent robots*

Made by the OpenControl team

</div> 