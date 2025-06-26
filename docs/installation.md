# Installation Guide

This guide provides complete installation instructions for OpenControl, covering everything from basic setup to advanced robotics integrations.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)

### Hardware Requirements (for Robotics)

- **Robot Platform**: Supported robot arm or mobile platform
- **Sensors**: RGB-D camera, force/torque sensor (optional)
- **Network**: Ethernet connection for robot communication
- **Real-time Capabilities**: RT kernel recommended for critical applications

## Basic Installation

### 1. Clone the Repository

```bash
git clone https://github.com/llamasearchai/OpenControl.git
cd OpenControl
```

### 2. Create Virtual Environment

#### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Using conda

```bash
conda create -n opencontrol python=3.11
conda activate opencontrol
```

#### Using venv

```bash
python -m venv opencontrol
source opencontrol/bin/activate  # On Windows: opencontrol\Scripts\activate
```

### 3. Install Dependencies

#### Basic Installation

```bash
uv pip install -e .
```

#### Development Installation

```bash
uv pip install -e ".[dev]"
```

#### Full Installation (with Robotics)

```bash
uv pip install -e ".[dev,robotics,monitoring,visualization]"
```

### 4. Verify Installation

```bash
# Run basic tests
pytest tests/unit/

# Check CLI
opencontrol --help

# Verify GPU support (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## GPU Setup

### NVIDIA CUDA

1. **Install CUDA Toolkit**:
   ```bash
   # Ubuntu
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. **Install cuDNN**:
   ```bash
   # Download from NVIDIA Developer website
   # Extract and copy files to CUDA directory
   ```

3. **Verify Installation**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

### AMD ROCm (Alternative)

```bash
# Ubuntu
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms
```

## Robotics Installation

### Robot Operating System (ROS)

#### ROS 2 Humble (Recommended)

```bash
# Ubuntu 22.04
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
```

#### ROS 1 Noetic (Legacy Support)

```bash
# Ubuntu 20.04
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full
```

### Robot-Specific Drivers

#### Universal Robots

```bash
# UR Modern Driver
sudo apt install ros-humble-ur-robot-driver

# Real-time kernel (recommended)
sudo apt install linux-lowlatency
```

#### Franka Emika Panda

```bash
# Install libfranka
git clone --recursive https://github.com/frankaemika/libfranka.git
cd libfranka
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install

# Install franka_ros
sudo apt install ros-humble-franka-ros
```

#### Kinova Arms

```bash
# Install Kinova API
wget https://artifactory.kinovaapps.com/artifactory/generic-public/cortex/API/2.6.0/kinova-api_2.6.0_amd64.deb
sudo dpkg -i kinova-api_2.6.0_amd64.deb
```

### Sensor Drivers

#### Intel RealSense

```bash
# Install RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev
```

#### Azure Kinect

```bash
# Install Azure Kinect SDK
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt update
sudo apt install k4a-tools libk4a1.4-dev
```

## Docker Installation

### Build Docker Image

```bash
# Basic image
docker build -t opencontrol:latest .

# Development image
docker build -f docker/Dockerfile.dev -t opencontrol:dev .

# GPU-enabled image
docker build -f docker/Dockerfile.gpu -t opencontrol:gpu .
```

### Run Docker Container

```bash
# Basic container
docker run -it --rm opencontrol:latest

# With GPU support
docker run -it --rm --gpus all opencontrol:gpu

# With robot hardware access
docker run -it --rm --privileged --network host opencontrol:latest
```

### Docker Compose

```bash
# Start full stack
docker-compose up -d

# Start with GPU support
docker-compose -f docker-compose.gpu.yml up -d
```

## Development Setup

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### IDE Configuration

#### VS Code

1. Install recommended extensions:
   - Python
   - Pylance
   - Black Formatter
   - GitLens

2. Configure settings (`.vscode/settings.json`):
   ```json
   {
     "python.defaultInterpreterPath": ".venv/bin/python",
     "python.formatting.provider": "black",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": false,
     "python.linting.flake8Enabled": true
   }
   ```

#### PyCharm

1. Configure Python interpreter to use virtual environment
2. Enable Black formatter
3. Configure pytest as test runner

### Testing Setup

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/robotics/      # Robot-specific tests

# Run with coverage
pytest --cov=opencontrol --cov-report=html

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Reinstall in development mode
pip uninstall opencontrol
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### CUDA Issues

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Robot Connection Issues

```bash
# Check network connectivity
ping <robot_ip>

# Verify robot is in correct mode
# Check robot teach pendant/interface

# Test with robot-specific tools
# UR: URSim simulator
# Franka: Desk interface
```

### Performance Optimization

#### System Tuning

```bash
# Increase shared memory (for large datasets)
sudo sysctl -w kernel.shmmax=68719476736

# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### Real-time Setup

```bash
# Install RT kernel
sudo apt install linux-lowlatency

# Configure RT priorities
sudo groupadd realtime
sudo usermod -a -G realtime $USER

# Add to /etc/security/limits.conf:
# @realtime soft rtprio 99
# @realtime soft priority 99
# @realtime soft memlock 102400
# @realtime hard rtprio 99
# @realtime hard priority 99
# @realtime hard memlock 102400
```

## Next Steps

After successful installation:

1. **Complete the [Robotics Tutorial](robotics_tutorial.md)** - Learn robot integration
2. **Try [Quick Start Examples](quickstart.md)** - Run sample code
3. **Configure [Hardware Setup](hardware_setup.md)** - Set up your robot
4. **Read [Training Guide](training.md)** - Train your first model

## Support

If you encounter issues:

- Check the [FAQ](faq.md)
- Search [GitHub Issues](https://github.com/llamasearchai/OpenControl/issues)
- Join our [Discord Community](https://discord.gg/opencontrol)
- Email support: nikjois@llamasearch.ai

---

**Author**: Nik Jois (nikjois@llamasearch.ai)  
**Last Updated**: December 2024 