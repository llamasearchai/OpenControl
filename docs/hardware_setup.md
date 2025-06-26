# Hardware Setup Guide

This guide covers the complete hardware setup process for integrating OpenControl with various robotic systems.

## Overview

OpenControl supports a wide range of robotic hardware configurations, from single-arm setups to complex multi-robot systems. This guide provides step-by-step instructions for setting up your hardware components.

## Supported Hardware

### Robot Arms

#### Universal Robots (UR Series)
- **Models**: UR3, UR3e, UR5, UR5e, UR10, UR10e, UR16e
- **Communication**: Ethernet TCP/IP
- **Control Frequency**: Up to 500Hz
- **Software**: URScript, Real-Time Data Exchange (RTDE)

#### Franka Emika Panda
- **Models**: Panda, Research 3
- **Communication**: Ethernet (Franka Control Interface)
- **Control Frequency**: 1000Hz
- **Software**: libfranka, franka_ros

#### Kinova Arms
- **Models**: Gen2, Gen3, MOVO
- **Communication**: USB, Ethernet
- **Control Frequency**: 200Hz
- **Software**: Kinova API, kortex_driver

#### ABB Robots
- **Models**: IRB series, YuMi
- **Communication**: Ethernet/IP, Robot Web Services
- **Control Frequency**: 250Hz
- **Software**: RobotStudio, ABB Robot Web Services

### Mobile Robots

#### TurtleBot Series
- **Models**: TurtleBot 2, 3, 4
- **Base**: Kobuki, Waffle, Waffle Pi
- **Sensors**: LiDAR, RGB-D camera, IMU
- **Software**: ROS/ROS2 navigation stack

#### Clearpath Robotics
- **Models**: Husky, Jackal, Ridgeback
- **Communication**: Ethernet, WiFi
- **Sensors**: GPS, IMU, cameras, LiDAR
- **Software**: ROS/ROS2 drivers

### End Effectors

#### Robotiq Grippers
- **Models**: 2F-85, 2F-140, 3F series, Hand-E
- **Communication**: Modbus RTU, Ethernet/IP
- **Force Control**: Adaptive grasping
- **Integration**: Direct robot mounting

#### Schunk Grippers
- **Models**: EGP, EGK, PGN series
- **Communication**: EtherCAT, Profinet
- **Precision**: High-precision positioning
- **Integration**: Tool changer compatible

### Sensors

#### Vision Sensors
- **Intel RealSense**: D435, D455, L515
- **Microsoft Azure Kinect**: DK
- **Zed Cameras**: Zed, Zed 2, Zed X
- **Industrial Cameras**: Basler, FLIR, IDS

#### Force/Torque Sensors
- **ATI**: Nano, Mini, Gamma series
- **Robotiq**: FT 300, FT 150
- **OnRobot**: HEX series
- **Weiss Robotics**: KMS series

#### Tactile Sensors
- **SynTouch**: BioTac sensors
- **Shadow Robot**: Tactile fingertips
- **Digit**: High-resolution tactile sensing
- **GelSight**: Vision-based tactile sensors

## Network Configuration

### Basic Network Setup

```bash
# Configure robot network interface
sudo ip addr add 192.168.1.10/24 dev eth0
sudo ip link set eth0 up

# Set up static routes (if needed)
sudo ip route add 192.168.1.0/24 via 192.168.1.1

# Configure firewall rules
sudo ufw allow from 192.168.1.0/24
sudo ufw allow out to 192.168.1.0/24
```

### Network Topology Examples

#### Single Robot Setup
```
Computer (192.168.1.10) ←→ Robot (192.168.1.100)
```

#### Multi-Robot Setup
```
Computer (192.168.1.10)
    ↓
Switch (192.168.1.1)
    ├── Robot 1 (192.168.1.100)
    ├── Robot 2 (192.168.1.101)
    └── Sensors (192.168.1.200+)
```

### Real-Time Network Configuration

```bash
# Optimize network for real-time communication
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 65536 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Disable network power management
sudo ethtool -s eth0 wol d
```

## Robot-Specific Setup

### Universal Robots Setup

#### Hardware Connections
1. **Power**: Connect robot to appropriate power supply
2. **Network**: Connect Ethernet cable to robot controller
3. **Emergency Stop**: Connect external e-stop if required
4. **Tool**: Mount gripper or tool with appropriate wiring

#### Software Configuration

```bash
# Install UR drivers
sudo apt install ros-humble-ur-robot-driver

# Configure robot IP
export UR_ROBOT_IP=192.168.1.100

# Test connection
ping $UR_ROBOT_IP
```

#### Teach Pendant Setup
1. Navigate to **Installation → Network**
2. Set **IP Address**: 192.168.1.100
3. Set **Subnet Mask**: 255.255.255.0
4. Set **Gateway**: 192.168.1.1
5. Enable **Remote Control**

#### Safety Configuration
```python
# Safety limits configuration
safety_config = {
    "joint_limits": {
        "velocity": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],  # rad/s
        "acceleration": [15, 15, 15, 15, 15, 15]  # rad/s²
    },
    "cartesian_limits": {
        "velocity": 1.0,      # m/s
        "acceleration": 5.0,  # m/s²
        "force": 150.0        # N
    },
    "workspace": {
        "x": [-0.8, 0.8],
        "y": [-0.8, 0.8],
        "z": [0.0, 1.2]
    }
}
```

### Franka Panda Setup

#### Hardware Connections
1. **Power**: Connect to Franka Control Unit
2. **Network**: Connect computer directly to Franka Control Interface
3. **Emergency Stop**: Connect shop floor e-stop
4. **Tool**: Install tool with proper mounting and wiring

#### Network Configuration
```bash
# Configure network for Franka
sudo ip addr add 172.16.0.1/16 dev eth0
sudo ip link set eth0 up

# Test Franka connection
ping 172.16.0.2  # Franka robot IP
```

#### Software Installation
```bash
# Install libfranka
git clone --recursive https://github.com/frankaemika/libfranka.git
cd libfranka
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
make -j4
sudo make install

# Install franka_ros
sudo apt install ros-humble-franka-ros
```

#### Desk Configuration
1. Open Franka Desk in web browser: `https://172.16.0.2`
2. **Activate FCI**: Enable Franka Control Interface
3. **Set Collision Behavior**: Configure collision thresholds
4. **Calibrate Tool**: Set tool center point and mass properties

### Kinova Setup

#### Hardware Connections
1. **Power**: Connect to Kinova power adapter
2. **Communication**: USB or Ethernet connection
3. **Tool**: Mount gripper with Kinova interface

#### Software Installation
```bash
# Install Kinova API
wget https://artifactory.kinovaapps.com/artifactory/generic-public/cortex/API/2.6.0/kinova-api_2.6.0_amd64.deb
sudo dpkg -i kinova-api_2.6.0_amd64.deb

# Install ROS drivers
sudo apt install ros-humble-kinova-ros
```

## Sensor Setup

### Intel RealSense Cameras

#### Hardware Installation
1. **Mounting**: Secure camera to robot or fixed mount
2. **USB Connection**: Use USB 3.0 for best performance
3. **Calibration**: Ensure proper camera-to-robot calibration

#### Software Setup
```bash
# Install RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev

# Test camera
realsense-viewer
```

#### Camera Configuration
```python
from opencontrol.sensors import RealSenseCamera

camera = RealSenseCamera(
    device_id=0,
    width=640,
    height=480,
    fps=30,
    enable_color=True,
    enable_depth=True,
    enable_infrared=False,
    depth_units=0.001,  # mm to m conversion
    clipping_distance=3.0  # meters
)
```

### Force/Torque Sensors

#### ATI Force/Torque Sensor Setup
```bash
# Install ATI drivers
# Download from ATI website and follow installation instructions

# Configure network interface
sudo ip addr add 192.168.1.10/24 dev eth0
```

```python
from opencontrol.sensors import ATIForceTorqueSensor

ft_sensor = ATIForceTorqueSensor(
    ip_address="192.168.1.101",
    calibration_file="FT12345.cal",
    sample_rate=1000,  # Hz
    filter_frequency=50  # Hz
)
```

#### Robotiq FT 300 Setup
```python
from opencontrol.sensors import RobotiqFT300

ft_sensor = RobotiqFT300(
    device="/dev/ttyUSB0",  # or IP address for network version
    baud_rate=115200,
    timeout=1.0
)
```

## Gripper Setup

### Robotiq 2F-85 Gripper

#### Hardware Installation
1. **Mounting**: Attach to robot flange using adapter plate
2. **Wiring**: Connect power and communication cables
3. **Tool Center Point**: Configure TCP in robot software

#### Software Configuration
```python
from opencontrol.grippers import Robotiq2F85

gripper = Robotiq2F85(
    device="/dev/ttyUSB0",  # or Modbus TCP IP
    baud_rate=115200,
    stroke=85,  # mm
    force_limit=235  # N
)

# Initialize and test
gripper.connect()
gripper.activate()
gripper.open()
gripper.close()
```

### Schunk EGP Gripper

#### Hardware Installation
1. **Mounting**: Use standard ISO flange
2. **Communication**: EtherCAT or Profinet connection
3. **Safety**: Configure safety functions

#### Software Configuration
```python
from opencontrol.grippers import SchunkEGP

gripper = SchunkEGP(
    ip_address="192.168.1.102",
    stroke=64,  # mm
    force_limit=140  # N
)
```

## Calibration Procedures

### Hand-Eye Calibration

#### Camera-to-Robot Calibration
```python
from opencontrol.calibration import HandEyeCalibration

# Collect calibration data
calibrator = HandEyeCalibration(
    robot=robot,
    camera=camera,
    calibration_target="checkerboard",
    target_size=(9, 6),
    square_size=0.025  # meters
)

# Collect calibration poses
poses = []
for i in range(20):
    # Move robot to calibration pose
    robot.move_to_pose(calibration_poses[i])
    
    # Capture image and detect target
    image = camera.get_frame().color_image
    pose_data = calibrator.detect_target(image)
    
    if pose_data is not None:
        poses.append({
            "robot_pose": robot.get_pose(),
            "target_pose": pose_data
        })

# Perform calibration
transformation = calibrator.calibrate(poses)
camera.set_extrinsics(transformation)
```

### Force Sensor Calibration

#### Bias Removal and Gravity Compensation
```python
from opencontrol.calibration import ForceSensorCalibration

# Initialize calibration
ft_calibrator = ForceSensorCalibration(
    force_sensor=ft_sensor,
    robot=robot
)

# Remove bias (no external forces)
ft_calibrator.remove_bias()

# Gravity compensation
tool_mass = 0.5  # kg
tool_com = [0.0, 0.0, 0.1]  # meters from sensor frame

ft_calibrator.set_tool_parameters(
    mass=tool_mass,
    center_of_mass=tool_com
)
```

## Safety Systems

### Emergency Stop Configuration

#### Hardware E-Stop
```python
from opencontrol.safety import EmergencyStopMonitor

# Configure emergency stop monitoring
estop_monitor = EmergencyStopMonitor(
    estop_pin=18,  # GPIO pin
    robot=robot,
    sensors=[camera, ft_sensor],
    safe_stop_mode="category_1"  # IEC 60204-1
)

estop_monitor.start_monitoring()
```

#### Software Safety Limits
```python
from opencontrol.safety import SafetyMonitor

safety_monitor = SafetyMonitor(
    robot=robot,
    limits={
        "workspace": {
            "x": [-0.8, 0.8],
            "y": [-0.8, 0.8],
            "z": [0.0, 1.2]
        },
        "velocity": 0.5,      # m/s
        "acceleration": 2.0,  # m/s²
        "force": 50.0,        # N
        "torque": 10.0        # Nm
    }
)
```

### Collision Detection

#### Vision-Based Collision Detection
```python
from opencontrol.safety import VisionCollisionDetector

collision_detector = VisionCollisionDetector(
    camera=camera,
    robot=robot,
    safety_distance=0.1,  # meters
    detection_frequency=30  # Hz
)

# Monitor for collisions
while robot.is_moving():
    if collision_detector.check_collision():
        robot.emergency_stop()
        break
```

## Workspace Setup

### Physical Workspace Configuration

#### Table and Mounting
1. **Stable Surface**: Use vibration-damped table
2. **Robot Mounting**: Secure robot base with appropriate bolts
3. **Cable Management**: Route cables safely away from robot motion
4. **Lighting**: Ensure adequate and consistent lighting for vision

#### Safety Barriers
1. **Physical Barriers**: Install safety fencing if required
2. **Light Curtains**: For collaborative applications
3. **Safety Mats**: Pressure-sensitive floor mats
4. **Warning Signs**: Appropriate safety signage

### Software Workspace Definition

```python
from opencontrol.workspace import WorkspaceManager

# Define workspace zones
workspace = WorkspaceManager()

# Safe zone for normal operation
workspace.add_zone(
    name="safe_zone",
    bounds={
        "x": [-0.6, 0.6],
        "y": [-0.6, 0.6],
        "z": [0.1, 0.8]
    },
    max_velocity=0.5
)

# Restricted zone near humans
workspace.add_zone(
    name="collaborative_zone",
    bounds={
        "x": [0.2, 0.6],
        "y": [-0.2, 0.2],
        "z": [0.1, 0.5]
    },
    max_velocity=0.1,
    force_limit=25.0
)

# Forbidden zone
workspace.add_zone(
    name="forbidden_zone",
    bounds={
        "x": [-0.8, -0.6],
        "y": [-0.8, 0.8],
        "z": [0.0, 1.2]
    },
    access="forbidden"
)
```

## Testing and Validation

### Hardware Tests

#### Connectivity Test
```python
def test_hardware_connectivity():
    """Test all hardware connections."""
    results = {}
    
    # Test robot connection
    try:
        robot.connect()
        robot.get_joint_positions()
        results["robot"] = "PASS"
    except Exception as e:
        results["robot"] = f"FAIL: {e}"
    
    # Test camera
    try:
        camera.start()
        frame = camera.get_frame()
        results["camera"] = "PASS"
    except Exception as e:
        results["camera"] = f"FAIL: {e}"
    
    # Test force sensor
    try:
        ft_sensor.start()
        reading = ft_sensor.get_reading()
        results["force_sensor"] = "PASS"
    except Exception as e:
        results["force_sensor"] = f"FAIL: {e}"
    
    return results
```

#### Motion Test
```python
def test_robot_motion():
    """Test robot motion capabilities."""
    # Joint space motion test
    home_joints = [0, -1.57, 0, -1.57, 0, 0, 0]
    robot.move_to_joints(home_joints)
    
    # Cartesian motion test
    poses = [
        Pose([0.4, 0.2, 0.3], [0, 1, 0, 0]),
        Pose([0.4, 0.0, 0.3], [0, 1, 0, 0]),
        Pose([0.4, -0.2, 0.3], [0, 1, 0, 0])
    ]
    
    for pose in poses:
        robot.move_to_pose(pose)
        time.sleep(1.0)
```

### Calibration Validation

#### Accuracy Test
```python
def validate_calibration():
    """Validate hand-eye calibration accuracy."""
    # Known target positions
    target_positions = [
        [0.4, 0.2, 0.1],
        [0.4, 0.0, 0.1],
        [0.4, -0.2, 0.1]
    ]
    
    errors = []
    for target_pos in target_positions:
        # Move robot to look at target
        robot.move_to_pose(Pose(target_pos + [0, 0, 0.2], [0, 1, 0, 0]))
        
        # Detect target in camera
        image = camera.get_frame().color_image
        detected_pos = detect_target_position(image)
        
        # Calculate error
        error = np.linalg.norm(np.array(target_pos) - np.array(detected_pos))
        errors.append(error)
    
    mean_error = np.mean(errors)
    print(f"Mean calibration error: {mean_error:.3f} m")
    
    return mean_error < 0.005  # 5mm tolerance
```

## Troubleshooting

### Common Hardware Issues

#### Robot Connection Problems
```bash
# Check network connectivity
ping <robot_ip>

# Check robot status
# For UR robots:
# - Check teach pendant for errors
# - Verify remote control is enabled
# - Check safety system status

# For Franka robots:
# - Check Franka Desk for errors
# - Verify FCI is activated
# - Check collision detection status
```

#### Sensor Issues
```bash
# Camera problems
lsusb  # Check USB connection
v4l2-ctl --list-devices  # List video devices

# Force sensor problems
dmesg | grep tty  # Check serial connections
netstat -an | grep <port>  # Check network connections
```

#### Gripper Problems
```python
# Diagnostics for Robotiq grippers
gripper.get_status()  # Check gripper status
gripper.reset()       # Reset gripper
gripper.activate()    # Reactivate gripper
```

### Performance Optimization

#### Real-Time Performance
```bash
# Install real-time kernel
sudo apt install linux-lowlatency

# Configure CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set process priorities
sudo chrt -f 80 python robot_control.py
```

#### Network Optimization
```bash
# Optimize network buffers
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Disable network interrupt coalescing
sudo ethtool -C eth0 rx-usecs 0 tx-usecs 0
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Checks
- Verify robot emergency stop functionality
- Check cable connections and routing
- Test basic robot motions
- Verify sensor readings

#### Weekly Checks
- Clean camera lenses and sensors
- Check robot calibration accuracy
- Verify safety system functionality
- Update system logs

#### Monthly Checks
- Perform full system calibration
- Update robot firmware if needed
- Check mechanical wear and tear
- Review safety procedures

### Preventive Maintenance

#### Robot Maintenance
```python
# Automated health monitoring
def monitor_robot_health():
    """Monitor robot health metrics."""
    health_data = {
        "joint_temperatures": robot.get_joint_temperatures(),
        "motor_currents": robot.get_motor_currents(),
        "error_log": robot.get_error_log(),
        "runtime_hours": robot.get_runtime_hours()
    }
    
    # Check for anomalies
    for joint, temp in enumerate(health_data["joint_temperatures"]):
        if temp > 70:  # Celsius
            logger.warning(f"Joint {joint} temperature high: {temp}°C")
    
    return health_data
```

## Next Steps

After completing hardware setup:

1. **System Integration**: Test complete system integration
2. **Calibration**: Perform comprehensive calibration procedures
3. **Safety Validation**: Validate all safety systems
4. **Performance Testing**: Measure system performance metrics
5. **Documentation**: Document your specific hardware configuration

## Resources

- [Robot Manufacturer Documentation]
- [Sensor Calibration Guide](calibration.md)
- [Safety Systems Guide](safety.md)
- [Troubleshooting Guide](troubleshooting.md)

---

**Author**: Nik Jois (nikjois@llamasearch.ai)  
**Last Updated**: December 2024 