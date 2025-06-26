# Robotics Tutorial: Getting Started with OpenControl

This tutorial will guide you through integrating OpenControl with real robots, from basic setup to advanced control scenarios.

## Prerequisites

- Completed [Installation Guide](installation.md)
- Access to a supported robot platform
- Basic understanding of robotics concepts
- Python programming experience

## Tutorial Overview

1. **Robot Connection Setup**
2. **Basic Robot Control**
3. **Sensor Integration**
4. **World Model Training**
5. **Visual Model Predictive Control**
6. **Advanced Applications**

## Part 1: Robot Connection Setup

### Supported Robot Platforms

OpenControl supports a wide range of robot platforms:

- **Universal Robots**: UR3, UR5, UR10, UR16
- **Franka Emika**: Panda, Research 3
- **Kinova**: Gen2, Gen3, MOVO
- **Custom Robots**: Via ROS/ROS2 interface

### Example: Connecting to a Universal Robots UR5

#### Step 1: Network Configuration

```bash
# Set robot IP address (example)
export ROBOT_IP=192.168.1.100

# Test connectivity
ping $ROBOT_IP
```

#### Step 2: Robot Preparation

1. Power on the robot
2. Access the teach pendant
3. Navigate to Installation → Network
4. Set IP address to match your network
5. Enable remote control

#### Step 3: OpenControl Connection

```python
from opencontrol.robots import UniversalRobot

# Initialize robot connection
robot = UniversalRobot(
    ip_address="192.168.1.100",
    control_frequency=125,  # Hz
    safety_limits={
        "max_velocity": 0.5,    # m/s
        "max_acceleration": 1.0, # m/s²
        "workspace_limits": {
            "x": [-0.8, 0.8],
            "y": [-0.8, 0.8], 
            "z": [0.0, 1.2]
        }
    }
)

# Connect to robot
robot.connect()
print(f"Robot connected: {robot.is_connected()}")

# Get current pose
current_pose = robot.get_pose()
print(f"Current pose: {current_pose}")
```

### Example: Connecting to a Franka Panda

```python
from opencontrol.robots import FrankaPanda

# Initialize Franka robot
robot = FrankaPanda(
    ip_address="172.16.0.2",
    control_mode="position",  # or "velocity", "torque"
    safety_config={
        "collision_threshold": 20.0,
        "force_threshold": 10.0
    }
)

# Connect and activate FCI
robot.connect()
robot.activate_fci()

# Home the robot
robot.move_to_home()
```

## Part 2: Basic Robot Control

### Joint Space Control

```python
import numpy as np
from opencontrol.control import JointController

# Initialize joint controller
joint_controller = JointController(robot)

# Define target joint angles (radians)
target_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Move to target position
joint_controller.move_to_joints(
    target_joints,
    velocity=0.1,  # rad/s
    acceleration=0.5  # rad/s²
)

# Wait for motion to complete
joint_controller.wait_for_motion()
```

### Cartesian Space Control

```python
from opencontrol.control import CartesianController
from opencontrol.utils import Pose

# Initialize cartesian controller
cartesian_controller = CartesianController(robot)

# Define target pose
target_pose = Pose(
    position=[0.4, 0.2, 0.3],  # x, y, z in meters
    orientation=[0.0, 1.0, 0.0, 0.0]  # quaternion [x, y, z, w]
)

# Move to target pose
cartesian_controller.move_to_pose(
    target_pose,
    velocity=0.1,  # m/s
    acceleration=0.2  # m/s²
)
```

### Trajectory Following

```python
from opencontrol.planning import TrajectoryPlanner

# Initialize trajectory planner
planner = TrajectoryPlanner(robot)

# Define waypoints
waypoints = [
    Pose([0.4, 0.2, 0.3], [0.0, 1.0, 0.0, 0.0]),
    Pose([0.4, 0.0, 0.3], [0.0, 1.0, 0.0, 0.0]),
    Pose([0.4, -0.2, 0.3], [0.0, 1.0, 0.0, 0.0]),
]

# Plan trajectory
trajectory = planner.plan_cartesian_path(
    waypoints,
    velocity_scaling=0.1,
    acceleration_scaling=0.1
)

# Execute trajectory
robot.execute_trajectory(trajectory)
```

## Part 3: Sensor Integration

### RGB-D Camera Setup

```python
from opencontrol.sensors import RealSenseCamera

# Initialize RealSense camera
camera = RealSenseCamera(
    width=640,
    height=480,
    fps=30,
    enable_depth=True,
    enable_color=True
)

# Start streaming
camera.start()

# Capture frame
frame = camera.get_frame()
rgb_image = frame.color_image
depth_image = frame.depth_image

# Get point cloud
point_cloud = camera.get_point_cloud()
```

### Force/Torque Sensor

```python
from opencontrol.sensors import ForceTorqueSensor

# Initialize F/T sensor
ft_sensor = ForceTorqueSensor(
    sensor_type="robotiq_ft300",
    tcp_ip="192.168.1.101"
)

# Start sensor
ft_sensor.start()

# Read force/torque values
ft_reading = ft_sensor.get_reading()
force = ft_reading.force  # [Fx, Fy, Fz] in Newtons
torque = ft_reading.torque  # [Tx, Ty, Tz] in Nm
```

### Multi-Modal Sensor Fusion

```python
from opencontrol.perception import MultiModalPerception

# Initialize perception system
perception = MultiModalPerception(
    camera=camera,
    force_sensor=ft_sensor,
    robot=robot
)

# Get unified sensor data
sensor_data = perception.get_sensor_data()
print(f"RGB shape: {sensor_data.rgb.shape}")
print(f"Depth shape: {sensor_data.depth.shape}")
print(f"Force: {sensor_data.force}")
print(f"Joint positions: {sensor_data.joint_positions}")
```

## Part 4: World Model Training

### Data Collection

```python
from opencontrol.data import DataCollector

# Initialize data collector
collector = DataCollector(
    robot=robot,
    sensors=[camera, ft_sensor],
    save_path="data/robot_demos"
)

# Start demonstration recording
collector.start_recording("pick_and_place_demo_001")

# Perform demonstration (manually or programmatically)
# ... robot movements ...

# Stop recording
collector.stop_recording()

# Collect multiple demonstrations
for i in range(10):
    collector.start_recording(f"demo_{i:03d}")
    # Perform demonstration
    collector.stop_recording()
```

### Dataset Preparation

```python
from opencontrol.data import RobotDataset

# Load dataset
dataset = RobotDataset(
    data_path="data/robot_demos",
    sequence_length=32,
    prediction_horizon=16,
    modalities=["rgb", "depth", "joints", "force"]
)

# Split dataset
train_dataset, val_dataset = dataset.split(train_ratio=0.8)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
```

### Model Training

```python
from opencontrol.training import WorldModelTrainer
from opencontrol.models import MultiModalWorldModel

# Initialize model
model = MultiModalWorldModel(
    vision_encoder="resnet50",
    sequence_length=32,
    prediction_horizon=16,
    hidden_dim=512,
    num_layers=8
)

# Initialize trainer
trainer = WorldModelTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config={
        "batch_size": 16,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "device": "cuda"
    }
)

# Start training
trainer.train()

# Save trained model
trainer.save_model("checkpoints/manipulation_model.pt")
```

## Part 5: Visual Model Predictive Control

### MPC Controller Setup

```python
from opencontrol.control import VisualMPCController

# Load trained world model
world_model = MultiModalWorldModel.load("checkpoints/manipulation_model.pt")

# Initialize Visual MPC controller
mpc_controller = VisualMPCController(
    world_model=world_model,
    robot=robot,
    sensors=[camera, ft_sensor],
    config={
        "horizon": 16,
        "num_samples": 1000,
        "temperature": 0.1,
        "control_frequency": 10  # Hz
    }
)
```

### Task Definition

```python
from opencontrol.tasks import PickAndPlaceTask

# Define pick and place task
task = PickAndPlaceTask(
    target_object="red_cube",
    destination="blue_box",
    success_criteria={
        "position_tolerance": 0.02,  # 2cm
        "orientation_tolerance": 0.1  # radians
    }
)

# Set task goal
mpc_controller.set_task(task)
```

### Execution Loop

```python
import time

# Main control loop
try:
    while not task.is_complete():
        # Get current sensor observations
        observations = perception.get_sensor_data()
        
        # Compute control action using MPC
        action = mpc_controller.compute_action(observations)
        
        # Execute action on robot
        robot.execute_action(action)
        
        # Check safety constraints
        if mpc_controller.check_safety_violation():
            robot.emergency_stop()
            break
            
        # Sleep to maintain control frequency
        time.sleep(0.1)  # 10Hz
        
    print("Task completed successfully!")
    
except Exception as e:
    print(f"Error during execution: {e}")
    robot.emergency_stop()
```

## Part 6: Advanced Applications

### Adaptive Control

```python
from opencontrol.adaptation import OnlineAdapter

# Initialize online adaptation
adapter = OnlineAdapter(
    world_model=world_model,
    adaptation_rate=0.01,
    buffer_size=100
)

# Adapt model based on recent experience
for i in range(1000):
    # Execute action and observe result
    action = mpc_controller.compute_action(observations)
    robot.execute_action(action)
    
    # Get new observations
    new_observations = perception.get_sensor_data()
    
    # Adapt model
    adapter.update(observations, action, new_observations)
    
    observations = new_observations
```

### Multi-Robot Coordination

```python
from opencontrol.coordination import MultiRobotCoordinator

# Initialize multiple robots
robot1 = UniversalRobot("192.168.1.100")
robot2 = UniversalRobot("192.168.1.101")

# Create coordinator
coordinator = MultiRobotCoordinator([robot1, robot2])

# Define collaborative task
collaborative_task = CollaborativeAssemblyTask(
    robots=[robot1, robot2],
    shared_workspace=True,
    collision_avoidance=True
)

# Execute coordinated actions
coordinator.execute_task(collaborative_task)
```

### Human-Robot Interaction

```python
from opencontrol.interaction import HumanRobotInterface

# Initialize HRI system
hri = HumanRobotInterface(
    robot=robot,
    input_modalities=["speech", "gesture", "gaze"],
    safety_monitor=True
)

# Start interaction loop
hri.start_interaction()

# Process human commands
while hri.is_active():
    command = hri.get_human_command()
    
    if command.type == "pick_object":
        # Execute pick command with safety checks
        hri.execute_safe_action(command)
    elif command.type == "stop":
        hri.stop_robot()
        break
```

## Safety Considerations

### Emergency Stop System

```python
from opencontrol.safety import EmergencyStopSystem

# Initialize emergency stop system
emergency_stop = EmergencyStopSystem(
    robot=robot,
    sensors=[camera, ft_sensor],
    triggers={
        "force_limit": 50.0,  # Newtons
        "velocity_limit": 2.0,  # m/s
        "workspace_violation": True,
        "collision_detection": True
    }
)

# Monitor safety in separate thread
emergency_stop.start_monitoring()
```

### Collision Avoidance

```python
from opencontrol.safety import CollisionAvoidance

# Initialize collision avoidance
collision_avoidance = CollisionAvoidance(
    robot=robot,
    camera=camera,
    safety_distance=0.1  # meters
)

# Check for potential collisions
is_safe = collision_avoidance.check_path_safety(planned_trajectory)
if not is_safe:
    # Replan trajectory
    safe_trajectory = collision_avoidance.replan_safe_path(planned_trajectory)
```

## Troubleshooting

### Common Issues

1. **Robot Connection Failed**
   ```bash
   # Check network connectivity
   ping <robot_ip>
   
   # Verify robot is in remote control mode
   # Check firewall settings
   ```

2. **High Control Latency**
   ```python
   # Reduce control frequency
   mpc_controller.config["control_frequency"] = 5  # Hz
   
   # Use lighter world model
   model = MultiModalWorldModel(hidden_dim=256, num_layers=4)
   ```

3. **Poor Task Performance**
   ```python
   # Collect more training data
   # Increase model capacity
   # Tune MPC parameters
   ```

## Next Steps

After completing this tutorial:

1. **Experiment with different robot platforms**
2. **Try advanced control algorithms**
3. **Implement custom tasks**
4. **Explore multi-robot scenarios**
5. **Contribute to the OpenControl community**

## Resources

- [Hardware Setup Guide](hardware_setup.md)
- [Training Guide](training.md)
- [API Reference](api/)
- [Example Code Repository](https://github.com/llamasearchai/OpenControl/tree/main/examples)

---

**Author**: Nik Jois (nikjois@llamasearch.ai)  
**Last Updated**: December 2024 