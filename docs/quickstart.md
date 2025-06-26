# Quick Start Examples

Get up and running with OpenControl in minutes with these ready-to-run examples.

## Prerequisites

```bash
# Install OpenControl with robotics support
pip install -e ".[robotics]"

# Verify installation
opencontrol --version
```

## Example 1: Basic Robot Connection

### Universal Robots UR5

```python
#!/usr/bin/env python3
"""
Basic UR5 connection and movement example.
"""

from opencontrol.robots import UniversalRobot
import time

def main():
    # Initialize robot
    robot = UniversalRobot(
        ip_address="192.168.1.100",  # Change to your robot's IP
        control_frequency=125
    )
    
    try:
        # Connect to robot
        print("Connecting to robot...")
        robot.connect()
        print(f"Connected: {robot.is_connected()}")
        
        # Get current pose
        current_pose = robot.get_pose()
        print(f"Current pose: {current_pose}")
        
        # Move to a safe position
        safe_joints = [0, -1.57, 0, -1.57, 0, 0, 0]
        robot.move_to_joints(safe_joints, velocity=0.1)
        
        print("Movement complete!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
```

### Franka Panda

```python
#!/usr/bin/env python3
"""
Basic Franka Panda connection and movement example.
"""

from opencontrol.robots import FrankaPanda
from opencontrol.utils import Pose

def main():
    # Initialize robot
    robot = FrankaPanda(ip_address="172.16.0.2")
    
    try:
        # Connect and activate
        robot.connect()
        robot.activate_fci()
        
        # Move to home position
        robot.move_to_home()
        
        # Move to a target pose
        target_pose = Pose(
            position=[0.4, 0.0, 0.4],
            orientation=[1.0, 0.0, 0.0, 0.0]
        )
        robot.move_to_pose(target_pose)
        
        print("Movement complete!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
```

## Example 2: Camera Integration

```python
#!/usr/bin/env python3
"""
RGB-D camera capture and display example.
"""

from opencontrol.sensors import RealSenseCamera
import cv2
import numpy as np

def main():
    # Initialize camera
    camera = RealSenseCamera(
        width=640,
        height=480,
        fps=30
    )
    
    try:
        # Start camera
        camera.start()
        print("Camera started. Press 'q' to quit.")
        
        while True:
            # Get frame
            frame = camera.get_frame()
            
            # Display RGB image
            cv2.imshow('RGB', frame.color_image)
            
            # Display depth image (normalized)
            depth_normalized = cv2.normalize(
                frame.depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            cv2.imshow('Depth', depth_normalized)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## Example 3: Simple Pick and Place

```python
#!/usr/bin/env python3
"""
Simple pick and place example with vision.
"""

from opencontrol.robots import UniversalRobot
from opencontrol.sensors import RealSenseCamera
from opencontrol.perception import ObjectDetector
from opencontrol.tasks import PickAndPlaceTask

def main():
    # Initialize components
    robot = UniversalRobot("192.168.1.100")
    camera = RealSenseCamera()
    detector = ObjectDetector(model="yolo_v8")
    
    try:
        # Connect and start
        robot.connect()
        camera.start()
        
        # Capture image
        frame = camera.get_frame()
        
        # Detect objects
        detections = detector.detect(frame.color_image)
        print(f"Found {len(detections)} objects")
        
        # Find target object
        target_object = None
        for detection in detections:
            if detection.class_name == "cube":
                target_object = detection
                break
        
        if target_object is None:
            print("No target object found!")
            return
        
        # Convert 2D detection to 3D pose
        object_pose = camera.pixel_to_world(
            target_object.center_x,
            target_object.center_y,
            frame.depth_image
        )
        
        # Execute pick and place
        pick_pose = Pose(
            position=[object_pose.x, object_pose.y, object_pose.z + 0.1],
            orientation=[0, 1, 0, 0]
        )
        
        place_pose = Pose(
            position=[object_pose.x + 0.2, object_pose.y, object_pose.z + 0.1],
            orientation=[0, 1, 0, 0]
        )
        
        # Pick
        robot.move_to_pose(pick_pose)
        robot.close_gripper()
        
        # Place
        robot.move_to_pose(place_pose)
        robot.open_gripper()
        
        print("Pick and place complete!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.disconnect()
        camera.stop()

if __name__ == "__main__":
    main()
```

## Example 4: Data Collection

```python
#!/usr/bin/env python3
"""
Collect robot demonstration data for training.
"""

from opencontrol.robots import UniversalRobot
from opencontrol.sensors import RealSenseCamera, ForceTorqueSensor
from opencontrol.data import DataCollector

def main():
    # Initialize components
    robot = UniversalRobot("192.168.1.100")
    camera = RealSenseCamera()
    ft_sensor = ForceTorqueSensor("robotiq_ft300", "192.168.1.101")
    
    # Initialize data collector
    collector = DataCollector(
        robot=robot,
        sensors=[camera, ft_sensor],
        save_path="data/demonstrations",
        recording_frequency=30  # Hz
    )
    
    try:
        # Connect all components
        robot.connect()
        camera.start()
        ft_sensor.start()
        
        print("Ready to collect demonstrations.")
        print("Press ENTER to start recording, 'q' to quit.")
        
        demo_count = 0
        while True:
            user_input = input("Command: ")
            
            if user_input.lower() == 'q':
                break
            elif user_input == '':
                # Start recording
                demo_name = f"demo_{demo_count:03d}"
                print(f"Recording {demo_name}...")
                
                collector.start_recording(demo_name)
                
                # Wait for user to complete demonstration
                input("Perform demonstration, then press ENTER to stop...")
                
                collector.stop_recording()
                print(f"Saved {demo_name}")
                demo_count += 1
        
        print(f"Collected {demo_count} demonstrations")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.disconnect()
        camera.stop()
        ft_sensor.stop()

if __name__ == "__main__":
    main()
```

## Example 5: World Model Training

```python
#!/usr/bin/env python3
"""
Train a world model on collected robot data.
"""

from opencontrol.data import RobotDataset
from opencontrol.models import MultiModalWorldModel
from opencontrol.training import WorldModelTrainer

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = RobotDataset(
        data_path="data/demonstrations",
        sequence_length=32,
        prediction_horizon=16,
        modalities=["rgb", "depth", "joints", "force"]
    )
    
    # Split dataset
    train_dataset, val_dataset = dataset.split(train_ratio=0.8)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = MultiModalWorldModel(
        vision_encoder="resnet18",
        sequence_length=32,
        prediction_horizon=16,
        hidden_dim=256,
        num_layers=6
    )
    
    # Initialize trainer
    trainer = WorldModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_interval": 10
        }
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model("checkpoints/world_model.pt")
    print("Training complete!")

if __name__ == "__main__":
    main()
```

## Example 6: Visual MPC Control

```python
#!/usr/bin/env python3
"""
Use trained world model for visual MPC control.
"""

from opencontrol.robots import UniversalRobot
from opencontrol.sensors import RealSenseCamera
from opencontrol.models import MultiModalWorldModel
from opencontrol.control import VisualMPCController
from opencontrol.tasks import ReachingTask

def main():
    # Initialize components
    robot = UniversalRobot("192.168.1.100")
    camera = RealSenseCamera()
    
    # Load trained world model
    world_model = MultiModalWorldModel.load("checkpoints/world_model.pt")
    
    # Initialize MPC controller
    mpc_controller = VisualMPCController(
        world_model=world_model,
        robot=robot,
        sensors=[camera],
        config={
            "horizon": 16,
            "num_samples": 500,
            "temperature": 0.1,
            "control_frequency": 10
        }
    )
    
    # Define task
    target_pose = Pose([0.5, 0.2, 0.3], [0, 1, 0, 0])
    task = ReachingTask(target_pose=target_pose)
    
    try:
        # Connect and start
        robot.connect()
        camera.start()
        
        # Set task
        mpc_controller.set_task(task)
        
        print("Starting MPC control...")
        
        # Control loop
        while not task.is_complete():
            # Get observations
            observations = {
                "rgb": camera.get_frame().color_image,
                "joints": robot.get_joint_positions(),
                "pose": robot.get_pose()
            }
            
            # Compute action
            action = mpc_controller.compute_action(observations)
            
            # Execute action
            robot.execute_action(action)
            
            # Check progress
            current_pose = robot.get_pose()
            distance = np.linalg.norm(
                np.array(current_pose.position) - np.array(target_pose.position)
            )
            print(f"Distance to target: {distance:.3f}m")
            
            time.sleep(0.1)  # 10Hz control
        
        print("Task completed!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.disconnect()
        camera.stop()

if __name__ == "__main__":
    main()
```

## Example 7: CLI Usage

### Interactive Dashboard

```bash
# Launch interactive dashboard
opencontrol dashboard --robot-ip 192.168.1.100

# Monitor robot status
opencontrol status --robot-type ur5 --ip 192.168.1.100

# Emergency stop
opencontrol emergency-stop --robot-ip 192.168.1.100
```

### Training Commands

```bash
# Train world model
opencontrol train \
    --config configs/models/manipulation.yaml \
    --data-path data/demonstrations \
    --output-dir checkpoints/

# Evaluate model
opencontrol evaluate \
    --model checkpoints/world_model.pt \
    --test-data data/test_episodes/

# Run benchmarks
opencontrol benchmark \
    --model checkpoints/world_model.pt \
    --tasks pick_place,reaching,stacking
```

### Deployment Commands

```bash
# Start model server
opencontrol serve \
    --model checkpoints/world_model.pt \
    --port 8000 \
    --workers 4

# Deploy to robot
opencontrol deploy \
    --robot-config configs/robots/ur5_config.yaml \
    --model checkpoints/world_model.pt \
    --task-config configs/tasks/pick_place.yaml
```

## Configuration Examples

### Robot Configuration

```yaml
# configs/robots/ur5_config.yaml
robot:
  type: "universal_robots"
  model: "ur5"
  ip_address: "192.168.1.100"
  control_frequency: 125
  
safety:
  max_velocity: 0.5
  max_acceleration: 1.0
  workspace_limits:
    x: [-0.8, 0.8]
    y: [-0.8, 0.8]
    z: [0.0, 1.2]
    
gripper:
  type: "robotiq_2f85"
  force_limit: 40.0
```

### Training Configuration

```yaml
# configs/training/world_model.yaml
model:
  type: "multimodal_world_model"
  vision_encoder: "resnet18"
  sequence_length: 32
  prediction_horizon: 16
  hidden_dim: 256
  num_layers: 6

training:
  batch_size: 16
  learning_rate: 1e-4
  num_epochs: 100
  device: "cuda"
  
data:
  modalities: ["rgb", "depth", "joints", "force"]
  augmentation: true
  normalization: true
```

## Troubleshooting

### Common Issues

1. **Robot Connection Failed**
   ```python
   # Check robot IP and network
   import subprocess
   result = subprocess.run(['ping', '-c', '1', '192.168.1.100'], 
                          capture_output=True, text=True)
   print(result.stdout)
   ```

2. **Camera Not Found**
   ```bash
   # List available cameras
   opencontrol list-cameras
   
   # Test camera
   opencontrol test-camera --device 0
   ```

3. **CUDA Out of Memory**
   ```python
   # Reduce batch size or model size
   config["batch_size"] = 4
   model = MultiModalWorldModel(hidden_dim=128, num_layers=4)
   ```

## Next Steps

1. **Explore more examples** in the [examples/](https://github.com/llamasearchai/OpenControl/tree/main/examples) directory
2. **Read the full [Robotics Tutorial](robotics_tutorial.md)**
3. **Check out [Hardware Setup](hardware_setup.md)** for your specific robot
4. **Join our community** on [Discord](https://discord.gg/opencontrol)

---

**Author**: Nik Jois (nikjois@llamasearch.ai)  
**Last Updated**: December 2024 