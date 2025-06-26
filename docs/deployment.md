# Deployment Guide: Production Robotics Systems

This guide covers deploying OpenControl in production environments, from edge devices to cloud infrastructure.

## Overview

OpenControl supports various deployment scenarios:
- **Edge Deployment**: Direct robot control with local inference
- **Cloud Deployment**: Centralized model serving and coordination
- **Hybrid Deployment**: Combination of edge and cloud components
- **Multi-Robot Systems**: Coordinated deployment across robot fleets

## Deployment Architecture

### Edge Deployment

```
Robot Hardware ←→ Edge Computer (OpenControl) ←→ Cloud (Optional)
```

**Advantages:**
- Low latency control
- Offline operation capability
- Real-time performance
- Data privacy

**Use Cases:**
- Manufacturing automation
- Warehouse robotics
- Autonomous vehicles

### Cloud Deployment

```
Robot Hardware ←→ Edge Gateway ←→ Cloud Services (OpenControl)
```

**Advantages:**
- Centralized management
- Scalable compute resources
- Easy updates and monitoring
- Multi-robot coordination

**Use Cases:**
- Robot fleets
- Remote monitoring
- Centralized learning

## Edge Deployment

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4-core ARM64 or x86_64
- **RAM**: 8GB
- **Storage**: 64GB SSD
- **GPU**: Optional (NVIDIA Jetson recommended)
- **Network**: Gigabit Ethernet

#### Recommended Requirements
- **CPU**: 8-core ARM64/x86_64
- **RAM**: 16GB+
- **Storage**: 256GB NVMe SSD
- **GPU**: NVIDIA RTX/Jetson series
- **Network**: Gigabit Ethernet + WiFi

### Supported Edge Platforms

#### NVIDIA Jetson Series
```bash
# Install JetPack
sudo apt update
sudo apt install nvidia-jetpack

# Install OpenControl
pip install opencontrol[robotics]

# Optimize for Jetson
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks   # Max clock speeds
```

#### Intel NUC
```bash
# Install Ubuntu 22.04
# Install OpenControl
pip install opencontrol[robotics]

# Install Intel GPU drivers (if applicable)
sudo apt install intel-media-va-driver-non-free
```

#### Raspberry Pi 4/5
```bash
# Install Ubuntu Server 22.04
# Install OpenControl (CPU-only)
pip install opencontrol --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Container Deployment

#### Docker Setup

```dockerfile
# Dockerfile.edge
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    librealsense2-dev \
    ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*

# Install OpenControl
COPY . /app/opencontrol
WORKDIR /app/opencontrol
RUN pip install -e .[robotics]

# Set up runtime environment
COPY configs/ /app/configs/
COPY models/ /app/models/

ENTRYPOINT ["python", "-m", "opencontrol.deployment.edge_server"]
```

```bash
# Build and run
docker build -f Dockerfile.edge -t opencontrol:edge .
docker run --gpus all --privileged --network host \
    -v /dev:/dev \
    -v ./configs:/app/configs \
    -v ./models:/app/models \
    opencontrol:edge
```

#### Kubernetes Deployment

```yaml
# edge-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencontrol-edge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: opencontrol-edge
  template:
    metadata:
      labels:
        app: opencontrol-edge
    spec:
      containers:
      - name: opencontrol
        image: opencontrol:edge
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config-volume
          mountPath: /app/configs
        - name: model-volume
          mountPath: /app/models
        - name: device-volume
          mountPath: /dev
        securityContext:
          privileged: true
      volumes:
      - name: config-volume
        configMap:
          name: opencontrol-config
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-storage
      - name: device-volume
        hostPath:
          path: /dev
```

## Cloud Deployment

### Architecture Components

#### Model Serving Service
```python
# cloud/model_server.py
from fastapi import FastAPI
from opencontrol.models import MultiModalWorldModel
from opencontrol.deployment import ModelServer

app = FastAPI()
model_server = ModelServer(
    model_path="models/world_model.pt",
    device="cuda",
    batch_size=16,
    max_workers=4
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    return await model_server.predict(request.data)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_server.is_ready()}
```

#### Robot Coordination Service
```python
# cloud/coordinator.py
from opencontrol.coordination import RobotFleetCoordinator

coordinator = RobotFleetCoordinator(
    robots=robot_fleet,
    task_queue=task_queue,
    optimization="multi_objective"
)

@app.post("/assign_task")
async def assign_task(task: Task):
    assignment = coordinator.assign_task(task)
    return {"robot_id": assignment.robot_id, "estimated_completion": assignment.eta}
```

### Cloud Platforms

#### AWS Deployment

```yaml
# aws/cloudformation.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'OpenControl Cloud Infrastructure'

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: opencontrol-cluster
      
  ModelService:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref ModelTaskDefinition
      DesiredCount: 3
      LaunchType: EC2
      
  ModelTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: opencontrol-model-server
      ContainerDefinitions:
        - Name: model-server
          Image: your-account.dkr.ecr.region.amazonaws.com/opencontrol:latest
          Memory: 8192
          Cpu: 4096
          Essential: true
          PortMappings:
            - ContainerPort: 8000
              Protocol: tcp
```

#### Google Cloud Platform

```yaml
# gcp/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencontrol-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opencontrol-model-server
  template:
    metadata:
      labels:
        app: opencontrol-model-server
    spec:
      containers:
      - name: model-server
        image: gcr.io/your-project/opencontrol:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "gs://your-bucket/models/world_model.pt"
---
apiVersion: v1
kind: Service
metadata:
  name: opencontrol-service
spec:
  selector:
    app: opencontrol-model-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Azure Deployment

```yaml
# azure/container-instances.yaml
apiVersion: '2019-12-01'
location: eastus
name: opencontrol-container-group
properties:
  containers:
  - name: model-server
    properties:
      image: your-registry.azurecr.io/opencontrol:latest
      resources:
        requests:
          cpu: 4
          memoryInGb: 8
          gpu:
            count: 1
            sku: V100
      ports:
      - port: 8000
        protocol: TCP
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
tags: {}
type: Microsoft.ContainerInstance/containerGroups
```

## Model Optimization

### Quantization

```python
from opencontrol.optimization import ModelQuantizer

# Post-training quantization
quantizer = ModelQuantizer(model)
quantized_model = quantizer.quantize(
    method="dynamic",  # or "static", "qat"
    backend="fbgemm",  # or "qnnpack" for mobile
    calibration_data=calibration_dataset
)

# Save quantized model
torch.jit.save(quantized_model, "models/world_model_quantized.pt")
```

### Model Pruning

```python
from opencontrol.optimization import ModelPruner

# Structured pruning
pruner = ModelPruner(model)
pruned_model = pruner.prune(
    sparsity=0.5,  # Remove 50% of parameters
    method="magnitude",
    structured=True
)

# Fine-tune pruned model
pruned_model = pruner.fine_tune(
    pruned_model,
    train_dataset,
    epochs=10
)
```

### TensorRT Optimization

```python
from opencontrol.optimization import TensorRTOptimizer

# Convert to TensorRT
trt_optimizer = TensorRTOptimizer()
trt_model = trt_optimizer.optimize(
    model,
    input_shapes={"rgb": (1, 3, 224, 224), "joints": (1, 7)},
    precision="fp16",  # or "int8"
    workspace_size=1 << 30  # 1GB
)
```

## Monitoring and Observability

### Metrics Collection

```python
from opencontrol.monitoring import MetricsCollector
import prometheus_client

# Initialize metrics
metrics = MetricsCollector()

# Custom metrics
inference_time = prometheus_client.Histogram(
    'opencontrol_inference_time_seconds',
    'Time spent on model inference'
)

task_success_rate = prometheus_client.Gauge(
    'opencontrol_task_success_rate',
    'Task success rate'
)

# Collect metrics
@metrics.track_inference_time
def predict(data):
    return model.predict(data)

@metrics.track_task_success
def execute_task(task):
    return robot.execute(task)
```

### Logging Configuration

```python
# logging_config.py
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### Health Checks

```python
from opencontrol.monitoring import HealthChecker

health_checker = HealthChecker([
    {"name": "model_server", "url": "http://localhost:8000/health"},
    {"name": "robot_connection", "check": robot.is_connected},
    {"name": "camera_feed", "check": camera.is_streaming},
    {"name": "force_sensor", "check": ft_sensor.is_active}
])

@app.get("/health")
async def health():
    return health_checker.check_all()
```

## Security

### Authentication and Authorization

```python
from opencontrol.security import JWTAuth, RoleBasedAuth

# JWT authentication
jwt_auth = JWTAuth(
    secret_key="your-secret-key",
    algorithm="HS256",
    expire_minutes=30
)

# Role-based authorization
role_auth = RoleBasedAuth({
    "admin": ["read", "write", "control"],
    "operator": ["read", "control"],
    "viewer": ["read"]
})

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    token = request.headers.get("Authorization")
    if token:
        user = jwt_auth.verify_token(token)
        request.state.user = user
    response = await call_next(request)
    return response
```

### Network Security

```yaml
# security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: opencontrol-network-policy
spec:
  podSelector:
    matchLabels:
      app: opencontrol
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: robot-gateway
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

## Scaling Strategies

### Horizontal Pod Autoscaling

```yaml
# scaling/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opencontrol-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opencontrol-model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing

```python
from opencontrol.deployment import LoadBalancer

# Custom load balancer with robot affinity
load_balancer = LoadBalancer(
    strategy="robot_affinity",  # Keep robot sessions on same server
    health_check_interval=30,
    timeout=60
)

@app.middleware("http")
async def load_balance_middleware(request: Request, call_next):
    server = load_balancer.select_server(request)
    request.state.target_server = server
    response = await call_next(request)
    return response
```

## Deployment Automation

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy OpenControl

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: |
        pip install -e .[dev]
        pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t opencontrol:${{ github.sha }} .
        docker tag opencontrol:${{ github.sha }} opencontrol:latest

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - name: Deploy to staging
      run: |
        kubectl set image deployment/opencontrol-model-server \
          model-server=opencontrol:${{ github.sha }}

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
    - name: Deploy to production
      run: |
        kubectl set image deployment/opencontrol-model-server \
          model-server=opencontrol:${{ github.sha }}
```

### Infrastructure as Code

```python
# infrastructure/pulumi_stack.py
import pulumi
import pulumi_aws as aws
import pulumi_kubernetes as k8s

# Create EKS cluster
cluster = aws.eks.Cluster(
    "opencontrol-cluster",
    version="1.28",
    instance_type="g4dn.xlarge",
    desired_capacity=3,
    min_size=1,
    max_size=10
)

# Deploy OpenControl application
app = k8s.apps.v1.Deployment(
    "opencontrol-app",
    spec=k8s.apps.v1.DeploymentSpecArgs(
        replicas=3,
        selector=k8s.meta.v1.LabelSelectorArgs(
            match_labels={"app": "opencontrol"}
        ),
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(
                labels={"app": "opencontrol"}
            ),
            spec=k8s.core.v1.PodSpecArgs(
                containers=[
                    k8s.core.v1.ContainerArgs(
                        name="opencontrol",
                        image="opencontrol:latest",
                        ports=[k8s.core.v1.ContainerPortArgs(container_port=8000)]
                    )
                ]
            )
        )
    )
)
```

## Best Practices

### Production Checklist

- [ ] **Security**: Authentication, authorization, encryption
- [ ] **Monitoring**: Metrics, logging, alerting
- [ ] **Backup**: Model checkpoints, configuration, data
- [ ] **Disaster Recovery**: Failover procedures, data recovery
- [ ] **Performance**: Load testing, optimization
- [ ] **Documentation**: Deployment procedures, troubleshooting
- [ ] **Compliance**: Safety standards, regulatory requirements

### Performance Optimization

1. **Model Optimization**: Quantization, pruning, distillation
2. **Caching**: Model predictions, sensor data
3. **Batching**: Request batching for efficiency
4. **Resource Management**: CPU/GPU utilization
5. **Network Optimization**: Compression, connection pooling

### Reliability

1. **Redundancy**: Multiple replicas, failover
2. **Health Checks**: Continuous monitoring
3. **Circuit Breakers**: Fault tolerance
4. **Graceful Degradation**: Fallback behaviors
5. **Recovery Procedures**: Automated recovery

---

**Author**: Nik Jois (nikjois@llamasearch.ai)  
**Last Updated**: December 2024 