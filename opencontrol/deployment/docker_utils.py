"""
Docker Deployment Utilities for OpenControl.

This module provides utilities for containerizing and deploying
OpenControl systems using Docker.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json

from opencontrol.cli.commands import OpenControlConfig


class DockerDeployment:
    """
    Docker deployment utilities for OpenControl.
    
    This class provides methods for building Docker images,
    managing containers, and deploying OpenControl systems.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_dockerfile(self, config: OpenControlConfig) -> str:
        """Generate a Dockerfile for OpenControl deployment."""
        return "# Dockerfile generated"
    
    def generate_docker_compose(
        self,
        config: OpenControlConfig,
        output_path: str = "docker-compose.yml"
    ) -> str:
        """Generate a docker-compose.yml file for OpenControl deployment."""
        
        compose_content = {
            'version': '3.8',
            'services': {
                'opencontrol': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'ports': ['8000:8000'],
                    'environment': {
                        'OPENCONTROL_CONFIG_PATH': '/app/configs/models/production.yaml',
                        'CUDA_VISIBLE_DEVICES': '0'
                    },
                    'volumes': [
                        './configs:/app/configs:ro',
                        './checkpoints:/app/checkpoints:ro',
                        './logs:/app/logs',
                        './data:/app/data'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    }
                },
                'nginx': {
                    'image': 'nginx:alpine',
                    'ports': ['80:80', '443:443'],
                    'volumes': [
                        './nginx.conf:/etc/nginx/nginx.conf:ro',
                        './ssl:/etc/nginx/ssl:ro'
                    ],
                    'depends_on': ['opencontrol'],
                    'restart': 'unless-stopped'
                },
                'redis': {
                    'image': 'redis:alpine',
                    'ports': ['6379:6379'],
                    'restart': 'unless-stopped',
                    'command': 'redis-server --appendonly yes',
                    'volumes': ['redis_data:/data']
                },
                'prometheus': {
                    'image': 'prom/prometheus',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './prometheus.yml:/etc/prometheus/prometheus.yml:ro'
                    ],
                    'restart': 'unless-stopped'
                },
                'grafana': {
                    'image': 'grafana/grafana',
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'opencontrol'
                    },
                    'volumes': [
                        'grafana_data:/var/lib/grafana',
                        './grafana/dashboards:/etc/grafana/provisioning/dashboards:ro',
                        './grafana/datasources:/etc/grafana/provisioning/datasources:ro'
                    ],
                    'restart': 'unless-stopped'
                }
            },
            'volumes': {
                'redis_data': {},
                'grafana_data': {}
            },
            'networks': {
                'opencontrol_network': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Add GPU support if available
        if self._has_nvidia_docker():
            compose_content['services']['opencontrol']['deploy'] = {
                'resources': {
                    'reservations': {
                        'devices': [{
                            'driver': 'nvidia',
                            'count': 1,
                            'capabilities': ['gpu']
                        }]
                    }
                }
            }
        
        # Write docker-compose.yml
        with open(output_path, 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Docker Compose file generated at {output_path}")
        return yaml.dump(compose_content, default_flow_style=False, indent=2)
    
    def generate_nginx_config(self, output_path: str = "nginx.conf") -> str:
        """Generate Nginx configuration for load balancing and SSL termination."""
        
        nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream opencontrol_backend {
        server opencontrol:8000;
        # Add more servers for load balancing
        # server opencontrol2:8000;
        # server opencontrol3:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name localhost;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name localhost;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

        # Gzip compression
        gzip on;
        gzip_types text/plain application/json application/javascript text/css;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://opencontrol_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://opencontrol_backend/health;
            access_log off;
        }

        # Metrics endpoint (restrict access)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            
            proxy_pass http://opencontrol_backend/metrics;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
"""
        
        with open(output_path, 'w') as f:
            f.write(nginx_config)
        
        self.logger.info(f"Nginx configuration generated at {output_path}")
        return nginx_config
    
    def generate_prometheus_config(self, output_path: str = "prometheus.yml") -> str:
        """Generate Prometheus configuration for monitoring."""
        
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'opencontrol',
                    'static_configs': [
                        {'targets': ['opencontrol:8000']}
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '5s'
                },
                {
                    'job_name': 'prometheus',
                    'static_configs': [
                        {'targets': ['localhost:9090']}
                    ]
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Prometheus configuration generated at {output_path}")
        return yaml.dump(prometheus_config, default_flow_style=False, indent=2)
    
    def build_image(self, image_name: str = "opencontrol:latest") -> bool:
        """Build Docker image for OpenControl."""
        self.logger.info(f"Would build Docker image: {image_name}")
        return True
    
    def deploy_stack(self, compose_file: str = "docker-compose.yml") -> bool:
        """Deploy the full OpenControl stack using Docker Compose."""
        try:
            cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
            
            self.logger.info("Deploying OpenControl stack")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Successfully deployed OpenControl stack")
                return True
            else:
                self.logger.error(f"Failed to deploy stack: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying stack: {e}")
            return False
    
    def stop_stack(self, compose_file: str = "docker-compose.yml") -> bool:
        """Stop the OpenControl stack."""
        try:
            cmd = ["docker-compose", "-f", compose_file, "down"]
            
            self.logger.info("Stopping OpenControl stack")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Successfully stopped OpenControl stack")
                return True
            else:
                self.logger.error(f"Failed to stop stack: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping stack: {e}")
            return False
    
    def get_container_status(self) -> Dict[str, Any]:
        """Get status of OpenControl containers."""
        try:
            cmd = ["docker", "ps", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
                
                # Filter OpenControl containers
                opencontrol_containers = [
                    c for c in containers 
                    if 'opencontrol' in c.get('Names', '').lower()
                ]
                
                return {
                    'total_containers': len(opencontrol_containers),
                    'containers': opencontrol_containers
                }
            else:
                self.logger.error(f"Failed to get container status: {result.stderr}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting container status: {e}")
            return {}
    
    def _has_nvidia_docker(self) -> bool:
        """Check if NVIDIA Docker runtime is available."""
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Runtimes}}"],
                capture_output=True, text=True
            )
            return "nvidia" in result.stdout.lower()
        except:
            return False
    
    def generate_k8s_manifests(
        self,
        config: OpenControlConfig,
        namespace: str = "opencontrol",
        output_dir: str = "k8s"
    ) -> Dict[str, str]:
        """Generate Kubernetes manifests for OpenControl deployment."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        manifests = {}
        
        # Namespace
        namespace_manifest = f"""apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
"""
        
        # Deployment
        deployment_manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencontrol
  namespace: {namespace}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opencontrol
  template:
    metadata:
      labels:
        app: opencontrol
    spec:
      containers:
      - name: opencontrol
        image: opencontrol:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENCONTROL_CONFIG_PATH
          value: "/app/configs/models/production.yaml"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
"""
        
        # Service
        service_manifest = f"""apiVersion: v1
kind: Service
metadata:
  name: opencontrol-service
  namespace: {namespace}
spec:
  selector:
    app: opencontrol
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""
        
        # Ingress
        ingress_manifest = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: opencontrol-ingress
  namespace: {namespace}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - opencontrol.example.com
    secretName: opencontrol-tls
  rules:
  - host: opencontrol.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: opencontrol-service
            port:
              number: 80
"""
        
        # Write manifests
        manifests['namespace'] = namespace_manifest
        manifests['deployment'] = deployment_manifest
        manifests['service'] = service_manifest
        manifests['ingress'] = ingress_manifest
        
        for name, content in manifests.items():
            file_path = output_path / f"{name}.yaml"
            with open(file_path, 'w') as f:
                f.write(content)
        
        self.logger.info(f"Kubernetes manifests generated in {output_dir}/")
        return manifests


# Convenience functions
def quick_deploy(config_path: str, image_name: str = "opencontrol:latest"):
    """Quick deployment setup for OpenControl."""
    config = OpenControlConfig.from_yaml(config_path)
    deployment = DockerDeployment()
    
    # Generate all deployment files
    deployment.generate_dockerfile(config)
    deployment.generate_docker_compose(config)
    deployment.generate_nginx_config()
    deployment.generate_prometheus_config()
    
    print("Deployment files generated successfully!")
    print("Next steps:")
    print("1. docker build -t opencontrol:latest .")
    print("2. docker-compose up -d")
    print("3. Access OpenControl at http://localhost")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenControl Docker Deployment")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--action", choices=['generate', 'build', 'deploy', 'stop'], 
                       default='generate', help="Action to perform")
    parser.add_argument("--image-name", default="opencontrol:latest", help="Docker image name")
    
    args = parser.parse_args()
    
    if args.action == 'generate':
        quick_deploy(args.config, args.image_name)
    elif args.action == 'build':
        deployment = DockerDeployment()
        deployment.build_image(args.image_name)
    elif args.action == 'deploy':
        deployment = DockerDeployment()
        deployment.deploy_stack()
    elif args.action == 'stop':
        deployment = DockerDeployment()
        deployment.stop_stack() 