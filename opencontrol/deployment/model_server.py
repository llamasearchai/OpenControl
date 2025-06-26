"""
Production Model Server for OpenControl.

This module provides a high-performance, production-ready server for serving
OpenControl world models and control systems via REST API and gRPC.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import numpy as np
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise ImportError("FastAPI dependencies required. Install with: pip install fastapi uvicorn")

from opencontrol.core.world_model import OpenControlWorldModel
from opencontrol.control.visual_mpc import ProductionVisualMPC
from opencontrol.cli.commands import OpenControlConfig
from .monitoring import ProductionMonitor


class ObservationRequest(BaseModel):
    """Request model for observations."""
    video: Optional[List[List[List[List[float]]]]] = None  # [B, T, C, H, W]
    audio: Optional[List[List[float]]] = None  # [B, T, L]
    actions: Optional[List[List[float]]] = None  # [B, T, A]
    proprioception: Optional[List[List[float]]] = None  # [B, T, P]
    prediction_horizon: int = Field(default=1, ge=1, le=100)


class ControlRequest(BaseModel):
    """Request model for control actions."""
    observation: Dict[str, Any]
    goal: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: Dict[str, List[Any]]
    uncertainty_estimates: Optional[Dict[str, List[Any]]] = None
    processing_time: float
    model_version: str


class ControlResponse(BaseModel):
    """Response model for control actions."""
    action: List[float]
    info: Dict[str, Any]
    processing_time: float
    safety_status: str


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime: float


class OpenControlModelServer:
    """
    Production-ready model server for OpenControl systems.
    
    This server provides:
    - High-performance model inference
    - Real-time control action computation
    - Health monitoring and metrics
    - Automatic batching and optimization
    - Error handling and recovery
    """
    
    def __init__(
        self,
        config: OpenControlConfig,
        model_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        self.config = config
        self.model_path = model_path
        self.host = host
        self.port = port
        
        # Initialize components
        self.world_model: Optional[OpenControlWorldModel] = None
        self.mpc_controller: Optional[ProductionVisualMPC] = None
        self.monitor = ProductionMonitor(config)
        
        # Server state
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        app = FastAPI(
            title="OpenControl Model Server",
            description="Production API for OpenControl world models and control systems",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes to the FastAPI app."""
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                memory_info = {}
                if torch.cuda.is_available():
                    memory_info = {
                        "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                        "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                        "gpu_memory_cached": torch.cuda.memory_cached() / 1024**3
                    }
                
                return HealthResponse(
                    status="healthy" if self.world_model is not None else "loading",
                    model_loaded=self.world_model is not None,
                    gpu_available=torch.cuda.is_available(),
                    memory_usage=memory_info,
                    uptime=time.time() - self.start_time
                )
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: ObservationRequest, background_tasks: BackgroundTasks):
            """Generate predictions from world model."""
            start_time = time.time()
            self.request_count += 1
            
            try:
                if self.world_model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                # Convert request to tensors
                observations = self._convert_request_to_tensors(request)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.world_model(observations, prediction_horizon=request.prediction_horizon)
                
                # Convert outputs to response format
                predictions = self._convert_tensors_to_response(outputs.predictions)
                uncertainty_estimates = None
                if hasattr(outputs, 'uncertainty_estimates') and outputs.uncertainty_estimates:
                    uncertainty_estimates = self._convert_tensors_to_response(outputs.uncertainty_estimates)
                
                processing_time = time.time() - start_time
                
                # Log metrics in background
                background_tasks.add_task(
                    self.monitor.log_prediction_metrics,
                    processing_time, len(predictions)
                )
                
                return PredictionResponse(
                    predictions=predictions,
                    uncertainty_estimates=uncertainty_estimates,
                    processing_time=processing_time,
                    model_version=getattr(self.config, 'model_version', '1.0.0')
                )
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Prediction failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @app.post("/control", response_model=ControlResponse)
        async def compute_control_action(request: ControlRequest, background_tasks: BackgroundTasks):
            """Compute control action using MPC."""
            start_time = time.time()
            self.request_count += 1
            
            try:
                if self.mpc_controller is None:
                    raise HTTPException(status_code=503, detail="Controller not loaded")
                
                # Convert observation to tensors
                observation = self._convert_dict_to_tensors(request.observation)
                goal = self._convert_dict_to_tensors(request.goal) if request.goal else None
                
                # Compute control action
                action, info = await self.mpc_controller.compute_action(
                    observation, goal, request.constraints
                )
                
                processing_time = time.time() - start_time
                
                # Determine safety status
                safety_status = "safe"
                if 'safety_info' in info and not info['safety_info'].get('is_safe', True):
                    safety_status = "unsafe"
                elif 'error' in info:
                    safety_status = "error"
                
                # Log metrics in background
                background_tasks.add_task(
                    self.monitor.log_control_metrics,
                    processing_time, info.get('cost', 0.0), safety_status
                )
                
                return ControlResponse(
                    action=action.cpu().numpy().tolist(),
                    info=self._sanitize_info_dict(info),
                    processing_time=processing_time,
                    safety_status=safety_status
                )
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Control computation failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Control computation failed: {str(e)}")
        
        @app.get("/metrics")
        async def get_metrics():
            """Get server metrics."""
            try:
                metrics = {
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(1, self.request_count),
                    "uptime": time.time() - self.start_time,
                    "model_metrics": self.monitor.get_metrics() if self.monitor else {}
                }
                
                if self.mpc_controller:
                    metrics["control_metrics"] = self.mpc_controller.get_performance_stats()
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Metrics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail="Metrics retrieval failed")
        
        @app.post("/reset")
        async def reset_controller():
            """Reset controller state."""
            try:
                if self.mpc_controller:
                    self.mpc_controller.reset_performance_stats()
                
                if self.monitor:
                    self.monitor.reset_metrics()
                
                return {"status": "reset_complete"}
                
            except Exception as e:
                self.logger.error(f"Reset failed: {e}")
                raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
    
    async def _startup(self):
        """Server startup logic."""
        self.logger.info("Starting OpenControl Model Server")
        
        try:
            # Load world model
            if self.model_path:
                self.logger.info(f"Loading model from {self.model_path}")
                self.world_model = self._load_model(self.model_path)
            else:
                self.logger.info("Creating new model")
                self.world_model = OpenControlWorldModel(self.config)
            
            # Initialize MPC controller
            self.logger.info("Initializing MPC controller")
            self.mpc_controller = ProductionVisualMPC(
                self.world_model, self.config, self.logger
            )
            
            self.logger.info("Server startup complete")
            
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}", exc_info=True)
            raise
    
    async def _shutdown(self):
        """Server shutdown logic."""
        self.logger.info("Shutting down OpenControl Model Server")
        
        # Save metrics
        if self.monitor:
            await self.monitor.save_metrics("server_shutdown_metrics.json")
        
        self.logger.info("Server shutdown complete")
    
    def _load_model(self, model_path: str) -> OpenControlWorldModel:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = OpenControlWorldModel(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model
    
    def _convert_request_to_tensors(self, request: ObservationRequest) -> Dict[str, torch.Tensor]:
        """Convert API request to tensor format."""
        observations = {}
        device = next(self.world_model.parameters()).device
        
        if request.video is not None:
            observations['video'] = torch.tensor(request.video, dtype=torch.float32, device=device)
        
        if request.audio is not None:
            observations['audio'] = torch.tensor(request.audio, dtype=torch.float32, device=device)
        
        if request.actions is not None:
            observations['actions'] = torch.tensor(request.actions, dtype=torch.float32, device=device)
        
        if request.proprioception is not None:
            observations['proprioception'] = torch.tensor(request.proprioception, dtype=torch.float32, device=device)
        
        return observations
    
    def _convert_dict_to_tensors(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert dictionary to tensor format."""
        if not data:
            return {}
        
        tensors = {}
        device = next(self.world_model.parameters()).device
        
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                tensors[key] = torch.tensor(value, dtype=torch.float32, device=device)
            elif isinstance(value, torch.Tensor):
                tensors[key] = value.to(device)
        
        return tensors
    
    def _convert_tensors_to_response(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, List[Any]]:
        """Convert tensors to API response format."""
        response = {}
        
        for key, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                response[key] = tensor.cpu().numpy().tolist()
            else:
                response[key] = tensor
        
        return response
    
    def _sanitize_info_dict(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize info dictionary for JSON serialization."""
        sanitized = {}
        
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                sanitized[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                sanitized[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        
        return sanitized
    
    def run(self, **kwargs):
        """Run the server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )
    
    async def run_async(self, **kwargs):
        """Run the server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )
        server = uvicorn.Server(config)
        await server.serve()


# Standalone server function for deployment
def create_server(
    config_path: str,
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000
) -> OpenControlModelServer:
    """Create a server instance from configuration file."""
    config = OpenControlConfig.from_yaml(config_path)
    return OpenControlModelServer(config, model_path, host, port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenControl Model Server")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--model", help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    server = create_server(args.config, args.model, args.host, args.port)
    server.run() 