"""
OpenControl Advanced Multi-Modal World Model.

A state-of-the-art transformer architecture for learning dynamics from
multi-modal sensory data, designed for scalability and performance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, Union
import logging

from .multimodal_encoder import MultiModalEncoder
from .attention_mechanisms import MultiModalAttention, RotaryPositionalEncoding
from .temporal_dynamics import TemporalDynamicsModule
from opencontrol.cli.commands import OpenControlConfig


class WorldModelOutput(dict):
    """
    A dictionary-like object for structured world model output.
    Allows both dict-style and attribute-style access.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value


class WorldModelTransformerLayer(nn.Module):
    """A single transformer layer with RoPE and multi-modal attention."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        # Multi-modal attention with RoPE
        self.self_attn = MultiModalAttention(
            self.config.model_dim,
            self.config.num_heads,
            dropout=self.config.dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.config.model_dim, self.config.model_dim * 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.model_dim * 4, self.config.model_dim),
            nn.Dropout(self.config.dropout)
        )
        
        # Layer normalizations (Pre-LN architecture)
        self.norm1 = nn.LayerNorm(self.config.model_dim)
        self.norm2 = nn.LayerNorm(self.config.model_dim)
        
        # RoPE encoder for this layer
        self.rope = RotaryPositionalEncoding(
            self.config.model_dim // self.config.num_heads,
            max_seq_len=self.config.max_sequence_length
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input tensor [batch, seq_len, model_dim]
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-LN: Apply layer norm before attention
        attn_input = self.norm1(x)
        
        # Self-attention with RoPE
        attn_output, attn_weights = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            rope_encoder=self.rope,
            return_attention_weights=return_attention_weights
        )
        
        # Residual connection
        x = x + attn_output
        
        # Pre-LN: Apply layer norm before FFN
        ffn_input = self.norm2(x)
        
        # Feed-forward network
        ffn_output = self.ffn(ffn_input)
        
        # Residual connection
        x = x + ffn_output
        
        return x, attn_weights


class WorldModelHead(nn.Module):
    """Prediction head for a specific modality."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OpenControlWorldModel(nn.Module):
    """
    The main world model class, integrating all components for multi-modal
    sequence modeling and prediction.
    """
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Multi-modal encoder
        self.multimodal_encoder = MultiModalEncoder(config)
        
        # Temporal dynamics module
        self.temporal_dynamics = TemporalDynamicsModule(
            model_dim=self.config.model_dim,
            num_layers=6,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_sequence_length
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            WorldModelTransformerLayer(config)
            for _ in range(self.config.num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.config.model_dim)
        
        # Prediction heads for each modality
        self.prediction_heads = self._create_prediction_heads()
        
        # Uncertainty estimation heads (if enabled)
        if self.config.uncertainty_estimation:
            self.uncertainty_heads = self._create_uncertainty_heads()
        else:
            self.uncertainty_heads = None
        
        # Value head for reinforcement learning (optional)
        self.value_head = nn.Linear(self.config.model_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model statistics
        self._log_model_stats()
    
    def _create_prediction_heads(self) -> nn.ModuleDict:
        """Create prediction heads for each output modality."""
        heads = nn.ModuleDict()
        
        # Video prediction head
        video_output_dim = 3 * self.config.video_resolution[0] * self.config.video_resolution[1]
        heads['video'] = WorldModelHead(
            self.config.model_dim,
            video_output_dim,
            hidden_dim=self.config.model_dim * 2
        )
        
        # Audio prediction head
        heads['audio'] = WorldModelHead(
            self.config.model_dim,
            self.config.audio_sample_rate // 10  # Simplified audio length
        )
        
        # Action prediction head
        heads['actions'] = WorldModelHead(
            self.config.model_dim,
            self.config.action_dim
        )
        
        # Proprioception prediction head
        heads['proprioception'] = WorldModelHead(
            self.config.model_dim,
            self.config.proprioception_dim
        )
        
        # Text prediction head (next token prediction)
        heads['text'] = WorldModelHead(
            self.config.model_dim,
            self.config.vocab_size
        )
        
        return heads
    
    def _create_uncertainty_heads(self) -> nn.ModuleDict:
        """Create uncertainty estimation heads (predict log variance)."""
        heads = nn.ModuleDict()
        
        for modality in self.prediction_heads.keys():
            heads[modality] = nn.Linear(self.config.model_dim, 1)
        
        return heads
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _log_model_stats(self):
        """Log model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.info(f"World Model initialized:")
        self.logger.info(f"  - Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        self.logger.info(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        self.logger.info(f"  - Model dimension: {self.config.model_dim}")
        self.logger.info(f"  - Number of layers: {self.config.num_layers}")
        self.logger.info(f"  - Number of heads: {self.config.num_heads}")
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        prediction_horizon: int = 1,
        return_attention: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        target_modalities: Optional[list] = None
    ) -> WorldModelOutput:
        """
        Forward pass for multi-modal sequence modeling and prediction.
        
        Args:
            inputs: Dictionary of input tensors (video, audio, actions, etc.)
            prediction_horizon: Number of future steps to predict
            return_attention: Whether to return attention weights
            attention_mask: Optional attention mask
            target_modalities: List of modalities to predict (None = all)
            
        Returns:
            WorldModelOutput containing predictions and intermediate states
        """
        # Encode multi-modal inputs
        tokens, modality_info = self.multimodal_encoder(inputs)
        batch_size, seq_len, model_dim = tokens.shape
        
        # Process through transformer layers
        hidden_states = tokens
        all_attention_weights = [] if return_attention else None
        
        for layer in self.transformer_layers:
            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states, attn_weights = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, return_attention
                )
            else:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_attention_weights=return_attention
                )
            
            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)
        
        # Apply output normalization
        hidden_states = self.output_norm(hidden_states)
        
        # Generate predictions for multiple time steps
        predictions = {}
        uncertainties = {} if self.uncertainty_heads else None
        
        # Determine which modalities to predict
        if target_modalities is None:
            target_modalities = list(self.prediction_heads.keys())
        
        # For simplicity, we'll predict from the last hidden state
        # In a full implementation, this would involve autoregressive generation
        last_hidden = hidden_states[:, -1, :]  # [batch, model_dim]
        
        for modality in target_modalities:
            if modality in self.prediction_heads:
                # Multi-step prediction
                modality_predictions = []
                
                for step in range(prediction_horizon):
                    # In a full implementation, this would use the previous prediction
                    # as input for the next step
                    pred = self.prediction_heads[modality](last_hidden)
                    modality_predictions.append(pred)
                
                predictions[modality] = torch.stack(modality_predictions, dim=1)
                
                # Uncertainty estimation
                if self.uncertainty_heads:
                    uncertainty = self.uncertainty_heads[modality](last_hidden)
                    uncertainties[modality] = uncertainty.expand(-1, prediction_horizon)
        
        # Compute value estimate (for RL applications)
        value_estimate = self.value_head(last_hidden)
        
        return WorldModelOutput(
            predictions=predictions,
            latent_states=hidden_states,
            attention_weights=all_attention_weights,
            uncertainty_estimates=uncertainties,
            value_estimate=value_estimate,
            modality_info=modality_info
        )
    
    def encode_sequence(
        self,
        inputs: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> WorldModelOutput:
        """
        Encode a sequence without prediction (for analysis or as input to other models).
        
        Args:
            inputs: Dictionary of input tensors
            return_attention: Whether to return attention weights
            
        Returns:
            WorldModelOutput with encoded sequence
        """
        return self.forward(
            inputs,
            prediction_horizon=0,
            return_attention=return_attention
        )
    
    def predict_next_step(
        self,
        inputs: Dict[str, torch.Tensor],
        target_modalities: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the next step for specified modalities.
        
        Args:
            inputs: Current observations
            target_modalities: Modalities to predict
            
        Returns:
            Dictionary of predictions
        """
        output = self.forward(
            inputs,
            prediction_horizon=1,
            target_modalities=target_modalities
        )
        
        # Return single-step predictions
        next_step_predictions = {}
        for modality, preds in output.predictions.items():
            next_step_predictions[modality] = preds[:, 0, :]  # First (and only) step
        
        return next_step_predictions
    
    def compute_loss(
        self,
        outputs: WorldModelOutput,
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-modal prediction losses.
        
        Args:
            outputs: Model outputs
            targets: Target values for each modality
            loss_weights: Optional weights for each modality loss
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        if loss_weights is None:
            loss_weights = {modality: 1.0 for modality in outputs.predictions.keys()}
        
        for modality, predictions in outputs.predictions.items():
            if modality in targets:
                target = targets[modality]
                
                if modality in ['video', 'audio', 'actions', 'proprioception']:
                    # Regression loss (MSE)
                    loss = F.mse_loss(predictions, target)
                elif modality == 'text':
                    # Classification loss (Cross-entropy)
                    loss = F.cross_entropy(
                        predictions.view(-1, predictions.size(-1)),
                        target.view(-1)
                    )
                else:
                    # Default to MSE
                    loss = F.mse_loss(predictions, target)
                
                # Apply uncertainty weighting if available
                if (self.uncertainty_heads and 
                    modality in outputs.uncertainty_estimates):
                    uncertainty = outputs.uncertainty_estimates[modality]
                    # Uncertainty-weighted loss (higher uncertainty = lower weight)
                    loss = loss * torch.exp(-uncertainty) + uncertainty
                
                # Apply modality weight
                weighted_loss = loss * loss_weights.get(modality, 1.0)
                losses[f'{modality}_loss'] = weighted_loss
                total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def generate_sequence(
        self,
        initial_inputs: Dict[str, torch.Tensor],
        num_steps: int,
        target_modalities: Optional[list] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a sequence autoregressively.
        
        Args:
            initial_inputs: Initial observations
            num_steps: Number of steps to generate
            target_modalities: Modalities to generate
            temperature: Sampling temperature
            
        Returns:
            Generated sequences
        """
        self.eval()
        generated = {modality: [] for modality in (target_modalities or self.prediction_heads.keys())}
        
        current_inputs = initial_inputs.copy()
        
        with torch.no_grad():
            for step in range(num_steps):
                # Predict next step
                predictions = self.predict_next_step(current_inputs, target_modalities)
                
                # Store predictions
                for modality, pred in predictions.items():
                    if temperature != 1.0 and modality == 'text':
                        # Apply temperature for text generation
                        pred = pred / temperature
                        pred = F.softmax(pred, dim=-1)
                    
                    generated[modality].append(pred)
                
                # Update current inputs for next step
                # (This is a simplified version - full implementation would be more complex)
                for modality in predictions:
                    if modality in current_inputs:
                        # Shift sequence and append new prediction
                        current_inputs[modality] = torch.cat([
                            current_inputs[modality][:, 1:],
                            predictions[modality].unsqueeze(1)
                        ], dim=1)
        
        # Stack generated sequences
        for modality in generated:
            if generated[modality]:
                generated[modality] = torch.stack(generated[modality], dim=1)
        
        return generated 