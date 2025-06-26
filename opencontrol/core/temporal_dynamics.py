"""
Temporal Dynamics Module for OpenControl World Model.

This module handles temporal modeling, state transitions, and dynamics
prediction within the world model architecture.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import math

from .attention_mechanisms import MultiModalAttention, RotaryPositionalEncoding


class TemporalConvolution(nn.Module):
    """1D temporal convolution for processing sequential data."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, channels]
        Returns:
            Output tensor [batch, seq_len, out_channels]
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # Back to [batch, seq_len, channels]
        
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention for temporal modeling."""
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
        
        # RoPE for temporal positions
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, model_dim]
            attention_mask: Optional attention mask
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.model_dim
        )
        out = self.out_proj(out)
        
        return out, attn_weights


class TemporalBlock(nn.Module):
    """A block combining temporal convolution and causal attention."""
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        conv_kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Temporal convolution branch
        self.temp_conv = TemporalConvolution(
            model_dim, model_dim, conv_kernel_size, dilation, dropout
        )
        
        # Causal attention branch
        self.causal_attn = CausalSelfAttention(
            model_dim, num_heads, dropout=dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        
        # Gating mechanism to combine conv and attention
        self.gate = nn.Linear(model_dim * 2, model_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, model_dim]
            attention_mask: Optional attention mask
        Returns:
            Tuple of (output, attention_weights)
        """
        residual = x
        
        # Temporal convolution branch
        conv_out = self.temp_conv(self.norm1(x))
        
        # Causal attention branch
        attn_out, attn_weights = self.causal_attn(self.norm2(x), attention_mask)
        
        # Combine branches with gating
        combined = torch.cat([conv_out, attn_out], dim=-1)
        gated = torch.sigmoid(self.gate(combined))
        mixed = gated * conv_out + (1 - gated) * attn_out
        
        # Residual connection
        x = residual + mixed
        
        # Feed-forward network
        x = x + self.ffn(self.norm3(x))
        
        return x, attn_weights


class StateTransitionPredictor(nn.Module):
    """Predicts state transitions based on current state and actions."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State and action encoders
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        
        # Transition network
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, state_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ])
        
        self.transition_net = nn.Sequential(*layers)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_dim, state_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Current state [batch, state_dim]
            action: Action [batch, action_dim]
        Returns:
            Tuple of (next_state_delta, uncertainty)
        """
        # Encode state and action
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        
        # Combine and predict transition
        combined = torch.cat([state_encoded, action_encoded], dim=-1)
        
        # Predict state change (residual learning)
        state_delta = self.transition_net(combined)
        
        # Predict uncertainty
        uncertainty = torch.exp(self.uncertainty_head(
            F.relu(combined.mean(dim=-1, keepdim=True).expand(-1, combined.shape[-1] // 2))
        ))
        
        return state_delta, uncertainty


class TemporalDynamicsModule(nn.Module):
    """
    Main temporal dynamics module that combines multiple temporal modeling
    techniques for robust dynamics learning.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_layers = num_layers
        
        # Stack of temporal blocks with increasing dilation
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                conv_kernel_size=3,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # State transition predictor
        self.state_predictor = StateTransitionPredictor(
            state_dim=model_dim,
            action_dim=model_dim,  # Actions are embedded to model_dim
            hidden_dim=model_dim * 2
        )
        
        # Output projection
        self.output_proj = nn.Linear(model_dim, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)
        
        # Temporal position embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, model_dim) * 0.02
        )
    
    def forward(
        self,
        x: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input sequence [batch, seq_len, model_dim]
            actions: Optional action sequence [batch, seq_len, action_dim]
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
        Returns:
            Dictionary containing outputs and optional attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Add temporal position embeddings
        if seq_len <= self.pos_embedding.shape[1]:
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # Interpolate position embeddings for longer sequences
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_emb
        
        # Process through temporal blocks
        all_attention_weights = []
        for block in self.temporal_blocks:
            x, attn_weights = block(x, attention_mask)
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Apply output normalization and projection
        x = self.output_norm(x)
        dynamics_output = self.output_proj(x)
        
        outputs = {
            'dynamics_output': dynamics_output,
            'temporal_features': x
        }
        
        # State transition prediction if actions are provided
        if actions is not None:
            # Use the last state for transition prediction
            current_state = x[:, -1, :]  # [batch, model_dim]
            action = actions[:, -1, :] if actions.dim() == 3 else actions
            
            state_delta, uncertainty = self.state_predictor(current_state, action)
            next_state = current_state + state_delta
            
            outputs.update({
                'predicted_next_state': next_state,
                'state_uncertainty': uncertainty,
                'state_delta': state_delta
            })
        
        if return_attention:
            outputs['attention_weights'] = all_attention_weights
        
        return outputs
    
    def predict_sequence(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        num_steps: int
    ) -> Dict[str, torch.Tensor]:
        """
        Predict a sequence of future states given initial state and actions.
        
        Args:
            initial_state: Initial state [batch, model_dim]
            actions: Action sequence [batch, num_steps, action_dim]
            num_steps: Number of steps to predict
        Returns:
            Dictionary containing predicted sequence and uncertainties
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        
        # Initialize sequence with initial state
        states = [initial_state]
        uncertainties = []
        
        current_state = initial_state
        
        for t in range(num_steps):
            action = actions[:, t, :]
            
            # Predict next state
            state_delta, uncertainty = self.state_predictor(current_state, action)
            next_state = current_state + state_delta
            
            states.append(next_state)
            uncertainties.append(uncertainty)
            
            current_state = next_state
        
        # Stack predictions
        predicted_states = torch.stack(states[1:], dim=1)  # [batch, num_steps, model_dim]
        predicted_uncertainties = torch.stack(uncertainties, dim=1)
        
        return {
            'predicted_states': predicted_states,
            'uncertainties': predicted_uncertainties,
            'final_state': current_state
        } 