"""
Advanced Attention Mechanisms for Multi-Modal Transformers.

This module implements state-of-the-art attention mechanisms including
Rotary Positional Embeddings (RoPE) for modern transformer architectures.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryPositionalEncoding(nn.Module):
    """
    Implements Rotary Positional Encoding (RoPE) as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute sine and cosine embeddings
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Pre-compute sine and cosine embeddings for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create embedding matrix [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
            
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        seq_len = x.shape[1]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[1]:
            self._build_cache(seq_len)
        
        # Get cached values for current sequence length
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        
        # Apply rotary embedding
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Ensure cos and sin have correct shape - take only the first half 
        cos_half = cos[..., :x1.shape[-1]]
        sin_half = sin[..., :x1.shape[-1]]
        
        return torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half
        ], dim=-1)


class MultiModalAttention(nn.Module):
    """
    Multi-head attention module with integrated support for RoPE and
    multi-modal input handling.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        scale_factor: Optional[float] = None
    ):
        super().__init__()
        
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = scale_factor or (self.head_dim ** -0.5)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(model_dim, model_dim, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_encoder: Optional[RotaryPositionalEncoding] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-modal attention.
        
        Args:
            x: Input tensor [batch, seq_len, model_dim]
            attention_mask: Optional mask [batch, seq_len, seq_len]
            rope_encoder: Optional RoPE encoder for positional encoding
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE if provided
        if rope_encoder is not None:
            q = rope_encoder(q)
            k = rope_encoder(k)
        
        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (set masked positions to large negative value)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # Reshape to original dimensions: [batch, seq_len, model_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.model_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output, None


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for attending between different modalities.
    Useful for fusion of video, audio, and other modalities.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-modal attention forward pass.
        
        Args:
            query: Query tensor from one modality [batch, seq_len_q, query_dim]
            key_value: Key-value tensor from another modality [batch, seq_len_kv, key_value_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor with cross-modal attention applied
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, -1)
        
        return self.out_proj(attn_output)


class FlashAttention(nn.Module):
    """
    Memory-efficient attention implementation similar to FlashAttention.
    This is a simplified version for educational purposes.
    """
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Memory-efficient attention computation.
        
        Note: This is a simplified implementation. Real FlashAttention
        requires custom CUDA kernels for optimal performance.
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V in one go
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch's scaled_dot_product_attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback to standard attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.model_dim)
        
        return self.out_proj(attn_output) 