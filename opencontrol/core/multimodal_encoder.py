"""
Unified Multi-Modal Encoder for OpenControl.

This module handles the encoding of various input modalities (video, audio, text, 
actions, proprioception) into a unified token sequence for the transformer.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from opencontrol.cli.commands import OpenControlConfig


class MockEncoder(nn.Module):
    """Mock encoder for when external libraries are not available."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.embed_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all dimensions except batch
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.projection(x_flat).unsqueeze(1)  # Add sequence dimension
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """For compatibility with timm models."""
        return self.forward(x)


class VideoEncoder(nn.Module):
    """Encodes video frames using vision transformer."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        if TIMM_AVAILABLE:
            try:
                self.encoder = timm.create_model(
                    self.config.video_encoder,
                    pretrained=True,
                    num_classes=0  # Remove classification head
                )
                self.feature_dim = self.encoder.embed_dim
            except Exception as e:
                print(f"Warning: Could not load {self.config.video_encoder}, using mock encoder: {e}")
                self.encoder = MockEncoder(
                    np.prod(self.config.video_resolution) * 3,  # H*W*C
                    self.config.model_dim
                )
                self.feature_dim = self.config.model_dim
        else:
            print("Warning: timm not available, using mock video encoder")
            self.encoder = MockEncoder(
                np.prod(self.config.video_resolution) * 3,
                self.config.model_dim
            )
            self.feature_dim = self.config.model_dim
        
        # Projection to model dimension
        self.projection = nn.Linear(self.feature_dim, self.config.model_dim)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.
        
        Args:
            video: Video tensor of shape [batch, time, channels, height, width]
            
        Returns:
            Encoded features of shape [batch, time * num_patches, model_dim]
        """
        batch_size, time_steps, channels, height, width = video.shape
        
        # Reshape to process all frames at once
        video_flat = video.view(batch_size * time_steps, channels, height, width)
        
        # Extract features
        if hasattr(self.encoder, 'forward_features'):
            features = self.encoder.forward_features(video_flat)
        else:
            features = self.encoder(video_flat)
        
        # Handle different output formats
        if features.dim() == 2:
            # Global features, add sequence dimension
            features = features.unsqueeze(1)
        
        # Project to model dimension
        features = self.projection(features)
        
        # Reshape back to [batch, time * num_patches, model_dim]
        num_patches = features.shape[1]
        features = features.view(batch_size, time_steps * num_patches, -1)
        
        return features


class AudioEncoder(nn.Module):
    """Encodes audio using pre-trained models."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = AutoModel.from_pretrained(self.config.audio_encoder)
                # Get hidden size from config
                if hasattr(self.encoder.config, 'hidden_size'):
                    self.feature_dim = self.encoder.config.hidden_size
                else:
                    self.feature_dim = 768  # Default
            except Exception as e:
                print(f"Warning: Could not load {self.config.audio_encoder}, using mock encoder: {e}")
                self.encoder = MockEncoder(
                    self.config.audio_sample_rate // 10,  # Simplified audio length
                    self.config.model_dim
                )
                self.feature_dim = self.config.model_dim
        else:
            print("Warning: transformers not available, using mock audio encoder")
            self.encoder = MockEncoder(
                self.config.audio_sample_rate // 10,
                self.config.model_dim
            )
            self.feature_dim = self.config.model_dim
        
        # Projection layer
        self.projection = nn.Linear(self.feature_dim, self.config.model_dim)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio sequences.
        
        Args:
            audio: Audio tensor of shape [batch, time, audio_length]
            
        Returns:
            Encoded features of shape [batch, time * audio_seq_len, model_dim]
        """
        batch_size, time_steps, audio_length = audio.shape
        
        # Flatten time dimension
        audio_flat = audio.view(batch_size * time_steps, audio_length)
        
        # Encode audio
        if hasattr(self.encoder, 'forward'):
            if isinstance(self.encoder, MockEncoder):
                features = self.encoder(audio_flat)
            else:
                # For transformers models
                outputs = self.encoder(audio_flat.unsqueeze(-1))  # Add feature dim
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state
                else:
                    features = outputs[0]
        else:
            features = self.encoder(audio_flat)
        
        # Handle shape
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        # Project to model dimension
        features = self.projection(features)
        
        # Reshape back
        audio_seq_len = features.shape[1]
        features = features.view(batch_size, time_steps * audio_seq_len, -1)
        
        return features


class ActionEncoder(nn.Module):
    """Encodes action sequences."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        # Simple MLP encoder for actions
        self.encoder = nn.Sequential(
            nn.Linear(self.config.action_dim, self.config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.model_dim // 2, self.config.model_dim),
            nn.LayerNorm(self.config.model_dim)
        )
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode action sequences.
        
        Args:
            actions: Action tensor of shape [batch, time, action_dim]
            
        Returns:
            Encoded actions of shape [batch, time, model_dim]
        """
        return self.encoder(actions)


class ProprioceptionEncoder(nn.Module):
    """Encodes proprioceptive information."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        # MLP encoder for proprioception
        self.encoder = nn.Sequential(
            nn.Linear(self.config.proprioception_dim, self.config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.model_dim // 2, self.config.model_dim),
            nn.LayerNorm(self.config.model_dim)
        )
    
    def forward(self, proprioception: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive data.
        
        Args:
            proprioception: Proprioception tensor of shape [batch, time, proprio_dim]
            
        Returns:
            Encoded proprioception of shape [batch, time, model_dim]
        """
        return self.encoder(proprioception)


class TextEncoder(nn.Module):
    """Encodes text using embeddings."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        # Simple embedding layer for text
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.model_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.config.max_sequence_length, self.config.model_dim)
        )
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode text sequences.
        
        Args:
            text: Text token indices of shape [batch, seq_len]
            
        Returns:
            Encoded text of shape [batch, seq_len, model_dim]
        """
        seq_len = text.shape[1]
        embeddings = self.embedding(text)
        
        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        
        return embeddings


class MultiModalEncoder(nn.Module):
    """
    Unified encoder that processes multiple modalities and creates a unified
    token sequence for the transformer.
    """
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config.model
        
        # Initialize modality encoders
        self.video_encoder = VideoEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.action_encoder = ActionEncoder(config)
        self.proprioception_encoder = ProprioceptionEncoder(config)
        self.text_encoder = TextEncoder(config)
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(6, self.config.model_dim)  # 6 modalities
        
        # Modality-specific layer norms
        self.layer_norms = nn.ModuleDict({
            'video': nn.LayerNorm(self.config.model_dim),
            'audio': nn.LayerNorm(self.config.model_dim),
            'actions': nn.LayerNorm(self.config.model_dim),
            'proprioception': nn.LayerNorm(self.config.model_dim),
            'text': nn.LayerNorm(self.config.model_dim),
        })
        
        # Define modality IDs
        self.modality_ids = {
            'video': 0,
            'audio': 1,
            'actions': 2,
            'proprioception': 3,
            'text': 4,
            'tokens': 5  # For pre-tokenized inputs
        }
    
    def forward(
        self, 
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process multi-modal inputs into a unified token sequence.
        
        Args:
            inputs: Dictionary of input tensors with keys like 'video', 'audio', etc.
            
        Returns:
            Tuple of (token_sequence, modality_info)
        """
        all_tokens = []
        modality_info = {
            'sequence_lengths': {},
            'modality_positions': {},
            'total_length': 0
        }
        
        current_pos = 0
        
        # Process each modality
        for modality_name, tensor in inputs.items():
            if modality_name == 'tokens':
                # Pre-tokenized inputs (used for inference/generation)
                tokens = tensor
            else:
                # Encode the modality
                if modality_name == 'video' and tensor is not None:
                    tokens = self.video_encoder(tensor)
                elif modality_name == 'audio' and tensor is not None:
                    tokens = self.audio_encoder(tensor)
                elif modality_name == 'actions' and tensor is not None:
                    tokens = self.action_encoder(tensor)
                elif modality_name == 'proprioception' and tensor is not None:
                    tokens = self.proprioception_encoder(tensor)
                elif modality_name == 'text' and tensor is not None:
                    tokens = self.text_encoder(tensor)
                else:
                    continue  # Skip unknown or None modalities
                
                # Apply modality-specific layer norm
                if modality_name in self.layer_norms:
                    tokens = self.layer_norms[modality_name](tokens)
            
            # Add modality type embedding
            if modality_name in self.modality_ids:
                modality_embed = self.modality_embeddings(
                    torch.tensor(
                        self.modality_ids[modality_name],
                        device=tokens.device,
                        dtype=torch.long
                    )
                )
                tokens = tokens + modality_embed.unsqueeze(0).unsqueeze(0)
            
            # Store sequence information
            seq_len = tokens.shape[1]
            modality_info['sequence_lengths'][modality_name] = seq_len
            modality_info['modality_positions'][modality_name] = (current_pos, current_pos + seq_len)
            current_pos += seq_len
            
            all_tokens.append(tokens)
        
        if not all_tokens:
            raise ValueError("No valid input modalities provided")
        
        # Concatenate all tokens
        final_tokens = torch.cat(all_tokens, dim=1)
        modality_info['total_length'] = final_tokens.shape[1]
        
        return final_tokens, modality_info
    
    def encode_single_modality(
        self,
        modality_name: str,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single modality. Useful for inference and analysis.
        
        Args:
            modality_name: Name of the modality
            data: Input data tensor
            
        Returns:
            Encoded tokens
        """
        inputs = {modality_name: data}
        tokens, _ = self.forward(inputs)
        return tokens
    
    def get_modality_mask(
        self,
        modality_info: Dict[str, Any],
        target_modality: str
    ) -> torch.Tensor:
        """
        Create a mask for a specific modality within the token sequence.
        
        Args:
            modality_info: Modality information from forward pass
            target_modality: Name of the target modality
            
        Returns:
            Boolean mask tensor
        """
        total_length = modality_info['total_length']
        mask = torch.zeros(total_length, dtype=torch.bool)
        
        if target_modality in modality_info['modality_positions']:
            start_pos, end_pos = modality_info['modality_positions'][target_modality]
            mask[start_pos:end_pos] = True
        
        return mask 