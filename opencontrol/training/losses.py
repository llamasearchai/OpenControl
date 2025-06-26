"""
Advanced Loss Functions for Multi-Modal World Model Training.

This module implements sophisticated loss functions that handle multiple modalities
with proper weighting, uncertainty estimation, and regularization terms.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from opencontrol.cli.commands import OpenControlConfig


class MultiModalLoss(nn.Module):
    """Base class for multi-modal loss functions."""
    
    def __init__(self, config: OpenControlConfig):
        super().__init__()
        self.config = config
        self.model_config = config.model
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class WorldModelLoss(MultiModalLoss):
    """
    Comprehensive loss function for world model training.
    
    This loss combines:
    - Reconstruction losses for each modality
    - Uncertainty-aware losses when available
    - Regularization terms
    - Temporal consistency losses
    """
    
    def __init__(self, config: OpenControlConfig):
        super().__init__(config)
        
        # Loss weights for different modalities
        self.loss_weights = {
            'video': 1.0,
            'audio': 0.5,
            'actions': 2.0,  # Higher weight for actions as they're critical for control
            'proprioception': 1.5,
            'text': 0.3
        }
        
        # Perceptual loss for video (if available)
        self.use_perceptual_loss = True
        try:
            from lpips import LPIPS
            self.lpips_loss = LPIPS(net='alex').eval()
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
        except ImportError:
            self.use_perceptual_loss = False
            
        # Temporal consistency weight
        self.temporal_weight = 0.1
        
        # Uncertainty loss weight
        self.uncertainty_weight = 0.01

    def video_loss(self, pred: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute video reconstruction loss with optional perceptual component.
        
        Args:
            pred: Predicted video frames [B, T, C, H, W]
            target: Target video frames [B, T, C, H, W]
            uncertainty: Predicted uncertainty [B, T, 1] (optional)
        """
        # Reshape for loss computation
        B, T, C, H, W = pred.shape
        pred_flat = pred.view(B * T, C, H, W)
        target_flat = target.view(B * T, C, H, W)
        
        # MSE loss
        mse_loss = F.mse_loss(pred_flat, target_flat, reduction='none')
        mse_loss = mse_loss.view(B, T, -1).mean(dim=-1)  # [B, T]
        
        # Perceptual loss (LPIPS)
        perceptual_loss = 0.0
        if self.use_perceptual_loss and pred_flat.shape[1] == 3:  # RGB images
            # Normalize to [-1, 1] for LPIPS
            pred_norm = pred_flat * 2.0 - 1.0
            target_norm = target_flat * 2.0 - 1.0
            
            # Compute LPIPS in chunks to avoid memory issues
            chunk_size = 16
            lpips_values = []
            for i in range(0, pred_norm.shape[0], chunk_size):
                chunk_pred = pred_norm[i:i+chunk_size]
                chunk_target = target_norm[i:i+chunk_size]
                with torch.no_grad():
                    lpips_chunk = self.lpips_loss(chunk_pred, chunk_target).squeeze()
                lpips_values.append(lpips_chunk)
            
            perceptual_loss = torch.cat(lpips_values).view(B, T)
        
        # Combine losses
        total_loss = mse_loss + 0.1 * perceptual_loss
        
        # Uncertainty weighting
        if uncertainty is not None:
            # Uncertainty should be log variance
            precision = torch.exp(-uncertainty.squeeze(-1))  # [B, T]
            total_loss = precision * total_loss + uncertainty.squeeze(-1)
        
        return total_loss.mean()

    def audio_loss(self, pred: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute audio reconstruction loss.
        
        Args:
            pred: Predicted audio [B, T, L]
            target: Target audio [B, T, L]
            uncertainty: Predicted uncertainty [B, T, 1] (optional)
        """
        # L1 loss works better for audio
        l1_loss = F.l1_loss(pred, target, reduction='none')
        l1_loss = l1_loss.mean(dim=-1)  # [B, T]
        
        # Spectral loss (optional)
        spectral_loss = 0.0
        if pred.shape[-1] > 1000:  # Only for longer sequences
            # Compute spectrograms
            pred_spec = torch.stft(pred.view(-1, pred.shape[-1]), n_fft=512, hop_length=256, return_complex=True)
            target_spec = torch.stft(target.view(-1, target.shape[-1]), n_fft=512, hop_length=256, return_complex=True)
            
            # Magnitude loss
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            spectral_loss = F.mse_loss(pred_mag, target_mag, reduction='none')
            spectral_loss = spectral_loss.mean(dim=(-2, -1)).view(pred.shape[0], pred.shape[1])
        
        total_loss = l1_loss + 0.1 * spectral_loss
        
        # Uncertainty weighting
        if uncertainty is not None:
            precision = torch.exp(-uncertainty.squeeze(-1))
            total_loss = precision * total_loss + uncertainty.squeeze(-1)
        
        return total_loss.mean()

    def action_loss(self, pred: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute action prediction loss.
        
        Args:
            pred: Predicted actions [B, T, A]
            target: Target actions [B, T, A]
            uncertainty: Predicted uncertainty [B, T, 1] (optional)
        """
        # MSE loss for actions
        mse_loss = F.mse_loss(pred, target, reduction='none')
        mse_loss = mse_loss.mean(dim=-1)  # [B, T]
        
        # Uncertainty weighting
        if uncertainty is not None:
            precision = torch.exp(-uncertainty.squeeze(-1))
            mse_loss = precision * mse_loss + uncertainty.squeeze(-1)
        
        return mse_loss.mean()

    def proprioception_loss(self, pred: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute proprioception prediction loss.
        
        Args:
            pred: Predicted proprioception [B, T, P]
            target: Target proprioception [B, T, P]
            uncertainty: Predicted uncertainty [B, T, 1] (optional)
        """
        # Smooth L1 loss for proprioception (robust to outliers)
        smooth_l1_loss = F.smooth_l1_loss(pred, target, reduction='none')
        smooth_l1_loss = smooth_l1_loss.mean(dim=-1)  # [B, T]
        
        # Uncertainty weighting
        if uncertainty is not None:
            precision = torch.exp(-uncertainty.squeeze(-1))
            smooth_l1_loss = precision * smooth_l1_loss + uncertainty.squeeze(-1)
        
        return smooth_l1_loss.mean()

    def text_loss(self, pred: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute text prediction loss (cross-entropy for tokens).
        
        Args:
            pred: Predicted logits [B, T, V]
            target: Target token ids [B, T]
            uncertainty: Predicted uncertainty [B, T, 1] (optional)
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(pred.view(-1, pred.shape[-1]), target.view(-1), reduction='none')
        ce_loss = ce_loss.view(target.shape[0], target.shape[1])  # [B, T]
        
        # Uncertainty weighting
        if uncertainty is not None:
            precision = torch.exp(-uncertainty.squeeze(-1))
            ce_loss = precision * ce_loss + uncertainty.squeeze(-1)
        
        return ce_loss.mean()

    def temporal_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal consistency loss to encourage smooth predictions.
        
        Args:
            predictions: Dictionary of predicted sequences
        """
        total_loss = 0.0
        count = 0
        
        for modality, pred in predictions.items():
            if pred.shape[1] > 1:  # Only if we have temporal dimension
                # Compute differences between consecutive frames
                diff = pred[:, 1:] - pred[:, :-1]
                # L2 norm of differences
                consistency_loss = torch.norm(diff.view(diff.shape[0], diff.shape[1], -1), dim=-1)
                total_loss += consistency_loss.mean()
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)

    def forward(self, outputs: Dict, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the total loss for world model training.
        
        Args:
            outputs: Model outputs containing predictions and uncertainties
            targets: Ground truth targets for each modality
        """
        predictions = outputs.get('predictions', {})
        uncertainties = outputs.get('uncertainty_estimates', {})
        
        total_loss = 0.0
        loss_components = {}
        
        # Compute losses for each modality
        for modality in predictions:
            if modality not in targets:
                continue
                
            pred = predictions[modality]
            target = targets[modality]
            uncertainty = uncertainties.get(modality) if uncertainties else None
            weight = self.loss_weights.get(modality, 1.0)
            
            if modality == 'video':
                loss = self.video_loss(pred, target, uncertainty)
            elif modality == 'audio':
                loss = self.audio_loss(pred, target, uncertainty)
            elif modality == 'actions':
                loss = self.action_loss(pred, target, uncertainty)
            elif modality == 'proprioception':
                loss = self.proprioception_loss(pred, target, uncertainty)
            elif modality == 'text':
                loss = self.text_loss(pred, target, uncertainty)
            else:
                # Default MSE loss
                loss = F.mse_loss(pred, target)
            
            weighted_loss = weight * loss
            total_loss += weighted_loss
            loss_components[f'{modality}_loss'] = loss.item()
        
        # Add temporal consistency loss
        if self.temporal_weight > 0:
            temporal_loss = self.temporal_consistency_loss(predictions)
            total_loss += self.temporal_weight * temporal_loss
            loss_components['temporal_loss'] = temporal_loss.item()
        
        # Add uncertainty regularization
        if uncertainties and self.uncertainty_weight > 0:
            uncertainty_reg = 0.0
            for modality, uncertainty in uncertainties.items():
                # Encourage reasonable uncertainty values (not too high or low)
                uncertainty_reg += torch.mean(torch.abs(uncertainty))
            
            total_loss += self.uncertainty_weight * uncertainty_reg
            loss_components['uncertainty_reg'] = uncertainty_reg.item()
        
        # Store loss components for logging
        if hasattr(self, 'loss_components'):
            self.loss_components = loss_components
        
        return total_loss


class ContrastiveLoss(MultiModalLoss):
    """
    Contrastive loss for learning aligned representations across modalities.
    
    This loss encourages representations from the same time step to be similar
    across different modalities, while pushing apart representations from
    different time steps.
    """
    
    def __init__(self, config: OpenControlConfig, temperature: float = 0.07):
        super().__init__(config)
        self.temperature = temperature
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss between modalities.
        
        Args:
            predictions: Dictionary of predicted features for each modality
            targets: Dictionary of target features (not used in contrastive loss)
        """
        if len(predictions) < 2:
            return torch.tensor(0.0, device=list(predictions.values())[0].device)
        
        # Get all modality features
        modalities = list(predictions.keys())
        features = []
        
        for modality in modalities:
            feat = predictions[modality]
            # Pool over spatial/temporal dimensions to get fixed-size features
            if len(feat.shape) > 3:
                feat = feat.mean(dim=tuple(range(2, len(feat.shape))))
            features.append(feat)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Compute contrastive loss between all pairs of modalities
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat1 = F.normalize(features[i], p=2, dim=-1)  # [B, T, D]
                feat2 = F.normalize(features[j], p=2, dim=-1)  # [B, T, D]
                
                B, T, D = feat1.shape
                
                # Reshape to [B*T, D]
                feat1_flat = feat1.view(B * T, D)
                feat2_flat = feat2.view(B * T, D)
                
                # Compute similarity matrix
                sim_matrix = torch.matmul(feat1_flat, feat2_flat.T) / self.temperature
                
                # Positive pairs are on the diagonal
                labels = torch.arange(B * T, device=feat1.device)
                
                # Contrastive loss (InfoNCE)
                loss = F.cross_entropy(sim_matrix, labels)
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class AdvancedWorldModelLoss(MultiModalLoss):
    """
    Advanced loss that combines reconstruction, contrastive, and additional terms.
    """
    
    def __init__(self, config: OpenControlConfig):
        super().__init__(config)
        self.reconstruction_loss = WorldModelLoss(config)
        self.contrastive_loss = ContrastiveLoss(config)
        self.contrastive_weight = 0.1
        
    def forward(self, outputs: Dict, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined loss."""
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(outputs, targets)
        
        # Contrastive loss
        predictions = outputs.get('predictions', {})
        contrast_loss = self.contrastive_loss(predictions, targets)
        
        total_loss = recon_loss + self.contrastive_weight * contrast_loss
        
        return total_loss 