"""
Comprehensive Evaluation Metrics Suite for World Models.

This module implements a wide range of evaluation metrics for assessing
world model performance across different modalities and tasks.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import time
from pathlib import Path
import json

if TYPE_CHECKING:
    from opencontrol.core.world_model import OpenControlWorldModel
    from opencontrol.data.dataset_manager import MultiModalDatasetManager

from opencontrol.cli.commands import OpenControlConfig


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation suite for world models.
    
    This class implements various metrics for evaluating world model performance:
    - Video prediction metrics (FVD, LPIPS, SSIM, PSNR)
    - Audio prediction metrics (spectral distance, MCD)
    - Action prediction metrics (MSE, trajectory similarity)
    - Multi-modal consistency metrics
    - Temporal coherence metrics
    """
    
    def __init__(
        self,
        world_model: "OpenControlWorldModel",
        dataset_manager: "MultiModalDatasetManager", 
        config: OpenControlConfig,
        logger: logging.Logger
    ):
        self.world_model = world_model
        self.dataset_manager = dataset_manager
        self.config = config
        self.logger = logger
        self.device = next(world_model.parameters()).device
        
        # Initialize metric computers
        self.video_metrics = VideoMetrics(config)
        self.audio_metrics = AudioMetrics(config)
        self.action_metrics = ActionMetrics(config)
        self.multimodal_metrics = MultiModalMetrics(config)
        
        # Results storage
        self.evaluation_results = {}
        
    async def run_comprehensive_evaluation(
        self,
        progress_callback: Optional[callable] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all metrics.
        
        Args:
            progress_callback: Optional callback for progress updates
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info("Starting comprehensive evaluation")
        start_time = time.time()
        
        results = {}
        
        # Get evaluation dataset
        eval_loader = self.dataset_manager.get_val_loader()
        
        # Collect predictions and ground truth
        predictions, ground_truth = await self._collect_predictions(eval_loader, progress_callback)
        
        # Compute metrics for each modality
        if 'video' in predictions:
            self.logger.info("Computing video metrics")
            results['video'] = await self.video_metrics.compute_metrics(
                predictions['video'], ground_truth['video']
            )
            
        if 'audio' in predictions:
            self.logger.info("Computing audio metrics")
            results['audio'] = await self.audio_metrics.compute_metrics(
                predictions['audio'], ground_truth['audio']
            )
            
        if 'actions' in predictions:
            self.logger.info("Computing action metrics")
            results['actions'] = await self.action_metrics.compute_metrics(
                predictions['actions'], ground_truth['actions']
            )
            
        if 'proprioception' in predictions:
            self.logger.info("Computing proprioception metrics")
            results['proprioception'] = await self.action_metrics.compute_metrics(
                predictions['proprioception'], ground_truth['proprioception']
            )
        
        # Compute multi-modal metrics
        self.logger.info("Computing multi-modal metrics")
        results['multimodal'] = await self.multimodal_metrics.compute_metrics(
            predictions, ground_truth
        )
        
        # Compute temporal metrics
        self.logger.info("Computing temporal metrics")
        results['temporal'] = await self._compute_temporal_metrics(predictions, ground_truth)
        
        # Overall summary
        results['summary'] = self._compute_summary_metrics(results)
        
        evaluation_time = time.time() - start_time
        results['meta'] = {
            'evaluation_time': evaluation_time,
            'timestamp': time.time(),
            'num_samples': len(predictions[list(predictions.keys())[0]]),
            'config': self.config.__dict__
        }
        
        self.evaluation_results = results
        
        if save_results:
            await self._save_results(results)
            
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        return results
    
    async def _collect_predictions(
        self, 
        eval_loader, 
        progress_callback: Optional[callable] = None
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Collect model predictions and ground truth from evaluation set."""
        predictions = {}
        ground_truth = {}
        
        self.world_model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get model predictions
                outputs = self.world_model(batch, prediction_horizon=1)
                
                # Store predictions and ground truth
                for modality, pred in outputs.predictions.items():
                    if modality not in predictions:
                        predictions[modality] = []
                        ground_truth[modality] = []
                    
                    predictions[modality].append(pred.cpu())
                    if modality in batch:
                        ground_truth[modality].append(batch[modality].cpu())
                
                if progress_callback:
                    progress = (batch_idx + 1) / len(eval_loader) * 100
                    progress_callback(progress)
                    
                # Limit evaluation size for efficiency
                if batch_idx >= 100:  # Evaluate on first 100 batches
                    break
        
        # Concatenate all predictions
        for modality in predictions:
            predictions[modality] = torch.cat(predictions[modality], dim=0)
            if modality in ground_truth:
                ground_truth[modality] = torch.cat(ground_truth[modality], dim=0)
        
        return predictions, ground_truth
    
    async def _compute_temporal_metrics(
        self, 
        predictions: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute temporal coherence metrics."""
        temporal_metrics = {}
        
        for modality in predictions:
            if modality in ground_truth and predictions[modality].shape[1] > 1:
                pred = predictions[modality]
                gt = ground_truth[modality]
                
                # Temporal consistency (smoothness)
                pred_diff = pred[:, 1:] - pred[:, :-1]
                gt_diff = gt[:, 1:] - gt[:, :-1]
                
                # L2 distance between consecutive frames
                pred_smoothness = torch.norm(pred_diff.view(pred_diff.shape[0], pred_diff.shape[1], -1), dim=-1).mean()
                gt_smoothness = torch.norm(gt_diff.view(gt_diff.shape[0], gt_diff.shape[1], -1), dim=-1).mean()
                
                temporal_metrics[f'{modality}_smoothness_ratio'] = (pred_smoothness / gt_smoothness).item()
                temporal_metrics[f'{modality}_temporal_mse'] = torch.nn.functional.mse_loss(pred_diff, gt_diff).item()
        
        return temporal_metrics
    
    def _compute_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall summary metrics."""
        summary = {}
        
        # Average MSE across modalities
        mse_values = []
        for modality_results in results.values():
            if isinstance(modality_results, dict) and 'mse' in modality_results:
                mse_values.append(modality_results['mse'])
        
        if mse_values:
            summary['average_mse'] = np.mean(mse_values)
        
        # Video-specific summary
        if 'video' in results:
            summary['video_quality_score'] = (
                (1.0 - results['video'].get('lpips', 1.0)) * 0.4 +
                results['video'].get('ssim', 0.0) * 0.3 +
                (1.0 - results['video'].get('fvd_normalized', 1.0)) * 0.3
            )
        
        # Action prediction accuracy
        if 'actions' in results:
            summary['action_prediction_score'] = 1.0 / (1.0 + results['actions'].get('mse', float('inf')))
        
        # Overall score (weighted combination)
        score_components = []
        if 'video_quality_score' in summary:
            score_components.append(summary['video_quality_score'] * 0.4)
        if 'action_prediction_score' in summary:
            score_components.append(summary['action_prediction_score'] * 0.6)
        
        if score_components:
            summary['overall_score'] = np.mean(score_components)
        
        return summary
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to disk."""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"evaluation_{timestamp}.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


class VideoMetrics:
    """Video prediction evaluation metrics."""
    
    def __init__(self, config: OpenControlConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize perceptual metrics
        self.lpips_loss = None
        try:
            from lpips import LPIPS
            self.lpips_loss = LPIPS(net='alex').to(self.device).eval()
        except ImportError:
            pass
    
    async def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Compute video prediction metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = torch.nn.functional.mse_loss(predictions, ground_truth).item()
        metrics['mae'] = torch.nn.functional.l1_loss(predictions, ground_truth).item()
        
        # PSNR
        mse = torch.nn.functional.mse_loss(predictions, ground_truth)
        metrics['psnr'] = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
        
        # SSIM (simplified implementation)
        metrics['ssim'] = self._compute_ssim(predictions, ground_truth)
        
        # LPIPS (perceptual similarity)
        if self.lpips_loss is not None:
            metrics['lpips'] = self._compute_lpips(predictions, ground_truth)
        
        # FVD (Fréchet Video Distance) - simplified
        metrics['fvd'] = self._compute_fvd(predictions, ground_truth)
        
        return metrics
    
    def _compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Structural Similarity Index (simplified)."""
        # Simplified SSIM computation
        mu1 = torch.mean(pred)
        mu2 = torch.mean(gt)
        
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(gt)
        sigma12 = torch.mean((pred - mu1) * (gt - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim.item()
    
    def _compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute LPIPS perceptual distance."""
        # Reshape for LPIPS computation
        B, T, C, H, W = pred.shape
        pred_flat = pred.view(B * T, C, H, W)
        gt_flat = gt.view(B * T, C, H, W)
        
        # Normalize to [-1, 1]
        pred_norm = pred_flat * 2.0 - 1.0
        gt_norm = gt_flat * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_values = self.lpips_loss(pred_norm, gt_norm)
        
        return lpips_values.mean().item()
    
    def _compute_fvd(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute simplified Fréchet Video Distance."""
        # Simplified FVD - in practice, this would use I3D features
        pred_features = torch.mean(pred.view(pred.shape[0], -1), dim=1)
        gt_features = torch.mean(gt.view(gt.shape[0], -1), dim=1)
        
        # Compute means and covariances
        mu1 = torch.mean(pred_features, dim=0)
        mu2 = torch.mean(gt_features, dim=0)
        
        sigma1 = torch.cov(pred_features.T)
        sigma2 = torch.cov(gt_features.T)
        
        # Simplified FVD computation
        fvd = torch.norm(mu1 - mu2)**2 + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 @ sigma2))
        return fvd.item()


class AudioMetrics:
    """Audio prediction evaluation metrics."""
    
    def __init__(self, config: OpenControlConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Compute audio prediction metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = torch.nn.functional.mse_loss(predictions, ground_truth).item()
        metrics['mae'] = torch.nn.functional.l1_loss(predictions, ground_truth).item()
        
        # Spectral distance
        metrics['spectral_distance'] = self._compute_spectral_distance(predictions, ground_truth)
        
        # Signal-to-noise ratio
        metrics['snr'] = self._compute_snr(predictions, ground_truth)
        
        return metrics
    
    def _compute_spectral_distance(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute spectral distance between predicted and ground truth audio."""
        # Compute spectrograms
        pred_spec = torch.stft(pred.view(-1, pred.shape[-1]), n_fft=512, hop_length=256, return_complex=True)
        gt_spec = torch.stft(gt.view(-1, gt.shape[-1]), n_fft=512, hop_length=256, return_complex=True)
        
        # Magnitude spectrograms
        pred_mag = torch.abs(pred_spec)
        gt_mag = torch.abs(gt_spec)
        
        # L2 distance
        spectral_distance = torch.nn.functional.mse_loss(pred_mag, gt_mag)
        return spectral_distance.item()
    
    def _compute_snr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute signal-to-noise ratio."""
        signal_power = torch.mean(gt ** 2)
        noise_power = torch.mean((pred - gt) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr.item()


class ActionMetrics:
    """Action and proprioception prediction metrics."""
    
    def __init__(self, config: OpenControlConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Compute action prediction metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = torch.nn.functional.mse_loss(predictions, ground_truth).item()
        metrics['mae'] = torch.nn.functional.l1_loss(predictions, ground_truth).item()
        
        # Per-dimension metrics
        dim_mse = torch.mean((predictions - ground_truth) ** 2, dim=(0, 1))
        metrics['max_dim_mse'] = torch.max(dim_mse).item()
        metrics['min_dim_mse'] = torch.min(dim_mse).item()
        
        # Trajectory similarity
        metrics['trajectory_similarity'] = self._compute_trajectory_similarity(predictions, ground_truth)
        
        # Velocity metrics
        if predictions.shape[1] > 1:
            pred_vel = predictions[:, 1:] - predictions[:, :-1]
            gt_vel = ground_truth[:, 1:] - ground_truth[:, :-1]
            metrics['velocity_mse'] = torch.nn.functional.mse_loss(pred_vel, gt_vel).item()
        
        return metrics
    
    def _compute_trajectory_similarity(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute trajectory similarity using dynamic time warping approximation."""
        # Simplified trajectory similarity using cosine similarity
        pred_flat = pred.view(pred.shape[0], -1)
        gt_flat = gt.view(gt.shape[0], -1)
        
        cos_sim = torch.nn.functional.cosine_similarity(pred_flat, gt_flat, dim=1)
        return torch.mean(cos_sim).item()


class MultiModalMetrics:
    """Multi-modal consistency and alignment metrics."""
    
    def __init__(self, config: OpenControlConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def compute_metrics(
        self, 
        predictions: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute multi-modal consistency metrics."""
        metrics = {}
        
        # Cross-modal consistency
        modalities = list(predictions.keys())
        
        if len(modalities) >= 2:
            # Compute pairwise consistency
            consistency_scores = []
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    mod1, mod2 = modalities[i], modalities[j]
                    consistency = self._compute_cross_modal_consistency(
                        predictions[mod1], predictions[mod2]
                    )
                    consistency_scores.append(consistency)
                    metrics[f'{mod1}_{mod2}_consistency'] = consistency
            
            metrics['average_consistency'] = np.mean(consistency_scores)
        
        # Temporal alignment across modalities
        if len(modalities) >= 2:
            alignment_scores = []
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    mod1, mod2 = modalities[i], modalities[j]
                    if predictions[mod1].shape[1] > 1 and predictions[mod2].shape[1] > 1:
                        alignment = self._compute_temporal_alignment(
                            predictions[mod1], predictions[mod2]
                        )
                        alignment_scores.append(alignment)
            
            if alignment_scores:
                metrics['average_temporal_alignment'] = np.mean(alignment_scores)
        
        return metrics
    
    def _compute_cross_modal_consistency(self, pred1: torch.Tensor, pred2: torch.Tensor) -> float:
        """Compute consistency between two modalities."""
        # Simplified consistency metric using correlation
        # In practice, this would use learned cross-modal embeddings
        
        # Flatten and normalize
        pred1_flat = pred1.view(pred1.shape[0], -1)
        pred2_flat = pred2.view(pred2.shape[0], -1)
        
        pred1_norm = torch.nn.functional.normalize(pred1_flat, p=2, dim=1)
        pred2_norm = torch.nn.functional.normalize(pred2_flat, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(pred1_norm, pred2_norm, dim=1)
        return torch.mean(similarity).item()
    
    def _compute_temporal_alignment(self, pred1: torch.Tensor, pred2: torch.Tensor) -> float:
        """Compute temporal alignment between two modalities."""
        # Compute temporal derivatives
        pred1_diff = pred1[:, 1:] - pred1[:, :-1]
        pred2_diff = pred2[:, 1:] - pred2[:, :-1]
        
        # Flatten
        pred1_diff_flat = pred1_diff.view(pred1_diff.shape[0], pred1_diff.shape[1], -1)
        pred2_diff_flat = pred2_diff.view(pred2_diff.shape[0], pred2_diff.shape[1], -1)
        
        # Compute correlation of temporal changes
        pred1_norm = torch.nn.functional.normalize(pred1_diff_flat, p=2, dim=2)
        pred2_norm = torch.nn.functional.normalize(pred2_diff_flat, p=2, dim=2)
        
        alignment = torch.sum(pred1_norm * pred2_norm, dim=2).mean()
        return alignment.item() 