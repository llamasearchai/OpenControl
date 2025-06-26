"""
High-Performance Data Pipeline for Multi-Modal Training.

This module manages dataset creation, preprocessing, augmentation, and
efficient data loading with distributed training support.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
import numpy as np
import logging
from typing import Dict, List, Optional
import asyncio

from opencontrol.cli.commands import OpenControlConfig


class MultiModalEpisodeDataset(Dataset):
    """A PyTorch Dataset for loading individual multi-modal robot episodes."""
    
    def __init__(self, episode_paths: List[Path], config: OpenControlConfig):
        super().__init__()
        self.episode_paths = episode_paths
        self.seq_len = config.data.sequence_length
        self.config = config  # Store full config
        self.data_config = config.data  # Store data config for convenience
        self.logger = logging.getLogger(self.__class__.__name__)

    def __len__(self) -> int:
        return len(self.episode_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_path = self.episode_paths[idx]
        
        try:
            # Load episode data
            data = np.load(episode_path)
            episode_len = data['actions'].shape[0]
            
            # Handle sequence length
            if episode_len <= self.seq_len:
                start_idx = 0
                end_idx = episode_len
                # Pad if necessary
                pad_length = self.seq_len - episode_len
            else:
                start_idx = np.random.randint(0, episode_len - self.seq_len)
                end_idx = start_idx + self.seq_len
                pad_length = 0
            
            # Extract sequence
            batch = {}
            
            if 'video' in data:
                video_seq = data['video'][start_idx:end_idx]
                if pad_length > 0:
                    pad_shape = (pad_length,) + video_seq.shape[1:]
                    video_pad = np.zeros(pad_shape, dtype=video_seq.dtype)
                    video_seq = np.concatenate([video_seq, video_pad], axis=0)
                batch['video'] = torch.from_numpy(video_seq).float()
            
            if 'audio' in data:
                audio_seq = data['audio'][start_idx:end_idx]
                if pad_length > 0:
                    pad_shape = (pad_length,) + audio_seq.shape[1:]
                    audio_pad = np.zeros(pad_shape, dtype=audio_seq.dtype)
                    audio_seq = np.concatenate([audio_seq, audio_pad], axis=0)
                batch['audio'] = torch.from_numpy(audio_seq).float()
            
            if 'actions' in data:
                action_seq = data['actions'][start_idx:end_idx]
                if pad_length > 0:
                    pad_shape = (pad_length,) + action_seq.shape[1:]
                    action_pad = np.zeros(pad_shape, dtype=action_seq.dtype)
                    action_seq = np.concatenate([action_seq, action_pad], axis=0)
                batch['actions'] = torch.from_numpy(action_seq).float()
            
            if 'proprioception' in data:
                proprio_seq = data['proprioception'][start_idx:end_idx]
                if pad_length > 0:
                    pad_shape = (pad_length,) + proprio_seq.shape[1:]
                    proprio_pad = np.zeros(pad_shape, dtype=proprio_seq.dtype)
                    proprio_seq = np.concatenate([proprio_seq, proprio_pad], axis=0)
                batch['proprioception'] = torch.from_numpy(proprio_seq).float()
            
            # Apply augmentations if enabled
            if self.data_config.enable_augmentation:
                batch = self._apply_augmentations(batch)
            
            return batch
            
        except Exception as e:
            self.logger.warning(f"Failed to load episode {episode_path}: {e}")
            # Return a dummy sample
            return self._create_dummy_sample()

    def _apply_augmentations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentations."""
        if 'video' in batch and self.data_config.video_jitter > 0:
            # Add random jitter to video
            noise = torch.randn_like(batch['video']) * self.data_config.video_jitter
            batch['video'] = torch.clamp(batch['video'] + noise, 0, 1)
        
        if 'audio' in batch and self.data_config.audio_noise > 0:
            # Add random noise to audio
            noise = torch.randn_like(batch['audio']) * self.data_config.audio_noise
            batch['audio'] = batch['audio'] + noise
        
        return batch

    def _create_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Create a dummy sample for error cases."""
        from opencontrol.cli.commands import OpenControlConfig
        
        batch = {}
        
        if hasattr(self.data_config, 'video_resolution'):
            batch['video'] = torch.zeros(
                self.seq_len, 3, *self.data_config.video_resolution
            ).float()
        
        batch['audio'] = torch.zeros(
            self.seq_len, self.data_config.audio_sample_rate // 10
        ).float()
        
        # Use actual config dimensions if available
        if hasattr(self, 'config') and hasattr(self.config, 'model'):
            action_dim = getattr(self.config.model, 'action_dim', 8)
            proprio_dim = getattr(self.config.model, 'proprioception_dim', 16)
        else:
            action_dim = 8
            proprio_dim = 16
            
        batch['actions'] = torch.zeros(self.seq_len, action_dim).float()
        batch['proprioception'] = torch.zeros(self.seq_len, proprio_dim).float()
        
        return batch


class MultiModalDatasetManager:
    """Manages the creation and handling of dataloaders for training and validation."""
    
    def __init__(self, config: OpenControlConfig):
        self.config = config
        self.train_dataset: Optional[MultiModalEpisodeDataset] = None
        self.val_dataset: Optional[MultiModalEpisodeDataset] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self) -> None:
        """Discover data files and set up the datasets."""
        data_path = Path(self.config.data.data_path)
        
        if not data_path.exists():
            self.logger.warning(f"Data path {data_path} not found. Creating mock data.")
            await self._create_mock_data(data_path)

        # Find episode files
        episode_paths = sorted([p for p in data_path.glob("*.npz")])
        
        if not episode_paths:
            self.logger.warning(f"No .npz episode files found in {data_path}. Creating mock data.")
            await self._create_mock_data(data_path)
            episode_paths = sorted([p for p in data_path.glob("*.npz")])

        if not episode_paths:
            raise FileNotFoundError(f"Still no .npz episode files found in {data_path}")

        # Split into train/val
        split_idx = int(len(episode_paths) * 0.9)
        train_paths = episode_paths[:split_idx] if split_idx > 0 else episode_paths
        val_paths = episode_paths[split_idx:] if split_idx < len(episode_paths) else episode_paths[:1]
        
        self.train_dataset = MultiModalEpisodeDataset(train_paths, self.config)
        self.val_dataset = MultiModalEpisodeDataset(val_paths, self.config)
        
        self.logger.info(
            f"Initialized datasets: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val samples."
        )

    async def _create_mock_data(self, path: Path) -> None:
        """Create dummy data for testing if real data is not found."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating mock data in {path}")
        
        video_res = self.config.data.video_resolution
        audio_len = self.config.data.audio_sample_rate // 10
        action_dim = self.config.model.action_dim
        proprio_dim = self.config.model.proprioception_dim
        
        for i in range(20):  # Create 20 mock episodes
            episode_length = np.random.randint(30, 100)  # Variable length episodes
            
            episode_data = {
                'video': np.random.rand(episode_length, 3, *video_res).astype(np.float32),
                'audio': np.random.randn(episode_length, audio_len).astype(np.float32) * 0.1,
                'actions': np.random.randn(episode_length, action_dim).astype(np.float32) * 0.5,
                'proprioception': np.random.randn(episode_length, proprio_dim).astype(np.float32) * 0.1
            }
            
            # Add some structure to make it more realistic
            # Actions should be smoother
            for j in range(1, episode_length):
                episode_data['actions'][j] = (
                    0.9 * episode_data['actions'][j-1] + 
                    0.1 * episode_data['actions'][j]
                )
            
            # Proprioception should be correlated with actions (if dimensions match)
            if action_dim == proprio_dim:
                episode_data['proprioception'] = (
                    episode_data['actions'] + 
                    np.random.randn(episode_length, proprio_dim).astype(np.float32) * 0.05
                )
            else:
                # Just generate independent proprioception data
                episode_data['proprioception'] = np.random.randn(episode_length, proprio_dim).astype(np.float32) * 0.1
            
            # Save episode
            episode_file = path / f"mock_episode_{i:03d}.npz"
            np.savez(episode_file, **episode_data)
        
        self.logger.info(f"Created {i+1} mock episodes")

    def get_train_loader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not initialized. Call initialize() first.")
        
        sampler = None
        shuffle = True
        
        if self.config.infrastructure.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False
        
        batch_size = self.config.training.batch_size // self.config.infrastructure.world_size
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            sampler=sampler,
            drop_last=True,
            prefetch_factor=self.config.data.prefetch_factor if self.config.data.num_workers > 0 else None
        )

    def get_val_loader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        if self.val_dataset is None:
            raise RuntimeError("Dataset not initialized. Call initialize() first.")
        
        sampler = None
        if self.config.infrastructure.world_size > 1:
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        
        batch_size = self.config.training.batch_size // self.config.infrastructure.world_size
        
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            sampler=sampler,
            drop_last=False
        )

    async def get_statistics(self) -> Dict[str, int]:
        """Returns basic statistics about the dataset."""
        if not self.train_dataset:
            await self.initialize()
        
        return {
            'total_episodes': len(self.train_dataset) + len(self.val_dataset),
            'train_episodes': len(self.train_dataset),
            'val_episodes': len(self.val_dataset)
        } 