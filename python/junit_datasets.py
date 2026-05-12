"""
JUnit Test Dataset Handling

This module provides PyTorch Dataset and DataLoader utilities for loading
and preprocessing JUnit test data for WGAN-GP training.

Features:
    - Loads CSV data with automatic NaN/Inf filtering
    - Normalizes test values to [-1, 1] range for stable GAN training
    - SAVES min/max statistics for reproducible denormalization
    - Provides statistics and feature analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class JUnitDataset(Dataset):
    """
    PyTorch Dataset for JUnit test data.
    
    Loads real JUnit test data from CSV and provides normalized tensors
    suitable for WGAN-GP training.
    
    CRITICAL: Saves normalization statistics (min/max) to JSON for reproducible
    denormalization of synthetic data.
    
    Attributes:
        raw_data: Original test data as numpy array
        normalized_data: [-1, 1] normalized tensors for training
        norm_stats: Dictionary with min/max for denormalization
    """
    
    def __init__(self, csv_path, stats_output_path=None, use_log_transform=True):
        """
        Initialize dataset.
        
        Args:
            csv_path (str): Path to CSV file with test data (no header, 3 columns: methods, branches, statements)
            stats_output_path (str, optional): Where to save normalization statistics JSON.
                                             If None, uses same directory as CSV with _norm_stats.json suffix
            use_log_transform (bool): Apply log1p before normalizing. Strongly recommended for
                                      count-type features with heavy right tails (skew>2).
                                      Transforms x → log(1+x) before [-1,1] normalization.
                                      Denormalization uses expm1(linear_denorm(x)).
        """
        df = pd.read_csv(csv_path, header=None)
        
        # Remove rows with NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Store raw test data (3 features: methods, branches, statements)
        self.raw_data = df.values
        self.use_log_transform = use_log_transform

        if use_log_transform:
            # log1p compresses heavy-tailed count distributions (skew 3-7 → ~1-2)
            # making them far more learnable for the WGAN Generator.
            log_data = np.log1p(df.values.astype(np.float64))
            self.data = torch.tensor(log_data, dtype=torch.float32)
            logger.info("log1p transform applied to all features")
        else:
            self.data = torch.tensor(df.values, dtype=torch.float32)
        
        # Compute normalization statistics BEFORE normalization (on possibly log-transformed data)
        self.norm_stats = self._compute_normalization_stats()
        self.norm_stats["log_transform"] = use_log_transform
        
        # Normalize data for GAN training
        self.normalized_data = self._normalize_data_with_stats()
        
        # Save statistics for later denormalization
        self._save_normalization_stats(csv_path, stats_output_path)

    def _compute_normalization_stats(self):
        """
        Compute min/max for each feature before normalization.
        
        Returns:
            Dict with min/max values for each feature column
        """
        stats = {
            'num_features': self.data.shape[1],
            'num_samples': len(self.data),
            'features': {}
        }
        
        for col_idx in range(self.data.shape[1]):
            col_data = self.data[:, col_idx]
            feature_min = col_data.min().item()
            feature_max = col_data.max().item()
            feature_range = feature_max - feature_min
            
            stats['features'][f'feature_{col_idx}'] = {
                'min': float(feature_min),
                'max': float(feature_max),
                'range': float(feature_range),
                'mean': float(col_data.mean().item()),
                'std': float(col_data.std().item()),
            }
            
            logger.info(f"Feature {col_idx}: min={feature_min:.2f}, max={feature_max:.2f}")
        
        return stats

    def _normalize_data_with_stats(self):
        """
        Normalize data to [-1, 1] range using computed statistics.
        
        Normalization formula: x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        
        This range is standard for GAN training as it helps with numerical stability
        and provides symmetric activation function ranges.
        
        Returns:
            torch.Tensor: Normalized data with any NaN/Inf replaced by zeros
        """
        normalized = self.data.clone()
        
        for col_idx in range(self.data.shape[1]):
            feature_stats = self.norm_stats['features'][f'feature_{col_idx}']
            feature_min = feature_stats['min']
            feature_max = feature_stats['max']
            feature_range = feature_max - feature_min
            
            # Avoid division by zero
            if feature_range == 0:
                feature_range = 1.0
            
            # Normalize to [-1, 1]
            normalized[:, col_idx] = 2 * (self.data[:, col_idx] - feature_min) / feature_range - 1
        
        # Remove any NaN or infinite values that might have been created
        normalized = torch.where(torch.isnan(normalized), torch.zeros_like(normalized), normalized)
        normalized = torch.where(torch.isinf(normalized), torch.zeros_like(normalized), normalized)
        
        return normalized

    def _save_normalization_stats(self, csv_path, stats_output_path=None):
        """
        Save normalization statistics to JSON file.
        
        These statistics are CRITICAL for denormalization of synthetic data.
        Without them, denormalization uses incorrect hardcoded ranges.
        """
        if stats_output_path is None:
            csv_path_obj = Path(csv_path)
            stats_output_path = csv_path_obj.parent / f"{csv_path_obj.stem}_norm_stats.json"
        else:
            stats_output_path = Path(stats_output_path)
        
        # Ensure parent directory exists
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(stats_output_path, 'w') as f:
                json.dump(self.norm_stats, f, indent=2)
            logger.info(f"✅ Saved normalization statistics to {stats_output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save normalization statistics: {e}")
            raise

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        return self.normalized_data[idx]

    def get_raw_data(self, idx):
        """Get raw unnormalized data"""
        return self.data[idx]

    def get_feature_statistics(self):
        """Get statistics about the test data for analysis"""
        return {
            'num_samples': len(self.data),
            'mean': self.data.mean(dim=0),
            'std': self.data.std(dim=0),
            'min': self.data.min(dim=0)[0],
            'max': self.data.max(dim=0)[0],
        }
    
    def denormalize(self, normalized_data):
        """
        Denormalize data from [-1, 1] back to original scale.
        
        If use_log_transform=True was set at construction, this also inverts
        the log1p transform via expm1: x_raw = expm1(linear_denorm(x_norm)).
        
        Args:
            normalized_data: Data in [-1, 1] range (numpy array or torch tensor)
        
        Returns:
            numpy array: Data in original raw scale
        """
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        denormalized = normalized_data.copy()
        
        for col_idx in range(denormalized.shape[1] if len(denormalized.shape) > 1 else 1):
            feature_stats = self.norm_stats['features'][f'feature_{col_idx}']
            feature_min = feature_stats['min']
            feature_max = feature_stats['max']
            feature_range = feature_max - feature_min
            
            if feature_range == 0:
                feature_range = 1.0
            
            # Inverse of normalization: x = (x_norm + 1) / 2 * range + min
            if len(denormalized.shape) == 1:
                denormalized = (denormalized + 1) / 2 * feature_range + feature_min
            else:
                denormalized[:, col_idx] = (normalized_data[:, col_idx] + 1) / 2 * feature_range + feature_min
        
        # If log1p was applied during loading, invert it now
        if getattr(self, 'use_log_transform', False) or self.norm_stats.get('log_transform', False):
            denormalized = np.expm1(np.clip(denormalized, 0, None))
        
        return denormalized


def load_normalization_stats(stats_path):
    """
    Load saved normalization statistics from JSON file.
    
    Args:
        stats_path: Path to JSON file with normalization statistics
    
    Returns:
        Dict with normalization statistics
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    logger.info(f"✅ Loaded normalization statistics from {stats_path}")
    return stats


def get_junit_data_loader(csv_path, batch_size=64, shuffle=True):
    """Load JUnit test data and create DataLoader"""
    dataset = JUnitDataset(csv_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader, dataset
