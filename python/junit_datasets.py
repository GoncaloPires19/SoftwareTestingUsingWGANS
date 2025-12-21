"""
JUnit Test Dataset Handling

This module provides PyTorch Dataset and DataLoader utilities for loading
and preprocessing JUnit test data for WGAN-GP training.

Features:
    - Loads CSV data with automatic NaN/Inf filtering
    - Normalizes test values to [-1, 1] range for stable GAN training
    - Provides statistics and feature analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class JUnitDataset(Dataset):
    """
    PyTorch Dataset for JUnit test data.
    
    Loads real JUnit test data from CSV and provides normalized tensors
    suitable for WGAN-GP training.
    
    Attributes:
        raw_data: Original test data as numpy array
        normalized_data: [-1, 1] normalized tensors for training
    """
    def __init__(self, csv_path):
        """
        Initialize dataset.
        
        Args:
            csv_path (str): Path to CSV file with test data (no header, columns as test features)
        """
        df = pd.read_csv(csv_path, header=None)
        
        # Remove rows with NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Store raw test data
        self.raw_data = df.values
        # Normalize data for better GAN training
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.normalized_data = self._normalize_data()

    def _normalize_data(self):
        """
        Normalize data to [-1, 1] range.
        
        Normalization formula: x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        
        This range is standard for GAN training as it helps with numerical stability
        and provides symmetric activation function ranges.
        
        Returns:
            torch.Tensor: Normalized data with any NaN/Inf replaced by zeros
        """
        data_min = self.data.min(dim=0)[0]
        data_max = self.data.max(dim=0)[0]
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        normalized = 2 * (self.data - data_min) / data_range - 1
        
        # Remove any NaN or infinite values that might have been created
        normalized = torch.where(torch.isnan(normalized), torch.zeros_like(normalized), normalized)
        normalized = torch.where(torch.isinf(normalized), torch.zeros_like(normalized), normalized)
        
        return normalized

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


def get_junit_data_loader(csv_path, batch_size=64, shuffle=True):
    """Load JUnit test data and create DataLoader"""
    dataset = JUnitDataset(csv_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader, dataset
