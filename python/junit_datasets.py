import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class JUnitDataset(Dataset):
    """Dataset for real JUnit test data"""
    def __init__(self, csv_path):
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
        """Normalize data to [-1, 1] range for better GAN training"""
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
