"""
Unified data loader for multiple datasets:
- Juliet (Java vulnerability benchmarks)
- OWASP Benchmark
- Defects4J (real bugs from open-source)
- Evosuite (automatically generated tests)
- Combined mode (mixture of all)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
from enum import Enum
import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


class DatasetType(Enum):
    JULIET = "juliet"
    OWASP = "owasp"
    DEFECTS4J = "defects4j"
    EVOSUITE = "evosuite"


class TestDataset(Dataset):
    """PyTorch Dataset for test code/features"""
    
    def __init__(self, data: np.ndarray, metadata: Optional[pd.DataFrame] = None):
        """
        Args:
            data: (N, D) array of feature vectors
            metadata: Optional DataFrame with test metadata (CWE, bug type, source, etc.)
        """
        self.data = torch.FloatTensor(data)
        self.metadata = metadata
        self.size = len(data)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict:
        item = {"features": self.data[idx]}
        
        if self.metadata is not None:
            for col in self.metadata.columns:
                item[col] = self.metadata.iloc[idx][col]
        
        return item


class UnifiedDataLoader:
    """Load and normalize data from multiple sources"""
    
    # Feature normalization constants (from JUnit/Java benchmarks)
    FEATURE_RANGES = {
        "methods": (1, 500),  # Number of methods
        "branches": (1, 1000),  # Code branches
        "statements": (1, 2000),  # Statements
    }
    
    def __init__(self, data_dir: Path = Path("data")):
        """
        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_juliet(self, path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load Juliet Java vulnerability benchmarks
        Expects CSV with columns: test_code, cwe, vulnerability_type, methods, branches, statements
        """
        if path is None:
            path = self.raw_dir / "juliet_java.csv"
        
        if not path.exists():
            logger.warning(f"Juliet dataset not found at {path}")
            return np.array([]).reshape(0, 3), pd.DataFrame()
        
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded Juliet: {len(df)} samples")
        
        # Extract numeric features
        features = self._extract_features(df)
        return features, df
    
    def load_owasp(self, path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load OWASP Benchmark data
        Expects CSV with test metadata and code metrics
        """
        if path is None:
            path = self.raw_dir / "owasp_benchmark.csv"
        
        if not path.exists():
            logger.warning(f"OWASP dataset not found at {path}")
            return np.array([]).reshape(0, 3), pd.DataFrame()
        
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded OWASP: {len(df)} samples")
        
        features = self._extract_features(df)
        return features, df
    
    def load_defects4j(self, path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load Defects4J real bugs
        Expects CSV with bug information and test metrics
        """
        if path is None:
            path = self.raw_dir / "defects4j_tests.csv"
        
        if not path.exists():
            logger.warning(f"Defects4J dataset not found at {path}")
            return np.array([]).reshape(0, 3), pd.DataFrame()
        
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded Defects4J: {len(df)} samples")
        
        features = self._extract_features(df)
        return features, df
    
    def load_evosuite(self, path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load EvoSuite generated tests
        Expects CSV with test characteristics
        """
        if path is None:
            path = self.raw_dir / "evosuite_generated.csv"
        
        if not path.exists():
            logger.warning(f"EvoSuite dataset not found at {path}")
            return np.array([]).reshape(0, 3), pd.DataFrame()
        
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded EvoSuite: {len(df)} samples")
        
        features = self._extract_features(df)
        return features, df
    
    def load_current_junit(self, path: Path = Path("python/data/junit_data_cleaned.csv")) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load current JUnit dataset (what we have now)
        """
        if not path.exists():
            logger.warning(f"JUnit dataset not found at {path}")
            return np.array([]).reshape(0, 3), pd.DataFrame()
        
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded JUnit: {len(df)} samples")
        
        features = self._extract_features(df)
        return features, df
    
    def _extract_features(self, df: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """
        Extract and normalize numeric features from DataFrame
        Args:
            df: Input DataFrame
            feature_cols: Columns to extract (defaults to methods, branches, statements)
        Returns:
            (N, D) normalized feature array
        """
        if feature_cols is None:
            feature_cols = ["methods", "branches", "statements"]
        
        # Use available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            # Try alternative column names
            col_mapping = {
                "methods": ["num_methods", "method_count", "methods_count"],
                "branches": ["num_branches", "branch_count", "branches_count"],
                "statements": ["num_statements", "statement_count", "statements_count"],
            }
            
            for target, alternatives in col_mapping.items():
                for alt in alternatives:
                    if alt in df.columns:
                        available_cols.append(alt)
                        break
        
        if not available_cols:
            # Generate synthetic features from any numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                available_cols = numeric_cols[:3]  # Use first 3 numeric columns
            else:
                # Fallback: generate random features
                logger.warning("No numeric columns found, generating synthetic features")
                return np.random.randn(len(df), 3)
        
        # Extract and pad/truncate to 3 features
        features = df[available_cols].fillna(0).values
        
        if features.shape[1] < 3:
            # Pad with random features
            pad = np.random.randn(len(features), 3 - features.shape[1])
            features = np.hstack([features, pad])
        elif features.shape[1] > 3:
            # Use first 3
            features = features[:, :3]
        
        # Normalize to [-1, 1] range
        features = self._normalize_features(features)
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range"""
        # Min-max normalization
        min_val = features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        
        # Avoid division by zero
        max_val = np.where(max_val == min_val, min_val + 1, max_val)
        
        normalized = 2 * (features - min_val) / (max_val - min_val) - 1
        
        # Clip to [-1, 1]
        normalized = np.clip(normalized, -1, 1)
        return normalized
    
    def load_combined(
        self,
        include_juliet: bool = True,
        include_owasp: bool = True,
        include_defects4j: bool = True,
        include_evosuite: bool = True,
        include_current_junit: bool = True,
        test_split: float = 0.15,
        val_split: float = 0.15,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load and combine multiple datasets
        Returns: train_features, val_features, test_features, combined_metadata
        """
        np.random.seed(seed)
        
        all_features = []
        all_metadata = []
        
        datasets_loaded = []
        
        # Load selected datasets
        if include_juliet:
            try:
                features, meta = self.load_juliet()
                if len(features) > 0:
                    all_features.append(features)
                    meta["source"] = "juliet"
                    all_metadata.append(meta)
                    datasets_loaded.append("Juliet")
            except Exception as e:
                logger.warning(f"Failed to load Juliet: {e}")
        
        if include_owasp:
            try:
                features, meta = self.load_owasp()
                if len(features) > 0:
                    all_features.append(features)
                    meta["source"] = "owasp"
                    all_metadata.append(meta)
                    datasets_loaded.append("OWASP")
            except Exception as e:
                logger.warning(f"Failed to load OWASP: {e}")
        
        if include_defects4j:
            try:
                features, meta = self.load_defects4j()
                if len(features) > 0:
                    all_features.append(features)
                    meta["source"] = "defects4j"
                    all_metadata.append(meta)
                    datasets_loaded.append("Defects4J")
            except Exception as e:
                logger.warning(f"Failed to load Defects4J: {e}")
        
        if include_evosuite:
            try:
                features, meta = self.load_evosuite()
                if len(features) > 0:
                    all_features.append(features)
                    meta["source"] = "evosuite"
                    all_metadata.append(meta)
                    datasets_loaded.append("EvoSuite")
            except Exception as e:
                logger.warning(f"Failed to load EvoSuite: {e}")
        
        if include_current_junit:
            try:
                features, meta = self.load_current_junit()
                if len(features) > 0:
                    all_features.append(features)
                    meta["source"] = "junit"
                    all_metadata.append(meta)
                    datasets_loaded.append("JUnit")
            except Exception as e:
                logger.warning(f"Failed to load current JUnit: {e}")
        
        # Combine
        if not all_features:
            logger.warning("No datasets loaded! Returning empty data.")
            return (np.array([]).reshape(0, 3), 
                    np.array([]).reshape(0, 3), 
                    np.array([]).reshape(0, 3), 
                    pd.DataFrame())
        
        combined_features = np.vstack(all_features)
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        
        logger.info(f"✅ Combined {len(datasets_loaded)} datasets: {', '.join(datasets_loaded)}")
        logger.info(f"   Total samples: {len(combined_features)}")
        
        # Split data
        n = len(combined_features)
        test_size = int(n * test_split)
        val_size = int(n * val_split)
        train_size = n - test_size - val_size
        
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        train_features = combined_features[train_idx]
        val_features = combined_features[val_idx]
        test_features = combined_features[test_idx]
        
        logger.info(f"📊 Data split:")
        logger.info(f"   Train: {len(train_features)} ({len(train_features)/n*100:.1f}%)")
        logger.info(f"   Val:   {len(val_features)} ({len(val_features)/n*100:.1f}%)")
        logger.info(f"   Test:  {len(test_features)} ({len(test_features)/n*100:.1f}%)")
        
        return train_features, val_features, test_features, combined_metadata
    
    def create_dataloaders(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        test_features: np.ndarray,
        batch_size: int = 32,
        train_metadata: Optional[pd.DataFrame] = None,
        val_metadata: Optional[pd.DataFrame] = None,
        test_metadata: Optional[pd.DataFrame] = None,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders
        """
        train_dataset = TestDataset(train_features, train_metadata)
        val_dataset = TestDataset(val_features, val_metadata)
        test_dataset = TestDataset(test_features, test_metadata)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("🔄 Testing unified data loader...")
    
    loader = UnifiedDataLoader()
    
    # Try loading combined data (will work with whatever is available)
    try:
        train, val, test, meta = loader.load_combined(
            include_juliet=False,
            include_owasp=False,
            include_defects4j=False,
            include_evosuite=False,
            include_current_junit=True,  # Load what we have
        )
        
        print(f"✅ Loaded {len(train) + len(val) + len(test)} samples")
        print(f"   Feature shape: {train[0].shape if len(train) > 0 else 'N/A'}")
        
        # Create DataLoaders
        train_loader, val_loader, test_loader = loader.create_dataloaders(
            train, val, test, batch_size=32
        )
        
        print(f"✅ DataLoaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
